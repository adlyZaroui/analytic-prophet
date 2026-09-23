#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <iostream>
#include <LBFGSB.h>
#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

// Column-for-column identical to Prophet's fourier_series: sin(order i) at
// column 2i, cos(order i) at 2i+1, with t_days measured from the Unix epoch.
// Must stay in lockstep with fourier_components() in customProphet.py --
// tests/test_prophet_agreement.py compares both against Prophet's own matrix.
Eigen::MatrixXd fourier_components(const Eigen::VectorXd& t_days, double period, int n) {
    // `2*pi/period` folded into one constant, matching
    // customProphet.fourier_components rather than Prophet's operation order.
    // Deliberate -- see README, "Where this deviates on purpose".
    Eigen::VectorXd orders = Eigen::VectorXd::LinSpaced(n, 1, n) * (2 * M_PI / period);
    Eigen::MatrixXd angles = t_days * orders.transpose();

    Eigen::MatrixXd result(t_days.size(), 2 * n);
    for (int i = 0; i < n; ++i) {
        result.col(2 * i) = angles.col(i).array().sin();
        result.col(2 * i + 1) = angles.col(i).array().cos();
    }
    return result;
}

// Parameter vector layout: [k, m, delta(S), beta(K), zeta], length 2 + S + K + 1.
// zeta = log(sigma_obs): liblbfgs has no box constraints, and optimizing the
// log enforces sigma_obs > 0 for free. Offsets are computed from S (and K,
// implied by the vector length) rather than hardcoded -- the literals this
// replaced (segment(2, 25), tail(size - 27)) would have silently absorbed the
// new trailing element into beta.
struct ModelParams {
    double k;
    double m;
    Eigen::VectorXd delta;
    Eigen::VectorXd beta;
    double zeta;
    double sigma_obs;   // exp(zeta), recovered once per evaluation
};

ModelParams extract_params(const Eigen::Ref<const Eigen::VectorXd>& params, int S, int K) {
    // S and K are passed rather than assumed: S comes from the changepoint
    // vector, K from the seasonality design matrix. Checking the length here
    // turns a downstream Eigen "invalid matrix product" abort into a
    // diagnosable error.
    if (params.size() != 2 + S + K + 1) {
        throw std::invalid_argument(
            "params has length " + std::to_string(params.size()) + ", but S=" +
            std::to_string(S) + " and K=" + std::to_string(K) + " require " +
            std::to_string(2 + S + K + 1) + ". The layout is "
            "[k, m, delta(S), beta(K), zeta], length 2 + S + K + 1.");
    }
    ModelParams p;
    p.k = params(0);
    p.m = params(1);
    p.delta = params.segment(2, S);
    p.beta = params.segment(2 + S, K);
    p.zeta = params(params.size() - 1);
    p.sigma_obs = std::exp(p.zeta);
    return p;
}

// The changepoint indicator. Depends only on t_scaled and the changepoint
// locations, so it is constant for a whole fit -- optimize() builds it once
// before the solver loop rather than on every evaluation (issue #28).
// >= , not > : Stan's get_changepoint_matrix uses t[i] >= t_change[j].
Eigen::MatrixXd changepoint_matrix(const Eigen::VectorXd& t_scaled_vec,
                                   const Eigen::VectorXd& change_points_vec) {
    return (t_scaled_vec.replicate(1, change_points_vec.size()).array()
            >= change_points_vec.transpose().replicate(t_scaled_vec.size(), 1).array())
           .cast<double>();
}

// Likewise constant for a whole fit. t_seasonality is days since the Unix
// epoch, so periods are in days -- 365.25 yearly, 7 weekly, 1 daily.
//
// One block per registered seasonality, concatenated left to right in the
// order given, which is the order the caller's `beta` is laid out in. Orders
// and periods are arguments, not constants: registering a second seasonality
// is the whole point of #16, and nothing here needs recompiling for it.
// `conditions` is T x n_components, one column per registered seasonality in
// the same order, holding 1 where the component applies and 0 where it does
// not. An empty matrix means no component is conditioned, which is the common
// case; carrying an all-ones matrix for it would be pure overhead, and
// carrying an index of which components have a condition would be a second
// thing to keep in step with the registry order.
Eigen::MatrixXd seasonality_matrix(const Eigen::VectorXd& t_seasonality_vec,
                                   const std::vector<int>& fourier_orders,
                                   const std::vector<double>& periods,
                                   const Eigen::Ref<const Eigen::MatrixXd>& conditions,
                                   const Eigen::Ref<const Eigen::MatrixXd>& data_columns) {
    if (fourier_orders.size() != periods.size()) {
        throw std::invalid_argument(
            "fourier_orders has " + std::to_string(fourier_orders.size()) +
            " entries but seasonality_periods has " + std::to_string(periods.size()) +
            "; there must be one period per seasonality");
    }

    // A negative order makes `2 * order` a negative block width and Eigen
    // aborts on the block assignment below; a non-positive or non-finite
    // period divides into 2*pi and silently fills the matrix with garbage or
    // NaN. Both are caller errors, and neither is visible in the result.
    for (std::size_t i = 0; i < fourier_orders.size(); ++i) {
        if (fourier_orders[i] <= 0) {
            throw std::invalid_argument(
                "fourier_orders[" + std::to_string(i) + "] is " +
                std::to_string(fourier_orders[i]) + "; every Fourier order must be "
                "positive (a component with no columns should be left out instead)");
        }
        if (!(periods[i] > 0.0) || !std::isfinite(periods[i])) {
            throw std::invalid_argument(
                "seasonality_periods[" + std::to_string(i) + "] is " +
                std::to_string(periods[i]) + "; every period must be positive and finite");
        }
    }

    const bool conditioned = conditions.size() != 0;
    if (conditioned &&
            (conditions.rows() != t_seasonality_vec.size() ||
             conditions.cols() != static_cast<Eigen::Index>(fourier_orders.size()))) {
        throw std::invalid_argument(
            "seasonality_conditions is " + std::to_string(conditions.rows()) + "x" +
            std::to_string(conditions.cols()) + " but there are " +
            std::to_string(t_seasonality_vec.size()) + " observations and " +
            std::to_string(fourier_orders.size()) + " seasonalities; it needs one "
            "column per seasonality, in the same order, or none at all");
    }

    if (data_columns.size() != 0 &&
            data_columns.rows() != t_seasonality_vec.size()) {
        throw std::invalid_argument(
            "data_columns has " + std::to_string(data_columns.rows()) +
            " rows but there are " + std::to_string(t_seasonality_vec.size()) +
            " observations");
    }

    int columns = 0;
    for (int order : fourier_orders) {
        columns += 2 * order;
    }
    // [fc] make_all_seasonality_features: holiday and extra-regressor columns
    // follow the seasonal ones. Both are data rather than anything derivable
    // from a period, so unlike the Fourier and changepoint blocks they are
    // passed in rather than built here -- there is nothing to recompute. They
    // arrive as one block because the objective does not distinguish them.
    const int n_data_columns = static_cast<int>(data_columns.cols());

    Eigen::MatrixXd x(t_seasonality_vec.size(), columns + n_data_columns);
    int offset = 0;
    for (std::size_t i = 0; i < fourier_orders.size(); ++i) {
        const int width = 2 * fourier_orders[i];
        Eigen::MatrixXd block =
            fourier_components(t_seasonality_vec, periods[i], fourier_orders[i]);
        // [fc] features[~df[condition_name]] = 0 -- the columns stay, so beta
        // keeps its width and only the excluded rows stop contributing.
        if (conditioned) {
            block.array().colwise() *= conditions.col(i).array();
        }
        x.block(0, offset, t_seasonality_vec.size(), width) = block;
        offset += width;
    }
    if (n_data_columns > 0) {
        x.block(0, offset, t_seasonality_vec.size(), n_data_columns) = data_columns;
    }
    return x;
}

// [stan] logistic_gamma: the piecewise offsets that keep the logistic curve
// continuous where the rate changes, and their Jacobian with respect to
// (k, m, delta). gamma[i] depends on every earlier gamma through m_pr, so the
// Jacobian is accumulated forward alongside gamma rather than written down.
void logistic_gamma_and_jacobian(double k, double m,
                                 const Eigen::VectorXd& delta,
                                 const Eigen::VectorXd& change_points,
                                 Eigen::VectorXd& gamma,
                                 Eigen::MatrixXd& dgamma) {
    const int S = static_cast<int>(delta.size());
    Eigen::VectorXd k_s(S + 1);
    k_s(0) = k;
    double running = k;
    for (int i = 0; i < S; ++i) {
        running += delta(i);
        k_s(i + 1) = running;
    }

    // d k_s[i] / d(k, m, delta): 1 for k, 0 for m, and delta_j enters k_s[i]
    // for every j < i
    Eigen::MatrixXd dk_s = Eigen::MatrixXd::Zero(S + 1, 2 + S);
    dk_s.col(0).setOnes();
    for (int i = 1; i <= S; ++i) {
        dk_s.row(i).segment(2, i).setOnes();
    }

    gamma.resize(S);
    dgamma.setZero(S, 2 + S);
    double m_pr = m;
    Eigen::VectorXd dm_pr = Eigen::VectorXd::Zero(2 + S);
    dm_pr(1) = 1.0;

    for (int i = 0; i < S; ++i) {
        const double c = 1.0 - k_s(i) / k_s(i + 1);
        const Eigen::VectorXd dc =
            -(dk_s.row(i).transpose() * k_s(i + 1) - dk_s.row(i + 1).transpose() * k_s(i))
            / (k_s(i + 1) * k_s(i + 1));

        gamma(i) = (change_points(i) - m_pr) * c;
        dgamma.row(i) = (-dm_pr * c + (change_points(i) - m_pr) * dc).transpose();

        m_pr += gamma(i);
        dm_pr += dgamma.row(i).transpose();
    }
}

// [stan] logistic_trend: cap .* inv_logit((k + A*delta) .* (t - (m + A*gamma))).
// Returns the trend and d(trend)/d(k, m, delta); the caller contracts the
// Jacobian with the residual, which is what lets the multiplicative multiplier
// be applied outside exactly as it is for the linear trend.
void logistic_trend_and_jacobian(double k, double m,
                                 const Eigen::VectorXd& delta,
                                 const Eigen::VectorXd& t_scaled,
                                 const Eigen::VectorXd& cap_scaled,
                                 const Eigen::MatrixXd& A,
                                 const Eigen::VectorXd& change_points,
                                 Eigen::VectorXd& trend,
                                 Eigen::MatrixXd& jacobian) {
    const int T = static_cast<int>(t_scaled.size());
    const int S = static_cast<int>(delta.size());

    Eigen::VectorXd gamma;
    Eigen::MatrixXd dgamma;
    logistic_gamma_and_jacobian(k, m, delta, change_points, gamma, dgamma);

    const Eigen::VectorXd rate = (k * Eigen::VectorXd::Ones(T) + A * delta);
    const Eigen::VectorXd offset = (m * Eigen::VectorXd::Ones(T) + A * gamma);
    const Eigen::VectorXd z = rate.array() * (t_scaled - offset).array();

    // exp(-|z|) form: the naive 1/(1+exp(-z)) overflows for z very negative
    Eigen::VectorXd sigmoid(T);
    for (int i = 0; i < T; ++i) {
        const double e = std::exp(-std::abs(z(i)));
        sigmoid(i) = z(i) >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
    }
    trend = cap_scaled.array() * sigmoid.array();

    Eigen::MatrixXd d_offset = A * dgamma;
    d_offset.col(1).array() += 1.0;
    Eigen::MatrixXd d_rate = Eigen::MatrixXd::Zero(T, 2 + S);
    d_rate.col(0).setOnes();
    d_rate.rightCols(S) = A;

    Eigen::MatrixXd dz = d_rate.array().colwise() * (t_scaled - offset).array();
    dz -= (d_offset.array().colwise() * rate.array()).matrix();
    jacobian = dz.array().colwise()
               * (cap_scaled.array() * sigmoid.array() * (1.0 - sigmoid.array()));
}

// Every dimension and scale the objective relies on, checked in one place.
//
// This exists because Eigen answers a size mismatch with an assertion, and an
// assertion calls abort(): the interpreter dies with no traceback, no line
// number and -- in a test run -- no failing test name. Three separate mismatches
// reached that state during #16 before this was consolidated (#43), each found
// by tripping over it rather than by a test.
//
// The scale checks are here for a different reason: a non-positive tau or
// prior scale does not abort, it divides into the objective and returns a
// plausible-looking wrong number, which is worse.
//
// Called on every objective evaluation, so it has to stay cheap: the only
// non-scalar work is one pass over `sigmas`, which is O(K) against the
// O(T*K) matrix products that follow.
void validate_inputs(const Eigen::VectorXd& t_scaled_vec,
                     const Eigen::VectorXd& change_points_vec,
                     const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& x,
                     const Eigen::VectorXd& normalized_y_vec,
                     double sigma_obs_prior_scale,
                     double sigma_k,
                     double sigma_m,
                     const Eigen::Ref<const Eigen::VectorXd>& sigmas,
                     const Eigen::Ref<const Eigen::VectorXd>& s_m,
                     double tau) {
    const Eigen::Index T = t_scaled_vec.size();
    if (T == 0) {
        throw std::invalid_argument("t_scaled is empty; there is nothing to fit");
    }
    if (normalized_y_vec.size() != T) {
        throw std::invalid_argument(
            "normalized_y has " + std::to_string(normalized_y_vec.size()) +
            " entries but t_scaled has " + std::to_string(T) + "; they index the "
            "same observations and must be the same length");
    }
    if (A.rows() != T || A.cols() != change_points_vec.size()) {
        throw std::invalid_argument(
            "the changepoint matrix is " + std::to_string(A.rows()) + "x" +
            std::to_string(A.cols()) + " but t_scaled has " + std::to_string(T) +
            " entries and change_points has " +
            std::to_string(change_points_vec.size()));
    }
    if (x.rows() != T) {
        throw std::invalid_argument(
            "the seasonality matrix has " + std::to_string(x.rows()) + " rows but "
            "t_scaled has " + std::to_string(T) + "; t_seasonality must cover the "
            "same observations as t_scaled");
    }
    if (sigmas.size() != x.cols()) {
        throw std::invalid_argument(
            "sigmas has " + std::to_string(sigmas.size()) + " entries but the "
            "design matrix has " + std::to_string(x.cols()) + " columns; there "
            "must be one prior scale per column");
    }
    if ((sigmas.array() <= 0.0).any()) {
        throw std::invalid_argument("every entry of sigmas must be positive");
    }
    // Empty means every column is additive, which is the common case and the
    // binding's default; anything else must cover the matrix exactly.
    if (s_m.size() != 0 && s_m.size() != x.cols()) {
        throw std::invalid_argument(
            "s_m has " + std::to_string(s_m.size()) + " entries but the design "
            "matrix has " + std::to_string(x.cols()) + " columns; there must be "
            "one mode indicator per column, or none at all");
    }
    if (s_m.size() != 0 && ((s_m.array() != 0.0) && (s_m.array() != 1.0)).any()) {
        throw std::invalid_argument(
            "s_m must hold only 0 (additive) or 1 (multiplicative)");
    }
    if (!(tau > 0.0)) {
        throw std::invalid_argument("tau must be positive");
    }
    if (!(sigma_obs_prior_scale > 0.0)) {
        throw std::invalid_argument("sigma_obs_prior_scale must be positive");
    }
    if (!(sigma_k > 0.0) || !(sigma_m > 0.0)) {
        throw std::invalid_argument("sigma_k and sigma_m must be positive");
    }
}

// include_l1_prior=false omits the Laplace (L1) prior on delta from both the
// objective and the gradient. The split reformulation in optimize() supplies
// that term itself, as a linear function of delta_pos and delta_neg.
//
// A and x are taken as arguments rather than rebuilt: see #28. The overload
// below keeps the standalone entry point self-contained.
void minus_log_posterior_and_gradient(const Eigen::VectorXd& params_vec,
                                      const Eigen::VectorXd& t_scaled_vec,
                                      const Eigen::VectorXd& change_points_vec,
                                      const Eigen::MatrixXd& A,
                                      const Eigen::MatrixXd& x,
                                      const Eigen::VectorXd& normalized_y_vec,
                                      double sigma_obs_prior_scale,
                                      double sigma_k,
                                      double sigma_m,
                                      const Eigen::Ref<const Eigen::VectorXd>& sigmas,
                                      const Eigen::Ref<const Eigen::VectorXd>& s_m,
                                      const Eigen::Ref<const Eigen::VectorXd>& cap_scaled,
                                      int trend_indicator,
                                      double tau,
                                      double& mlp_out,
                                      Eigen::Ref<Eigen::VectorXd> grad_out,
                                      bool include_l1_prior = true) {
    validate_inputs(t_scaled_vec, change_points_vec, A, x, normalized_y_vec,
                    sigma_obs_prior_scale, sigma_k, sigma_m, sigmas, s_m, tau);
    const ModelParams p = extract_params(params_vec,
                                         static_cast<int>(change_points_vec.size()),
                                         static_cast<int>(x.cols()));
    const double k = p.k;
    const double m = p.m;
    const Eigen::VectorXd& delta = p.delta;
    const Eigen::VectorXd& beta = p.beta;
    const double sigma_obs = p.sigma_obs;
    const double T = static_cast<double>(t_scaled_vec.size());

    // Trend component. Linear keeps its own expression rather than going
    // through the Jacobian form: it is the common case, and its fits are
    // sensitive in the last bits (see the note on the additive branch).
    // [stan] trend_indicator: 0 linear, 1 logistic, 2 flat.
    const bool logistic = trend_indicator == 1;
    const bool flat = trend_indicator == 2;
    if (trend_indicator < 0 || trend_indicator > 2) {
        throw std::invalid_argument(
            "trend_indicator is " + std::to_string(trend_indicator) +
            "; it must be 0 (linear), 1 (logistic) or 2 (flat)");
    }
    Eigen::VectorXd g;
    Eigen::MatrixXd trend_jacobian;
    if (logistic) {
        if (cap_scaled.size() != t_scaled_vec.size()) {
            throw std::invalid_argument(
                "cap_scaled has " + std::to_string(cap_scaled.size()) + " entries "
                "but t_scaled has " + std::to_string(t_scaled_vec.size()) +
                "; logistic growth needs one capacity per observation");
        }
        logistic_trend_and_jacobian(k, m, delta, t_scaled_vec, cap_scaled, A,
                                    change_points_vec, g, trend_jacobian);
    } else if (flat) {
        // [stan] flat_trend: rep_vector(m, T). k and delta stay parameters and
        // keep their priors; the likelihood does not see them.
        g = Eigen::VectorXd::Constant(t_scaled_vec.size(), m);
    } else {
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(t_scaled_vec.size());
        Eigen::VectorXd gamma = -delta.array() * change_points_vec.array();
        g = (k * ones + A * delta).array() * t_scaled_vec.array() + (m * ones + A * gamma).array();
    }

    // [stan] y ~ normal_id_glm(X_sa, trend .* (1 + X_sm * beta), beta, ...):
    // additive columns add to the trend, multiplicative ones scale it.
    //
    // With every column additive the multiplier is exactly 1, so the whole
    // thing collapses to g + x*beta. That case takes a separate branch rather
    // than multiplying by a vector of ones: it is the common one, and going
    // through the general path would allocate two K-column matrices per
    // evaluation and perturb every existing fit in the last bits.
    const bool multiplicative = s_m.size() != 0 && (s_m.array() != 0.0).any();

    Eigen::MatrixXd x_sm, x_sa;
    Eigen::VectorXd multiplier, y_pred;
    if (multiplicative) {
        x_sm = x * s_m.asDiagonal();
        x_sa = x - x_sm;
        multiplier = Eigen::VectorXd::Ones(t_scaled_vec.size()) + x_sm * beta;
        y_pred = g.array() * multiplier.array() + (x_sa * beta).array();
    } else {
        y_pred = g + x * beta;
    }
    Eigen::VectorXd r = normalized_y_vec - y_pred;
    // d(yhat)/d(a trend parameter) = d(g)/d(that parameter) * multiplier, so
    // every trend block below scales the residual by it.
    Eigen::VectorXd r_scaled = multiplicative
        ? Eigen::VectorXd(r.array() * multiplier.array())
        : r;

    double sum_squared_diff = r.array().square().sum();

    // T * zeta is the Gaussian likelihood's normalization. It was droppable
    // only while sigma_obs was a constant; with sigma_obs free, omitting it
    // leaves nothing penalizing growth -- the residual term shrinks
    // monotonically as sigma_obs rises and the optimizer inflates it badly.
    // The last term is the half-normal prior, sigma_obs ~ normal(0, 0.5).
    double minus_log_posterior_value = T * p.zeta +
                                       sum_squared_diff / (2 * std::pow(sigma_obs, 2)) +
                                       std::pow(sigma_obs, 2) / (2 * std::pow(sigma_obs_prior_scale, 2)) +
                                       std::pow(k, 2) / (2 * std::pow(sigma_k, 2)) +
                                       std::pow(m, 2) / (2 * std::pow(sigma_m, 2)) +
                                       (beta.array().square() /
                                        (2 * sigmas.array().square())).sum();

    if (include_l1_prior) {
        minus_log_posterior_value += delta.array().abs().sum() / tau;
    }

    // Set minus log posterior value
    mlp_out = minus_log_posterior_value;

    // Compute gradients
    grad_out.resize(params_vec.size());

    // Compute dk and dm
    Eigen::VectorXd ddelta;
    if (flat) {
        grad_out(0) = k / (sigma_k * sigma_k);
        grad_out(1) = -r_scaled.sum() / (sigma_obs * sigma_obs) + m / (sigma_m * sigma_m);
        ddelta = Eigen::VectorXd::Zero(delta.size());
    } else if (logistic) {
        // one contraction for all three blocks: the Jacobian already holds
        // d(trend)/d(k, m, delta), including gamma's recursion
        const Eigen::VectorXd d_trend =
            -(r_scaled.transpose() * trend_jacobian).transpose() / (sigma_obs * sigma_obs);
        grad_out(0) = d_trend(0) + k / (sigma_k * sigma_k);
        grad_out(1) = d_trend(1) + m / (sigma_m * sigma_m);
        ddelta = d_trend.tail(delta.size());
    } else {
        grad_out(0) = -r_scaled.dot(t_scaled_vec) / (sigma_obs * sigma_obs) + k / (sigma_k * sigma_k);
        grad_out(1) = -r_scaled.sum() / (sigma_obs * sigma_obs) + m / (sigma_m * sigma_m);

        Eigen::MatrixXd t_diff = t_scaled_vec.replicate(1, change_points_vec.size()).array().rowwise() - change_points_vec.transpose().array();
        Eigen::MatrixXd delta_contrib = t_diff.array() * A.array();
        ddelta = -(r_scaled.transpose() * delta_contrib).transpose() / (sigma_obs * sigma_obs);
    }

    if (include_l1_prior) {
        ddelta += (delta.array().sign() / tau).matrix();
    }

    grad_out.segment(2, delta.size()) = ddelta;

    // Compute dbeta
    int beta_start_index = 2 + delta.size(); // Dynamically calculate the starting index for beta
    // beta's gradient picks up the trend on its multiplicative columns.
    Eigen::VectorXd dbeta =
        (multiplicative
             ? Eigen::VectorXd(-(x_sm.transpose() * (r.array() * g.array()).matrix()
                                 + x_sa.transpose() * r) / (sigma_obs * sigma_obs))
             : Eigen::VectorXd(-(x.transpose() * r) / (sigma_obs * sigma_obs)))
        + (beta.array() / sigmas.array().square()).matrix();

    grad_out.segment(beta_start_index, beta.size()) = dbeta;

    // Compute dzeta, by chain rule from d/d_sigma_obs using
    // d(sigma_obs)/d(zeta) = sigma_obs:
    //   d/d_sigma_obs = T/sigma_obs - sum(r^2)/sigma_obs^3 + sigma_obs/scale^2
    //   d/d_zeta      = T - sum(r^2)/sigma_obs^2 + sigma_obs^2/scale^2
    // The four blocks above need no re-derivation: they already divide by
    // sigma_obs^2, which now simply varies between steps instead of being pinned.
    grad_out(grad_out.size() - 1) = T
        - sum_squared_diff / (sigma_obs * sigma_obs)
        + std::pow(sigma_obs, 2) / std::pow(sigma_obs_prior_scale, 2);
}

// Stan's L-BFGS convergence criteria, with CmdStan's default values --
// Prophet's CmdStanPyBackend.fit calls optimize(algorithm='LBFGS', iter=1e4)
// and sets no tolerances, so these are what the original actually runs with.
// [stan] src/stan/optimization/bfgs.hpp, ConvergenceOptions + step()
//
// Stan stops as soon as ANY of them holds. liblbfgs natively offers only two
// (epsilon on a relative gradient norm, and past/delta on relative objective
// change), which is how this implementation ended up with a single criterion
// -- and that one unreachable, so runs terminated on line-search exhaustion
// tens of thousands of iterations past convergence. They are evaluated in the
// progress callback instead, which liblbfgs lets us stop the run from.
namespace stan_convergence {
    constexpr double EPS = 2.220446049250313e-16;   // machine epsilon, as Stan uses it
    constexpr double F_SCALE = 1.0;                 // ConvergenceOptions::fScale
    constexpr double TOL_ABS_F = 1e-12;             // tol_obj
    constexpr double TOL_REL_F = 1e+4;              // tol_rel_obj, scaled by EPS below
    constexpr double TOL_ABS_GRAD = 1e-8;           // tol_grad
    constexpr double TOL_REL_GRAD = 1e+7;           // tol_rel_grad, scaled by EPS below
    constexpr double TOL_ABS_X = 1e-8;              // tol_param
    constexpr int MAX_ITERATIONS = 10000;           // Prophet passes iter=int(1e4)
    constexpr int HISTORY_SIZE = 5;                 // history_size
}


// The Laplace prior puts |delta|/tau in the objective, which is not
// differentiable at delta = 0 -- and that is where the optimum sits, since the
// prior is what drives changepoint rates to zero. Splitting delta into
// non-negative parts,
//
//     delta = delta_pos - delta_neg,   delta_pos, delta_neg >= 0
//
// turns |delta| into (delta_pos + delta_neg): smooth, with the non-smoothness
// moved into box constraints. This is the same reformulation fit() uses on the
// Python side, and it replaced OWL-QN here (issue #23) -- OWL-QN needed 5614
// iterations and 7.6s on the 2905-point series where this needs 2021 and 1.4s,
// and reaches a slightly better optimum besides.
//
// Box constraints are why the solver changed too: liblbfgs solves the
// unconstrained problem only and cannot express delta_pos >= 0. LBFGSpp
// provides L-BFGS-B, is header-only, and works in Eigen types directly.
//
// The split is an implementation detail of the optimizer: callers pass and
// receive the natural layout, [k, m, delta(S), beta(K), zeta].
// Self-contained overload: builds A and x, then delegates. Used by the
// exposed minus_log_posterior_and_gradient entry point, where there is no fit
// to amortize the construction over.
void minus_log_posterior_and_gradient(const Eigen::VectorXd& params_vec,
                                      const Eigen::VectorXd& t_scaled_vec,
                                      const Eigen::VectorXd& change_points_vec,
                                      const Eigen::VectorXd& t_seasonality_vec,
                                      const Eigen::VectorXd& normalized_y_vec,
                                      double sigma_obs_prior_scale,
                                      double sigma_k,
                                      double sigma_m,
                                      const Eigen::Ref<const Eigen::VectorXd>& sigmas,
                                      const Eigen::Ref<const Eigen::VectorXd>& s_m,
                                      const Eigen::Ref<const Eigen::VectorXd>& cap_scaled,
                                      int trend_indicator,
                                      double tau,
                                      const std::vector<int>& fourier_orders,
                                      const std::vector<double>& seasonality_periods,
                                      const Eigen::Ref<const Eigen::MatrixXd>& seasonality_conditions,
                                      const Eigen::Ref<const Eigen::MatrixXd>& data_columns,
                                      double& mlp_out,
                                      Eigen::Ref<Eigen::VectorXd> grad_out,
                                      bool include_l1_prior = true) {
    minus_log_posterior_and_gradient(params_vec, t_scaled_vec, change_points_vec,
                                     changepoint_matrix(t_scaled_vec, change_points_vec),
                                     seasonality_matrix(t_seasonality_vec, fourier_orders,
                                                        seasonality_periods,
                                                        seasonality_conditions,
                                                        data_columns),
                                     normalized_y_vec, sigma_obs_prior_scale, sigma_k,
                                     sigma_m, sigmas, s_m, cap_scaled, trend_indicator,
                                     tau, mlp_out, grad_out,
                                     include_l1_prior);
}

struct SplitObjective {
    const Eigen::VectorXd& t_scaled;
    const Eigen::VectorXd& change_points;
    // Built once by optimize() before the solver loop: constant for the whole
    // fit, and rebuilding them was the bulk of every evaluation (#28).
    Eigen::MatrixXd A;
    Eigen::MatrixXd x;
    const Eigen::VectorXd& normalized_y;
    double sigma_obs_prior_scale;
    double sigma_k;
    double sigma_m;
    Eigen::VectorXd sigmas;
    Eigen::VectorXd s_m;
    Eigen::VectorXd cap_scaled;
    int trend_indicator;
    double tau;
    int S;
    int K;

    // Best objective seen so far, recorded at each evaluation. LBFGSpp has no
    // per-iteration hook, so this is a monotone lower envelope over objective
    // evaluations rather than the per-iteration trace liblbfgs's progress
    // callback used to give. It still converges to the same final value and
    // still only ever decreases.
    std::vector<double> loss_envelope;

    Eigen::VectorXd to_natural(const Eigen::VectorXd& z) const {
        Eigen::VectorXd natural(2 + S + K + 1);
        natural(0) = z(0);
        natural(1) = z(1);
        natural.segment(2, S) = z.segment(2, S) - z.segment(2 + S, S);
        natural.segment(2 + S, K) = z.segment(2 + 2 * S, K);
        natural(natural.size() - 1) = z(z.size() - 1);
        return natural;
    }

    double operator()(const Eigen::VectorXd& z, Eigen::VectorXd& grad) {
        const Eigen::VectorXd natural = to_natural(z);

        double value = 0.0;
        Eigen::VectorXd natural_grad(natural.size());
        minus_log_posterior_and_gradient(natural, t_scaled, change_points, A, x,
                                         normalized_y, sigma_obs_prior_scale, sigma_k,
                                         sigma_m, sigmas, s_m, cap_scaled, trend_indicator,
                                         tau, value, natural_grad,
                                         // the split form supplies the L1 term itself
                                         /*include_l1_prior=*/false);

        // |delta| == delta_pos + delta_neg on the feasible set, so the Laplace
        // prior becomes linear here.
        value += (z.segment(2, S).sum() + z.segment(2 + S, S).sum()) / tau;

        // d/d(delta_pos) = d/d(delta) + 1/tau,  d/d(delta_neg) = -d/d(delta) + 1/tau
        const Eigen::VectorXd ddelta = natural_grad.segment(2, S);
        grad.resize(z.size());
        grad(0) = natural_grad(0);
        grad(1) = natural_grad(1);
        grad.segment(2, S) = ddelta.array() + 1.0 / tau;
        grad.segment(2 + S, S) = -ddelta.array() + 1.0 / tau;
        grad.segment(2 + 2 * S, K) = natural_grad.segment(2 + S, K);
        grad(grad.size() - 1) = natural_grad(natural_grad.size() - 1);

        if (loss_envelope.empty() || value < loss_envelope.back()) {
            loss_envelope.push_back(value);
        }
        return value;
    }
};

struct OptimizeResult {
    Eigen::VectorXd params;
    std::vector<double> loss_trace;
    int n_iterations;
    int status;
    std::string status_message;
};

OptimizeResult optimize(Eigen::VectorXd params,
                        const Eigen::Ref<const Eigen::VectorXd>& t_scaled,
                        const Eigen::Ref<const Eigen::VectorXd>& change_points,
                        const Eigen::Ref<const Eigen::VectorXd>& t_seasonality,
                        const Eigen::Ref<const Eigen::VectorXd>& normalized_y,
                        double sigma_obs_prior_scale,
                        double sigma_k,
                        double sigma_m,
                        const Eigen::Ref<const Eigen::VectorXd>& sigmas,
                        const Eigen::Ref<const Eigen::VectorXd>& s_m,
                        const Eigen::Ref<const Eigen::VectorXd>& cap_scaled,
                        int trend_indicator,
                        double tau,
                        const std::vector<int>& fourier_orders,
                        const std::vector<double>& seasonality_periods,
                        const Eigen::Ref<const Eigen::MatrixXd>& seasonality_conditions,
                        const Eigen::Ref<const Eigen::MatrixXd>& data_columns,
                        bool verbose) {

        const int params_size = static_cast<int>(params.size());
        const int S = static_cast<int>(change_points.size());
        const int K = params_size - 3 - S;

        if (params_size < 2 + S + 2) {
            throw std::invalid_argument("params is too short for the given number of change points");
        }

        // Built here rather than in the solver loop (#28), which also means
        // the shared checks can run against the real matrices before any
        // setup work -- a caller gets the error from the call, not from the
        // first objective evaluation inside it.
        const Eigen::MatrixXd A = changepoint_matrix(t_scaled, change_points);
        const Eigen::MatrixXd x = seasonality_matrix(t_seasonality, fourier_orders,
                                                     seasonality_periods,
                                                     seasonality_conditions,
                                                     data_columns);
        validate_inputs(t_scaled, change_points, A, x, normalized_y,
                        sigma_obs_prior_scale, sigma_k, sigma_m, sigmas, s_m, tau);

        // Specific to this entry point: the split vector is laid out from K,
        // which is read off `params` rather than from the design matrix.
        if (K != x.cols()) {
            throw std::invalid_argument(
                "params leaves room for " + std::to_string(K) + " seasonality "
                "coefficients, but the registered seasonalities need " +
                std::to_string(x.cols()));
        }

        // Natural -> split. delta splits into its positive and negative parts,
        // so a starting delta of zero starts both at zero.
        const int n = 2 + 2 * S + K + 1;
        Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
        z(0) = params(0);
        z(1) = params(1);
        for (int j = 0; j < S; ++j) {
            const double delta_j = params(2 + j);
            z(2 + j) = std::max(delta_j, 0.0);
            z(2 + S + j) = std::max(-delta_j, 0.0);
        }
        z.segment(2 + 2 * S, K) = params.segment(2 + S, K);
        z(n - 1) = params(params_size - 1);

        Eigen::VectorXd lower = Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
        Eigen::VectorXd upper = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
        lower.segment(2, 2 * S).setZero();   // delta_pos, delta_neg >= 0

        // Stan's convergence criteria, as far as LBFGSpp expresses them.
        // Its past/delta test is Stan's TERM_RELF exactly -- same inequality,
        // same max(|f_k|, |f_past|, 1) denominator at fScale = 1. Its epsilon
        // is an inf-norm on the projected gradient, which is the right
        // stationarity measure for a bound-constrained problem and, unlike
        // scipy's equivalent, does not fire early here (verified: identical
        // results at 0, 1e-8 and 1e-5).
        //
        // Stan's remaining tests -- absolute objective change and parameter
        // change -- need a per-iteration hook, which LBFGSpp does not provide.
        // TERM_RELF is the one that binds in practice.
        LBFGSpp::LBFGSBParam<double> param;
        param.m = stan_convergence::HISTORY_SIZE;
        param.max_iterations = stan_convergence::MAX_ITERATIONS;
        param.epsilon = stan_convergence::TOL_ABS_GRAD;
        param.epsilon_rel = 0.0;
        param.past = 1;
        param.delta = stan_convergence::TOL_REL_F * stan_convergence::EPS;
        param.max_linesearch = 60;

        SplitObjective objective{t_scaled, change_points, A, x,
                                 normalized_y, sigma_obs_prior_scale,
                                 sigma_k, sigma_m, sigmas, s_m, cap_scaled,
                                 trend_indicator, tau, S, K, {}};
        LBFGSpp::LBFGSBSolver<double> solver(param);

        double fx = 0.0;
        int iterations = 0;
        int status = 0;
        std::string message = "converged";
        try {
            iterations = solver.minimize(objective, z, fx, lower, upper);
            // LBFGSpp returns the iteration count silently when it runs out of
            // iterations rather than raising, so reaching the cap is
            // indistinguishable from converging unless it is checked for. Left
            // unchecked this reports "converged" on a fit that simply ran out
            // of budget -- the same false success as #13.
            if (iterations >= stan_convergence::MAX_ITERATIONS) {
                status = 2;
                message = "reached max_iterations without meeting a convergence test";
            }
        } catch (const std::exception& error) {
            status = 1;
            message = std::string("optimizer failed: ") + error.what();
        }

        if (verbose) {
            std::cout << "L-BFGS-B terminated after " << iterations
                      << " iterations: " << message << "\n  fx = " << fx << "\n";
        }

        return OptimizeResult{
            objective.to_natural(z),
            std::move(objective.loss_envelope),
            iterations,
            status,
            message,
        };
}

std::pair<double, Eigen::VectorXd> minus_log_posterior_and_gradient_py(
        const Eigen::Ref<const Eigen::VectorXd>& params,
        const Eigen::Ref<const Eigen::VectorXd>& t_scaled,
        const Eigen::Ref<const Eigen::VectorXd>& change_points,
        const Eigen::Ref<const Eigen::VectorXd>& t_seasonality,
        const Eigen::Ref<const Eigen::VectorXd>& normalized_y,
        double sigma_obs_prior_scale,
        double sigma_k,
        double sigma_m,
        const Eigen::Ref<const Eigen::VectorXd>& sigmas,
        const Eigen::Ref<const Eigen::VectorXd>& s_m,
        const Eigen::Ref<const Eigen::VectorXd>& cap_scaled,
        int trend_indicator,
        double tau,
        const std::vector<int>& fourier_orders,
        const std::vector<double>& seasonality_periods,
        const Eigen::Ref<const Eigen::MatrixXd>& seasonality_conditions,
        const Eigen::Ref<const Eigen::MatrixXd>& data_columns,
        bool include_l1_prior) {
    double mlp = 0.0;
    Eigen::VectorXd gradient(params.size());
    minus_log_posterior_and_gradient(params, t_scaled, change_points, t_seasonality,
                                     normalized_y, sigma_obs_prior_scale, sigma_k, sigma_m, sigmas,
                                     s_m, cap_scaled, trend_indicator, tau, fourier_orders, seasonality_periods,
                                     seasonality_conditions, data_columns,
                                     mlp, gradient,
                                     include_l1_prior);
    return {mlp, gradient};
}

PYBIND11_MODULE(analytic_prophet_cpp, m) {
    m.doc() = "Analytic Prophet's C++ core: an L-BFGS/OWL-QN MAP optimizer for the "
              "Prophet posterior, plus the objective and analytic gradient it drives.";

    py::class_<OptimizeResult>(m, "OptimizeResult",
            "Outcome of one optimize() run.")
        .def_readonly("params", &OptimizeResult::params,
                      "Optimized (k, m, delta, beta) vector.")
        .def_readonly("loss_trace", &OptimizeResult::loss_trace,
                      "Monotone trace of the minus-log-posterior: the best value seen so "
                      "far, recorded whenever an objective evaluation improves on it. "
                      "LBFGSpp has no per-iteration hook, so this samples evaluations "
                      "rather than iterations and is generally a little longer than "
                      "n_iterations.")
        .def_readonly("n_iterations", &OptimizeResult::n_iterations)
        .def_readonly("status", &OptimizeResult::status,
                      "Raw liblbfgs status code; 0 is LBFGS_SUCCESS.")
        .def_readonly("status_message", &OptimizeResult::status_message,
                      "Human-readable form of `status`.")
        .def("__repr__", [](const OptimizeResult& r) {
            return "<OptimizeResult status=" + std::to_string(r.status) +
                   " (" + r.status_message + ") n_iterations=" +
                   std::to_string(r.n_iterations) + ">";
        });

    m.def("optimize", &optimize,
          py::arg("params"),
          py::arg("t_scaled"),
          py::arg("change_points"),
          py::arg("t_seasonality"),
          py::arg("normalized_y"),
          py::arg("sigma_obs_prior_scale"),
          py::arg("sigma_k"),
          py::arg("sigma_m"),
          py::arg("sigmas"),
          py::arg("s_m") = Eigen::VectorXd(),
          py::arg("cap_scaled") = Eigen::VectorXd(),
          py::arg("trend_indicator") = 0,
          py::arg("tau"),
          py::arg("fourier_orders"),
          py::arg("seasonality_periods"),
          py::arg("seasonality_conditions") = Eigen::MatrixXd(),
          py::arg("data_columns") = Eigen::MatrixXd(),
          py::arg("verbose") = false,
          // The optimizer touches no Python objects, so let other threads run.
          py::call_guard<py::gil_scoped_release>(),
          "Run L-BFGS (OWL-QN) on the Prophet minus-log-posterior and return an "
          "OptimizeResult. `params` is not modified in place; the optimized vector "
          "comes back on the result.");

    m.def("minus_log_posterior_and_gradient", &minus_log_posterior_and_gradient_py,
          py::arg("params"),
          py::arg("t_scaled"),
          py::arg("change_points"),
          py::arg("t_seasonality"),
          py::arg("normalized_y"),
          py::arg("sigma_obs_prior_scale"),
          py::arg("sigma_k"),
          py::arg("sigma_m"),
          py::arg("sigmas"),
          py::arg("s_m") = Eigen::VectorXd(),
          py::arg("cap_scaled") = Eigen::VectorXd(),
          py::arg("trend_indicator") = 0,
          py::arg("tau"),
          py::arg("fourier_orders"),
          py::arg("seasonality_periods"),
          py::arg("seasonality_conditions") = Eigen::MatrixXd(),
          py::arg("data_columns") = Eigen::MatrixXd(),
          py::arg("include_l1_prior") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Return (minus_log_posterior, gradient) at `params`. With "
          "include_l1_prior=False the Laplace prior on delta is omitted from both, "
          "which is what OWL-QN requires of the callback it drives.");
}

// To compile, run the following command:
// c++ -std=c++17 -shared -fPIC -O3 -undefined dynamic_lookup \
//     $(python -m pybind11 --includes) -I/opt/homebrew/opt/eigen/include/eigen3 \
//     -I$(brew --prefix lbfgspp)/include \
//     optimize.cpp -o analytic_prophet_cpp$(python3-config --extension-suffix)
// (drop -undefined dynamic_lookup off macOS; tests/conftest.py builds it this way.)
//
// Header-only now: LBFGSpp replaced liblbfgs in #23, so there is nothing left
// to link against. That also removes a macOS trap -- LBFGSpp ships LBFGS.h,
// which shadows liblbfgs's lbfgs.h on a case-insensitive filesystem when both
// include directories are on the search path.
