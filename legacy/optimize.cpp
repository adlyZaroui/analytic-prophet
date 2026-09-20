#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <iostream>
#include <lbfgs.h>
#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

Eigen::MatrixXd fourier_components(const Eigen::VectorXd& t_days, double period, int n) {
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n, 1, n) * (2 * M_PI / period);
    Eigen::MatrixXd angles = t_days * x.transpose();
    
    Eigen::MatrixXd result(t_days.size(), 2 * n);
    result.leftCols(n) = angles.array().cos();
    result.rightCols(n) = angles.array().sin();
    
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

ModelParams extract_params(const Eigen::Ref<const Eigen::VectorXd>& params, int S) {
    const int K = static_cast<int>(params.size()) - 3 - S;
    // The Fourier order is still hardcoded at 10 further down (issue #14), so
    // K is pinned at 20. Checking it here turns a downstream Eigen
    // "invalid matrix product" abort into a diagnosable error.
    if (K != 2 * 10) {
        throw std::invalid_argument(
            "params has length " + std::to_string(params.size()) + " with S=" +
            std::to_string(S) + ", implying K=" + std::to_string(K) +
            " seasonality columns; expected 20. The layout is "
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

// include_l1_prior=false omits the Laplace (L1) prior on delta from both the
// objective and the gradient. OWL-QN adds that term itself and handles its
// kink at delta=0 via orthant projection, so the callback it drives must not
// include it -- see the orthantwise_c setup in optimize() below.
void minus_log_posterior_and_gradient(const Eigen::VectorXd& params_vec,
                                      const Eigen::VectorXd& t_scaled_vec,
                                      const Eigen::VectorXd& change_points_vec,
                                      double scale_period,
                                      const Eigen::VectorXd& normalized_y_vec,
                                      double sigma_obs_prior_scale,
                                      double sigma_k,
                                      double sigma_m,
                                      double sigma,
                                      double tau,
                                      double& mlp_out,
                                      Eigen::Ref<Eigen::VectorXd> grad_out,
                                      bool include_l1_prior = true) {
    const ModelParams p = extract_params(params_vec, static_cast<int>(change_points_vec.size()));
    const double k = p.k;
    const double m = p.m;
    const Eigen::VectorXd& delta = p.delta;
    const Eigen::VectorXd& beta = p.beta;
    const double sigma_obs = p.sigma_obs;
    const double T = static_cast<double>(t_scaled_vec.size());

    // Trend component
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(t_scaled_vec.size());
    // >= , not > : Stan's get_changepoint_matrix uses t[i] >= t_change[j]
    Eigen::MatrixXd A = (t_scaled_vec.replicate(1, change_points_vec.size()).array() >= change_points_vec.transpose().replicate(t_scaled_vec.size(), 1).array()).cast<double>();
    Eigen::VectorXd gamma = -delta.array() * change_points_vec.array();
    Eigen::VectorXd g = (k * ones + A * delta).array() * t_scaled_vec.array() + (m * ones + A * gamma).array();

    // Seasonality component
    double period = 365.25 / scale_period;
    Eigen::MatrixXd x = fourier_components(t_scaled_vec, period, 10);
    Eigen::VectorXd s = x * beta;

    Eigen::VectorXd y_pred = g + s;
    Eigen::VectorXd r = normalized_y_vec - y_pred;

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
                                       beta.array().square().sum() / (2 * std::pow(sigma, 2));

    if (include_l1_prior) {
        minus_log_posterior_value += delta.array().abs().sum() / tau;
    }

    // Set minus log posterior value
    mlp_out = minus_log_posterior_value;

    // Compute gradients
    grad_out.resize(params_vec.size());

    // Compute dk and dm
    grad_out(0) = -r.dot(t_scaled_vec) / (sigma_obs * sigma_obs) + k / (sigma_k * sigma_k);
    grad_out(1) = -r.sum() / (sigma_obs * sigma_obs) + m / (sigma_m * sigma_m);

    // Compute ddelta
    Eigen::MatrixXd t_diff = t_scaled_vec.replicate(1, change_points_vec.size()).array().rowwise() - change_points_vec.transpose().array();
    Eigen::MatrixXd delta_contrib = t_diff.array() * A.array();
    Eigen::VectorXd ddelta = -(r.transpose() * delta_contrib).transpose() / (sigma_obs * sigma_obs);

    if (include_l1_prior) {
        ddelta += (delta.array().sign() / tau).matrix();
    }

    grad_out.segment(2, delta.size()) = ddelta;

    // Compute dbeta
    int beta_start_index = 2 + delta.size(); // Dynamically calculate the starting index for beta
    Eigen::VectorXd dbeta = -(x.transpose() * r) / (sigma_obs * sigma_obs) + beta / (sigma * sigma);

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

struct OptimizationData {
    Eigen::VectorXd t_scaled;
    Eigen::VectorXd change_points;
    double scale_period;
    Eigen::VectorXd normalized_y;
    double sigma_obs_prior_scale;
    double sigma_k;
    double sigma_m;
    double sigma;
    double tau;
    // Per-iteration loss, so callers can compare the optimizer's trajectory
    // against the Python reference instead of scraping it from stdout. A
    // std::vector grows as needed -- the ctypes binding this replaced needed a
    // caller-preallocated buffer plus its length, since it could only pass
    // pointers.
    std::vector<double> loss_over_iterations;
    int n_iterations;
    bool verbose;
    // Previous iterate, for Stan's TERM_ABSF and TERM_ABSX tests
    double previous_f;
    Eigen::VectorXd previous_x;
    bool has_previous;
    // Which of Stan's tests stopped the run, empty if none did
    std::string termination;
};

lbfgsfloatval_t evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) {
    const Eigen::Map<const Eigen::VectorXd> params_vec(x, n);
    Eigen::Map<Eigen::VectorXd> grad_out(g, n);

    // Extract additional arguments from the instance
    auto* data = static_cast<OptimizationData*>(instance);

    double mlp;
    minus_log_posterior_and_gradient(params_vec,
                                     data->t_scaled,
                                     data->change_points,
                                     data->scale_period,
                                     data->normalized_y,
                                     data->sigma_obs_prior_scale,
                                     data->sigma_k,
                                     data->sigma_m,
                                     data->sigma,
                                     data->tau, mlp, grad_out,
                                     // OWL-QN contributes the Laplace prior on delta itself
                                     false);

    return mlp;
}

int progress(void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls) {
    auto* data = static_cast<OptimizationData*>(instance);

    // fx already includes the L1 term: OWL-QN adds orthantwise_c * |x|_1 to
    // whatever evaluate() returned, so this is the full minus-log-posterior.
    data->loss_over_iterations.push_back(fx);
    data->n_iterations = k;

    if (data->verbose) {
        std::cout << "Iteration " << k << ": fx = " << fx << ", xnorm = " << xnorm << ", gnorm = " << gnorm << ", step = " << step << std::endl;
    }

    // Stan's convergence tests, in the order bfgs.hpp checks them. Returning
    // non-zero stops the run.
    using namespace stan_convergence;
    const Eigen::Map<const Eigen::VectorXd> current_x(x, n);

    if (gnorm < TOL_ABS_GRAD) {
        data->termination = "converged: gradient norm below tol_grad";
        return 1;
    }

    if (data->has_previous) {
        const double objective_change = std::fabs(data->previous_f - fx);

        if (objective_change < TOL_ABS_F) {
            data->termination = "converged: objective change below tol_obj";
            return 1;
        }

        // rel_obj_decrease(), [stan] bfgs.hpp
        const double relative_change = objective_change
            / std::max(std::fabs(data->previous_f), std::max(std::fabs(fx), F_SCALE));
        if (relative_change < TOL_REL_F * EPS) {
            data->termination = "converged: relative objective change below tol_rel_obj";
            return 1;
        }

        if ((current_x - data->previous_x).norm() < TOL_ABS_X) {
            data->termination = "converged: parameter change below tol_param";
            return 1;
        }
    }

    data->previous_f = fx;
    data->previous_x = current_x;
    data->has_previous = true;
    return 0;
}

// Result of one optimization run. Bound as a Python object with named
// attributes, so the caller reads result.status instead of unpacking
// out-parameters the way the ctypes binding forced.
struct OptimizeResult {
    Eigen::VectorXd params;
    std::vector<double> loss_over_iterations;
    int n_iterations;
    int status;
    std::string status_message;
};

// liblbfgs only returns a bare integer code. Spelling them out here means a
// caller sees "the line search stepped across a non-differentiable point"
// rather than -1001, which previously had to be looked up in lbfgs.h by hand.
std::string lbfgs_status_message(int code) {
    switch (code) {
        case LBFGS_SUCCESS:                 return "converged";
        case LBFGS_STOP:                    return "stopped by the convergence test in progress()";
        case LBFGS_ALREADY_MINIMIZED:       return "the initial point is already a minimizer";
        case LBFGSERR_MAXIMUMITERATION:     return "reached max_iterations before converging";
        case LBFGSERR_MAXIMUMLINESEARCH:    return "line search hit max_linesearch evaluations";
        case LBFGSERR_ROUNDING_ERROR:       return "rounding error in the line search; no step satisfied the "
                                                   "sufficient-decrease and curvature conditions";
        case LBFGSERR_MINIMUMSTEP:          return "line search step became smaller than min_step";
        case LBFGSERR_MAXIMUMSTEP:          return "line search step grew larger than max_step";
        case LBFGSERR_INCREASEGRADIENT:     return "the current search direction increases the objective";
        case LBFGSERR_INVALID_LINESEARCH:   return "invalid line search for the configured method";
        case LBFGSERR_OUTOFMEMORY:          return "out of memory";
        default:                            return "liblbfgs error code " + std::to_string(code);
    }
}

OptimizeResult optimize(Eigen::VectorXd params,
                        const Eigen::Ref<const Eigen::VectorXd>& t_scaled,
                        const Eigen::Ref<const Eigen::VectorXd>& change_points,
                        double scale_period,
                        const Eigen::Ref<const Eigen::VectorXd>& normalized_y,
                        double sigma_obs_prior_scale,
                        double sigma_k,
                        double sigma_m,
                        double sigma,
                        double tau,
                        bool verbose) {

        const int params_size = static_cast<int>(params.size());
        const int change_points_size = static_cast<int>(change_points.size());

        // Checks the ctypes binding could not make: it received bare pointers
        // with caller-supplied lengths, so a mismatch corrupted memory silently
        // instead of raising.
        if (t_scaled.size() != normalized_y.size()) {
            throw std::invalid_argument("t_scaled and normalized_y must have the same length");
        }
        // 2 + S + K + 1, with at least one beta column
        if (params_size < 2 + change_points_size + 2) {
            throw std::invalid_argument("params is too short for the given number of change points");
        }
        if (tau <= 0.0) {
            throw std::invalid_argument("tau must be positive");
        }
        if (sigma_obs_prior_scale <= 0.0) {
            throw std::invalid_argument("sigma_obs_prior_scale must be positive");
        }

        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);

        // The Laplace (double-exponential) prior on delta puts a |delta|/tau
        // term in the objective, so the posterior is NOT differentiable at
        // delta = 0 -- and the optimum sits right on those kinks, since the
        // prior is what drives most changepoint rates to exactly zero.
        //
        // More-Thuente (LBFGS_LINESEARCH_DEFAULT) assumes a smooth objective:
        // it narrows an interval of uncertainty until the strong Wolfe
        // conditions hold, and across a kink that interval collapses instead,
        // so the search bails out with LBFGSERR_ROUNDING_ERROR after a couple
        // of iterations, thousands short of convergence.
        //
        // OWL-QN is liblbfgs's answer to exactly this: it takes the L1
        // coefficient itself and handles the non-differentiable point by
        // projecting each step onto the current orthant. It requires the
        // backtracking line search, and requires that evaluate() report the
        // objective and gradient WITHOUT the L1 term (see the
        // include_l1_prior=false call above); the library adds it back, so the
        // fx it reports is still the full minus-log-posterior.
        param.orthantwise_c = 1.0 / tau;
        param.orthantwise_start = 2;                        // protect k, m
        param.orthantwise_end = 2 + change_points_size;     // delta only, not beta
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;   // required by OWL-QN

        // Prophet's iter=int(1e4). This was raised to 100000 as a workaround
        // while the convergence tests were missing; with them in place the run
        // stops on a criterion long before the cap, as Stan's does.
        param.max_iterations = stan_convergence::MAX_ITERATIONS;
        param.m = stan_convergence::HISTORY_SIZE;   // Stan's history_size

        // liblbfgs's own relative-gradient test, used as the stand-in for
        // Stan's TERM_RELGRAD. Stan's form is -p_k.g_k / max(|f_k|, fScale),
        // which needs the search direction; liblbfgs exposes only
        // ||g|| / max(1, ||x||) and no way to reach p_k from the callback, so
        // this is the closest available and is the one Stan test not
        // reproduced exactly. The other four are in progress() above.
        param.epsilon = stan_convergence::TOL_REL_GRAD * stan_convergence::EPS;

        // Stan's relative-objective test is implemented exactly in progress(),
        // so liblbfgs's cruder version of it stays off.
        param.past = 0;
        param.max_linesearch = 30;          // Maximum number of line search trials
        param.min_step = 1e-20;             // Minimum step size for line search
        param.max_step = 1e+20;             // Maximum step size for line search
        param.ftol = 1e-4;                  // Accuracy parameter for line search (decrease function value)
        param.wolfe = 0.9;                  // Wolfe condition parameter for line search

        lbfgsfloatval_t fx;

        OptimizationData data = {
            t_scaled,
            change_points,
            scale_period,
            normalized_y,
            sigma_obs_prior_scale, sigma_k, sigma_m, sigma, tau,
            {}, 0, verbose,
            0.0, Eigen::VectorXd(), false, ""
        };

        int ret = lbfgs(params_size, params.data(), &fx, evaluate, progress, &data, &param);

        const std::string message = data.termination.empty()
            ? lbfgs_status_message(ret)
            : data.termination;

        if (verbose) {
            std::cout << "L-BFGS optimization terminated: " << message << "\n";
            std::cout << "  fx = " << fx << "\n";
        }

        return OptimizeResult{
            std::move(params),
            std::move(data.loss_over_iterations),
            data.n_iterations,
            ret,
            message,
        };
}

// Objective and analytic gradient on their own, for cross-checking against the
// Python reference implementation. This is the very function the optimizer
// drives (see evaluate() above), not a copy of it.
std::pair<double, Eigen::VectorXd> minus_log_posterior_and_gradient_py(
        const Eigen::Ref<const Eigen::VectorXd>& params,
        const Eigen::Ref<const Eigen::VectorXd>& t_scaled,
        const Eigen::Ref<const Eigen::VectorXd>& change_points,
        double scale_period,
        const Eigen::Ref<const Eigen::VectorXd>& normalized_y,
        double sigma_obs_prior_scale,
        double sigma_k,
        double sigma_m,
        double sigma,
        double tau,
        bool include_l1_prior) {
    double mlp = 0.0;
    Eigen::VectorXd gradient(params.size());
    minus_log_posterior_and_gradient(params, t_scaled, change_points, scale_period,
                                     normalized_y, sigma_obs_prior_scale, sigma_k, sigma_m, sigma,
                                     tau, mlp, gradient, include_l1_prior);
    return {mlp, gradient};
}

PYBIND11_MODULE(analytic_prophet_cpp, m) {
    m.doc() = "Analytic Prophet's C++ core: an L-BFGS/OWL-QN MAP optimizer for the "
              "Prophet posterior, plus the objective and analytic gradient it drives.";

    py::class_<OptimizeResult>(m, "OptimizeResult",
            "Outcome of one optimize() run.")
        .def_readonly("params", &OptimizeResult::params,
                      "Optimized (k, m, delta, beta) vector.")
        .def_readonly("loss_over_iterations", &OptimizeResult::loss_over_iterations,
                      "Minus-log-posterior after each iteration, including the L1 term.")
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
          py::arg("scale_period"),
          py::arg("normalized_y"),
          py::arg("sigma_obs_prior_scale"),
          py::arg("sigma_k"),
          py::arg("sigma_m"),
          py::arg("sigma"),
          py::arg("tau"),
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
          py::arg("scale_period"),
          py::arg("normalized_y"),
          py::arg("sigma_obs_prior_scale"),
          py::arg("sigma_k"),
          py::arg("sigma_m"),
          py::arg("sigma"),
          py::arg("tau"),
          py::arg("include_l1_prior") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Return (minus_log_posterior, gradient) at `params`. With "
          "include_l1_prior=False the Laplace prior on delta is omitted from both, "
          "which is what OWL-QN requires of the callback it drives.");
}

// To compile, run the following command:
// c++ -std=c++17 -shared -fPIC -O3 -undefined dynamic_lookup \
//     $(python -m pybind11 --includes) -I/opt/homebrew/opt/eigen/include/eigen3 \
//     optimize.cpp -llbfgs -o analytic_prophet_cpp$(python3-config --extension-suffix)
// (drop -undefined dynamic_lookup off macOS; tests/conftest.py builds it this way.)