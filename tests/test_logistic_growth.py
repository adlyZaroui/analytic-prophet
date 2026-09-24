"""
Issue #16 task 12: logistic growth.

    [stan] cap .* inv_logit((k + A*delta) .* (t - (m + A*gamma)))

Saturating growth toward a capacity the caller supplies per row, in a `cap`
column on every frame passed to fit() and predict().

This is much harder to differentiate than the linear trend, and for one reason:
`gamma`, the piecewise offsets that keep the curve continuous where the rate
changes, is defined by a *recursion*.

    k_s        = [k, k + cumsum(delta)]
    gamma[i]   = (changepoints_t[i] - m_pr_i) * (1 - k_s[i]/k_s[i+1])
    m_pr_{i+1} = m_pr_i + gamma[i]

In the linear case gamma is just `-changepoints_t * delta` and each entry is
independent. Here gamma[i] depends on every earlier gamma through `m_pr`, so
d(gamma)/d(k, m, delta) is accumulated forward alongside gamma itself rather
than written down -- an S x (2 + S) Jacobian, which the trend's own chain rule
then carries through `inv_logit`.

The derivation was checked against finite differences standalone before it was
written into the model, and is checked again here in place.
"""
import numpy as np
import pandas as pd
import pytest

from analytic_prophet import (AnalyticProphet, canonical_to_cpp, predict_trend,
                           logistic_gamma_and_jacobian, logistic_growth_init,
                           logistic_trend_and_jacobian)


def with_cap(df, multiple=1.25):
    return df.assign(cap=df["y"].max() * multiple)


def logistic_model():
    model = AnalyticProphet()
    model.growth = "logistic"
    return model


# -- gamma's recursion --------------------------------------------------

def test_gamma_matches_stans_recursion():
    """Transcribed directly from [stan] logistic_gamma, independently of the
    implementation, so a transcription error in one shows against the other."""
    rng = np.random.default_rng(0)
    k, m = 1.7, 0.3
    delta = rng.normal(scale=0.08, size=6)
    changepoints_t = np.sort(rng.uniform(0.05, 0.9, 6))

    k_s = np.concatenate(([k], k + np.cumsum(delta)))
    expected = np.empty(6)
    m_pr = m
    for i in range(6):
        expected[i] = (changepoints_t[i] - m_pr) * (1 - k_s[i] / k_s[i + 1])
        m_pr += expected[i]

    gamma, _ = logistic_gamma_and_jacobian(k, m, delta, changepoints_t)
    np.testing.assert_allclose(gamma, expected, rtol=0, atol=0)


def test_gamma_jacobian_matches_finite_differences():
    """The part that cannot be written down: gamma[i] depends on every earlier
    gamma, so a Jacobian that forgot the carry would still look plausible."""
    rng = np.random.default_rng(1)
    theta = np.concatenate(([1.7], [0.3], rng.normal(scale=0.08, size=6)))
    changepoints_t = np.sort(rng.uniform(0.05, 0.9, 6))

    _, analytic = logistic_gamma_and_jacobian(theta[0], theta[1], theta[2:], changepoints_t)

    step = 1e-7
    numerical = np.empty_like(analytic)
    for j in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[j] += step
        down[j] -= step
        numerical[:, j] = (
            logistic_gamma_and_jacobian(up[0], up[1], up[2:], changepoints_t)[0]
            - logistic_gamma_and_jacobian(down[0], down[1], down[2:], changepoints_t)[0]) / (2 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-7)


def test_the_trend_is_continuous_at_every_changepoint():
    """What gamma exists for. A discontinuity would mean the offsets are being
    computed independently rather than carried forward."""
    rng = np.random.default_rng(2)
    changepoints_t = np.array([0.2, 0.5, 0.75])
    delta = rng.normal(scale=0.3, size=3)
    t = np.sort(np.concatenate([np.linspace(0, 1, 400), changepoints_t - 1e-9,
                                changepoints_t + 1e-9]))
    A = (t[:, None] >= changepoints_t) * 1.0

    trend, _ = logistic_trend_and_jacobian(1.5, 0.4, delta, t, np.full(len(t), 1.3),
                                           A, changepoints_t)

    for breakpoint in changepoints_t:
        before = trend[np.searchsorted(t, breakpoint) - 1]
        after = trend[np.searchsorted(t, breakpoint)]
        assert abs(after - before) < 1e-6


# -- the trend and its gradient -----------------------------------------

def test_the_trend_saturates_at_the_capacity():
    t = np.linspace(0, 40, 500)
    cap = np.full(len(t), 2.0)
    trend, _ = logistic_trend_and_jacobian(1.0, 0.0, np.zeros(0), t, cap,
                                           np.zeros((len(t), 0)), np.zeros(0))

    assert trend[-1] == pytest.approx(2.0, rel=1e-6)
    assert np.all(np.diff(trend) >= -1e-12)
    assert np.all(trend <= cap + 1e-12)


def test_the_sigmoid_does_not_overflow_on_extreme_arguments():
    """The naive 1/(1+exp(-z)) overflows for z very negative, which a fit can
    reach while the optimizer is far from the optimum."""
    t = np.linspace(-500, 500, 200)
    trend, jacobian = logistic_trend_and_jacobian(
        3.0, 0.0, np.zeros(0), t, np.full(len(t), 1.0), np.zeros((len(t), 0)), np.zeros(0))

    assert np.all(np.isfinite(trend))
    assert np.all(np.isfinite(jacobian))
    assert trend[0] == pytest.approx(0.0, abs=1e-12)
    assert trend[-1] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("multiplicative", [False, True])
def test_analytic_gradient_matches_finite_differences(peyton_manning_df,
                                                      compiled_optimizer_module,
                                                      multiplicative):
    """The acceptance criterion. Run in both modes, since the multiplicative
    multiplier is applied outside the trend Jacobian and a mistake in how the
    two compose would only show with both on."""
    df = with_cap(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = logistic_model()
    if multiplicative:
        model.seasonality_mode = "multiplicative"
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    rng = np.random.default_rng(0)
    params = np.concatenate(([1.2], [-0.4],
                             rng.normal(scale=0.02, size=model.layout.n_changepoints),
                             [0.6],
                             rng.normal(scale=0.2, size=model.layout.n_regressor_columns)))

    analytic = model._gradient(params, include_l1_prior=False)
    step = 1e-6
    numerical = np.empty_like(params)
    for i in range(len(params)):
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        numerical[i] = (model._minus_log_posterior(up, include_l1_prior=False)
                        - model._minus_log_posterior(down, include_l1_prior=False)) / (2 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-5)


def test_both_languages_agree_under_logistic_growth(peyton_manning_df,
                                                    cpp_mlp_and_gradient,
                                                    compiled_optimizer_module):
    """The recursion is transcribed twice. This is the only thing that says
    the two transcriptions agree."""
    df = with_cap(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = logistic_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    rng = np.random.default_rng(0)
    params = np.concatenate(([1.2], [-0.4],
                             rng.normal(scale=0.02, size=model.layout.n_changepoints),
                             [0.6],
                             rng.normal(scale=0.2, size=model.layout.n_regressor_columns)))

    expected, expected_gradient = model._minus_log_posteriorAndGradient(params)
    value, gradient = cpp_mlp_and_gradient(model, canonical_to_cpp(params, model.layout))

    assert value == pytest.approx(expected, rel=1e-12)
    sigma_obs = params[model.layout.sigma_obs_idx]
    reordered = np.concatenate((
        expected_gradient[:2 + model.layout.n_changepoints],
        expected_gradient[model.layout.beta],
        [expected_gradient[model.layout.sigma_obs_idx] * sigma_obs]))
    np.testing.assert_allclose(gradient, reordered, rtol=1e-9, atol=1e-8)


# -- the data it needs --------------------------------------------------

def test_logistic_growth_requires_a_cap_column(peyton_manning_df,
                                               compiled_optimizer_module):
    model = logistic_model()
    with pytest.raises(ValueError, match='column "cap"'):
        model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                      lib_path=compiled_optimizer_module)


def test_a_cap_below_the_floor_is_rejected(peyton_manning_df,
                                           compiled_optimizer_module):
    df = peyton_manning_df.iloc[:300].reset_index(drop=True).assign(cap=-1.0)
    with pytest.raises(ValueError, match="cap must be greater than floor"):
        logistic_model().fit_cpp(df, lib_path=compiled_optimizer_module)


def test_an_unsupported_growth_mode_says_so(peyton_manning_df,
                                            compiled_optimizer_module):
    """'flat' was on this list until #16 task 13; anything outside the three
    Stan knows about still is."""
    model = AnalyticProphet()
    model.growth = "quadratic"
    with pytest.raises(ValueError, match="flat, linear, logistic"):
        model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                      lib_path=compiled_optimizer_module)


def test_predict_requires_the_cap_column(peyton_manning_df, compiled_optimizer_module):
    """The future capacity is data and may differ from the history's, so it
    comes from the frame passed to predict rather than being carried over."""
    df = with_cap(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = logistic_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=30)
    with pytest.raises(ValueError, match='column "cap"'):
        model.predict(future)

    forecast = model.predict(future.assign(cap=df["cap"].iloc[0]))
    assert np.all(np.isfinite(forecast["yhat"].values))


def test_a_forecast_stays_under_a_raised_capacity(peyton_manning_df,
                                                  compiled_optimizer_module):
    """The point of the mode: the trend cannot exceed the capacity it is
    given, and a different future capacity changes the forecast."""
    df = with_cap(peyton_manning_df.iloc[:1000].reset_index(drop=True))
    model = logistic_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=365)
    low = model.predict(future.assign(cap=df["cap"].iloc[0]))
    high = model.predict(future.assign(cap=df["cap"].iloc[0] * 2))

    assert np.all(low["trend"].values <= df["cap"].iloc[0] + 1e-8)
    assert high["trend"].values[-1] > low["trend"].values[-1]


def test_compute_trend_agrees_with_the_fitted_trend(peyton_manning_df,
                                                    compiled_optimizer_module):
    """predict() and trend_forecast_uncertainty() share predict_trend, which
    has to produce the same logistic curve the objective fitted."""
    df = with_cap(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = logistic_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    k, m = model.params["k"][0][0], model.params["m"][0][0]
    delta = model.params["delta"][0]
    A, _ = model._design_matrices()

    direct, _ = logistic_trend_and_jacobian(k, m, delta, model.t,
                                            model.cap_scaled, A, model.changepoints_t)
    shared = predict_trend(k, m, delta, model.changepoints_t, model.t,
                           model.y_scale, model.cap_scaled, model.floor)

    np.testing.assert_allclose(shared, direct * model.y_scale, rtol=1e-12)


# -- initialization -----------------------------------------------------

def test_logistic_init_puts_the_curve_through_the_endpoints():
    """[fc] logistic_growth_init."""
    t = np.linspace(0, 1, 50)
    cap = np.full(50, 2.0)
    y = cap / (1 + np.exp(-(1.3 * (t - 0.45))))

    k, m = logistic_growth_init(t, y, cap)
    fitted, _ = logistic_trend_and_jacobian(k, m, np.zeros(0), t, cap,
                                            np.zeros((50, 0)), np.zeros(0))

    assert fitted[0] == pytest.approx(y[0], rel=0.02)
    assert fitted[-1] == pytest.approx(y[-1], rel=0.02)


def test_init_clamps_y_outside_the_capacity():
    """[fc] forces y into (0, cap) before taking logs, since a y above cap
    would make log(r - 1) undefined."""
    t = np.linspace(0, 1, 20)
    cap = np.full(20, 1.0)
    y = np.full(20, 5.0)             # entirely above the capacity

    k, m = logistic_growth_init(t, y, cap)
    assert np.isfinite(k) and np.isfinite(m)


# -- linear growth is untouched -----------------------------------------

def test_linear_growth_takes_its_own_path(peyton_manning_df, compiled_optimizer_module):
    model = AnalyticProphet()
    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.growth == "linear"
    assert model.cap_scaled is None


# -- against Prophet ----------------------------------------------------

def test_objective_and_posterior_agree_with_prophet(prophet_comparison,
                                                    compiled_optimizer_module):
    """The acceptance criterion's other half: our objective is Stan's under
    logistic growth too."""
    Prophet, common, bridge = prophet_comparison
    df = with_cap(common.load_data(1000))
    kwargs = dict(common.PROPHET_KWARGS)
    kwargs["growth"] = "logistic"

    prophet_model = Prophet(**kwargs)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    assert stan_data["trend_indicator"] == 1
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, prophet_params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = logistic_model()
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    np.testing.assert_allclose(ours.cap_scaled,
                               np.asarray(stan_data["cap"], dtype=float), rtol=1e-12)

    rng = np.random.default_rng(0)
    sums = []
    for scale in (0.0, 0.02, 0.1):
        point = ours.get_parameters().copy()
        if scale:
            point[:2] += rng.normal(scale=scale, size=2)
            point[ours.layout.delta] += rng.normal(scale=scale * 0.05, size=25)
            point[ours.layout.beta] += rng.normal(
                scale=scale, size=ours.layout.n_regressor_columns)
            point[ours.layout.sigma_obs_idx] = abs(point[ours.layout.sigma_obs_idx]) + 0.01
        sums.append(ours._minus_log_posterior(point)
                    + bridge.stan_log_prob(stan_model, stan_data, point[0], point[1],
                                           point[ours.layout.delta],
                                           point[ours.layout.sigma_obs_idx],
                                           point[ours.layout.beta]))

    assert abs(sums[0]) < 1e-3, f"objectives differ by {sums[0]} at the optimum"
    assert max(sums) - min(sums) < 1e-2, f"the difference varies across points: {sums}"

    lp_ours = bridge.stan_log_prob(stan_model, stan_data, ours.params["k"][0][0],
                                   ours.params["m"][0][0], ours.params["delta"][0],
                                   ours.sigma_obs, ours.params["beta"][0])
    assert lp_ours >= lp_prophet - 1e-6
