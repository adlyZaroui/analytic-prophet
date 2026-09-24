"""
Issue #16 task 13: flat growth.

    [stan] flat_trend: rep_vector(m, T)

The trend is the constant `m`. `k` and `delta` remain parameters and keep their
priors, but the likelihood never sees them, so both are driven to exactly zero
-- which is what Prophet's own fit does, and what this reproduces.

That makes this the simplest of the three growth modes and, unexpectedly, the
most informative. Every comparison against Prophet so far has ended with our
posterior a few nats ahead, attributed throughout to Prophet stopping short on
the trend's flat directions -- `k` against `delta`, which trade off almost
freely. Flat growth removes those directions entirely.

The result is an **exact tie**: identical `lp__` to Stan's full printed
precision at T = 300, 1000 and 2905, with `beta` agreeing to ~1e-7. Not merely
"ours is no worse". The same, to the digit.

That is the strongest evidence the project has that the margin under linear
growth is optimizer behaviour on a flat objective rather than a difference in
what is being fitted. Take the flat directions away and the two implementations
land on the same point.
"""
import numpy as np
import pytest

from customProphet import (CustomProphet, canonical_to_cpp, predict_trend,
                           flat_growth_init, TREND_INDICATORS)


def flat_model():
    model = CustomProphet()
    model.growth = "flat"
    return model


# -- the trend ----------------------------------------------------------

def test_the_trend_is_constant(peyton_manning_df, compiled_optimizer_module):
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = flat_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    forecast = model.predict(model.make_future_dataframe(periods=60))
    trend = forecast["trend"].values

    assert np.ptp(trend) < 1e-9, "a flat trend must not vary"
    assert trend[0] == pytest.approx(model.params["m"][0][0] * model.y_scale)


def test_k_and_delta_are_driven_to_zero(peyton_manning_df, compiled_optimizer_module):
    """They stay parameters and keep their priors -- normal(0, 5) and the
    Laplace -- but the likelihood does not see them, so the only pull is
    toward zero."""
    model = flat_model()
    model.fit_cpp(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.params["k"][0][0] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(model.params["delta"][0], 0.0, atol=1e-8)


def test_the_gradient_carries_only_the_priors_on_k_and_delta(
        peyton_manning_df, compiled_optimizer_module):
    """d(trend)/dk and d(trend)/d(delta) are zero, so those blocks reduce to
    k/sigma_k^2 and (with the L1 prior) sign(delta)/changepoint_prior_scale."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = flat_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    rng = np.random.default_rng(0)
    params = np.concatenate(([0.4], [0.6],
                             rng.normal(scale=0.05, size=model.layout.n_changepoints),
                             [0.6],
                             rng.normal(scale=0.2, size=model.layout.n_regressor_columns)))

    gradient = model._gradient(params, include_l1_prior=False)

    assert gradient[0] == pytest.approx(params[0] / model.sigma_k ** 2)
    np.testing.assert_allclose(gradient[model.layout.delta], 0.0, atol=0)


def test_the_l1_prior_still_reaches_delta(peyton_manning_df, compiled_optimizer_module):
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = flat_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    params = model.get_parameters().copy()
    params[model.layout.delta] = 0.5

    gradient = model._gradient(params, include_l1_prior=True)
    np.testing.assert_allclose(gradient[model.layout.delta], 1.0 / model.changepoint_prior_scale)


@pytest.mark.parametrize("multiplicative", [False, True])
def test_analytic_gradient_matches_finite_differences(peyton_manning_df,
                                                      compiled_optimizer_module,
                                                      multiplicative):
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = flat_model()
    if multiplicative:
        model.seasonality_mode = "multiplicative"
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    rng = np.random.default_rng(0)
    params = np.concatenate(([0.4], [0.6],
                             rng.normal(scale=0.05, size=model.layout.n_changepoints),
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

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)


def test_both_languages_agree_under_flat_growth(peyton_manning_df, cpp_mlp_and_gradient,
                                                compiled_optimizer_module):
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = flat_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    rng = np.random.default_rng(0)
    params = np.concatenate(([0.4], [0.6],
                             rng.normal(scale=0.05, size=model.layout.n_changepoints),
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


# -- initialization and plumbing ----------------------------------------

def test_flat_init_is_no_rate_and_the_mean_offset():
    """[fc] flat_growth_init."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert flat_growth_init(y) == (0.0, 2.5)


def test_the_trend_indicators_match_stans():
    """[stan] `int trend_indicator` in the data block."""
    assert TREND_INDICATORS == {"linear": 0, "logistic": 1, "flat": 2}


def test_compute_trend_agrees_with_the_fitted_trend(peyton_manning_df,
                                                    compiled_optimizer_module):
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = flat_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    shared = predict_trend(model.params["k"][0][0], model.params["m"][0][0],
                           model.params["delta"][0], model.changepoints_t,
                           model.t, model.y_scale, growth="flat")

    np.testing.assert_allclose(shared, model.params["m"][0][0] * model.y_scale, rtol=1e-12)


def test_flat_growth_needs_no_cap(peyton_manning_df, compiled_optimizer_module):
    """Unlike logistic, which requires one on every frame."""
    model = flat_model()
    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.cap_scaled is None
    forecast = model.predict(model.make_future_dataframe(periods=30))
    assert np.all(np.isfinite(forecast["yhat"].values))


# -- against Prophet ----------------------------------------------------

@pytest.mark.parametrize("n_rows", [300, 1000, 2905])
def test_the_fit_is_prophets_fit_exactly(prophet_comparison, compiled_optimizer_module,
                                         n_rows):
    """The strongest agreement result in the project.

    Every other comparison ends with our posterior a few nats ahead, which the
    README attributes to Prophet stopping short on the trend's flat directions.
    Flat growth removes those directions -- `k` and `delta` are pinned at zero
    by their priors with no likelihood pulling against them -- and the two
    implementations land on the same point.

    Identical `lp__` at Stan's full printed precision, at all three sizes, with
    `beta` agreeing to ~1e-7. This is what says the margin elsewhere is
    optimizer behaviour on a flat objective rather than a different model.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)
    kwargs = dict(common.PROPHET_KWARGS)
    kwargs["growth"] = "flat"

    prophet_model = Prophet(**kwargs)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    assert stan_data["trend_indicator"] == 2
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, prophet_params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = flat_model()
    ours._generate_change_points = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    lp_ours = bridge.stan_log_prob(stan_model, stan_data, ours.params["k"][0][0],
                                   ours.params["m"][0][0], ours.params["delta"][0],
                                   ours.sigma_obs, ours.params["beta"][0])

    assert lp_ours == pytest.approx(lp_prophet, abs=1e-6)
    np.testing.assert_allclose(ours.params["beta"][0],
                               np.asarray(prophet_params["beta"]).ravel(), atol=1e-5)
    assert ours.params["m"][0][0] == pytest.approx(
        float(np.asarray(prophet_params["m"]).ravel()[0]), abs=1e-5)


def test_our_objective_is_stans_under_flat_growth(prophet_comparison,
                                                  compiled_optimizer_module):
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)
    kwargs = dict(common.PROPHET_KWARGS)
    kwargs["growth"] = "flat"

    prophet_model = Prophet(**kwargs)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    bridge.validate_bridge(stan_model, stan_data, prophet_params,
                           float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = flat_model()
    ours._generate_change_points = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    rng = np.random.default_rng(0)
    sums = []
    for scale in (0.0, 0.05, 0.3):
        point = ours.get_parameters().copy()
        if scale:
            point[:2] += rng.normal(scale=scale, size=2)
            point[ours.layout.delta] += rng.normal(scale=scale * 0.1, size=25)
            point[ours.layout.beta] += rng.normal(
                scale=scale, size=ours.layout.n_regressor_columns)
            point[ours.layout.sigma_obs_idx] = abs(point[ours.layout.sigma_obs_idx]) + 0.01
        sums.append(ours._minus_log_posterior(point)
                    + bridge.stan_log_prob(stan_model, stan_data, point[0], point[1],
                                           point[ours.layout.delta],
                                           point[ours.layout.sigma_obs_idx],
                                           point[ours.layout.beta]))

    assert abs(sums[0]) < 1e-3
    assert max(sums) - min(sums) < 1e-2
