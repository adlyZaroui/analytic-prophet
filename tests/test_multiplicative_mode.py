"""
Issue #16 task 11: multiplicative seasonality, holidays and mode indicators.

Until now every regressor column added to the trend. Prophet lets each one
either add or *scale*:

    [stan] y ~ normal_id_glm(X_sa, trend .* (1 + X_sm * beta), beta, sigma_obs)

with `s_a` and `s_m` marking each of the K columns. A multiplicative component's
contribution grows with the level of the series, which is what you want when a
weekly pattern is "10% above trend" rather than "+3 units".

This is the first task since the registry work that moves the mathematics, and
it moves all of it. Writing `M = 1 + X_sm·beta`:

    yhat            = g .* M + X_sa·beta
    d yhat / dk     = t .* M
    d yhat / dm     = M
    d yhat / ddelta = (A .* (t - s)) .* M
    d yhat / dbeta  = g .* X_sm + X_sa

so every trend block picks up `M` and `beta`'s picks up the trend. With `s_m`
zero, `M` is 1 and each reduces to the additive form.

That reduction is not just checked, it is *taken*: both languages branch on
whether any column is multiplicative. The reason is not speed. `y - g - s` and
`y - (g*1 + s)` differ in the last bits, and on this objective's flat
directions that was enough to move the scipy path to a point 2.96 nats worse --
a regression the size of the project's whole margin over Prophet.
"""
import numpy as np
import pytest

from analytic_prophet import (AnalyticProphet, canonical_to_cpp, seasonality,
                           seasonality_modes)


def multiplicative_model():
    model = AnalyticProphet()
    model.seasonality_mode = "multiplicative"
    return model


# -- the mode vector ----------------------------------------------------

def test_modes_repeat_across_each_components_block():
    registry = {"a": seasonality(365.25, 10),
                "b": seasonality(7.0, 3, mode="multiplicative")}
    modes = seasonality_modes(registry)

    assert modes.shape == (26,)
    np.testing.assert_array_equal(modes[:20], 0.0)
    np.testing.assert_array_equal(modes[20:], 1.0)


def test_s_a_and_s_m_partition_the_columns(peyton_manning_df, compiled_optimizer_module):
    """[stan] s_a and s_m are complementary indicators over the K columns."""
    model = AnalyticProphet()
    model.add_seasonality("monthly", 30.5, 5, mode="multiplicative")
    model.fit(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.s_m.shape == (model.layout.n_regressor_columns,)
    np.testing.assert_array_equal(model.s_a + model.s_m, 1.0)
    assert set(np.unique(model.s_m)) <= {0.0, 1.0}
    np.testing.assert_array_equal(model.s_m[:10], 1.0)     # monthly, registered first


def test_the_model_wide_mode_reaches_auto_selected_components(
        peyton_manning_df, compiled_optimizer_module):
    """[fc] set_auto_seasonalities registers with self.seasonality_mode."""
    model = multiplicative_model()
    model.fit(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert list(model.seasonalities) == ["yearly", "weekly"]
    assert all(p["mode"] == "multiplicative" for p in model.seasonalities.values())
    np.testing.assert_array_equal(model.s_m, 1.0)


def test_holidays_carry_their_own_mode(peyton_manning_df, compiled_optimizer_module):
    """[fc] holidays_mode, defaulting to seasonality_mode. Holidays may be
    multiplicative while the seasonality is additive, and vice versa."""
    import pandas as pd

    frame = pd.DataFrame({"holiday": "bump",
                          "ds": [pd.Timestamp(f"{y}-03-15") for y in (2008, 2009, 2010)],
                          "lower_window": 0, "upper_window": 0})
    model = AnalyticProphet().add_holidays(frame, mode="multiplicative")
    model.fit(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    np.testing.assert_array_equal(model.s_m[model.layout.seasonality_block], 0.0)
    np.testing.assert_array_equal(model.s_m[model.layout.holiday_block], 1.0)


def test_an_unrecognized_mode_is_rejected_rather_than_read_as_additive():
    """The check that survives now nothing is refused as unimplemented: an
    unknown mode would otherwise reach the design matrix as "not
    multiplicative", which is silently additive."""
    from analytic_prophet import check_seasonality_supported
    with pytest.raises(ValueError, match="additive"):
        check_seasonality_supported({"x": seasonality(7.0, 3, mode="sideways")})


# -- the gradient -------------------------------------------------------

def non_optimal_point(model, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate(([0.3], [-0.5],
                           rng.normal(scale=0.02, size=model.layout.n_changepoints),
                           [0.6],
                           rng.normal(scale=0.2, size=model.layout.n_regressor_columns)))


@pytest.mark.parametrize("mode", ["multiplicative", "mixed"])
def test_analytic_gradient_matches_finite_differences(peyton_manning_df,
                                                      compiled_optimizer_module, mode):
    """The acceptance criterion. Every block changed, so every block is checked
    -- a `multiplier` left off the trend blocks, or a trend factor left off
    beta's, shows up here and essentially nowhere else.
    """
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = AnalyticProphet()
    if mode == "multiplicative":
        model.seasonality_mode = "multiplicative"
    else:
        # one of each, so a formula correct only when every column shares a
        # mode does not pass
        model.weekly_seasonality = False
        model.add_seasonality("weekly", 7.0, 3, mode="multiplicative")
    model.fit(df, lib_path=compiled_optimizer_module)
    assert model._multiplicative

    params = non_optimal_point(model)
    analytic = model._gradient(params, include_l1_prior=False)

    step = 1e-6
    numerical = np.empty_like(params)
    for i in range(len(params)):
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        numerical[i] = (model._minus_log_posterior(up, include_l1_prior=False)
                        - model._minus_log_posterior(down, include_l1_prior=False)) / (2 * step)

    # delta sits at the L1 kink, but the prior is off on both sides here, so
    # every coordinate is comparable
    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-5)


def test_the_fused_objective_agrees_with_the_separate_ones(peyton_manning_df,
                                                           compiled_optimizer_module):
    """Three methods compute this; they must not drift apart."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = multiplicative_model()
    model.fit(df, lib_path=compiled_optimizer_module)
    params = non_optimal_point(model)

    fused_value, fused_gradient = model._minus_log_posteriorAndGradient(params)

    assert fused_value == pytest.approx(model._minus_log_posterior(params), rel=1e-15)
    np.testing.assert_array_equal(fused_gradient, model._gradient(params))


def test_both_languages_agree_in_multiplicative_mode(peyton_manning_df,
                                                     cpp_mlp_and_gradient,
                                                     compiled_optimizer_module):
    """The derivation is written twice. This is the only thing that says the
    two transcriptions agree."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = multiplicative_model()
    model.fit(df, lib_path=compiled_optimizer_module)
    params = non_optimal_point(model)

    expected, expected_gradient = model._minus_log_posteriorAndGradient(params)
    value, gradient = cpp_mlp_and_gradient(model, canonical_to_cpp(params, model.layout))

    assert value == pytest.approx(expected, rel=1e-12)
    sigma_obs = params[model.layout.sigma_obs_idx]
    reordered = np.concatenate((
        expected_gradient[:2 + model.layout.n_changepoints],
        expected_gradient[model.layout.beta],
        [expected_gradient[model.layout.sigma_obs_idx] * sigma_obs]))
    np.testing.assert_allclose(gradient, reordered, rtol=1e-9, atol=1e-8)


# -- the additive reduction ---------------------------------------------

def test_an_all_additive_model_takes_the_additive_branch(peyton_manning_df,
                                                         compiled_optimizer_module):
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert not model._multiplicative
    np.testing.assert_array_equal(model.s_m, 0.0)


def test_an_explicitly_additive_model_fits_exactly_as_before(peyton_manning_df,
                                                             compiled_optimizer_module):
    """Setting mode='additive' everywhere must be indistinguishable from not
    setting it -- same branch, same arithmetic, same fit."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)

    default = AnalyticProphet()
    default.fit(df, lib_path=compiled_optimizer_module)

    explicit = AnalyticProphet()
    explicit.seasonality_mode = "additive"
    explicit.fit(df, lib_path=compiled_optimizer_module)

    np.testing.assert_array_equal(default.get_parameters(), explicit.get_parameters())
    assert default.opt.n_iterations == explicit.opt.n_iterations


def test_the_multiplicative_formulas_reduce_to_the_additive_ones(peyton_manning_df,
                                                                 compiled_optimizer_module):
    """The branch is an optimization, not a different model. Forcing the
    general path on an all-additive model has to give the same answer, to
    floating point -- if it did not, the branch would be hiding a discrepancy
    rather than avoiding a rounding difference.
    """
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = AnalyticProphet()
    model.fit(df, lib_path=compiled_optimizer_module)
    params = non_optimal_point(model)

    fast_value, fast_gradient = model._minus_log_posteriorAndGradient(params)

    model._multiplicative = True          # take the general path on s_m = 0
    general_value, general_gradient = model._minus_log_posteriorAndGradient(params)

    assert general_value == pytest.approx(fast_value, rel=1e-12)
    np.testing.assert_allclose(general_gradient, fast_gradient, rtol=1e-9, atol=1e-9)


# -- predict and Prophet ------------------------------------------------

def test_a_multiplicative_component_scales_with_the_trend(peyton_manning_df,
                                                          compiled_optimizer_module):
    """The behaviour the mode exists for: the same coefficients produce a
    larger effect where the trend is larger."""
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    model = multiplicative_model()
    model.fit(df, lib_path=compiled_optimizer_module)

    forecast = model.predict(model.make_future_dataframe(periods=0))
    trend = forecast["trend"].values
    effect = np.abs(forecast["seasonality"].values)

    low, high = trend < np.percentile(trend, 25), trend > np.percentile(trend, 75)
    assert effect[high].mean() > effect[low].mean()

    # and yhat is still trend + what seasonality reports
    np.testing.assert_allclose(forecast["yhat"].values,
                               trend + forecast["seasonality"].values, rtol=1e-10)


def test_objective_and_posterior_agree_with_prophet(prophet_comparison,
                                                    compiled_optimizer_module):
    """Our objective is Stan's in multiplicative mode too: `ours + stan_lp__`
    is zero at the optimum and constant away from it."""
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)
    kwargs = dict(common.PROPHET_KWARGS)
    kwargs["seasonality_mode"] = "multiplicative"

    prophet_model = Prophet(**kwargs)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, prophet_params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = multiplicative_model()
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=compiled_optimizer_module)

    np.testing.assert_array_equal(ours.s_m, np.asarray(stan_data["s_m"], dtype=float))
    np.testing.assert_array_equal(ours.s_a, np.asarray(stan_data["s_a"], dtype=float))

    rng = np.random.default_rng(0)
    sums = []
    for scale in (0.0, 0.05, 0.3):
        point = ours.get_parameters().copy()
        if scale:
            point[:2] += rng.normal(scale=scale, size=2)
            point[ours.layout.delta] += rng.normal(scale=scale * 0.1, size=25)
            point[ours.layout.beta] += rng.normal(
                scale=scale * 0.05, size=ours.layout.n_regressor_columns)
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
