"""
Issue #16 task 7: conditional seasonalities.

`condition_name` names a boolean column on the frame passed to fit() and
predict(). Rows where it is False have that component's features zeroed, so the
component contributes nothing there while keeping its columns -- `beta` does not
change width, only which rows inform it.

Prophet's own example is an NFL series where a weekly pattern exists during the
season and not outside it. Fitting one weekly component over both regimes would
average them into something that describes neither.

The acceptance criterion is at the bottom: the feature matrix reproduces
Prophet's column for column, checked by equality rather than a tolerance.
"""
import numpy as np
import pandas as pd
import pytest

from customProphet import (CustomProphet, condition_masks, condition_matrix,
                           fourier_components, seasonal_time,
                           seasonality_design_matrix)

CONDITION = "on_season"


def with_condition(df, months=(9, 10, 11, 12, 1)):
    """Prophet's documented example: an on-season flag by month."""
    return df.assign(**{CONDITION: pd.to_datetime(df["ds"]).dt.month.isin(months)})


def conditioned_model():
    return CustomProphet().add_seasonality("on_season_weekly", 7, 3,
                                           condition_name=CONDITION)


# -- validating the condition column ------------------------------------

def test_a_missing_condition_column_is_rejected(peyton_manning_df, compiled_optimizer_module):
    """[fc] "Condition {name!r} missing from dataframe"."""
    model = conditioned_model()

    with pytest.raises(ValueError, match=f"Condition '{CONDITION}' missing"):
        model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                      lib_path=compiled_optimizer_module)


def test_a_non_boolean_condition_column_is_rejected(peyton_manning_df):
    """[fc] "Found non-boolean in column {name!r}"."""
    df = peyton_manning_df.iloc[:100].reset_index(drop=True).assign(**{CONDITION: 2.5})

    with pytest.raises(ValueError, match=f"non-boolean in column '{CONDITION}'"):
        condition_masks(conditioned_model().seasonalities, df)


@pytest.mark.parametrize("column,expected", [
    ([True, False, True], [True, False, True]),
    ([1, 0, 1], [True, False, True]),           # 1 == True, so Prophet accepts these
    ([1.0, 0.0, 1.0], [True, False, True]),
])
def test_integer_indicators_are_accepted_like_prophet(column, expected):
    """[fc] the test is `isin([True, False])`, and `1 == True` in pandas, so an
    integer or float 0/1 indicator passes. Worth pinning: a stricter check
    would reject a column Prophet accepts."""
    registry = conditioned_model().seasonalities
    masks = condition_masks(registry, pd.DataFrame({CONDITION: column}))

    np.testing.assert_array_equal(masks["on_season_weekly"], expected)


def test_nan_in_the_condition_column_is_rejected():
    with pytest.raises(ValueError, match="non-boolean"):
        condition_masks(conditioned_model().seasonalities,
                        pd.DataFrame({CONDITION: [True, np.nan, False]}))


def test_an_unconditioned_registry_needs_no_columns(peyton_manning_df,
                                                    compiled_optimizer_module):
    """The common case must not start demanding columns."""
    model = CustomProphet()
    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.condition_masks == {}


# -- what the mask does to the matrix -----------------------------------

def test_excluded_rows_are_zeroed_and_the_columns_stay(peyton_manning_df):
    """[fc] `features[~df[condition_name]] = 0`. Zeroing rows rather than
    dropping columns is what keeps `beta` the same width."""
    df = with_condition(peyton_manning_df.iloc[:400].reset_index(drop=True))
    registry = conditioned_model().seasonalities
    masks = condition_masks(registry, df)
    mask = masks["on_season_weekly"]

    x = seasonality_design_matrix(seasonal_time(df["ds"]), registry, masks)

    assert x.shape == (len(df), 6)
    assert np.all(x[~mask] == 0)
    assert np.all(np.abs(x[mask]).sum(axis=1) > 0)
    assert 0 < mask.sum() < len(df), "the fixture must exercise both regimes"


def test_an_unconditioned_component_is_untouched(peyton_manning_df):
    df = with_condition(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = conditioned_model().add_seasonality("yearly", 365.25, 10)
    masks = condition_masks(model.seasonalities, df)

    t = seasonal_time(df["ds"])
    x = seasonality_design_matrix(t, model.seasonalities, masks)

    np.testing.assert_array_equal(x[:, 6:], fourier_components(t, 365.25, 10))


def test_an_all_false_condition_removes_the_component_entirely(peyton_manning_df):
    df = peyton_manning_df.iloc[:100].reset_index(drop=True).assign(**{CONDITION: False})
    registry = conditioned_model().seasonalities

    x = seasonality_design_matrix(seasonal_time(df["ds"]), registry,
                                  condition_masks(registry, df))
    assert np.all(x == 0)


def test_the_condition_matrix_is_empty_when_nothing_is_conditioned():
    """Empty means "no conditions" to the C++, so the common case carries no
    T x n matrix of ones."""
    model = CustomProphet().add_seasonality("monthly", 30.5, 5)
    assert condition_matrix(model.seasonalities, {}, 100).shape == (100, 0)


def test_the_condition_matrix_fills_unconditioned_components_with_ones():
    """One uniform rule in the C++ rather than an index of which components
    carry a condition -- so a component without one is all ones."""
    model = conditioned_model().add_seasonality("yearly", 365.25, 10)
    masks = {"on_season_weekly": np.array([True, False, True])}

    matrix = condition_matrix(model.seasonalities, masks, 3)

    assert matrix.shape == (3, 2)
    np.testing.assert_array_equal(matrix[:, 0], [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(matrix[:, 1], [1.0, 1.0, 1.0])


# -- the mask reaching the C++ ------------------------------------------

def test_both_objectives_agree_with_a_conditioned_component(peyton_manning_df,
                                                            cpp_module):
    """The Python and C++ design matrices are built separately; a mask applied
    in one and not the other would surface nowhere until the fits diverged."""
    from customProphet import SIGMA, canonical_to_cpp

    df = with_condition(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = conditioned_model()
    model.y = df["y"].values
    model.ds = pd.to_datetime(df["ds"])
    model.t_scaled = np.array((model.ds - model.ds.min()) / (model.ds.max() - model.ds.min()))
    model.T = len(df)
    model.t_seasonality = seasonal_time(model.ds)
    model._normalize_y()
    model.condition_masks = condition_masks(model.seasonalities, df)
    model._build_layout()
    model._generate_change_points()

    rng = np.random.default_rng(0)
    params = np.concatenate(([0.3], [-0.7], rng.normal(scale=0.01, size=25), [0.8],
                             rng.normal(scale=0.5, size=6)))
    expected, expected_gradient = model._minus_log_posteriorAndGradient(params)

    value, gradient = cpp_module.minus_log_posterior_and_gradient(
        params=canonical_to_cpp(params, model.layout), t_scaled=model.t_scaled,
        change_points=model.change_points, t_seasonality=model.t_seasonality,
        normalized_y=model.normalized_y, sigma_obs_prior_scale=0.5,
        sigma_k=model.sigma_k, sigma_m=model.sigma_m,
        sigmas=np.full(6, SIGMA), tau=model.tau,
        fourier_orders=[3], seasonality_periods=[7.0],
        seasonality_conditions=condition_matrix(model.seasonalities,
                                                model.condition_masks, model.T))

    assert value == pytest.approx(expected, rel=1e-12)
    sigma_obs = params[model.layout.sigma_obs_idx]
    reordered = np.concatenate((expected_gradient[:27], expected_gradient[model.layout.beta],
                                [expected_gradient[model.layout.sigma_obs_idx] * sigma_obs]))
    np.testing.assert_allclose(gradient, reordered, rtol=1e-9, atol=1e-8)


def test_cpp_rejects_a_condition_matrix_of_the_wrong_shape(prepared_model, cpp_module):
    from customProphet import SIGMA

    with pytest.raises(ValueError, match="one column per seasonality"):
        cpp_module.minus_log_posterior_and_gradient(
            params=np.zeros(prepared_model.layout.size), t_scaled=prepared_model.t_scaled,
            change_points=prepared_model.change_points,
            t_seasonality=prepared_model.t_seasonality,
            normalized_y=prepared_model.normalized_y, sigma_obs_prior_scale=0.5,
            sigma_k=prepared_model.sigma_k, sigma_m=prepared_model.sigma_m,
            sigmas=np.full(20, SIGMA), tau=prepared_model.tau,
            fourier_orders=[10], seasonality_periods=[365.25],
            seasonality_conditions=np.ones((len(prepared_model.t_scaled), 2)))


# -- fit and predict ----------------------------------------------------

def test_the_condition_reaches_the_fit(peyton_manning_df, compiled_optimizer_module):
    df = with_condition(peyton_manning_df.iloc[:1000].reset_index(drop=True))

    model = conditioned_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert "on_season_weekly" in model.condition_masks
    assert list(model.seasonalities) == ["on_season_weekly", "yearly", "weekly"]
    assert model.layout.n_seasonality_columns == 2 * (3 + 10 + 3)
    assert np.all(np.isfinite(model.opt_params))


def test_predict_requires_the_condition_column(peyton_manning_df,
                                               compiled_optimizer_module):
    """[fc] predict() re-runs setup_dataframe, so the column is required there
    too. make_future_dataframe emits only `ds`, so the caller must add it."""
    df = with_condition(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = conditioned_model()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=30)
    with pytest.raises(ValueError, match="missing from dataframe"):
        model.predict(future)

    forecast = model.predict(with_condition(future))
    assert len(forecast) == len(df) + 30
    assert np.all(np.isfinite(forecast["yhat"].values))


def test_predict_uses_the_mask_it_is_given(peyton_manning_df, compiled_optimizer_module):
    """The masks come from the frame passed to predict, not from the fit: a
    caller decides on which future rows the component applies.

    Auto-selection is off so the conditioned component is the only one --
    `seasonality` in the forecast is the total across components, so an
    unconditioned weekly term would mask the effect being measured.
    """
    df = with_condition(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = conditioned_model()
    model.yearly_seasonality = model.weekly_seasonality = model.daily_seasonality = False
    model.fit_cpp(df, lib_path=compiled_optimizer_module)
    assert list(model.seasonalities) == ["on_season_weekly"]

    future = model.make_future_dataframe(periods=30)
    on = model.predict(future.assign(**{CONDITION: True}))
    off = model.predict(future.assign(**{CONDITION: False}))

    assert np.all(off["seasonality"].values == 0)
    assert np.any(np.abs(on["seasonality"].values) > 1e-8)


# -- the acceptance criterion -------------------------------------------

# The Fourier basis is the same function as Prophet's but evaluated in a
# different order -- `(2*pi/period) * t*(i+1)` rather than `(i+1)/period *
# (2*pi*t)` -- so the two agree to floating point rather than exactly. See
# README, "Where this deviates on purpose", for the measurement behind keeping
# it. `t` is days since 1970, so angles reach ~15000 radians and one ULP there
# is ~1e-12 in sin/cos; 1e-10 is a couple of orders above that and still 1e8
# below anything the model cares about.
BASIS_TOLERANCE = 1e-10


def test_fourier_features_match_prophets_to_floating_point(prophet_comparison,
                                                           peyton_manning_df):
    """Not exactly equal, and deliberately so.

    An earlier revision matched Prophet's order of operations to make these
    bit-identical. It was reverted: measured against exact rational arithmetic
    neither order is more accurate -- 0.470 against 0.498 ULP, ours ahead in
    five of nine configurations and behind in four -- so the exactness bought
    nothing numerically while perturbing every fit by ~4e-5 relative.
    """
    Prophet, _, _ = prophet_comparison
    ds = pd.to_datetime(peyton_manning_df["ds"])

    for period, order in ((7.0, 3), (365.25, 10), (30.5, 5)):
        ours = fourier_components(seasonal_time(ds), period, order)
        theirs = Prophet.fourier_series(ds, period, order)
        assert np.max(np.abs(ours - theirs)) < BASIS_TOLERANCE
        assert ours.shape == theirs.shape


def test_the_feature_matrix_reproduces_prophets_column_for_column(
        prophet_comparison, compiled_optimizer_module):
    """The acceptance criterion for this task.

    Prophet is fitted on the same conditioned specification, and the matrix it
    hands Stan is compared against the one this implementation builds -- same
    shape, same column order, same values, including the zeroed rows.

    "Column for column" is checked to BASIS_TOLERANCE rather than exactly, for
    the reason given above; the zeroed rows are checked exactly, since zero is
    zero in either order of operations.
    """
    Prophet, common, bridge = prophet_comparison
    df = with_condition(common.load_data(1000))

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.add_seasonality("on_season_weekly", 7, 3, condition_name=CONDITION)
    _, stan_data, _ = bridge.capture_stan_model(prophet_model, df)
    X_stan = np.asarray(stan_data["X"], dtype=float)

    ours = conditioned_model()
    ours._generate_change_points = lambda: setattr(
        ours, "change_points", np.asarray(stan_data["t_change"], dtype=float))
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert list(ours.seasonalities) == list(prophet_model.seasonalities)
    X_ours = seasonality_design_matrix(ours.t_seasonality, ours.seasonalities,
                                       ours.condition_masks)

    assert X_ours.shape == X_stan.shape
    assert np.max(np.abs(X_ours - X_stan)) < BASIS_TOLERANCE

    # the conditioning itself is exact: an excluded row is zero on both sides
    mask = ours.condition_masks["on_season_weekly"]
    np.testing.assert_array_equal(X_ours[~mask, :6], 0.0)
    np.testing.assert_array_equal(X_stan[~mask, :6], 0.0)


def test_posterior_agrees_with_prophet_on_a_conditioned_model(prophet_comparison,
                                                              compiled_optimizer_module):
    Prophet, common, bridge = prophet_comparison
    df = with_condition(common.load_data(1000))

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.add_seasonality("on_season_weekly", 7, 3, condition_name=CONDITION)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, prophet_params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    t_change = np.asarray(stan_data["t_change"], dtype=float)

    ours = conditioned_model()
    ours._generate_change_points = lambda: setattr(ours, "change_points", t_change.copy())
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    lp_ours = bridge.stan_log_prob(stan_model, stan_data, ours.opt_params[0],
                                   ours.opt_params[1], ours.opt_params[2:2 + len(t_change)],
                                   ours.sigma_obs, ours.opt_params[ours.layout.beta])
    assert lp_ours >= lp_prophet - 1e-6, (
        f"our posterior is worse on a conditioned model: {lp_ours} < {lp_prophet}")
