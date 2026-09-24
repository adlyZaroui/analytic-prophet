"""
Issue #16 task 8: holidays.

A holiday is an indicator column: 1 on the day it falls, 0 elsewhere, with an
optional window of days either side that each become their own column. The
columns join the seasonal ones in the same design matrix, with their own prior
scales in `sigmas`.

The objective and gradient needed no change for this, which is worth stating
because it is the kind of thing that looks like it should. Stan writes
`y ~ normal(X*beta, sigma_obs)` and `beta ~ normal(0, sigmas)` over *all* K
regressor columns, and this implementation has matched that shape since `K`
became registry-driven (task 2) and `sigmas` became per-column (task 5). A
holiday column is a column. What task 8 adds is how those columns are built and
kept aligned between fit and predict -- not the mathematics.

The awkward parts are alignment, not arithmetic:

  * Columns are sorted by *name*, not by the holidays frame's row order, so
    `beta` is indexed by the sort. Anything else misassigns every coefficient.
  * A window offset landing outside the frame still gets an all-zero column --
    fit and predict must present the same columns.
  * A holiday the fit never saw is dropped at predict time, and one the fit saw
    but the future frame lacks is kept, empty.
"""
import numpy as np
import pandas as pd
import pytest

from customProphet import (CustomProphet, make_holiday_features, validate_holidays_frame)

YEARS = range(2008, 2017)


def superbowls(lower=0, upper=0, **extra):
    frame = pd.DataFrame({
        "holiday": "superbowl",
        "ds": [pd.Timestamp(f"{y}-02-07") for y in YEARS],
        "lower_window": lower,
        "upper_window": upper,
    })
    for key, value in extra.items():
        frame[key] = value
    return frame


def two_holidays():
    playoffs = pd.DataFrame({
        "holiday": "playoff",
        "ds": [pd.Timestamp(f"{y}-01-12") for y in YEARS],
        "lower_window": 0,
        "upper_window": 2,
    })
    return pd.concat([superbowls(-1, 1), playoffs], ignore_index=True)


# -- validating the frame -----------------------------------------------

def test_a_frame_without_the_required_columns_is_rejected():
    model = CustomProphet()
    with pytest.raises(ValueError, match='"ds" and "holiday" columns'):
        model.add_holidays(pd.DataFrame({"holiday": ["x"]}))
    with pytest.raises(ValueError, match='"ds" and "holiday" columns'):
        model.add_holidays(pd.DataFrame({"ds": [pd.Timestamp("2020-01-01")]}))


def test_a_nan_anywhere_in_the_frame_is_rejected():
    with pytest.raises(ValueError, match="Found a NaN"):
        CustomProphet().add_holidays(pd.DataFrame(
            {"holiday": ["x", None], "ds": pd.to_datetime(["2020-01-01", "2020-01-02"])}))


def test_one_window_without_the_other_is_rejected():
    """[fc] "Holidays must have both lower_window and upper_window, or neither"."""
    frame = superbowls().drop(columns=["upper_window"])
    with pytest.raises(ValueError, match="both lower_window and upper_window"):
        CustomProphet().add_holidays(frame)


@pytest.mark.parametrize("lower,upper,message", [
    (1, 1, "lower_window should be <= 0"),
    (-1, -1, "upper_window should be >= 0"),
])
def test_windows_must_point_the_right_way(lower, upper, message):
    with pytest.raises(ValueError, match=message):
        CustomProphet().add_holidays(superbowls(lower, upper))


def test_a_holiday_may_not_take_a_reserved_or_taken_name():
    """The same validate_column_name add_seasonality uses (#16 task 6), so a
    holiday cannot shadow a seasonality or a predict() output column."""
    with pytest.raises(ValueError, match="reserved"):
        CustomProphet().add_holidays(superbowls().assign(holiday="trend"))

    model = CustomProphet().add_seasonality("monthly", 30.5, 5)
    with pytest.raises(ValueError, match="already used for a seasonality"):
        model.add_holidays(superbowls().assign(holiday="monthly"))


def test_the_same_holiday_may_repeat_across_years():
    """check_holidays=False on the name check: repeating a holiday is how its
    occurrences are listed, and must not read as a collision."""
    frame = validate_holidays_frame(superbowls(), CustomProphet().validate_column_name)
    assert len(frame) == len(list(YEARS))


def test_adding_holidays_after_a_fit_is_refused(peyton_manning_df, compiled_optimizer_module):
    model = CustomProphet()
    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    with pytest.raises(RuntimeError, match="before fitting"):
        model.add_holidays(superbowls())


def test_holidays_mode_is_validated_and_accepted():
    """Refused as unimplemented until task 11; fitted now."""
    with pytest.raises(ValueError, match='"additive" or "multiplicative"'):
        CustomProphet().add_holidays(superbowls(), mode="sideways")

    model = CustomProphet().add_holidays(superbowls(), mode="multiplicative")
    assert model.holidays_mode == "multiplicative"


def test_a_non_positive_prior_scale_is_refused():
    with pytest.raises(ValueError, match="Prior scale must be > 0"):
        CustomProphet().add_holidays(superbowls(), prior_scale=0)


# -- building the columns -----------------------------------------------

@pytest.mark.parametrize("frame_factory,label", [
    (lambda: superbowls(), "no windows"),
    (lambda: superbowls(-2, 1), "windows"),
    (lambda: two_holidays(), "two holidays, different windows"),
    (lambda: superbowls(-1, 1, prior_scale=2.0), "per-holiday prior scale"),
])
def test_columns_match_prophets_exactly(prophet_comparison, frame_factory, label):
    """Against Prophet's own make_holiday_features: same columns, same order,
    same values, same prior scale per column."""
    Prophet, _, _ = prophet_comparison
    frame = frame_factory()
    dates = pd.Series(pd.date_range("2010-01-01", "2012-12-31"))

    prophet_model = Prophet(holidays=frame)
    theirs, their_scales, _ = prophet_model.make_holiday_features(
        dates, prophet_model.construct_holiday_dataframe(dates))
    ours, our_scales, _ = make_holiday_features(dates, frame, 10.0)

    assert ours.shape == theirs.shape, label
    np.testing.assert_array_equal(ours, theirs.values)
    assert our_scales == their_scales


def test_a_window_becomes_one_column_per_offset():
    """[fc] `{holiday}_delim_{+/-}{n}`, one per offset in [lower, upper]."""
    dates = pd.Series(pd.date_range("2010-02-01", "2010-02-15"))
    features, scales, names = make_holiday_features(dates, superbowls(-2, 1), 10.0)

    assert features.shape == (15, 4)          # offsets -2, -1, 0, +1
    assert names == ["superbowl"]
    assert len(scales) == 4
    # 2010-02-07 is the holiday; offsets land on the 5th through the 8th
    hit_rows = np.flatnonzero(features.sum(axis=1))
    np.testing.assert_array_equal(dates.iloc[hit_rows].dt.day.values, [5, 6, 7, 8])


def test_a_holiday_outside_the_dates_still_gets_its_columns():
    """Fit and predict must present the same columns; an occurrence that misses
    the frame gives an all-zero column rather than no column."""
    dates = pd.Series(pd.date_range("2010-06-01", "2010-06-30"))
    features, _, _ = make_holiday_features(dates, superbowls(-1, 1), 10.0)

    assert features.shape == (30, 3)
    assert np.all(features == 0)


def test_columns_are_sorted_by_name_not_by_row_order():
    """Prophet sorts the columns, so `beta` is indexed by the sort. A frame
    listing holidays in a different order must give the same matrix."""
    dates = pd.Series(pd.date_range("2010-01-01", "2010-12-31"))
    forward = two_holidays()
    reversed_rows = forward.iloc[::-1].reset_index(drop=True)

    a, scales_a, _ = make_holiday_features(dates, forward, 10.0)
    b, scales_b, _ = make_holiday_features(dates, reversed_rows, 10.0)

    np.testing.assert_array_equal(a, b)
    assert scales_a == scales_b


def test_an_inconsistent_prior_scale_for_one_holiday_is_rejected():
    """[fc] "does not have consistent prior scale specification"."""
    frame = superbowls(prior_scale=2.0)
    frame.loc[0, "prior_scale"] = 5.0

    with pytest.raises(ValueError, match="consistent prior scale"):
        make_holiday_features(pd.Series(pd.date_range("2010-01-01", "2010-12-31")), frame, 10.0)


# -- reaching the fit ---------------------------------------------------

def test_holiday_columns_extend_the_parameter_vector(peyton_manning_df,
                                                     compiled_optimizer_module):
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)

    plain = CustomProphet()
    plain.fit_cpp(df, lib_path=compiled_optimizer_module)

    with_holidays = CustomProphet().add_holidays(two_holidays())
    with_holidays.fit_cpp(df, lib_path=compiled_optimizer_module)

    layout = with_holidays.layout
    assert layout.n_seasonality_columns == plain.layout.n_seasonality_columns
    assert layout.n_holiday_columns > 0
    assert layout.n_regressor_columns == (layout.n_seasonality_columns
                                          + layout.n_holiday_columns)
    assert with_holidays.get_parameters().shape == (layout.size,)
    assert np.all(np.isfinite(with_holidays.get_parameters()))


def test_the_holiday_prior_scale_lands_on_the_holiday_columns(peyton_manning_df,
                                                              compiled_optimizer_module):
    """`sigmas` spans every regressor column; the holiday block carries
    holidays_prior_scale, the seasonal block seasonality_prior_scale."""
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)

    model = CustomProphet().add_holidays(two_holidays(), prior_scale=3.0)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert model.sigmas.shape == (model.layout.n_regressor_columns,)
    np.testing.assert_array_equal(model.sigmas[model.layout.holiday_block], 3.0)
    np.testing.assert_array_equal(model.sigmas[model.layout.seasonality_block], 10.0)


def test_the_two_layout_slices_are_not_interchangeable():
    """`holidays` indexes the parameter vector, `holiday_block` the design
    matrix and sigmas. They differ by 3 + S, and mixing them returns a wrong
    slice silently rather than raising -- which is what this pins."""
    from customProphet import ParameterLayout
    layout = ParameterLayout(25, 26, 6)

    assert layout.holidays == slice(54, 60)
    assert layout.holiday_block == slice(26, 32)
    assert layout.seasonality_block == slice(0, 26)
    assert layout.n_regressor_columns == 32


def test_both_objectives_agree_with_holidays(peyton_manning_df, cpp_mlp_and_gradient,
                                             compiled_optimizer_module):
    """The holiday block is passed to the C++ rather than rebuilt there, since
    indicator columns are data and not derivable from a period. Python and C++
    must still land on the same matrix."""
    from customProphet import canonical_to_cpp

    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    model = CustomProphet().add_holidays(two_holidays())
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    rng = np.random.default_rng(0)
    params = np.concatenate((
        [0.3], [-0.7], rng.normal(scale=0.01, size=model.layout.n_changepoints),
        [0.8], rng.normal(scale=0.5, size=model.layout.n_regressor_columns)))

    expected, expected_gradient = model._minus_log_posteriorAndGradient(params)
    value, gradient = cpp_mlp_and_gradient(model, canonical_to_cpp(params, model.layout))

    assert value == pytest.approx(expected, rel=1e-12)
    sigma_obs = params[model.layout.sigma_obs_idx]
    reordered = np.concatenate((
        expected_gradient[:2 + model.layout.n_changepoints],
        expected_gradient[model.layout.beta],
        [expected_gradient[model.layout.sigma_obs_idx] * sigma_obs]))
    np.testing.assert_allclose(gradient, reordered, rtol=1e-9, atol=1e-8)


def test_a_holiday_effect_is_actually_fitted(peyton_manning_df, compiled_optimizer_module):
    """A synthetic bump on known dates has to show up in the holiday
    coefficients, or the columns are being carried and ignored."""
    df = peyton_manning_df.iloc[:1500].reset_index(drop=True).copy()
    bump_dates = [pd.Timestamp(f"{y}-03-15") for y in (2008, 2009, 2010, 2011)]
    is_bump = pd.to_datetime(df["ds"]).isin(bump_dates)
    df.loc[is_bump, "y"] = df.loc[is_bump, "y"] + 4.0

    frame = pd.DataFrame({"holiday": "bump", "ds": bump_dates,
                          "lower_window": 0, "upper_window": 0})
    model = CustomProphet().add_holidays(frame)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    coefficients = model.params["beta"][0][model.layout.holiday_block]
    assert coefficients.shape == (1,)
    # the bump is +4 on a series scaled by max|y|; the coefficient is in
    # normalized units, so compare there
    assert coefficients[0] > 0.5 * 4.0 / model.y_scale


# -- predict ------------------------------------------------------------

def test_predict_carries_the_holiday_columns(peyton_manning_df, compiled_optimizer_module):
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    model = CustomProphet().add_holidays(two_holidays())
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    forecast = model.predict(model.make_future_dataframe(periods=60))

    assert len(forecast) == len(df) + 60
    assert np.all(np.isfinite(forecast["yhat"].values))


def test_predict_drops_a_holiday_the_fit_never_saw(peyton_manning_df,
                                                   compiled_optimizer_module):
    """[fc] construct_holiday_dataframe drops names absent from training --
    there is no coefficient for them, so including them would misalign beta."""
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    model = CustomProphet().add_holidays(superbowls(-1, 1))
    model.fit_cpp(df, lib_path=compiled_optimizer_module)
    fitted_columns = model.layout.n_holiday_columns

    # a holiday appearing only now must not widen the design matrix
    model.holidays = pd.concat([model.holidays, pd.DataFrame(
        {"holiday": "newcomer", "ds": [pd.Timestamp("2016-06-01")],
         "lower_window": 0, "upper_window": 0})], ignore_index=True)

    features, _ = model._holiday_design(pd.Series(pd.to_datetime(df["ds"])))
    assert features.shape[1] == fitted_columns


def test_a_training_holiday_absent_from_the_future_keeps_its_empty_column(
        peyton_manning_df, compiled_optimizer_module):
    """[fc] holidays_to_add with ds as NA. The column must stay so `beta` keeps
    its alignment, even with nothing in it."""
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    model = CustomProphet().add_holidays(two_holidays())
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    far_future = pd.DataFrame({"ds": pd.date_range("2030-06-01", periods=30)})
    features, _ = model._holiday_design(far_future["ds"])

    assert features.shape == (30, model.layout.n_holiday_columns)
    assert np.all(features == 0)


# -- against Prophet ----------------------------------------------------

def test_design_matrix_and_sigmas_match_prophets(prophet_comparison,
                                                 compiled_optimizer_module):
    """The acceptance check: the full design matrix, seasonal columns and
    holiday columns together, against what Prophet hands Stan."""
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)
    frame = two_holidays()

    prophet_model = Prophet(holidays=frame, **common.PROPHET_KWARGS)
    _, stan_data, _ = bridge.capture_stan_model(prophet_model, df)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = CustomProphet().add_holidays(frame)
    ours._generate_change_points = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    _, X_ours = ours._design_matrices()
    X_stan = np.asarray(stan_data["X"], dtype=float)
    assert X_ours.shape == X_stan.shape

    # The seasonal block agrees to floating point rather than exactly: the
    # Fourier basis is evaluated in a different order from Prophet's, which is
    # a deliberate choice (README, "Where this deviates on purpose").
    seasonal, holiday = ours.layout.seasonality_block, ours.layout.holiday_block
    assert np.max(np.abs(X_ours[:, seasonal] - X_stan[:, seasonal])) < 1e-10

    # The holiday block is exact, and has to be: these are 0/1 indicators, so
    # no order of operations can perturb them. A difference here would be a
    # column landing on the wrong day, not arithmetic.
    np.testing.assert_array_equal(X_ours[:, holiday], X_stan[:, holiday])

    np.testing.assert_array_equal(ours.sigmas, np.asarray(stan_data["sigmas"], dtype=float))


def test_our_objective_is_stans_with_holidays(prophet_comparison,
                                              compiled_optimizer_module):
    """The objective needed no change for holidays -- a holiday column is just
    a column of X. This is what says so: `ours + stan_lp__` is zero at the
    optimum and constant away from it, exactly as it is without holidays.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)
    frame = two_holidays()

    prophet_model = Prophet(holidays=frame, **common.PROPHET_KWARGS)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, prophet_params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = CustomProphet().add_holidays(frame)
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

    assert abs(sums[0]) < 1e-3, f"objectives differ by {sums[0]} at the optimum"
    assert max(sums) - min(sums) < 1e-2, f"the difference varies across points: {sums}"

    lp_ours = bridge.stan_log_prob(stan_model, stan_data, ours.params["k"][0][0],
                                   ours.params["m"][0][0], ours.params["delta"][0],
                                   ours.sigma_obs, ours.params["beta"][0])
    assert lp_ours >= lp_prophet - 1e-6
