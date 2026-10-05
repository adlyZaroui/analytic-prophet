"""
Issue #16 task 9: built-in country holidays.

`add_country_holidays('US')` is not shorthand for writing the dates out. A frame
passed to `add_holidays` contains the occurrences it lists and no others, so a
forecast past the end of it silently loses them. A country is *resolved* against
whatever years a frame covers, at fit and at predict alike, so the holidays keep
coming in the future.

The generation itself is the `holidays` package, the same one Prophet uses --
imported lazily, because trend, seasonality and a hand-written holidays frame
all work without it.

Everything below is checked against Prophet's own `make_holidays_df` and
`get_holiday_names` rather than against expected dates: those move between
releases of `holidays`, and a test asserting that Juneteenth is a US holiday
from 2021 would be testing the package rather than this code.
"""
import numpy as np
import pandas as pd
import pytest

from analytic_prophet import (COUNTRY_CODE_SUBSTITUTIONS, AnalyticProphet,
                           get_holiday_names, make_holidays_df)

COUNTRIES = ["US", "FR", "UK", "DE", "IN"]


@pytest.fixture(scope="module")
def prophet_holiday_helpers(prophet_comparison):
    """Prophet's two, under names that cannot be mistaken for ours.

    Both implementations now call these `make_holidays_df` and
    `get_holiday_names` (#54), so importing Prophet's at module scope would
    shadow ours and leave these tests comparing Prophet against itself --
    passing while verifying nothing.
    """
    from prophet.make_holidays import get_holiday_names as prophet_names
    from prophet.make_holidays import make_holidays_df as prophet_frame
    return prophet_frame, prophet_names


# -- generation ---------------------------------------------------------

@pytest.mark.parametrize("country", COUNTRIES)
def test_the_frame_matches_prophets(prophet_holiday_helpers, country):
    prophet_frame, _ = prophet_holiday_helpers
    years = [2015, 2016]

    def normalize(frame):
        return frame.sort_values(["ds", "holiday"]).reset_index(drop=True)

    assert normalize(make_holidays_df(years, country)).equals(
        normalize(prophet_frame(years, country)))


@pytest.mark.parametrize("country", COUNTRIES)
def test_the_name_set_matches_prophets(prophet_holiday_helpers, country):
    _, prophet_names = prophet_holiday_helpers
    assert get_holiday_names(country) == prophet_names(country)


def test_one_date_carrying_several_names_becomes_several_rows():
    """[fc] make_holidays_df explodes the name list, so a date with two
    holidays gets a row each -- and therefore a column each."""
    frame = make_holidays_df(range(2010, 2021), "US")
    counts = frame.groupby("ds").size()

    assert frame["ds"].is_unique or counts.max() > 1
    assert not frame["holiday"].isnull().any()
    assert frame["ds"].dtype.kind == "M"


def test_an_unsupported_country_says_so():
    with pytest.raises(AttributeError, match="not currently supported"):
        make_holidays_df([2020], "Atlantis")


def test_the_turkey_code_substitution_is_carried():
    """[fc] the one substitution in get_country_holidays_class."""
    assert COUNTRY_CODE_SUBSTITUTIONS == {"TU": "TR"}
    assert get_holiday_names("TU") == get_holiday_names("TR")


# -- registering --------------------------------------------------------

def test_add_country_holidays_returns_self_and_records_the_country():
    model = AnalyticProphet()
    assert model.add_country_holidays("US") is model
    assert model.country_holidays == "US"


def test_registering_a_second_country_replaces_the_first_and_warns(caplog):
    """[fc] one country at a time, with a warning rather than a silent swap."""
    model = AnalyticProphet().add_country_holidays("US")

    with caplog.at_level("WARNING", logger="analytic_prophet"):
        model.add_country_holidays("FR")

    assert model.country_holidays == "FR"
    assert any("Changing country holidays" in record.message for record in caplog.records)


def test_registering_the_same_country_twice_is_quiet(caplog):
    model = AnalyticProphet().add_country_holidays("US")
    with caplog.at_level("WARNING", logger="analytic_prophet"):
        model.add_country_holidays("US")
    assert not [r for r in caplog.records if "Changing country holidays" in r.message]


def test_a_seasonality_may_not_take_a_country_holiday_name():
    """The names are validated when the country is registered, so the
    collision is found before any frame is built around it."""
    model = AnalyticProphet().add_country_holidays("US")

    with pytest.raises(ValueError, match="is a holiday name in US"):
        model.add_seasonality("Christmas Day", 30.5, 5)


def test_a_country_may_be_merged_with_a_hand_written_frame():
    """check_holidays=False when validating the country's own names: merging
    is the documented behaviour, so naming the same holiday in both is fine."""
    frame = pd.DataFrame({"holiday": "Christmas Day",
                          "ds": [pd.Timestamp("2020-12-25")],
                          "lower_window": -1, "upper_window": 1})
    model = AnalyticProphet().add_holidays(frame).add_country_holidays("US")

    assert model.country_holidays == "US"
    assert model.holidays is not None


def test_adding_a_country_after_a_fit_is_refused(peyton_manning_df,
                                                 compiled_optimizer_module):
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    with pytest.raises(RuntimeError, match="before fitting"):
        model.add_country_holidays("US")


# -- reaching the fit ---------------------------------------------------

def test_country_holidays_extend_the_design_matrix(peyton_manning_df,
                                                   compiled_optimizer_module):
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)

    plain = AnalyticProphet()
    plain.fit(df, lib_path=compiled_optimizer_module)

    with_country = AnalyticProphet().add_country_holidays("US")
    with_country.fit(df, lib_path=compiled_optimizer_module)

    assert with_country.layout.n_holiday_columns > 0
    assert plain.layout.n_holiday_columns == 0
    assert with_country.layout.n_seasonality_columns == plain.layout.n_seasonality_columns
    assert np.all(np.isfinite(with_country.get_parameters()))


def test_a_country_and_a_frame_combine(peyton_manning_df, compiled_optimizer_module):
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    frame = pd.DataFrame({"holiday": "custom",
                          "ds": [pd.Timestamp(f"{y}-03-15") for y in (2008, 2009, 2010)],
                          "lower_window": 0, "upper_window": 0})

    country_only = AnalyticProphet().add_country_holidays("US")
    country_only.fit(df, lib_path=compiled_optimizer_module)

    both = AnalyticProphet().add_holidays(frame).add_country_holidays("US")
    both.fit(df, lib_path=compiled_optimizer_module)

    assert both.layout.n_holiday_columns == country_only.layout.n_holiday_columns + 1
    assert "custom" in both.train_holiday_names


def test_holidays_keep_coming_past_the_end_of_the_history(peyton_manning_df,
                                                          compiled_optimizer_module):
    """The reason a country is not shorthand for a frame: it is resolved
    against whatever years a frame covers, so a forecast into a year the
    history never saw still gets its holidays."""
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    model = AnalyticProphet().add_country_holidays("US")
    model.fit(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=400)
    features, _ = model._holiday_design(future["ds"])

    assert features.shape == (len(df) + 400, model.layout.n_holiday_columns)
    assert features[len(df):].sum() > 0, "no holidays generated beyond the history"


def test_predict_runs_with_country_holidays(peyton_manning_df, compiled_optimizer_module):
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    model = AnalyticProphet().add_country_holidays("US")
    model.fit(df, lib_path=compiled_optimizer_module)

    forecast = model.predict(model.make_future_dataframe(periods=400))

    assert len(forecast) == len(df) + 400
    assert np.all(np.isfinite(forecast["yhat"].values))


# -- against Prophet ----------------------------------------------------

def test_design_matrix_and_posterior_match_prophets(prophet_comparison,
                                                    compiled_optimizer_module):
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.add_country_holidays("US")
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, prophet_params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = AnalyticProphet().add_country_holidays("US")
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=compiled_optimizer_module)

    _, X_ours = ours._design_matrices()
    X_stan = np.asarray(stan_data["X"], dtype=float)
    assert X_ours.shape == X_stan.shape

    # exact on the holiday block -- these are 0/1 indicators, so a difference
    # would be a column on the wrong day rather than arithmetic
    holiday = ours.layout.holiday_block
    np.testing.assert_array_equal(X_ours[:, holiday], X_stan[:, holiday])
    seasonal = ours.layout.seasonality_block
    assert np.max(np.abs(X_ours[:, seasonal] - X_stan[:, seasonal])) < 1e-10

    np.testing.assert_array_equal(ours.sigmas, np.asarray(stan_data["sigmas"], dtype=float))

    lp_ours = bridge.stan_log_prob(stan_model, stan_data, ours.params["k"][0][0],
                                   ours.params["m"][0][0], ours.params["delta"][0],
                                   ours.sigma_obs, ours.params["beta"][0])
    assert lp_ours >= lp_prophet - 1e-6
