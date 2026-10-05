"""
Issue #52: the constructor takes Prophet's arguments.

It previously took none. Every setting was an attribute assigned afterwards, so
a script ported from Prophet had to be rewritten line by line — and the rewrite
was not mechanical, since one argument had a different name and one was a
method.

Two rules decide the shape of this:

  * **Prophet's name on the argument, Stan's on the attribute.**
    `changepoint_prior_scale` is what a user writes; `changepoint_prior_scale` is what
    `prophet.stan` calls it and what every derivation in this repository calls
    it. Renaming the attribute would make the gradient comments harder to
    follow than the API mismatch was worth.

  * **Accept and reject, rather than omit.** Arguments for things this does not
    do — `mcmc_samples`, `stan_backend`, `scaling='minmax'`, an explicit
    `changepoints` list — raise `NotImplementedError`. Omitting them would fail
    with an `AttributeError` that says nothing about why, and a ported script
    should fail where it is actually wrong.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet, N_CHANGE_POINTS, SIGMA, TAU


# -- the signature ------------------------------------------------------

def test_every_prophet_argument_is_accepted(prophet_comparison):
    """Name for name against the installed Prophet, so an argument added
    upstream shows up as a failure here rather than as a TypeError for a user."""
    Prophet, _, _ = prophet_comparison
    theirs = set(inspect.signature(Prophet.__init__).parameters) - {"self"}
    ours = set(inspect.signature(AnalyticProphet.__init__).parameters) - {"self"}

    assert theirs <= ours, f"missing from this constructor: {sorted(theirs - ours)}"


def test_the_defaults_match_prophets(prophet_comparison):
    """A default that silently differs is worse than a missing argument: the
    model fits something else and nothing says so."""
    Prophet, _, _ = prophet_comparison
    theirs = inspect.signature(Prophet.__init__).parameters
    ours = inspect.signature(AnalyticProphet.__init__).parameters

    # changepoint_prior_scale is Prophet's name for changepoint_prior_scale; the default is the
    # same value, checked below with the rest
    for name, parameter in theirs.items():
        if name == "self":
            continue
        assert ours[name].default == parameter.default, name


def test_the_constants_are_the_defaults():
    assert inspect.signature(AnalyticProphet.__init__).parameters["n_changepoints"].default == N_CHANGE_POINTS
    assert inspect.signature(AnalyticProphet.__init__).parameters["seasonality_prior_scale"].default == SIGMA
    assert inspect.signature(AnalyticProphet.__init__).parameters["changepoint_prior_scale"].default == TAU


# -- what the arguments set ---------------------------------------------

@pytest.mark.parametrize("argument,value,attribute", [
    ("growth", "logistic", "growth"),
    ("n_changepoints", 5, "n_changepoints"),
    ("changepoint_range", 0.9, "changepoint_range"),
    ("yearly_seasonality", False, "yearly_seasonality"),
    ("weekly_seasonality", 7, "weekly_seasonality"),
    ("daily_seasonality", True, "daily_seasonality"),
    ("seasonality_mode", "multiplicative", "seasonality_mode"),
    ("seasonality_prior_scale", 3.0, "seasonality_prior_scale"),
    ("holidays_prior_scale", 4.0, "holidays_prior_scale"),
    ("holidays_mode", "additive", "holidays_mode"),
    ("interval_width", 0.95, "interval_width"),
    ("uncertainty_samples", 200, "uncertainty_samples"),
])
def test_an_argument_reaches_its_attribute(argument, value, attribute):
    assert getattr(AnalyticProphet(**{argument: value}), attribute) == value


def test_changepoint_prior_scale_is_the_name_on_both_sides():
    """The argument and the attribute are both Prophet's name.

    This briefly kept Stan's `tau` on the attribute, on the grounds that the
    derivations in this repository are written against `prophet.stan`. That was
    reversed in #54: a user porting a script, or reading a traceback beside
    Prophet's, meets the attribute far more often than the derivations do.
    `[stan] tau` is now a comment where the arithmetic needs it.
    """
    model = AnalyticProphet(changepoint_prior_scale=0.01)

    assert model.changepoint_prior_scale == 0.01
    assert not hasattr(model, "tau"), (
        "keeping both would give two names for one number, which is how they "
        "drift apart")


def test_holidays_can_be_given_to_the_constructor(peyton_manning_df,
                                                  compiled_optimizer_module):
    """Prophet has no add_holidays method -- the frame is a constructor
    argument, and that is how a ported script will pass it."""
    frame = pd.DataFrame({"holiday": "bump",
                          "ds": [pd.Timestamp(f"{y}-03-15") for y in (2008, 2009)],
                          "lower_window": 0, "upper_window": 0})

    model = AnalyticProphet(holidays=frame)
    assert model.holidays is not None

    model.fit(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    # one column per (holiday, window offset), not per occurrence: two dates of
    # the same holiday with a zero window share the column `bump_delim_+0`
    assert model.layout.n_holiday_columns == 1


def test_a_bad_holidays_frame_is_rejected_at_construction():
    """Validated where it is given, rather than at fit time."""
    with pytest.raises(ValueError, match='"ds" and "holiday" columns'):
        AnalyticProphet(holidays=pd.DataFrame({"holiday": ["x"]}))


# -- what is accepted and then refused ----------------------------------

@pytest.mark.parametrize("kwargs,fragment", [
    ({"mcmc_samples": 100}, "MAP only"),
    ({"stan_backend": "cmdstanpy"}, "no Stan here"),
    ({"scaling": "minmax"}, "only 'absmax'"),
])
def test_unsupported_features_are_refused_not_ignored(kwargs, fragment):
    with pytest.raises(NotImplementedError, match=fragment):
        AnalyticProphet(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"mcmc_samples": 0},
    {"scaling": "absmax"},
    {"changepoints": None},
    {"stan_backend": None},
    {"changepoints": ["2008-01-01"]},   # supported as of #15
])
def test_the_supported_value_of_each_is_accepted(kwargs):
    """The refusals must not reject Prophet's own defaults, or every ported
    script fails immediately."""
    assert AnalyticProphet(**kwargs) is not None


def test_an_unknown_argument_is_a_plain_type_error():
    with pytest.raises(TypeError, match="unexpected keyword"):
        AnalyticProphet(not_a_setting=1)


# -- the interval ------------------------------------------------------

def test_interval_width_sets_the_quantiles(peyton_manning_df,
                                           compiled_optimizer_module):
    """[fc] predict_uncertainty: the edges are (1 -+ interval_width) / 2.
    These were hardcoded at [2.5, 97.5] -- a 95% interval where Prophet's
    default is 80%, so every interval this produced was wider than the one a
    user asked for."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)

    narrow = AnalyticProphet(interval_width=0.5, uncertainty_samples=400)
    narrow.fit(df, lib_path=compiled_optimizer_module)
    wide = AnalyticProphet(interval_width=0.95, uncertainty_samples=400)
    wide.fit(df, lib_path=compiled_optimizer_module)

    future = narrow.make_future_dataframe(periods=60)
    narrow_band = narrow.predict(future)
    wide_band = wide.predict(future)

    narrow_width = (narrow_band["trend_upper"] - narrow_band["trend_lower"]).mean()
    wide_width = (wide_band["trend_upper"] - wide_band["trend_lower"]).mean()
    assert wide_width > narrow_width


def test_uncertainty_samples_is_the_draw_count(peyton_manning_df,
                                               compiled_optimizer_module):
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    model = AnalyticProphet(uncertainty_samples=7)
    model.fit(df, lib_path=compiled_optimizer_module)

    class CountingGenerator:
        """Generator's methods are read-only, so the whole thing is wrapped."""

        def __init__(self, inner):
            self.inner = inner
            self.laplace_calls = 0

        def laplace(self, *args, **kwargs):
            self.laplace_calls += 1
            return self.inner.laplace(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    counting = CountingGenerator(model.rng)
    model.rng = counting
    model.trend_forecast_uncertainty(horizon=10)

    assert counting.laplace_calls == 7


def test_the_draws_come_from_the_models_own_generator(peyton_manning_df,
                                                      compiled_optimizer_module):
    """One of the two draws used the global numpy generator, so seeding the
    model did not make its intervals reproducible."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    model = AnalyticProphet(uncertainty_samples=50)
    model.fit(df, lib_path=compiled_optimizer_module)

    state = model.rng.bit_generator.state
    _, first = model.trend_forecast_uncertainty(horizon=20)
    model.rng.bit_generator.state = state
    _, second = model.trend_forecast_uncertainty(horizon=20)

    np.testing.assert_array_equal(first, second)


def test_the_uncertainty_trend_follows_the_fitted_growth_mode(
        peyton_manning_df, compiled_optimizer_module):
    """The sampler called predict_trend without the growth mode, so a logistic
    fit got a linear interval -- a band around a curve the model never
    produced. It also could not see the capacity, which is per row."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    df = df.assign(cap=df["y"].max() * 1.25)

    model = AnalyticProphet(growth="logistic", uncertainty_samples=200)
    model.fit(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=200).assign(cap=df["cap"].iloc[0])
    forecast = model.predict(future)

    cap = df["cap"].iloc[0]
    assert np.all(forecast["trend_upper"].values <= cap + 1e-8), (
        "a logistic interval cannot exceed the capacity; a linear one can")
    assert np.all(forecast["trend_lower"].values <= forecast["trend"].values + 1e-8)
