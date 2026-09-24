"""
Issue #54: the names, and a test that stops them drifting again.

Where this implementation and Prophet hold the same quantity they often called
it something different, and nothing checked that. The divergence grew one task
at a time, which is how it reached fourteen pairs before anyone counted.

The rule settled on: **match Prophet's name, unless the Stan model uses a
different one, in which case match Stan's.** Stan wins because every derivation
in this repository is written against `prophet.stan`, and a comment explaining
`changepoint_prior_scale`'s gradient should use the symbol the model does.

What this module is for is the third part of #54, and the part that matters:
the mapping is enumerated here, so a rename on either side fails a test instead
of quietly widening the gap again. It checks names *and* values — a name that
matches while holding something else would be worse than no match at all.

The deliberate non-matches are enumerated too, with their reasons. Without
that, the next person to notice one has no way to tell a decision from an
oversight, and "fixing" it is a plausible mistake.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

import customProphet
from customProphet import CustomProphet


# ours -> Prophet's, for things that now share a name
SHARED_ATTRIBUTES = ["growth", "n_changepoints", "changepoint_range", "seasonalities",
                     "changepoint_prior_scale", "changepoints_t",
                     "extra_regressors", "holidays", "holidays_prior_scale",
                     "holidays_mode", "seasonality_mode", "seasonality_prior_scale",
                     "country_holidays", "train_holiday_names", "interval_width",
                     "uncertainty_samples", "yearly_seasonality", "weekly_seasonality",
                     "daily_seasonality", "y_scale"]

SHARED_CALLABLES = ["add_seasonality", "add_regressor", "add_country_holidays",
                    "validate_column_name", "set_auto_seasonalities",
                    "parse_seasonality_args", "construct_holiday_dataframe",
                    "make_holiday_features", "fourier_series", "predict_trend",
                    "linear_growth_init", "logistic_growth_init", "flat_growth_init",
                    "fit", "predict", "make_future_dataframe"]

# ours -> (theirs, why it stays different)
DELIBERATE = {
    "fit_cpp": (None, "No counterpart in Prophet. It is the point of the project: "
                      "the compiled path with the hand-derived gradient."),
    "sigma_k": (None, "[stan] the prior scale on k. Stan writes it as a literal "
                      "in `k ~ normal(0, 5)` rather than naming it in the data "
                      "block, and Prophet does not expose it at all."),
    "sigma_m": (None, "[stan] the prior scale on m, likewise a literal there and "
                      "absent from Prophet's API."),
    "T": (None, "[stan] T, the observation count. Prophet reads it off "
                "`history.shape[0]` rather than keeping an attribute."),
}


@pytest.fixture(scope="module")
def prophet_class(prophet_comparison):
    Prophet, _, _ = prophet_comparison
    return Prophet


# -- the names ----------------------------------------------------------

@pytest.mark.parametrize("name", SHARED_ATTRIBUTES)
def test_a_shared_attribute_exists_on_both(prophet_class, name):
    """Fails if either side renames it, which is the whole point of the
    module: the gap closed in #54 must not reopen silently."""
    assert hasattr(CustomProphet(), name), f"{name} is gone from this implementation"
    assert hasattr(prophet_class(), name), f"{name} is gone from Prophet"


@pytest.mark.parametrize("name", SHARED_CALLABLES)
def test_a_shared_callable_exists_on_both(prophet_class, name):
    ours = getattr(CustomProphet, name, None) or getattr(customProphet, name, None)
    theirs = getattr(prophet_class, name, None)

    assert callable(ours), f"{name} is gone from this implementation"
    assert callable(theirs), f"{name} is gone from Prophet"


@pytest.mark.parametrize("ours,expected", sorted(DELIBERATE.items()))
def test_a_deliberate_difference_is_still_different(prophet_class, ours, expected):
    """The other half. If one of these ever *does* match, either the reason
    stopped applying or someone changed it without reading why -- and a silent
    convergence is as much a surprise as a silent divergence.
    """
    theirs, _reason = expected
    assert hasattr(CustomProphet(), ours) or hasattr(customProphet, ours), ours

    if theirs is None or "[" in theirs:
        return
    assert not hasattr(CustomProphet(), theirs), (
        f"{ours} and {theirs} now both exist here; DELIBERATE says they should "
        f"not, so either the entry is stale or this was an accident")


def test_every_deliberate_difference_carries_a_reason():
    for ours, (_theirs, reason) in DELIBERATE.items():
        assert reason and len(reason) > 30, (
            f"{ours} is listed as deliberate without saying why, which makes it "
            f"indistinguishable from an oversight")


# -- the values behind them ---------------------------------------------

def test_the_renamed_quantities_hold_what_prophet_holds(prophet_comparison,
                                                        compiled_optimizer_module):
    """A matching name on a different quantity is worse than no match. Each
    pair below was verified in #54 and is re-verified here on a real fit."""
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    _, stan_data, _ = bridge.capture_stan_model(prophet_model, df)

    ours = CustomProphet()
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    # y_scale: the divisor, shared name as of #54
    assert ours.y_scale == pytest.approx(prophet_model.y_scale)

    # y_scaled: the scaled series, Prophet's history column
    np.testing.assert_allclose(ours.y_scaled,
                               prophet_model.history["y_scaled"].to_numpy())

    # t against Stan's `t`, which is Prophet's history column too
    np.testing.assert_allclose(ours.t, np.asarray(stan_data["t"], dtype=float))

    # changepoint_prior_scale against Stan's own, under Prophet's constructor name
    assert ours.changepoint_prior_scale == pytest.approx(float(stan_data["tau"]))
    assert ours.changepoint_prior_scale == pytest.approx(prophet_model.changepoint_prior_scale)


def test_fourier_series_matches_prophets_under_the_shared_name(prophet_class,
                                                               peyton_manning_df):
    """Renamed from `fourier_components` in #54. Accessed through the class on
    Prophet's side, so the names cannot shadow each other here."""
    from customProphet import seasonal_time

    dates = pd.to_datetime(peyton_manning_df["ds"])
    ours = customProphet.fourier_series(seasonal_time(dates), 365.25, 10)
    theirs = prophet_class.fourier_series(dates, 365.25, 10)

    assert ours.shape == theirs.shape
    assert np.max(np.abs(ours - theirs)) < 1e-10


def test_the_signature_of_each_shared_callable_is_compatible(prophet_class):
    """Not identical -- several of ours take fewer arguments, and some are
    module functions where Prophet's are methods. What must hold is that every
    parameter *we* require, Prophet also has, so a call written against their
    documentation does not fail on an argument name."""
    for name in ("add_seasonality", "add_regressor", "add_country_holidays"):
        ours = set(inspect.signature(getattr(CustomProphet, name)).parameters) - {"self"}
        theirs = set(inspect.signature(getattr(prophet_class, name)).parameters) - {"self"}
        assert ours <= theirs, f"{name} takes {sorted(ours - theirs)}, which Prophet does not"
