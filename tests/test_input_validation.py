"""
Issues #102 and #104: what the data path does with input it cannot use.

Two issues, one PR, because they land in the same five lines at the top of
`preprocess` and could not be merged independently. #104 is the ergonomic half
-- every mistake a new user is likely to make surfaced as a traceback from
inside numpy or pandas, several frames below anything they wrote. #102 is the
correctness half: a single NaN in `y` fitted happily to NaN parameters and
returned an all-NaN forecast, raising nothing and logging nothing.

The second is the worse of the two, and it is the reason these tests assert
*messages* rather than exception types. A `pytest.raises(ValueError)` passes
just as well when the message is unhelpful, and the old behaviour passed every
existing test in the suite -- because no test ever fed it a gap.

Prophet's rules are copied rather than approximated, wording included, so that
someone porting a script recognises the error and can search for it. They are
three-way where it would be tempting to be uniform: a missing `y` is
**dropped**, a missing `ds` **raises**, and an infinite `y` **raises**.
Dropping is right for `y` because the trend is a function of `t` -- a dropped
row leaves a real gap in the design matrix rather than shifting everything
after it up by one. It is wrong for `ds`, where there is no timestamp to leave
a gap at.

[fc] Prophet.preprocess, setup_dataframe and validate_inputs between them.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet, CONSTANT_SERIES_SIGMA_OBS
from analytic_prophet.serialize import model_from_json, model_to_json


# Each row of #104's table, with the sentence it must now produce. The frames
# are built per test so a mutation in one cannot reach another.
BAD_FRAMES = {
    "no y column": (
        lambda df: df[["ds"]],
        'Dataframe must have columns "ds" and "y" with the dates and values '
        'respectively.'),
    "no ds column": (
        lambda df: df[["y"]],
        'Dataframe must have columns "ds" and "y" with the dates and values '
        'respectively.'),
    "empty frame": (
        lambda df: df.iloc[:0],
        'Dataframe has less than 2 non-NaN rows.'),
    "one row": (
        lambda df: df.iloc[:1],
        'Dataframe has less than 2 non-NaN rows.'),
    "one usable row": (
        lambda df: df.assign(y=[df["y"].iloc[0]] + [np.nan] * (len(df) - 1)),
        'Dataframe has less than 2 non-NaN rows.'),
    "all y missing": (
        lambda df: df.assign(y=np.nan),
        'Dataframe has less than 2 non-NaN rows.'),
    "infinity in y": (
        lambda df: df.assign(y=df["y"].mask(df.index == 3, np.inf)),
        'Found infinity in column y.'),
    "negative infinity in y": (
        lambda df: df.assign(y=df["y"].mask(df.index == 3, -np.inf)),
        'Found infinity in column y.'),
    "NaT in ds": (
        lambda df: df.assign(ds=pd.to_datetime(df["ds"]).mask(df.index == 3)),
        'Found NaN in column ds.'),
    "timezone in ds": (
        lambda df: df.assign(ds=pd.to_datetime(df["ds"]).dt.tz_localize("UTC")),
        'Column ds has timezone specified, which is not supported. '
        'Remove timezone.'),
    # both wrong at once, which is the only case that can tell the order of the
    # two checks apart. [fc] setup_dataframe names the timezone first.
    "timezone and NaT in ds": (
        lambda df: df.assign(
            ds=pd.to_datetime(df["ds"]).dt.tz_localize("UTC").mask(df.index == 3)),
        'Column ds has timezone specified, which is not supported. '
        'Remove timezone.'),
}


@pytest.fixture
def short_df(peyton_manning_df):
    """Enough rows to fit, few enough to fit quickly."""
    return peyton_manning_df.iloc[:60].reset_index(drop=True)


@pytest.mark.parametrize("case", list(BAD_FRAMES))
def test_bad_input_raises_prophets_sentence(case, short_df):
    """The message, not merely the type. Both are asserted; only the first
    would have caught the behaviour this replaces."""
    build, message = BAD_FRAMES[case]
    model = AnalyticProphet()
    with pytest.raises(ValueError) as raised:
        model.fit(build(short_df), analytic=True)
    assert str(raised.value) == message


@pytest.mark.parametrize("case", list(BAD_FRAMES))
def test_both_fit_paths_reject_identically(case, short_df, compiled_optimizer_module):
    """The checks live in the shared `preprocess` (#77), so the compiled path
    gets them for free -- and raises before the library is even loaded."""
    build, message = BAD_FRAMES[case]
    model = AnalyticProphet()
    with pytest.raises(ValueError) as raised:
        model.fit_cpp(build(short_df), lib_path=compiled_optimizer_module)
    assert str(raised.value) == message


def test_a_gappy_series_fits_on_the_rows_it_has(peyton_manning_df):
    """#102's headline case: a frame with holes in `y`.

    Before, this fitted to NaN parameters and returned an all-NaN forecast
    with nothing raised -- the failure mode of #13, on the input side.
    """
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    gappy = df.assign(y=df["y"].mask(df.index % 7 == 0))
    usable = int(gappy["y"].notnull().sum())
    assert 0 < usable < len(gappy)        # the test would be vacuous otherwise

    model = AnalyticProphet()
    model.fit(gappy, analytic=True)
    assert model.T == usable
    assert len(model.y) == usable
    assert np.all(np.isfinite(model.get_parameters()))

    forecast = model.predict(model.make_future_dataframe(periods=10))
    assert np.all(np.isfinite(forecast["yhat"]))
    assert np.all(np.isfinite(forecast[["yhat_lower", "yhat_upper"]].to_numpy()))


def test_dropped_rows_are_logged(peyton_manning_df, caplog):
    """Not [fc] -- Prophet drops the rows silently.

    This is the one place the data path says more than Prophet does, and the
    reason is #102: using fewer rows than the caller passed is exactly the
    event whose silence made the NaN case so hard to notice. It is INFO, so it
    changes nothing for a caller who is not listening.
    """
    df = peyton_manning_df.iloc[:60].reset_index(drop=True)
    gappy = df.assign(y=df["y"].mask(df.index < 4))
    with caplog.at_level(logging.INFO, logger="analytic_prophet"):
        AnalyticProphet().fit(gappy, analytic=True)
    assert any("4" in record.message and "missing y" in record.message
               for record in caplog.records)


def test_preprocess_returns_the_cleaned_history(peyton_manning_df):
    """The frame the fit paths pass on to the regressor models.

    This is the part of #102 with teeth: `preprocess` dropped rows into
    `self.y` while `_fit_regressor_models` kept being handed the caller's
    frame, so a nested regressor model was fitted over rows the optimizer
    never saw.
    """
    df = peyton_manning_df.iloc[:60].reset_index(drop=True)
    gappy = df.assign(y=df["y"].mask(df.index % 5 == 0),
                      temperature=np.linspace(0.0, 1.0, 60))
    model = AnalyticProphet().add_regressor("temperature",
                                            regressor_predictor={})
    cleaned = model.preprocess(gappy)

    assert len(cleaned) == int(gappy["y"].notnull().sum())
    assert cleaned["y"].notnull().all()
    assert list(cleaned.index) == list(range(len(cleaned)))
    assert "temperature" in cleaned            # the regressor column survives

    model_with_regressor = AnalyticProphet().add_regressor(
        "temperature", regressor_predictor={})
    model_with_regressor.fit(gappy, analytic=True)
    assert len(model_with_regressor._regressor_history) == len(cleaned)


@pytest.mark.parametrize("growth", ["linear", "flat"])
def test_a_constant_series_skips_the_optimizer(peyton_manning_df, growth):
    """[fc] Prophet.fit: `y.min() == y.max()` under linear or flat growth
    keeps the initial parameters, sets sigma_obs = 1e-9 and never calls Stan.

    Measured against Prophet on a 200-point constant series, both sides put
    yhat on the constant exactly and report the same interval width, 7.7e-9.
    """
    df = peyton_manning_df.iloc[:200].reset_index(drop=True).assign(y=3.0)
    model = AnalyticProphet(growth=growth)
    model.fit(df, analytic=True)

    assert model.sigma_obs == CONSTANT_SERIES_SIGMA_OBS
    assert model.optimizer_used is None       # nothing ran to name
    assert model.loss_over_iterations == []

    forecast = model.predict(model.make_future_dataframe(periods=10))
    assert np.allclose(forecast["yhat"].to_numpy(), 3.0)


def test_the_compiled_path_short_circuits_too(peyton_manning_df):
    """And without loading the library: the C++ core would otherwise be asked
    to minimise a function with no interior minimum."""
    df = peyton_manning_df.iloc[:200].reset_index(drop=True).assign(y=3.0)
    model = AnalyticProphet()
    model.fit_cpp(df, lib_path="/nonexistent/path/to/no/library.so")

    assert model.sigma_obs == CONSTANT_SERIES_SIGMA_OBS
    assert np.allclose(model.predict(
        model.make_future_dataframe(periods=5))["yhat"].to_numpy(), 3.0)


def test_logistic_growth_does_not_short_circuit(peyton_manning_df):
    """Excluded there and here: the logistic initializer solves for a curve
    approaching `cap`, which a constant series is not."""
    df = peyton_manning_df.iloc[:60].reset_index(drop=True)
    df = df.assign(y=3.0, cap=10.0)
    model = AnalyticProphet(growth="logistic")
    model.fit(df, analytic=True)

    assert model.sigma_obs != CONSTANT_SERIES_SIGMA_OBS
    assert model.optimizer_used is not None
    assert np.all(np.isfinite(model.get_parameters()))


def test_an_all_zero_series_forecasts_zero(peyton_manning_df):
    """[fc] `if self.y_scale == 0: self.y_scale = 1.0`.

    A count of something that never happened over the window being fitted is a
    real series, and without that guard every y_scaled was 0/0 = nan: nan
    parameters, nan forecast, nothing raised (#102).
    """
    df = peyton_manning_df.iloc[:60].reset_index(drop=True).assign(y=0.0)
    model = AnalyticProphet()
    model.fit(df, analytic=True)

    assert model.y_scale == 1.0
    assert np.all(np.isfinite(model.y_scaled))
    forecast = model.predict(model.make_future_dataframe(periods=5))
    assert np.allclose(forecast["yhat"].to_numpy(), 0.0)


NOT_FITTED = {
    "predict": lambda model: model.predict(
        pd.DataFrame({"ds": pd.date_range("2013-01-01", periods=3)})),
    "make_future_dataframe": lambda model: model.make_future_dataframe(periods=3),
    "get_parameters": lambda model: model.get_parameters(),
}


@pytest.mark.parametrize("call", list(NOT_FITTED))
def test_calling_before_fit_says_so(call):
    """[fc] Prophet's `'Model has not been fit.'`, which it raises as a bare
    `Exception`; this raises `ValueError`, a subclass, so `except Exception`
    written against Prophet still catches it. Recorded in docs/deviations.md.
    """
    with pytest.raises(ValueError) as raised:
        NOT_FITTED[call](AnalyticProphet())
    assert str(raised.value) == 'Model has not been fit.'


def test_a_history_with_a_floor_requires_one_at_predict(peyton_manning_df):
    """[fc] setup_dataframe's `'Expected column "floor".'`

    A history fitted with a floor is fitted on `y - floor`, so a future frame
    that omits the column is not defaulting to zero -- it is asking for a
    different model. Dropping it moved the last yhat of this fit from 6.92 to
    1.90, with nothing raised.
    """
    df = peyton_manning_df.iloc[:150].reset_index(drop=True)
    df = df.assign(floor=df["y"].min() - 1.0, cap=df["y"].max() + 2.0)
    model = AnalyticProphet(growth="logistic")
    model.fit(df, analytic=True)
    assert model.logistic_floor

    future = model.make_future_dataframe(periods=5).assign(cap=df["cap"].iloc[0])
    with pytest.raises(ValueError) as raised:
        model.predict(future)
    assert str(raised.value) == 'Expected column "floor".'

    # and with the column present it predicts as before
    forecast = model.predict(future.assign(floor=df["floor"].iloc[0]))
    assert np.all(np.isfinite(forecast["yhat"]))

    # the flag is fit-derived state, so it has to survive a save and load --
    # otherwise a restored model answers a frame it should have rejected
    restored = model_from_json(model_to_json(model))
    assert restored.logistic_floor
    with pytest.raises(ValueError) as raised:
        restored.predict(future)
    assert str(raised.value) == 'Expected column "floor".'


def test_a_history_without_a_floor_still_does_not_require_one(peyton_manning_df):
    """The flag follows the history, so the common logistic case -- cap only,
    floor left at its default of zero -- is unaffected."""
    df = peyton_manning_df.iloc[:60].reset_index(drop=True)
    df = df.assign(cap=df["y"].max() + 2.0)
    model = AnalyticProphet(growth="logistic")
    model.fit(df, analytic=True)
    assert not model.logistic_floor

    future = model.make_future_dataframe(periods=5).assign(cap=df["cap"].iloc[0])
    assert np.all(np.isfinite(model.predict(future)["yhat"]))
