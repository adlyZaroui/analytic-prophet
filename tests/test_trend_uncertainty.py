"""
Issue #58: where the trend interval's changepoints come from.

`trend_forecast_uncertainty` widens the trend band by drawing changepoints the
future might contain. It drew them in the wrong place:

    probability_changepoint = self.n_changepoints / self.T     # 25/1000
    sample = self.rng.random(future_t_scaled.shape)            # the WHOLE grid
    new_changepoints = future_t_scaled[sample <= probability_changepoint]

`future_t_scaled` spans the history as well as the horizon, so roughly 2.5% of
*every* point became a new changepoint with its own rate jump. Most landed
inside the history, and every draw rewrote the fitted past before extrapolating
from it. The band measured ~170x Prophet's -- 8.04 against 0.0473 on a series
whose values run 6 to 12, wider than the data.

[fc] sample_predictive_trend places them as a Poisson process on `(1, T]`,
where `T` is the largest scaled time in the frame being forecast:

    n_changes = Poisson(S * (T - 1))
    times     = 1 + rand(n_changes) * (T - 1)

so the count scales with how far past the history the forecast reaches, and is
zero when it does not reach past it at all.

The sharpest observable consequence, and what most of these tests check: with
every new changepoint past `t = 1`, the sampled trend over the *history* is the
same in every draw, so the band there is exactly zero and all of the width is
in the future.
"""
import numpy as np
import pandas as pd
import pytest

from customProphet import CustomProphet


@pytest.fixture(scope="module")
def fitted(compiled_optimizer_module):
    """One fit shared by the module: several of these tests draw a few hundred
    samples each, and refitting for every one of them dominates the runtime."""
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parent / "data" / "peyton_manning.csv")
    model = CustomProphet(uncertainty_samples=400)
    model.fit_cpp(df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    return model


# -- where the changepoints land ----------------------------------------

def test_a_history_only_frame_gets_no_new_changepoints(fitted):
    """T = 1 exactly, so `Poisson(S * (T - 1))` is `Poisson(0)`. Every draw is
    the fitted trend, and the band collapses."""
    forecast = fitted.predict(fitted.make_future_dataframe(periods=0))
    band = forecast["trend_upper"].values - forecast["trend_lower"].values

    np.testing.assert_array_equal(band, 0.0)


def test_the_history_keeps_a_zero_width_band_under_a_forecast(fitted):
    """The sharp consequence of placing changepoints past t = 1: whatever the
    horizon, the sampled trend over the history is identical in every draw.

    Under the old rule this was the opposite -- the history carried most of the
    sampled changepoints, so it carried most of the width."""
    n_history = fitted.T
    forecast = fitted.predict(fitted.make_future_dataframe(periods=180))
    band = forecast["trend_upper"].values - forecast["trend_lower"].values

    # Zero to floating point rather than exactly: each draw appends its new
    # changepoints, so `A` gains columns and `A @ delta` sums a different
    # number of terms -- all of them zero over the history, but not summed in
    # the same order. Measured at 1.8e-15.
    assert np.abs(band[:n_history]).max() < 1e-12
    assert band[n_history:].max() > 0.0


def test_the_band_widens_with_the_horizon(fitted):
    """`Poisson(S * (T - 1))` scales with how far past the history the frame
    reaches, so a longer forecast is less certain rather than equally so."""
    widths = []
    for horizon in (30, 90, 365):
        forecast = fitted.predict(fitted.make_future_dataframe(periods=horizon))
        band = (forecast["trend_upper"] - forecast["trend_lower"]).values
        widths.append(band[fitted.T:].mean())

    assert widths[0] < widths[1] < widths[2]


def test_the_number_of_new_changepoints_follows_the_poisson_rate(fitted):
    """Checked at the draw rather than through the band, since the band also
    depends on where they land and how large their deltas are."""
    class RecordingGenerator:
        def __init__(self, inner):
            self.inner = inner
            self.counts = []

        def poisson(self, rate, *args, **kwargs):
            drawn = self.inner.poisson(rate, *args, **kwargs)
            self.counts.append((rate, drawn))
            return drawn

        def __getattr__(self, name):
            return getattr(self.inner, name)

    recording = RecordingGenerator(fitted.rng)
    original, fitted.rng = fitted.rng, recording
    try:
        future = fitted.make_future_dataframe(periods=100)
        t_scaled = ((pd.to_datetime(future["ds"]) - fitted.ds.min())
                    / (fitted.ds.max() - fitted.ds.min())).values
        fitted.trend_forecast_uncertainty(t_scaled=t_scaled, n_samples=300)
    finally:
        fitted.rng = original

    rates = {rate for rate, _ in recording.counts}
    assert len(rates) == 1, "the rate must not vary between draws"
    rate = rates.pop()

    expected = len(fitted.change_points) * (t_scaled.max() - 1.0)
    assert rate == pytest.approx(expected)

    drawn = np.array([count for _, count in recording.counts])
    assert drawn.mean() == pytest.approx(rate, rel=0.25)   # Poisson mean is its rate


def test_the_laplace_scale_has_prophets_epsilon(peyton_manning_df,
                                                compiled_optimizer_module):
    """[fc] `lambda_ = np.mean(np.abs(deltas)) + 1e-8`.

    A series straight enough that no changepoint is active gives mean|delta| of
    exactly zero, and Laplace(0, 0) is degenerate. Without the epsilon the band
    is identically zero rather than merely narrow -- a forecast claiming perfect
    certainty.
    """
    straight = pd.DataFrame({
        "ds": pd.date_range("2015-01-01", periods=400),
        "y": np.linspace(10.0, 20.0, 400),
    })
    model = CustomProphet(uncertainty_samples=200, yearly_seasonality=False,
                          weekly_seasonality=False, daily_seasonality=False)
    model.fit_cpp(straight, lib_path=compiled_optimizer_module)

    assert np.abs(model.opt_params[model.layout.delta]).mean() < 1e-6, (
        "the fixture must actually produce a flat delta for this to test anything")

    forecast = model.predict(model.make_future_dataframe(periods=180))
    band = (forecast["trend_upper"] - forecast["trend_lower"]).values[len(straight):]

    assert np.all(np.isfinite(band))
    assert band.max() > 0.0, "a zero-width band claims certainty the model does not have"
    assert band.max() < 1e-3, "and it should still be narrow"


# -- reproducibility ----------------------------------------------------

def test_the_band_is_reproducible_from_the_models_generator(fitted):
    """One of the two draws used the global numpy generator until #52."""
    future = fitted.make_future_dataframe(periods=60)
    t_scaled = ((pd.to_datetime(future["ds"]) - fitted.ds.min())
                / (fitted.ds.max() - fitted.ds.min())).values

    state = fitted.rng.bit_generator.state
    _, first = fitted.trend_forecast_uncertainty(t_scaled=t_scaled, n_samples=100)
    fitted.rng.bit_generator.state = state
    _, second = fitted.trend_forecast_uncertainty(t_scaled=t_scaled, n_samples=100)

    np.testing.assert_array_equal(first, second)


# -- #35, fixed alongside -----------------------------------------------

def test_predict_does_not_write_into_the_callers_frame(fitted):
    """#35: predict added a `t_scaled` column to the frame it was handed."""
    future = fitted.make_future_dataframe(periods=30)
    before = list(future.columns)

    fitted.predict(future)

    assert list(future.columns) == before
    assert "t_scaled" not in future.columns


def test_predict_does_not_write_into_a_frame_carrying_extra_columns(
        peyton_manning_df, compiled_optimizer_module):
    """The copy has to happen whether or not there are regressors to fill --
    the regressor path already copied, the plain path did not."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    model = CustomProphet()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=10)
    future["note"] = "unchanged"
    snapshot = future.copy()

    model.predict(future)

    pd.testing.assert_frame_equal(future, snapshot)


# -- against Prophet ----------------------------------------------------

def test_the_band_agrees_with_prophets(prophet_comparison, compiled_optimizer_module):
    """What the issue measured at 170x.

    The remaining difference is the fitted `delta`, not the sampler: the
    Laplace scale is `mean|delta|`, the band is linear in it, and ours runs
    about 1.17x Prophet's on this series because the two optimizers land on
    different -- sparser but larger -- rate adjustments. The bound allows for
    that and for the sampling noise of a few hundred draws.
    """
    Prophet, common, _ = prophet_comparison
    df = common.load_data(1000)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.fit(df)
    prophet_forecast = prophet_model.predict(
        prophet_model.make_future_dataframe(periods=365))

    ours = CustomProphet(uncertainty_samples=1000)
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)
    our_forecast = ours.predict(ours.make_future_dataframe(periods=365))

    theirs = (prophet_forecast["trend_upper"] - prophet_forecast["trend_lower"]).values[len(df):]
    mine = (our_forecast["trend_upper"] - our_forecast["trend_lower"]).values[len(df):]

    ratio = mine.mean() / theirs.mean()
    assert 0.5 < ratio < 2.0, (
        f"trend band is {ratio:.2f}x Prophet's; it was 170x before #58")


def test_prophet_also_collapses_the_band_without_a_horizon(prophet_comparison,
                                                           compiled_optimizer_module):
    """The zero-width history band is Prophet's behaviour too, not a quirk of
    this implementation -- worth pinning, since it looks like a bug on its own."""
    Prophet, common, _ = prophet_comparison
    df = common.load_data(300)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.fit(df)
    theirs = prophet_model.predict(prophet_model.make_future_dataframe(periods=0))

    ours = CustomProphet(uncertainty_samples=200)
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)
    mine = ours.predict(ours.make_future_dataframe(periods=0))

    assert (theirs["trend_upper"] - theirs["trend_lower"]).abs().max() < 1e-9
    np.testing.assert_array_equal(
        mine["trend_upper"].values - mine["trend_lower"].values, 0.0)
