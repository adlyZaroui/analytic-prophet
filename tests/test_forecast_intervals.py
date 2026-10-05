"""
Issue #63: what `yhat`'s interval is made of.

It used to be the trend's band, shifted by a deterministic seasonality:

    forecast['yhat_lower'] = forecast['trend_lower'] * multiplier + seasonality * y_scale

so `yhat_upper - yhat_lower` was exactly `trend_upper - trend_lower`. It
contained no observation noise, which is the term that dominates it — measured
at **0.056x** Prophet's band, an 18-fold overstatement of precision.

That is the dangerous direction. #58 made the trend band ~170x too *wide*,
which is visibly absurd; this made `yhat` 18x too *narrow*, which looks entirely
reasonable and is not.

`predict` now draws `yhat` rather than deriving it, [fc] `sample_model`:

    yhat_draw = trend_draw * multiplier + seasonality_draw + normal(0, sigma_obs)

Two things enter through that loop which could not before — the observation
noise here, and a regressor's own forecast uncertainty (#16 task 14a, in
tests/test_regressor_predictor.py).
"""
import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet


@pytest.fixture(scope="module")
def fitted(compiled_optimizer_module):
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parent / "data" / "peyton_manning.csv")
    model = AnalyticProphet(uncertainty_samples=600)
    model.fit(df.iloc[:1000].reset_index(drop=True), lib_path=compiled_optimizer_module)
    return model


def test_the_yhat_band_is_wider_than_the_trend_band(fitted):
    """The whole point: they were identical, which is what said the noise was
    missing."""
    forecast = fitted.predict(fitted.make_future_dataframe(periods=90))
    horizon = slice(fitted.T, None)

    trend_band = (forecast["trend_upper"] - forecast["trend_lower"]).values[horizon]
    yhat_band = (forecast["yhat_upper"] - forecast["yhat_lower"]).values[horizon]

    assert yhat_band.mean() > 10 * trend_band.mean()


def test_the_yhat_band_is_non_zero_over_the_history(fitted):
    """The trend band is zero there -- no changepoints are sampled inside the
    history (#58) -- but the observation noise applies everywhere, so a fitted
    point is not claimed to be exact."""
    forecast = fitted.predict(fitted.make_future_dataframe(periods=30))
    history = slice(None, fitted.T)

    assert np.abs(forecast["trend_upper"].values[history]
                  - forecast["trend_lower"].values[history]).max() < 1e-12
    assert (forecast["yhat_upper"] - forecast["yhat_lower"]).values[history].min() > 0.0


def test_the_band_collapses_to_the_trend_band_without_noise(peyton_manning_df,
                                                            compiled_optimizer_module):
    """#63's acceptance criterion. With sigma_obs at zero the only remaining
    source of width is the trend, so the two bands must coincide."""
    model = AnalyticProphet(uncertainty_samples=600)
    model.fit(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    # exactly zero, not merely small: the first few horizon points have no
    # sampled changepoints yet, so their trend band is exactly 0 and any
    # residual noise there fails a relative comparison
    model.params["sigma_obs"][0][0] = 0.0
    # seeded, so the draw is the same one every time. Unseeded, this test was
    # nondeterministic in a way that mattered below: it passed in one CI run
    # and failed in the next on 3.9 and 3.13 (#103).
    model.rng = np.random.default_rng(0)

    forecast = model.predict(model.make_future_dataframe(periods=90))
    horizon = slice(model.T, None)

    trend_band = (forecast["trend_upper"] - forecast["trend_lower"]).values[horizon]
    yhat_band = (forecast["yhat_upper"] - forecast["yhat_lower"]).values[horizon]

    # `atol=0` is the part that matters and is kept: where the trend band is
    # exactly 0 the two must agree exactly, which is what catches noise leaking
    # into a point that should have none.
    #
    # `rtol` was 1e-12, which is below float arithmetic rather than below
    # anything meaningful. The two bands are built from the *same* samples --
    # yhat is trend plus a noise term that is identically zero here -- so they
    # differ only in summation order, by 1.78e-15 absolute: eight times eps,
    # one element in ninety. On a point whose band is ~9e-5 that is 2e-11
    # relative, and the comparison failed for it. 1e-9 is still nine orders
    # tighter than any difference this test exists to catch.
    np.testing.assert_allclose(yhat_band, trend_band, rtol=1e-9, atol=0)


def test_the_width_matches_the_normal_interval_it_should_be(peyton_manning_df,
                                                            compiled_optimizer_module):
    """Over the history the trend contributes nothing, so the band there is a
    centred interval on normal(0, sigma_obs) and can be checked against the
    normal quantile rather than against another implementation."""
    from scipy.stats import norm

    model = AnalyticProphet(uncertainty_samples=2000, interval_width=0.8)
    model.fit(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    forecast = model.predict(model.make_future_dataframe(periods=0))
    band = (forecast["yhat_upper"] - forecast["yhat_lower"]).values.mean()

    expected = 2 * norm.ppf(0.9) * model.sigma_obs * model.y_scale
    assert band == pytest.approx(expected, rel=0.05)


def test_interval_width_still_controls_it(peyton_manning_df, compiled_optimizer_module):
    model = AnalyticProphet(uncertainty_samples=1000, interval_width=0.5)
    model.fit(peyton_manning_df.iloc[:400].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    narrow = model.predict(model.make_future_dataframe(periods=30))

    wide_model = AnalyticProphet(uncertainty_samples=1000, interval_width=0.95)
    wide_model.fit(peyton_manning_df.iloc[:400].reset_index(drop=True),
                       lib_path=compiled_optimizer_module)
    wide = wide_model.predict(wide_model.make_future_dataframe(periods=30))

    assert ((wide["yhat_upper"] - wide["yhat_lower"]).mean()
            > 2 * (narrow["yhat_upper"] - narrow["yhat_lower"]).mean())


def test_yhat_itself_is_unchanged_by_the_draws(peyton_manning_df,
                                               compiled_optimizer_module):
    """`yhat` is the fitted point forecast, not a mean of draws -- sampling
    must not move it."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    few = AnalyticProphet(uncertainty_samples=10)
    few.fit(df, lib_path=compiled_optimizer_module)
    many = AnalyticProphet(uncertainty_samples=500)
    many.fit(df, lib_path=compiled_optimizer_module)

    future = few.make_future_dataframe(periods=30)
    np.testing.assert_array_equal(few.predict(future)["yhat"].values,
                                  many.predict(future)["yhat"].values)


def test_the_band_agrees_with_prophets(prophet_comparison, compiled_optimizer_module):
    """0.056x before. The trend contributes about 5% of this width, so it is
    mostly a check that the noise term is right."""
    Prophet, common, _ = prophet_comparison
    df = common.load_data(1000)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.fit(df)
    theirs = prophet_model.predict(prophet_model.make_future_dataframe(periods=90))

    ours = AnalyticProphet(uncertainty_samples=1000)
    ours.fit(df, lib_path=compiled_optimizer_module)
    mine = ours.predict(ours.make_future_dataframe(periods=90))

    horizon = slice(len(df), None)
    their_band = (theirs["yhat_upper"] - theirs["yhat_lower"]).values[horizon].mean()
    my_band = (mine["yhat_upper"] - mine["yhat_lower"]).values[horizon].mean()

    assert 0.85 < my_band / their_band < 1.15, (
        f"yhat band is {my_band / their_band:.3f}x Prophet's")
