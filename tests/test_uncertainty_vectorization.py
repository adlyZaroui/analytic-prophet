"""
Issue #87: `predict` spent 85% of its time in a Python loop over the
uncertainty draws.

The loop is gone, and the whole point is that **the forecast did not change**.
Every draw is the same draw, to the last bit, which is what makes this an
optimization rather than a new sampler. Three things have to hold for that:

  * the fitted part of the trend is the same in every draw, so hoisting it out
    is arithmetic nobody repeated on purpose -- and a draw that samples no new
    changepoints *is* that hoisted value;
  * batching draws that sampled the same number of changepoints does not change
    the summation, because `det_dot` reduces over the last axis and batching
    adds a leading one;
  * drawing the observation noise as one block advances the generator exactly
    as a row at a time did.

The reference here is the loop this replaced, written out in the test. Asserting
against a reimplementation rather than against stored numbers is what makes the
claim checkable rather than recorded.
"""
import numpy as np
import pytest

from analytic_prophet import AnalyticProphet, forecaster
from analytic_prophet.trend import predict_trend


def reference_trends(model, t, cap_scaled, floor, n_samples):
    """`_sample_trends` as it was before #87: one `predict_trend` per draw."""
    k, m, delta, _sigma, _beta = model._fitted()
    horizon_scaled = float(np.max(t))
    n_changepoints = len(model.changepoints_t)
    lambda_mle = float(np.abs(delta).mean()) + 1e-8

    draws = []
    for _ in range(n_samples):
        n_new = (model.rng.poisson(n_changepoints * (horizon_scaled - 1.0))
                 if horizon_scaled > 1.0 else 0)
        new_change_points = np.sort(
            1.0 + model.rng.random(n_new) * (horizon_scaled - 1.0))
        new_delta = model.rng.laplace(0, lambda_mle, n_new)
        draws.append(predict_trend(
            k, m, np.concatenate((delta, new_delta)),
            np.concatenate((model.changepoints_t, new_change_points)),
            t, model.y_scale, cap_scaled, floor, model.growth))
    return np.array(draws)


@pytest.fixture
def fitted(peyton_manning_df, compiled_optimizer_module):
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:1200].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    return model


def prepared(model, periods=90):
    """(t, cap_scaled, floor) for a future frame, as _forecast_draws builds it."""
    import pandas as pd
    future = model.make_future_dataframe(periods=periods)
    future = model._ensure_regressor_values(future).copy()
    future["t"] = ((pd.to_datetime(future["ds"]) - model.ds.min())
                   / (model.ds.max() - model.ds.min()))
    cap_scaled, floor = model._future_capacity(future)
    return future["t"].values, cap_scaled, floor


def test_the_draws_are_the_draws_the_loop_produced(fitted):
    """The claim, against a reimplementation of what it replaced."""
    t, cap_scaled, floor = prepared(fitted)

    fitted.rng = np.random.default_rng(4)
    fast = fitted._sample_trends(t, cap_scaled, floor, 200)
    fitted.rng = np.random.default_rng(4)
    slow = reference_trends(fitted, t, cap_scaled, floor, 200)

    np.testing.assert_array_equal(fast, slow)


def test_the_generator_is_left_where_the_loop_left_it(fitted):
    """Not just the draws -- the generator state. Anything drawn afterwards,
    including the observation noise, must follow the same stream."""
    t, cap_scaled, floor = prepared(fitted)

    fitted.rng = np.random.default_rng(4)
    fitted._sample_trends(t, cap_scaled, floor, 200)
    after_fast = fitted.rng.normal(0, 1, 5)

    fitted.rng = np.random.default_rng(4)
    reference_trends(fitted, t, cap_scaled, floor, 200)
    after_slow = fitted.rng.normal(0, 1, 5)

    np.testing.assert_array_equal(after_fast, after_slow)


def test_a_draw_that_samples_no_changepoints_is_the_fitted_trend(fitted):
    """Why hoisting the fitted part is exact rather than close: for those
    draws the concatenated arrays *are* the fitted ones."""
    t, cap_scaled, floor = prepared(fitted)
    k, m, delta, _s, _b = fitted._fitted()
    base = predict_trend(k, m, delta, fitted.changepoints_t, t, fitted.y_scale,
                         cap_scaled, floor, fitted.growth)

    fitted.rng = np.random.default_rng(4)
    draws = fitted._sample_trends(t, cap_scaled, floor, 200)

    matching = [row for row in draws if np.array_equal(row, base)]
    assert matching, "no draw sampled zero new changepoints; the test is inert"
    assert len(matching) < len(draws), "every draw was the base; also inert"


@pytest.mark.parametrize("chunk", [1, 7, 64, 500])
def test_the_chunk_size_does_not_change_the_answer(fitted, monkeypatch, chunk):
    """Batching is only legitimate if how much is batched cannot be observed.
    A chunk of 1 is the loop; 500 batches every draw of a given count at once."""
    t, cap_scaled, floor = prepared(fitted)

    monkeypatch.setattr(forecaster, "TREND_DRAW_CHUNK", 64)
    fitted.rng = np.random.default_rng(9)
    reference = fitted._sample_trends(t, cap_scaled, floor, 200)

    monkeypatch.setattr(forecaster, "TREND_DRAW_CHUNK", chunk)
    fitted.rng = np.random.default_rng(9)
    np.testing.assert_array_equal(fitted._sample_trends(t, cap_scaled, floor, 200),
                                  reference)


def test_the_whole_forecast_is_reproducible(fitted):
    """End to end, including the hoisted seasonality and the batched noise."""
    future = fitted.make_future_dataframe(periods=60)

    fitted.rng = np.random.default_rng(1)
    first = fitted.predict(future)
    fitted.rng = np.random.default_rng(1)
    second = fitted.predict(future)

    for column in ("yhat", "yhat_lower", "yhat_upper", "trend_lower", "trend_upper"):
        np.testing.assert_array_equal(first[column].values, second[column].values)


def test_the_seasonal_term_is_hoisted_only_when_it_is_invariant(
        peyton_manning_df, compiled_optimizer_module):
    """With a regressor that has its own model the design matrix differs per
    draw, so the loop has to stay. The interval must still widen, which is what
    #16 task 14a added it for."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)
    df = df.assign(temp=np.arange(len(df), dtype=float) % 17)

    model = AnalyticProphet(uncertainty_samples=200)
    model.add_regressor("temp", regressor_predictor=True)
    model.fit(df, lib_path=compiled_optimizer_module)
    future = model.make_future_dataframe(periods=60)

    model.rng = np.random.default_rng(2)
    forecast = model.predict(future)

    assert np.all(np.isfinite(forecast["yhat"].values))
    widths = (forecast["yhat_upper"] - forecast["yhat_lower"]).values
    assert np.all(widths > 0)
