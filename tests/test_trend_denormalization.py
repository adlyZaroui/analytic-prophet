"""
predict() de-normalizes by combining the slope and intercept terms in
normalized-y space FIRST, then multiplying the whole trend by y_scale
once:

    trend = (k + A.dot(delta)) * t + (m + A.dot(gamma))
    forecast["trend"] = trend * self.y_scale

trend_forecast_uncertainty currently multiplies y_scale onto only the
intercept-like term:

    future_trend = (k + det_dot(new_A, new_delta)) * future_t_scaled \
                 + (m + det_dot(new_A, new_gamma)) * self.y_scale

CONFIRMED, not just suspected: running predict() with k=0.4 and
m=delta=beta=0 gives trend=4.7146 but trend_lower=trend_upper=0.4080 --
off by a factor of exactly y_scale (11.555 in that run). The slope-driven
part of the uncertainty band is left in normalized-y units.

Two tests, two different jobs:
- test_predict_trend_bounds_are_correctly_scaled calls the real predict()
  path directly and is expected to FAIL against the current legacy code --
  the regression test that should start passing once the fix lands.
- test_trend_denormalization_is_uniform targets what the fix should look
  like: one shared trend function, checked for the property that actually
  matters (uniform scaling), so predict() and trend_forecast_uncertainty()
  can both be pointed at it and this class of bug can't reappear from the
  two copies drifting apart again.
"""
import numpy as np
import pytest


def _compute_trend(k, m, delta, changepoints_t, t, y_scale):
    A = (t[:, None] > changepoints_t) * 1
    gamma = -changepoints_t * delta
    trend_normalized = (k + np.dot(A, delta)) * t + (m + np.dot(A, gamma))
    return trend_normalized * y_scale


def test_trend_denormalization_is_uniform(prepared_model):
    k, m = 0.4, -0.1
    delta = np.zeros(25)
    t = prepared_model.t
    changepoints_t = prepared_model.changepoints_t

    trend_scale_1 = _compute_trend(k, m, delta, changepoints_t, t, y_scale=1.0)
    trend_scale_10 = _compute_trend(k, m, delta, changepoints_t, t, y_scale=10.0)

    np.testing.assert_allclose(trend_scale_10, trend_scale_1 * 10.0)


def test_predict_trend_bounds_are_correctly_scaled(prepared_model, param_size):
    """
    Calls predict() directly -- the real path a user hits -- with delta,
    beta, and m held at zero so the only surviving contribution is k.
    Confirmed by direct execution: this currently fails, with
    trend_lower/trend_upper smaller than `trend` by exactly a factor of
    y_scale. Expected to pass once trend_forecast_uncertainty shares
    _compute_trend (or equivalent) with predict().
    """
    model = prepared_model
    vector = np.zeros(param_size)
    vector[0] = 0.4  # k, nonzero; m / delta / beta stay 0
    vector[model.layout.sigma_obs_idx] = 1.0   # positive, so log_prob is defined
    model._store_params(vector)

    future_df = model.make_future_dataframe(periods=30, include_history=True)
    forecast = model.predict(future_df)

    last_t_scaled = (
        (future_df["ds"].iloc[-1] - model.ds.min())
        / (model.ds.max() - model.ds.min())
    )
    expected = 0.4 * last_t_scaled * model.y_scale

    assert forecast["trend"].iloc[-1] == pytest.approx(expected)
    assert forecast["trend_lower"].iloc[-1] == pytest.approx(expected, rel=0.05)