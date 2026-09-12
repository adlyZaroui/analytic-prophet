"""
predict() de-normalizes by combining the slope and intercept terms in
normalized-y space FIRST, then multiplying the whole trend by y_absmax
once:

    trend = (k + A.dot(delta)) * t_scaled + (m + A.dot(gamma))
    forecast["trend"] = trend * self.y_absmax

trend_forecast_uncertainty currently multiplies y_absmax onto only the
intercept-like term:

    future_trend = (k + det_dot(new_A, new_delta)) * future_t_scaled \
                 + (m + det_dot(new_A, new_gamma)) * self.y_absmax

If unintentional, the slope-driven part of the simulated band is left in
normalized-y units while the intercept part is in original units.

Rather than testing around that asymmetry directly, this targets what the
fix should look like: one shared trend function used by both methods,
checked for the property that actually matters -- scaling y_absmax by any
factor should scale the de-normalized trend by exactly that factor,
uniformly, regardless of how much of the trend at a given point comes
from k vs. from m/delta. Wire predict() and trend_forecast_uncertainty()
to both call this instead of repeating the formula, and this test (and
the bug) apply to both at once.
"""
import numpy as np


def _compute_trend(k, m, delta, change_points, t_scaled, y_absmax):
    A = (t_scaled[:, None] > change_points) * 1
    gamma = -change_points * delta
    trend_normalized = (k + np.dot(A, delta)) * t_scaled + (m + np.dot(A, gamma))
    return trend_normalized * y_absmax


def test_trend_denormalization_is_uniform(prepared_model):
    k, m = 0.4, -0.1
    delta = np.zeros(25)
    t_scaled = prepared_model.t_scaled
    change_points = prepared_model.change_points

    trend_scale_1 = _compute_trend(k, m, delta, change_points, t_scaled, y_absmax=1.0)
    trend_scale_10 = _compute_trend(k, m, delta, change_points, t_scaled, y_absmax=10.0)

    np.testing.assert_allclose(trend_scale_10, trend_scale_1 * 10.0)
