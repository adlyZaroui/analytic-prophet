"""The three growth modes: their initializations, their trends, and the
derivatives the analytic gradient needs.

Logistic is the substantial one -- `gamma` is a recursion, so its Jacobian is
accumulated forward rather than written down, and the trend carries that
S x (2 + S) matrix through `inv_logit`. [fc] prophet/forecaster.py keeps the
trends but never needs the derivatives; Stan differentiates them.
"""
import numpy as np


# [stan] `int trend_indicator` in the data block: 0 linear, 1 logistic, 2 flat.
TREND_INDICATORS = {"linear": 0, "logistic": 1, "flat": 2}

def flat_growth_init(y_scaled):
    """[fc] flat_growth_init: no rate, and the offset at the mean of y."""
    return 0.0, float(np.mean(y_scaled))

def logistic_gamma_and_jacobian(k, m, delta, changepoints_t):
    """Piecewise offsets for the logistic trend, and their Jacobian.

    [stan] logistic_gamma. Each segment's offset is chosen so the curve stays
    continuous where the rate changes:

        k_s        = [k, k + cumsum(delta)]                 S+1 segment rates
        gamma[i]   = (changepoints_t[i] - m_pr_i) * (1 - k_s[i]/k_s[i+1])
        m_pr_{i+1} = m_pr_i + gamma[i],   m_pr_1 = m

    Unlike the linear case, where gamma is just `-changepoints_t * delta`, this is a
    recursion: gamma[i] depends on every earlier gamma through `m_pr`. So its
    Jacobian is accumulated forward alongside it rather than written down.

    Returns (gamma, dgamma) with dgamma of shape (S, 2 + S), columns ordered
    (k, m, delta).
    """
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    k_s = np.concatenate(([k], k + np.cumsum(delta)))

    # d k_s[i] / d(k, m, delta): 1 for k, 0 for m, and delta_j enters k_s[i]
    # for every j < i
    dk_s = np.zeros((n + 1, 2 + n))
    dk_s[:, 0] = 1.0
    for i in range(1, n + 1):
        dk_s[i, 2:2 + i] = 1.0

    gamma = np.empty(n)
    dgamma = np.zeros((n, 2 + n))
    m_pr = m
    dm_pr = np.zeros(2 + n)
    dm_pr[1] = 1.0

    for i in range(n):
        c = 1.0 - k_s[i] / k_s[i + 1]
        dc = -(dk_s[i] * k_s[i + 1] - k_s[i] * dk_s[i + 1]) / k_s[i + 1] ** 2

        gamma[i] = (changepoints_t[i] - m_pr) * c
        dgamma[i] = -dm_pr * c + (changepoints_t[i] - m_pr) * dc

        m_pr = m_pr + gamma[i]
        dm_pr = dm_pr + dgamma[i]
    return gamma, dgamma

def _logistic_pieces(k, m, delta, t, cap_scaled, A, changepoints_t):
    """Everything both logistic entry points need, computed once.

    The overflow-safe sigmoid is the subtle part, and having it in two places
    is how the two would eventually disagree.
    """
    gamma, dgamma = logistic_gamma_and_jacobian(k, m, delta, changepoints_t)
    rate = k + np.dot(A, delta)
    offset = m + np.dot(A, gamma)
    z = rate * (t - offset)

    # exp(-|z|) form: the naive 1/(1+exp(-z)) overflows for z very negative
    sigmoid = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                       np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
    return gamma, dgamma, rate, offset, sigmoid


def logistic_trend(k, m, delta, t, cap_scaled, A, changepoints_t):
    """The logistic trend alone, for callers that do not want the Jacobian.

    [stan] logistic_trend: cap .* inv_logit((k + A*delta) .* (t - (m + A*gamma))).

    The objective is one such caller, and it is the one that matters: it runs
    on every optimizer iteration and used to call
    `logistic_trend_and_jacobian` and discard the second return value. The
    Jacobian is **52% of that call** -- 508 us against 242 us at T = 2905 with
    25 changepoints -- so a logistic fit was doing about twice the arithmetic
    it needed on the objective. Only the gradient wants the Jacobian (#105).
    """
    _gamma, _dgamma, _rate, _offset, sigmoid = _logistic_pieces(
        k, m, delta, t, cap_scaled, A, changepoints_t)
    return cap_scaled * sigmoid


def logistic_trend_and_jacobian(k, m, delta, t, cap_scaled, A, changepoints_t):
    """The logistic trend and d(trend)/d(k, m, delta).

    [stan] logistic_trend: cap .* inv_logit((k + A*delta) .* (t - (m + A*gamma))).

    Returns (trend, jacobian) with jacobian of shape (T, 2 + S). The caller
    contracts it with the residual; keeping the full Jacobian here rather than
    the contracted gradient is what lets the multiplicative multiplier be
    applied outside, exactly as it is for the linear trend.
    """
    gamma, dgamma, rate, offset, sigmoid = _logistic_pieces(
        k, m, delta, t, cap_scaled, A, changepoints_t)
    trend = cap_scaled * sigmoid

    d_offset = np.dot(A, dgamma)
    d_offset[:, 1] += 1.0                      # offset = m + A*gamma
    d_rate = np.zeros((len(t), 2 + len(delta)))
    d_rate[:, 0] = 1.0
    d_rate[:, 2:] = A

    dz = d_rate * (t - offset)[:, None] - rate[:, None] * d_offset
    jacobian = (cap_scaled * sigmoid * (1.0 - sigmoid))[:, None] * dz
    return trend, jacobian

def logistic_growth_init(t, y_scaled, cap_scaled):
    """[fc] logistic_growth_init: put the curve through the first and last
    points, clamping y into (0, cap) first so the logs are defined."""
    i0, i1 = int(np.argmin(t)), int(np.argmax(t))
    span = t[i1] - t[i0]

    c0, c1 = cap_scaled[i0], cap_scaled[i1]
    y0 = max(0.01 * c0, min(0.99 * c0, y_scaled[i0]))
    y1 = max(0.01 * c1, min(0.99 * c1, y_scaled[i1]))

    r0, r1 = c0 / y0, c1 / y1
    if abs(r0 - r1) <= 0.01:
        r0 = 1.05 * r0

    l0, l1 = np.log(r0 - 1), np.log(r1 - 1)
    return (l0 - l1) / span, l0 * span / (l0 - l1)

def linear_growth_init(t, y_scaled):
    """Prophet's deterministic starting point for (k, m): the line through the
    first and last points of the scaled series.

    [fc] Prophet.linear_growth_init -- it indexes by argmin/argmax of ds rather
    than assuming the frame is sorted, so this does the same via t.

        k = (y_scaled[i1] - y_scaled[i0]) / (t[i1] - t[i0])
        m = y_scaled[i0] - k * t[i0]

    Prophet always passes this in explicitly, which is why Stan's random
    `init_r * N(0, 1)` default never applies.
    """
    i0 = int(np.argmin(t))
    i1 = int(np.argmax(t))
    span = t[i1] - t[i0]
    k = (y_scaled[i1] - y_scaled[i0]) / span
    m = y_scaled[i0] - k * t[i0]
    return float(k), float(m)

def det_dot(a, b):
    return (a * b[None, :]).sum(axis=-1)

def predict_trend(k, m, delta, changepoints_t, t, y_scale,
                  cap_scaled=None, floor=None, growth='linear'):
    """The trend in normalized-y space, de-normalized once at the end.

    Shared by predict() and trend_forecast_uncertainty() so the
    de-normalization can't drift apart between the two again. `cap_scaled`
    selects logistic growth; without it the trend is piecewise linear.
    """
    A = (t[:, None] >= changepoints_t) * 1
    if growth == 'flat':
        trend_normalized = np.full(len(t), m)
    elif cap_scaled is None:
        gamma = -changepoints_t * delta
        trend_normalized = (k + det_dot(A, delta)) * t + (m + det_dot(A, gamma))
    else:
        trend_normalized = logistic_trend_and_jacobian(
            k, m, delta, t, cap_scaled, A, changepoints_t)[0]
    trend = trend_normalized * y_scale
    return trend if floor is None else trend + floor
