"""Series generated from the model, with the parameters that made them.

Tier 1 needs ground truth, and this is the only way to have it: on real data you
can measure whether two implementations *agree*, never whether either is
*right*, because there is no true parameter vector to compare against.

The generator runs the model forward rather than approximating it, so the series
it produces are exactly what the model says a series looks like:

    y_scaled = (k + A.delta).t + (m + A.gamma)  +  X.beta  +  normal(0, sigma)

Two details make the recovered parameters comparable to the generating ones.

**The changepoints are the ones both implementations will choose.** `delta`
indexes breakpoints, so a recovered `delta` means nothing unless the breakpoints
are the same. The frame is passed through `preprocess` once before anything is
generated, and the changepoints and design matrix that come back are what the
series is built from -- so both implementations place them where the truth is
rather than near it. Since #15 our placement is Prophet's, so this holds for
both sides.

**The truth is recorded in normalized space.** Both implementations fit
`y / max|y|`, and every parameter scales with that divisor, so the generating
vector is divided by the same `y_scale` the fit computes. Comparing raw against
fitted would otherwise report an error that is entirely the normalization.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytic_prophet import AnalyticProphet


@dataclass
class Truth:
    """A generated series and the parameters that made it."""
    frame: pd.DataFrame
    theta: np.ndarray            # canonical layout, normalized by y_scale
    layout: object
    changepoints_t: np.ndarray
    sigma_obs: float             # normalized, as theta's entry is
    active: int                  # changepoints with a non-negligible rate change
    label: str


def _scaffold(dates):
    """A preprocessed model over these dates, for its changepoints and design.

    `y` is arbitrary here -- the design matrix is a function of the dates alone,
    and the changepoints of the dates and the row count. It has to be non-zero
    only because `_normalize_y` divides by max|y|.
    """
    model = AnalyticProphet()
    model.preprocess(pd.DataFrame({"ds": dates, "y": np.ones(len(dates))}))
    return model


def generate(n_rows, noise=0.1, seed=0, start="2015-01-01", freq="D",
             changepoint_rate=0.6, seasonal_amplitude=0.5, slope=1.0):
    """One series, with the parameters that made it.

    `changepoint_rate` is the scale of the Laplace draws for `delta`, relative
    to the trend slope: a series with no rate changes is one where the L1 prior
    is doing all the work and neither optimizer has anything to find.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_rows, freq=freq)
    scaffold = _scaffold(dates)
    layout, changepoints_t = scaffold.layout, scaffold.changepoints_t
    design = np.ascontiguousarray(
        scaffold.make_all_seasonality_features(
            pd.DataFrame({"ds": dates, "y": np.ones(n_rows)}))[0].to_numpy(dtype=float))

    t = np.asarray(scaffold.t, dtype=float)
    k = slope * rng.normal(1.0, 0.2)
    m = rng.normal(0.0, 0.5)
    delta = rng.laplace(0.0, changepoint_rate * abs(k), layout.n_changepoints)
    # most changepoints inactive, which is the regime the Laplace prior is for
    delta[rng.random(delta.size) < 0.6] = 0.0
    beta = rng.normal(0.0, seasonal_amplitude, layout.n_regressor_columns)

    A = (t[:, None] >= changepoints_t) * 1
    gamma = -changepoints_t * delta
    trend = (k + A @ delta) * t + (m + A @ gamma)
    signal = trend + design @ beta
    y = signal + rng.normal(0.0, noise, n_rows)

    # every parameter scales with the divisor the fit will compute
    y_scale = float(np.max(np.abs(y)))
    theta = np.concatenate(([k], [m], delta, [noise], beta)) / y_scale
    # ... except that sigma_obs' entry is a standard deviation in the same
    # normalized units, which is the same division
    return Truth(
        frame=pd.DataFrame({"ds": dates, "y": y}),
        theta=theta,
        layout=layout,
        changepoints_t=changepoints_t,
        sigma_obs=noise / y_scale,
        active=int(np.sum(np.abs(delta) > 1e-12)),
        label=f"synthetic[T={n_rows},noise={noise},seed={seed}]",
    )
