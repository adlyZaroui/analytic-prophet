"""Tier 2 -- forecast accuracy, held out (#79).

**Does the better MAP point generalize?** This is the question the README
explicitly refuses to answer, and the only adjacent evidence points the other
way: Prophet fits the training data marginally better while we score better on
the posterior, because we find a sparser trend. Tier 1 adds that on synthetic
data the truth is sparser than Prophet's fit and denser than ours -- neither is
calibrated. Whether any of that helps a forecast is what this measures.

Two decisions keep the comparison honest, and both amount to using Prophet's own
machinery rather than writing a second version of it:

  * **the cutoffs come from `prophet.diagnostics.generate_cutoffs`** and are
    computed once, then handed to both. Rolling-origin evaluation has enough
    knobs -- initial window, period, horizon -- that two implementations
    choosing their own would be comparing splits as much as models.
  * **the scoring is `prophet.diagnostics.performance_metrics`**, applied to
    both sides' forecasts. Writing our own RMSE is a chance to write a different
    RMSE; `coverage` in particular has a definition worth not re-deriving.

Metrics per series, per horizon: MAE, RMSE, MAPE, sMAPE and the coverage of the
80% interval. Coverage matters as much as point error here -- #63 shipped a
`yhat` band 18 times too narrow and it looked entirely reasonable.

Reported as **paired per-series differences**, because forecast errors are
heavy-tailed across series and a mean over them measures the worst series rather
than the method.
"""
import tempfile
import warnings

import numpy as np
import pandas as pd

import corpora
import harness
import metrics
from harness import Measurement

# Sized so the tier runs in minutes rather than hours: a rolling origin over a
# handful of cutoffs, on a sample of each frequency.
N_SERIES = 20
HORIZONS = {"Weekly": ("52 W", "104 W", "52 W"), "Daily": ("90 D", "730 D", "180 D")}
FREQUENCIES = ("Weekly", "Daily")


def _dated(df):
    """`ds` as datetimes.

    `cross_validation` coerces internally and `generate_cutoffs` does not, so
    calling the second directly on a frame read from CSV fails on a string
    minus a Timedelta. The M4 loader builds its dates, which is why this only
    shows up on the calendar-real series.
    """
    return df.assign(ds=pd.to_datetime(df["ds"])).reset_index(drop=True)


def _our_cross_validation(df, cutoffs, horizon, lib_path):
    """[fc] prophet.diagnostics.cross_validation, for AnalyticProphet.

    Deliberately not a port of the whole function: the cutoffs are given rather
    than derived, which is the part that has to match, and the rest is a fit and
    a predict per cutoff.
    """
    from analytic_prophet import AnalyticProphet

    df = _dated(df)
    predictions = []
    for cutoff in cutoffs:
        history = df[df["ds"] <= cutoff].reset_index(drop=True)
        future = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + horizon)]
        if len(history) < 2 or future.empty:
            continue
        model = AnalyticProphet(**harness.PROPHET_KWARGS)
        model.rng = np.random.default_rng(harness.SEED)
        model.fit(history, lib_path=lib_path)
        forecast = model.predict(future[["ds"]].reset_index(drop=True))
        predictions.append(pd.DataFrame({
            "ds": future["ds"].to_numpy(),
            "yhat": forecast["yhat"].to_numpy(),
            "yhat_lower": forecast["yhat_lower"].to_numpy(),
            "yhat_upper": forecast["yhat_upper"].to_numpy(),
            "y": future["y"].to_numpy(),
            "cutoff": cutoff,
        }))
    return pd.concat(predictions, ignore_index=True) if predictions else None


def _score(cv_frame):
    """Prophet's own scoring, so neither side can be using a different RMSE."""
    from prophet.diagnostics import performance_metrics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return performance_metrics(cv_frame, rolling_window=1.0)


def _one_series(name, df, frequency, lib_path):
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, generate_cutoffs

    df = _dated(df)
    horizon_text, initial_text, period_text = HORIZONS[frequency]
    horizon = pd.Timedelta(horizon_text)
    try:
        cutoffs = generate_cutoffs(df, horizon, pd.Timedelta(initial_text),
                                   pd.Timedelta(period_text))
    except ValueError:
        return []                      # less data than horizon; not a failure
    if not cutoffs:
        return []

    rows = []

    def row(implementation, metric, value, unit=""):
        if value is not None and np.isfinite(value):
            rows.append(Measurement(2, name, frequency, implementation,
                                    metric, float(value), unit))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Prophet's uncertainty draws come from numpy's *global* generator, so
        # without this its intervals differ between runs and the committed
        # results contradict what this suite promises about them -- that an
        # unchanged rerun produces an unchanged file. Ours is seeded per fit in
        # _our_cross_validation; this is the other half.
        np.random.seed(harness.SEED % (2 ** 32))
        prophet_cv = cross_validation(
            Prophet(**harness.PROPHET_KWARGS).fit(df), horizon=horizon_text,
            cutoffs=cutoffs, disable_tqdm=True)
    ours_cv = _our_cross_validation(df, cutoffs, horizon, lib_path)
    if ours_cv is None or prophet_cv.empty:
        return []

    row("both", "cutoffs", len(cutoffs))
    row("both", "observations", len(df))
    row("both", "forecast_points", len(ours_cv))

    for implementation, frame in (("prophet", prophet_cv), ("analytic_prophet", ours_cv)):
        scores = _score(frame)
        for metric in ("mae", "rmse", "mape", "smape", "coverage"):
            if metric in scores:
                row(implementation, metric, scores[metric].iloc[-1])
        row(implementation, "interval_width",
            float(np.mean(frame["yhat_upper"] - frame["yhat_lower"])))

    # The trend's sparsity, as two quantities that do not depend on where a
    # line is drawn. A count of `|delta| > 1e-6` used to stand here and was
    # reported as the mechanism behind the accuracy result; it is almost
    # entirely a threshold artifact, because Prophet's rates are never exactly
    # zero and ours often are (#95).
    from analytic_prophet import AnalyticProphet
    ours = AnalyticProphet(**harness.PROPHET_KWARGS)
    ours.fit(df, lib_path=lib_path)
    theirs = Prophet(**harness.PROPHET_KWARGS).fit(df)
    # `implementation` rather than `name`: `row` closes over this function's
    # `name`, which is the *series*, and a loop variable called `name` shadows
    # it -- every row then files itself under the implementation's name and the
    # per-series pairing silently finds nothing to pair.
    for implementation, rates in (
            ("analytic_prophet", np.abs(ours.params["delta"][0])),
            ("prophet", np.abs(np.ravel(theirs.params["delta"])))):
        row(implementation, "sum_abs_delta", rates.sum())
        row(implementation, "exact_zeros", np.sum(rates == 0.0))
        row(implementation, "l1_penalty",
            rates.sum() / ours.changepoint_prior_scale, "nats")
    return rows


def collect(frequencies=FREQUENCIES, n_series=N_SERIES, lib_path=None, download=True):
    lib_path = lib_path or harness.build_extension(tempfile.mkdtemp())
    measurements = []
    for frequency in frequencies:
        series = list(corpora.m4(frequency, n_series=n_series,
                                 seed=harness.SEED, download=download))
        if not series:
            print(f"  {frequency}: no corpus available, skipped")
            continue
        print(f"  {frequency}: {len(series)} series")
        for index, (name, df) in enumerate(series, 1):
            measurements += _one_series(name, df, frequency, lib_path)
            if index % 5 == 0:
                print(f"    {index}/{len(series)}")
    measurements += _paired(measurements)
    return measurements


def _paired(measurements):
    """The summary the tier is actually about: per-series differences.

    Negative means we win, since every metric here is an error except coverage.
    """
    by_series = {}
    for m in measurements:
        by_series.setdefault((m.series, m.metric), {})[m.implementation] = m.value

    summaries = []
    for metric in ("mae", "rmse", "mape", "smape", "coverage", "interval_width",
                   "sum_abs_delta", "exact_zeros", "l1_penalty"):
        differences = [v["analytic_prophet"] - v["prophet"]
                       for (_, name), v in by_series.items()
                       if name == metric and len(v) == 2]
        if len(differences) < 2:
            continue
        summary = metrics.paired_summary(differences)
        for field in ("n", "median", "iqr_low", "iqr_high", "wins", "losses", "p_value"):
            summaries.append(Measurement(2, "paired", "all", "difference",
                                         f"{metric}_{field}", summary[field]))
    return summaries


harness.register(2, "accuracy", collect)
