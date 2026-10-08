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
import itertools
import json
import os
import tempfile
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import corpora
import harness
import metrics
from harness import Measurement

# Sized so the tier runs in minutes rather than hours: a rolling origin over a
# handful of cutoffs, on a sample of each frequency.
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


def _measure(job):
    """One series, in whatever process the pool put it in.

    Top level and taking a single tuple because that is what a process pool can
    pickle. Returns the series name alongside its rows so the caller can
    checkpoint by name without trusting completion order.
    """
    name, frame, frequency, lib_path = job
    return name, _one_series(name, frame, frequency, lib_path)


def _checkpoint_path(directory, name):
    return Path(directory) / f"{name}.json"


def _save_checkpoint(directory, name, measurements):
    """One series' rows, written as soon as they exist.

    A census of 3008 series is roughly fifteen hours of fitting serially, and
    the eventual target is M4 entire. A run that loses everything to one
    failure at hour fourteen is not a run anybody repeats, so each series is
    durable the moment it finishes and a restart picks up the rest (#164).
    """
    path = _checkpoint_path(directory, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(m) for m in measurements]
    path.write_text(json.dumps(payload))


def _load_checkpoint(directory, name):
    rows = json.loads(_checkpoint_path(directory, name).read_text())
    return [Measurement(**row) for row in rows]


def _jobs(frequencies, manifest, lib_path, download):
    for frequency in frequencies:
        for name, frame in corpora.m4_census(frequency, manifest,
                                             download=download):
            yield name, frame, frequency, lib_path


def collect(frequencies=FREQUENCIES, lib_path=None, download=True,
            workers=None, checkpoint=None, limit=None, manifest=None):
    """The tier, over the frozen corpus, in parallel and resumably.

    **The corpus is a census and it is pre-registered** (#164). Which series
    are measured is decided by `evaluation/corpus/m4_census_v1.json`, written
    before any of these results existed and carrying a digest of its own
    membership. There is no `n_series` and no sampling seed any more, because
    there is nothing to sample: every M4 Weekly and Daily series the protocol
    can measure is in, all 3008.

    **Parallel, because the unit of work is a whole series and they do not
    interact.** Each worker fits both implementations over one series'
    rolling-origin cutoffs. Submission is bounded rather than eager so the
    parent holds a few frames rather than three thousand, which is what lets
    the same code reach M4 entire.

    **Deterministic regardless of worker count.** Every series is seeded from
    `harness.SEED` inside `_one_series`, completion order is discarded by
    sorting on the way out, and the paired summaries sort their differences by
    series name before computing -- a median does not care, but a file that
    differs between runs would, and this suite promises it does not.
    """
    lib_path = lib_path or harness.build_extension(tempfile.mkdtemp())
    manifest = manifest or corpora.load_manifest()
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    jobs = _jobs(frequencies, manifest, lib_path, download)
    if limit:
        jobs = itertools.islice(jobs, limit)

    done, measurements = {}, []
    if checkpoint:
        Path(checkpoint).mkdir(parents=True, exist_ok=True)

    def record(name, rows):
        done[name] = rows
        if checkpoint:
            _save_checkpoint(checkpoint, name, rows)

    pending, submitted, finished = set(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job in jobs:
            name = job[0]
            if checkpoint and _checkpoint_path(checkpoint, name).exists():
                done[name] = _load_checkpoint(checkpoint, name)
                finished += 1
                continue
            while len(pending) >= workers * 2:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    record(*future.result())
                    finished += 1
                    if finished % 50 == 0:
                        print(f"    {finished} series done")
            pending.add(pool.submit(_measure, job))
            submitted += 1
        for future in as_completed(pending):
            record(*future.result())
            finished += 1

    for name in sorted(done):
        measurements += done[name]
    print(f"  {finished} series measured "
          f"({submitted} fitted, {finished - submitted} from checkpoints)")

    measurements += _corpus_rows(manifest, done)
    measurements += _paired(measurements)
    measurements += _paired_by_frequency(measurements, manifest)
    return measurements


def _corpus_rows(manifest, done):
    """What was actually measured, so the results can be tied to the manifest.

    The digest lives in the manifest rather than here -- a `Measurement` holds
    a float -- but the counts belong with the numbers they describe, and
    `tests/test_corpus_census.py` checks that the set of series in the results
    is exactly the set the manifest froze.
    """
    rows = [Measurement(2, "corpus", "all", "both", "series_measured",
                        float(len(done)))]
    for frequency, identifiers in sorted(manifest["series"].items()):
        prefix = f"m4_{frequency.lower()}_"
        rows.append(Measurement(2, "corpus", frequency, "both", "series_frozen",
                                float(len(identifiers))))
        rows.append(Measurement(2, "corpus", frequency, "both", "series_measured",
                                float(sum(1 for name in done
                                          if name.startswith(prefix)))))
    return rows


METRICS = ("mae", "rmse", "mape", "smape", "coverage", "interval_width",
           "sum_abs_delta", "exact_zeros", "l1_penalty")


def _differences(measurements, metric, keep=None):
    """Per-series `ours - theirs` for one metric, in series order.

    **Sorted by series name, which is not cosmetic.** Under a process pool the
    completion order is whatever the scheduler did, and an unsorted list would
    make this file differ between runs of the same corpus. A median does not
    care; the promise that an unchanged rerun produces an unchanged diff does.
    """
    by_series = {}
    for m in measurements:
        if m.metric == metric and m.series not in ("paired", "corpus"):
            if keep is None or keep(m.series):
                by_series.setdefault(m.series, {})[m.implementation] = m.value
    return [by_series[name]["analytic_prophet"] - by_series[name]["prophet"]
            for name in sorted(by_series)
            if len(by_series[name]) == 2]


def _summarise(measurements, label, keep=None):
    summaries = []
    for metric in METRICS:
        differences = _differences(measurements, metric, keep)
        if len(differences) < 2:
            continue
        summary = metrics.paired_summary(differences)
        for field in ("n", "median", "iqr_low", "iqr_high", "wins", "losses", "p_value"):
            summaries.append(Measurement(2, "paired", label, "difference",
                                         f"{metric}_{field}", summary[field]))
    return summaries


def _paired(measurements):
    """The summary the tier is actually about: per-series differences.

    Negative means we win, since every metric here is an error except coverage.
    """
    return _summarise(measurements, "all")


def _paired_by_frequency(measurements, manifest):
    """The same summary within each stratum.

    **At 3008 series the pooled p-value stops being the finding** (#164). Any
    consistent difference clears every threshold at that n, so a reader who
    takes significance as the headline learns nothing from the corpus getting
    bigger. What a census buys is the right to ask where the difference holds,
    and that is a per-stratum question.
    """
    rows = []
    for frequency in sorted(manifest["series"]):
        prefix = f"m4_{frequency.lower()}_"
        rows += _summarise(measurements, frequency,
                           keep=lambda name, prefix=prefix: name.startswith(prefix))
    return rows


harness.register(2, "accuracy", collect)
