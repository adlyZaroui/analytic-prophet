"""Tier 3 -- what it costs to fit and to predict (#79).

Two claims the README already makes -- speed does not regress, memory improves
-- measured across sizes and across model widths rather than at one point, and
with CPU time added, which no existing benchmark reports.

**CPU time is the part that needs care.** Prophet does the arithmetic in a
cmdstan subprocess, so `time.process_time()` in this process reads essentially
zero for its fit. Adding CPU time naively would show Prophet using no processor
at all and hand this project a flattering result that is entirely an artefact.
Every measurement here sums RUSAGE_SELF with RUSAGE_CHILDREN, which is what
benchmark_memory.py already does and benchmark_fit_time.py does not, because it
only reports wall clock.

**Wall time and CPU time answer different questions.** Prophet's wall clock
includes spawning that subprocess, which at T=100 is a large fraction of the
0.026 s it takes in total. That is a real cost to a user and not a property of
the method, so both are reported and labelled rather than one being chosen.

`K` is varied by configuration on a real series rather than by generating
synthetic ones, which keeps this tier independent of Tier 1.
"""
import json
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

import corpora
import harness
from harness import Measurement

SIZES = (50, 100, 300, 1000, 2905)
REPEATS = 3

# name -> kwargs for both constructors, widening the design matrix. Prophet
# takes the same arguments, so one dict configures both.
WIDTHS = {
    "yearly only": dict(yearly_seasonality=10, weekly_seasonality=False,
                        daily_seasonality=False),
    "yearly+weekly": dict(yearly_seasonality=10, weekly_seasonality=3,
                          daily_seasonality=False),
    "yearly+weekly+daily": dict(yearly_seasonality=10, weekly_seasonality=3,
                                daily_seasonality=4),
}
WIDTH_SIZE = 1000


def _usage():
    """Processor seconds used by this process *and its children*."""
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


def timed(call, repeats=REPEATS):
    """(best wall seconds, cpu seconds at that run).

    Best-of rather than mean: the distribution is bounded below by the real cost
    and has a long tail of scheduling noise, so the minimum is the more stable
    estimate of what the work takes.
    """
    best = (float("inf"), float("inf"))
    for _ in range(repeats):
        cpu_before, wall_before = _usage(), time.perf_counter()
        call()
        wall = time.perf_counter() - wall_before
        cpu = _usage() - cpu_before
        if wall < best[0]:
            best = (wall, cpu)
    return best


MEMORY_PROBE = """
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = {paths!r}
import harness, corpora
baseline = harness.peak_rss_bytes()
df = corpora.peyton_manning({size!r})
if {name!r} == "prophet":
    from prophet import Prophet
    imported = harness.peak_rss_bytes()
    Prophet(**harness.PROPHET_KWARGS).fit(df)
else:
    from analytic_prophet import AnalyticProphet
    imported = harness.peak_rss_bytes()
    model = AnalyticProphet(**harness.PROPHET_KWARGS)
    if {name!r} == "compiled":
        model.fit(df, lib_path={lib!r})
    else:
        model.fit(df, backend="python", analytic=True)
print("RESULT" + json.dumps({{"baseline": baseline, "imported": imported,
                              "peak": harness.peak_rss_bytes()}}))
"""


def peak_memory(name, size, lib_path):
    """Peak resident bytes, split into what the import cost and what the fit did.

    The split is the whole point. The README's claim is about the *fit* --
    reverse-mode autodiff retains a tape and a closed-form gradient does not --
    so a baseline taken before the import measures something else entirely, and
    would have this project losing on a number it is not making a claim about.
    Reported separately rather than combined, because a library that is
    expensive to merely import is still expensive to deploy.

    A fresh process per measurement, because peak RSS is a high-water mark: run
    two fits in one process and the second one's delta is whatever the first
    left behind. `peak_rss_bytes` sums RUSAGE_SELF with RUSAGE_CHILDREN, which
    is what catches cmdstan.
    """
    script = MEMORY_PROBE.format(paths=[str(harness.REPO), str(harness.EVALUATION),
                                        str(harness.REPO / "benchmark")],
                                 size=size, name=name, lib=lib_path)
    completed = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=900)
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT"):
            payload = json.loads(line[len("RESULT"):])
            return {
                "import_rss": payload["imported"] - payload["baseline"],
                "fit_peak_rss_added": payload["peak"] - payload["imported"],
                "peak_rss": payload["peak"],
            }
    return None


def _fitters(df, lib_path, **kwargs):
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    settings = dict(harness.PROPHET_KWARGS, **kwargs)

    def prophet():
        Prophet(**settings).fit(df)

    def compiled():
        AnalyticProphet(**settings).fit(df, lib_path=lib_path)

    def python():
        AnalyticProphet(**settings).fit(df, backend="python", analytic=True)

    return {"prophet": prophet, "compiled": compiled, "python": python}


def _predictors(df, lib_path, **kwargs):
    """Fitted models with a future frame ready, for the inference measurement.

    **Both implementations are measured twice, and the pairing is
    load-bearing.** `predict(vectorized=True)` is the default on both sides and
    is not a faster form of the exact sampler -- it is a different one, and the
    two disagree on the interval bounds by about 1.4% in Prophet and a similar
    amount here. Only the diagonal comparisons mean anything: approximate
    against approximate, exact against exact. Timing our exact sampler against
    Prophet's approximate one is what produced the "2.4x slower" claim of #87,
    and #93 is why there are four entries here rather than three.
    """
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    settings = dict(harness.PROPHET_KWARGS, **kwargs)
    prophet_model = Prophet(**settings).fit(df)
    ours = AnalyticProphet(**settings)
    ours.fit(df, lib_path=lib_path)
    prophet_future = prophet_model.make_future_dataframe(periods=90)
    our_future = ours.make_future_dataframe(periods=90)
    return {
        "prophet": lambda: prophet_model.predict(prophet_future),
        "prophet(exact)": lambda: prophet_model.predict(prophet_future,
                                                        vectorized=False),
        "compiled": lambda: ours.predict(our_future),
        "compiled(exact)": lambda: ours.predict(our_future, vectorized=False),
    }


def collect(sizes=SIZES, repeats=REPEATS, lib_path=None, with_memory=True):
    lib_path = lib_path or harness.build_extension(tempfile.mkdtemp())
    measurements = []

    def row(series, configuration, implementation, metric, value, unit):
        measurements.append(Measurement(3, series, configuration, implementation,
                                        metric, float(value), unit))

    # --- cost against the length of the series --------------------------
    for size in sizes:
        series = f"peyton_manning[:{size}]"
        df = corpora.peyton_manning(size)
        fitters = _fitters(df, lib_path)
        for name, call in fitters.items():
            call()                                   # warm up, discarded
            wall, cpu = timed(call, repeats)
            row(series, "default", name, "fit_wall", wall, "s")
            row(series, "default", name, "fit_cpu", cpu, "s")
            row(series, "default", name, "observations", size, "")
        for name, call in _predictors(df, lib_path).items():
            call()
            wall, cpu = timed(call, repeats)
            row(series, "default", name, "predict_wall", wall, "s")
            row(series, "default", name, "predict_cpu", cpu, "s")
        if with_memory:
            for name in ("prophet", "compiled", "python"):
                usage = peak_memory(name, size, lib_path)
                if usage is not None:
                    for metric, value in usage.items():
                        row(series, "default", name, metric, value, "bytes")
        print(f"    T={size} done")

    # --- cost against the width of the design matrix --------------------
    df = corpora.peyton_manning(WIDTH_SIZE)
    for label, settings in WIDTHS.items():
        from analytic_prophet import AnalyticProphet
        shape = AnalyticProphet(**dict(harness.PROPHET_KWARGS, **settings))
        shape.preprocess(df)
        columns = shape.layout.n_regressor_columns
        series = f"peyton_manning[:{WIDTH_SIZE}]"
        for name, call in _fitters(df, lib_path, **settings).items():
            call()
            wall, cpu = timed(call, repeats)
            row(series, label, name, "fit_wall", wall, "s")
            row(series, label, name, "fit_cpu", cpu, "s")
            row(series, label, name, "design_columns", columns, "")
        print(f"    K={columns} ({label}) done")

    measurements += _scaling(measurements)
    return measurements


def _scaling(measurements):
    """Fitted exponents of time against T, from a log-log regression.

    A number for "how does this grow", which reading five timings off a table
    does not give.
    """
    points = {}
    for m in measurements:
        if m.configuration != "default":
            continue
        if m.metric in ("fit_wall", "fit_cpu", "fit_peak_rss_added"):
            size = int(m.series.split(":")[1].rstrip("]"))
            points.setdefault((m.implementation, m.metric), []).append((size, m.value))

    rows = []
    for (implementation, metric), pairs in sorted(points.items()):
        pairs = [(t, v) for t, v in pairs if t > 0 and v > 0]
        if len(pairs) < 3:
            continue
        sizes, values = np.log(np.array([p[0] for p in pairs], dtype=float)), \
            np.log(np.array([p[1] for p in pairs], dtype=float))
        slope = float(np.polyfit(sizes, values, 1)[0])
        rows.append(Measurement(3, "scaling", "default", implementation,
                                f"{metric}_exponent", slope, "d log / d log T"))
    return rows


harness.register(3, "cost", collect)
