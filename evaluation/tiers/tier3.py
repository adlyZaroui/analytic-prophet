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
only reports wall clock. The memory measurement records the two sides separately
as well as summed: the sum is the footprint a machine must provide, and the
split is the only way to say whether what grows with the series grows in the
process that runs the autodiff (#125).

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
baseline = harness.peak_rss_split()
df = corpora.peyton_manning({size!r})
if {name!r} == "prophet":
    from prophet import Prophet
    imported = harness.peak_rss_split()
    Prophet(**harness.PROPHET_KWARGS).fit(df)
else:
    from analytic_prophet import AnalyticProphet
    if {name!r} == "python":
        # scipy is an import cost of this backend, not a cost of its fit, and
        # since #130 it is imported when the backend runs. Binding it before
        # the mark keeps this row's "fit added" comparable with the others --
        # left to fall where it lands, it charged scipy's 36 MiB to the fit and
        # gave this row a 34 MiB fixed cost in the decomposition.
        from analytic_prophet import forecaster
        forecaster._ensure_scipy()
    imported = harness.peak_rss_split()
    model = AnalyticProphet(**harness.PROPHET_KWARGS)
    if {name!r} == "compiled":
        model.fit(df, lib_path={lib!r})
    else:
        model.fit(df, backend="python", analytic=True)
print("RESULT" + json.dumps({{"baseline": baseline, "imported": imported,
                              "peak": harness.peak_rss_split()}}))
"""


def peak_memory(name, size, lib_path):
    """Peak resident bytes, split two ways: import against fit, parent against child.

    **The import/fit split is what the README claims about.** Reverse-mode
    autodiff retains a tape and a closed-form gradient does not -- that is a
    statement about the fit, so a baseline taken before the import measures
    something else entirely and would have this project losing on a number it
    is not making a claim about. Reported separately rather than combined,
    because a library that is expensive to merely import is still expensive to
    deploy.

    **The parent/child split is what lets the claim be checked.** Prophet's
    optimizer -- and therefore its tape, if that is what the gap is -- runs in
    the cmdstan child, while the parent side is cmdstanpy marshalling a data
    file out and draws back in. Recording one number for the two would make it
    impossible to say which of them grows (#125). Ours spawns nothing once the
    extension is built, so its child side is zero by construction.

    A fresh process per measurement, because peak RSS is a high-water mark: run
    two fits in one process and the second one's delta is whatever the first
    left behind. The same property is why the child side needs the fresh process
    more than the parent does -- `RUSAGE_CHILDREN` is a maximum over every child
    ever waited on, so one compiler invocation would set a floor for the rest of
    the run.
    """
    script = MEMORY_PROBE.format(paths=[str(harness.REPO), str(harness.EVALUATION),
                                        str(harness.REPO / "benchmark")],
                                 size=size, name=name, lib=lib_path)
    completed = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=900)
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT"):
            payload = json.loads(line[len("RESULT"):])
            baseline, imported, peak = (payload["baseline"], payload["imported"],
                                        payload["peak"])
            return {
                "fit_peak_rss_added": sum(peak) - sum(imported),
                "fit_peak_rss_added_self": peak[0] - imported[0],
                "fit_peak_rss_added_children": peak[1] - imported[1],
                "peak_rss": sum(peak),
            }
    return None


# `VmHWM` is read inline rather than through `harness.peak_rss_split`: this
# probe measures what one import costs, and importing the harness to ask would
# pull numpy, pandas, scipy and this package in before the question is put.
IMPORT_PROBE = """
import json, resource, sys, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = {paths!r}
{imports}
peak = 0
try:
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            peak = int(line.split()[1]) * 1024
except OSError:
    pass
if not peak:
    scale = 1 if sys.platform == "darwin" else 1024
    peak = scale * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("RESULT" + json.dumps(peak))
"""
IMPORT_REPEATS = 3

# label -> what a fresh interpreter imports, and nothing else. The two
# reference levels are measured rather than assumed so that the comparison
# between the libraries can be read against them.
IMPORT_LEVELS = {
    "bare": "",
    "numpy+pandas": "import numpy, pandas",
    "prophet": "import prophet",
    "analytic_prophet": "import analytic_prophet",
}


def _import_peak(imports):
    """Median peak RSS of a fresh interpreter that imported exactly `imports`.

    The median of a few runs rather than one, since the levels are compared
    with each other and noise in any of them is not divided away.
    """
    peaks = []
    for _ in range(IMPORT_REPEATS):
        completed = subprocess.run(
            [sys.executable, "-c", IMPORT_PROBE.format(
                paths=[str(harness.REPO)], imports=imports)],
            capture_output=True, text=True, timeout=900)
        for line in completed.stdout.splitlines():
            if line.startswith("RESULT"):
                peaks.append(json.loads(line[len("RESULT"):]))
    return float(np.median(peaks)) if peaks else None


def import_levels():
    """Peak RSS of one fresh interpreter per library, as levels, not deltas.

    **Reported as a level rather than a delta, because a level is a quantity
    and the delta was an artefact.** Measured as two marks inside one
    interpreter -- import numpy and pandas, read the peak, import the library,
    read it again -- it gave 58 MiB on macOS/arm64 and exactly zero in all
    seven Linux CI jobs. The cause is `ru_maxrss`, and it is spelled out in
    `_common._own_peak_bytes`: the value is inherited across `fork` and `exec`
    does not reset it, so a child launched from a large parent reports the
    parent's peak, both marks inside it read that same inherited constant, and
    the difference is zero. Reading `VmHWM` fixes the measurement; reporting a
    level rather than a difference means a reader can see what is being
    compared with what, which is worth keeping independently of the bug.

    What survives is the level itself. Each process imports one thing and
    nothing else, so its peak is a quantity with a meaning, and the four of
    them are directly comparable: a bare interpreter and a numpy+pandas one
    are measured alongside the two libraries so a reader can see what the
    comparison is against instead of taking a subtraction on trust.

    Before #125 this was read off the memory probe as the step between its
    baseline and its first fit. That stopped measuring anything the day
    `_common.py` grew a module-level `from analytic_prophet.build import
    ToolchainMissing`: the probe imports `harness`, `harness` reaches through
    to `_common`, so our package and scipy were already resident and the step
    read zero while the report went on quoting 57.9 MiB.
    """
    levels = {}
    for label, imports in IMPORT_LEVELS.items():
        peak = _import_peak(imports)
        if peak is not None:
            levels[label] = peak
    return levels


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

    # Import cost, once rather than per length: it does not depend on T.
    if with_memory:
        for label, peak in import_levels().items():
            row("imports", "default", label, "import_peak_rss", peak, "bytes")
        # Recorded so a reader of the committed CSV can tell whether the
        # parent/child split in it means anything, rather than having to know
        # which platform produced it.
        row("environment", "default", "both", "child_peak_rss_reliable",
            float(harness.child_peak_rss_is_reliable()), "")

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
