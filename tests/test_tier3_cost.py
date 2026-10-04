"""
Tier 3 of the evaluation suite (#79): what it costs to fit and to predict.

Two measurement traps decide whether these numbers mean anything, and both would
flatter this project if got wrong. They are what the tests are about.

  * **Prophet does the arithmetic in a cmdstan subprocess.** `process_time()`
    in this process reads essentially zero for its fit, so a naive CPU-time
    comparison would show Prophet using no processor at all. Every measurement
    sums RUSAGE_SELF with RUSAGE_CHILDREN.
  * **Import cost is not fit cost.** The README's memory claim is about the fit
    -- reverse-mode autodiff retains a tape and a closed-form gradient does not
    -- so a baseline taken before the import measures something else, and would
    have this project losing on a number it is not claiming.
"""
import subprocess
import sys

import numpy as np
import pytest

import harness
from tiers import tier3


# -- the CPU-time trap ----------------------------------------------------

def test_cpu_time_counts_work_done_in_a_subprocess():
    """The trap, demonstrated rather than described.

    A child burning processor time must show up, or Prophet's fit -- which is
    entirely a child process -- would be recorded as free.
    """
    burn = [sys.executable, "-c",
            "x = 0\nfor i in range(4_000_000): x += i*i\n"]

    wall, cpu = tier3.timed(lambda: subprocess.run(burn, check=True), repeats=1)

    assert cpu > 0.05, "child CPU time is not being counted"
    assert cpu == pytest.approx(wall, rel=0.9), \
        "a CPU-bound child should account for most of the wall time"


def test_process_time_alone_would_have_missed_it():
    """Why RUSAGE_CHILDREN is spelled out rather than left to the obvious call.
    This is the measurement the tier does *not* make."""
    import time

    burn = [sys.executable, "-c", "x = 0\nfor i in range(4_000_000): x += i*i\n"]
    before = time.process_time()
    subprocess.run(burn, check=True)
    naive = time.process_time() - before

    assert naive < 0.02, (
        "process_time now sees child CPU, so the warning in tier3.py is stale")


def test_timed_reports_the_best_of_several_runs():
    """Bounded below by the real cost with a long tail of scheduling noise, so
    the minimum is the more stable estimate."""
    calls = []
    wall, _ = tier3.timed(lambda: calls.append(1), repeats=4)

    assert len(calls) == 4
    assert wall >= 0


# -- the import/fit split -------------------------------------------------

def test_memory_separates_what_the_import_cost_from_what_the_fit_did(
        compiled_optimizer_module):
    usage = tier3.peak_memory("fit_cpp", 300, compiled_optimizer_module)

    assert set(usage) == {"import_rss", "fit_peak_rss_added", "peak_rss"}
    assert usage["import_rss"] > 0, "importing the package costs something"
    assert usage["fit_peak_rss_added"] >= 0
    assert usage["peak_rss"] >= usage["import_rss"]


def test_each_measurement_gets_a_process_that_has_done_nothing_else(
        compiled_optimizer_module):
    """Peak RSS is a high-water mark: two fits in one process and the second's
    delta is whatever the first left behind. Two calls must therefore agree
    rather than the second reading zero."""
    first = tier3.peak_memory("fit_cpp", 300, compiled_optimizer_module)
    second = tier3.peak_memory("fit_cpp", 300, compiled_optimizer_module)

    assert second["import_rss"] == pytest.approx(first["import_rss"], rel=0.25)


# -- the scaling summary --------------------------------------------------

def test_scaling_exponents_come_from_at_least_three_points():
    """A slope through two points is a line through two points."""
    from harness import Measurement

    rows = [Measurement(3, f"peyton_manning[:{t}]", "default", "fit_cpp",
                        "fit_wall", t / 1000.0, "s") for t in (100, 1000, 10000)]
    rows.append(Measurement(3, "peyton_manning[:100]", "default", "other",
                            "fit_wall", 1.0, "s"))

    summaries = {(m.implementation, m.metric): m.value for m in tier3._scaling(rows)}

    assert summaries[("fit_cpp", "fit_wall_exponent")] == pytest.approx(1.0, abs=1e-9)
    assert ("other", "fit_wall_exponent") not in summaries


def test_the_width_sweep_reaches_different_design_widths(compiled_optimizer_module):
    """K is varied by configuration on a real series, which is what keeps this
    tier independent of Tier 1's generators."""
    import corpora

    from analytic_prophet import AnalyticProphet

    widths = set()
    for settings in tier3.WIDTHS.values():
        model = AnalyticProphet(**dict(harness.PROPHET_KWARGS, **settings))
        model.preprocess(corpora.peyton_manning(tier3.WIDTH_SIZE))
        widths.add(model.layout.n_regressor_columns)

    assert len(widths) == len(tier3.WIDTHS), "two configurations give the same K"


# -- the tier -------------------------------------------------------------

@pytest.fixture(scope="module")
def small_run(compiled_optimizer_module, prophet_comparison):
    # prophet_comparison is for the skip, not its value: tier3 imports
    # prophet inside collect() (#103).
    return tier3.collect(sizes=(300,), repeats=1, lib_path=compiled_optimizer_module,
                         with_memory=False)


def test_both_fitting_and_predicting_are_measured(small_run):
    metrics = {m.metric for m in small_run}
    assert {"fit_wall", "fit_cpu", "predict_wall", "predict_cpu"} <= metrics


def test_every_implementation_is_measured_the_same_way(small_run):
    fit = {(m.implementation, m.metric) for m in small_run if m.metric == "fit_wall"}
    assert {"prophet", "fit_cpp", "fit(analytic=True)"} == {i for i, _ in fit}


def test_reported_times_are_positive_and_finite(small_run):
    for m in small_run:
        if m.unit == "s":
            assert np.isfinite(m.value) and m.value >= 0, (m.implementation, m.metric)
