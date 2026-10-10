"""
Issue #174: one Newton method, two parameterizations, reproducible.

The result used to exist only as prose -- 66 nats short after 10,000
iterations -- from a version that was never committed. It is now
`paper/experiments/newton.py`, and the rerun corrected it: the natural
parameterization stops 1.2 to 7.1 nats short, early, on Stan's own tests.

Checked here:

  * **the instrument** -- the iterate recorder reads accepted points off
    `projected_newton`'s calls, and on a problem with a known answer the
    points it recovers reproduce the loss trace;
  * **the committed results** -- complete, internally consistent with the
    traces, and holding the typed claims the paper and the docs make ("at no
    length at the optimum", "no rate exactly zero", "reports success");
  * **the code** -- the shortest length refitted and compared, exactly where
    the results were recorded and to the quoted precision elsewhere (#186);
  * **every document quoting the range** moves with the data.
"""
import csv
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"
RESULTS = PAPER / "results" / "newton.csv"
TRACES = PAPER / "results" / "newton_traces.csv"


def _newton():
    sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("newton", EXPERIMENTS / "newton.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


newton = _newton()


@pytest.fixture(scope="module")
def rows():
    return newton.read()


@pytest.fixture(scope="module")
def traces():
    by = {}
    with open(TRACES, newline="") as handle:
        for row in csv.DictReader(handle):
            by.setdefault((int(row["observations"]), row["arm"]), []).append(
                float(row["objective"]))
    return by


def _arm(rows, arm):
    return [row for row in rows if row["arm"] == arm]


# ---------------------------------------------------------------------------
# The instrument, on a problem with a known answer.

def test_the_recorder_recovers_projected_newtons_iterates():
    """A bounded quadratic: the recovered iterates must reproduce the loss
    trace point for point, or run_arm refuses -- and the run must stop at the
    known minimizer with the bound active."""
    A = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.5]])
    b = np.array([1.0, -4.0, 2.0])
    objective = lambda x: 0.5 * x @ A @ x - b @ x
    gradient = lambda x: A @ x - b
    lower = np.array([-np.inf, 0.0, -np.inf])
    arm = newton.run_arm(objective, gradient, np.zeros(3), lower, np.full(3, np.inf),
                         lambda x: x, slice(0, 3))
    from scipy.optimize import minimize
    reference = minimize(objective, np.zeros(3), jac=gradient, method="L-BFGS-B",
                         bounds=[(None, None), (0, None), (None, None)], tol=1e-14)
    np.testing.assert_allclose(arm["canonical"], reference.x, atol=1e-6)
    assert arm["canonical"][1] == 0.0
    assert len(arm["trace"]) == arm["iterations"] + 1


def test_the_recorder_refuses_a_call_pattern_it_does_not_recognise(monkeypatch):
    """If projected_newton ever evaluates the gradient differently, the
    recovered points stop matching the trace and the run is refused rather than
    reported with the wrong iterates."""
    import analytic_prophet.optimizer as optimizer

    original = optimizer.finite_difference_hessian

    def one_extra_call(gradient_fn, x):
        gradient_fn(x)
        return original(gradient_fn, x)

    monkeypatch.setattr(optimizer, "finite_difference_hessian", one_extra_call)
    A, b = np.diag([2.0, 3.0]), np.array([1.0, 1.0])
    with pytest.raises(RuntimeError, match="call pattern"):
        newton.run_arm(lambda x: 0.5 * x @ A @ x - b @ x, lambda x: A @ x - b,
                       np.zeros(2), np.full(2, -np.inf), np.full(2, np.inf),
                       lambda x: x, slice(0, 2))


# ---------------------------------------------------------------------------
# The committed results.

def test_every_length_has_every_arm(rows):
    found = {(int(row["observations"]), row["arm"]) for row in rows}
    assert found == {(size, arm) for size in newton.LENGTHS for arm in newton.ARMS}


def test_the_split_arm_is_the_shipped_fit(rows):
    """Same code, same start: the split arm lands exactly where
    fit(backend="python", algorithm="Newton") does, at every length."""
    assert all(row["matches_shipped"] == "True" for row in _arm(rows, "split"))


def test_the_traces_are_the_runs_and_end_at_minus_lp(rows, traces):
    """Each trace has one entry per iteration plus the start, never rises
    (Newton here accepts only improving steps), and ends at -lp__: this
    implementation's objective is Stan's negative log density with no constant
    offset, so the two rulers agree on where each run ended."""
    for row in _arm(rows, "split") + _arm(rows, "natural"):
        trace = np.array(traces[(int(row["observations"]), row["arm"])])
        assert len(trace) == int(float(row["iterations"])) + 1
        assert np.all(np.diff(trace) < 0)
        assert trace[-1] == pytest.approx(-float(row["lp"]), abs=1e-6)
        tail = -np.diff(trace)[-newton.TAIL:]
        assert float(row["median_progress_tail"]) == pytest.approx(np.median(tail), rel=1e-12)


def test_the_natural_parameterization_stops_short_and_reports_success(rows):
    """'It does not exhaust Prophet's budget; it stops early and reports
    success' -- typed in the paper and the docs, held to the data here."""
    for row in _arm(rows, "natural"):
        assert float(row["shortfall"]) > 1.0
        assert row["stopped_by"] != "iteration cap"
        assert int(float(row["iterations"])) < 10_000
        assert int(row["exact_zeros"]) == 0
        assert int(float(row["sign_flips"])) > 1000


def test_stans_newton_comes_closer_but_never_arrives(rows):
    """'Closer than the natural-parameterisation Newton at every length, and at
    no length at the optimum, with no rate exactly zero.'"""
    natural = {row["observations"]: float(row["shortfall"]) for row in _arm(rows, "natural")}
    for row in _arm(rows, "stan_newton"):
        assert 0 < float(row["shortfall"]) < natural[row["observations"]]
        assert int(row["exact_zeros"]) == 0


def test_the_split_arm_reaches_exact_zeros(rows):
    assert all(int(row["exact_zeros"]) >= 1 for row in _arm(rows, "split"))


def test_every_document_quoting_the_range_moves_with_it(rows):
    """The docs, two code comments and a test docstring quote the natural
    arm's shortfall as a range. A rerun that moves it has to move them."""
    shortfalls = [float(row["shortfall"]) for row in _arm(rows, "natural")]
    quoted = f"{min(shortfalls):.1f} to {max(shortfalls):.1f}"
    for path in ("docs/non-smooth-objective.md", "analytic_prophet/optimizer.py",
                 "analytic_prophet/optimize.cpp", "tests/test_short_series.py"):
        assert quoted in (REPO / path).read_text(), f"{path} does not quote {quoted}"


# ---------------------------------------------------------------------------
# The code.

def _recorded_here():
    """OS, architecture and Prophet release, as in test_paper_margin (#186)."""
    from importlib.metadata import version

    meta = json.loads(RESULTS.with_suffix(".meta.json").read_text())
    return (meta["platform"].split("-")[0] == platform.platform().split("-")[0]
            and meta["machine"] == platform.machine()
            and meta["versions"]["prophet"] == version("prophet"))


def test_the_code_reproduces_the_shortest_length(rows, prophet_comparison):
    """T = 50 refitted from the code. Exactly where the results were recorded;
    elsewhere the natural arm's oscillation has nothing to pin its path to the
    last bit, so it is held to what the paper quotes."""
    fresh, _ = newton.measure(50)
    committed = {row["arm"]: row for row in rows if int(row["observations"]) == 50}
    for row in fresh:
        old = committed[row["arm"]]
        if _recorded_here():
            assert row["lp"] == pytest.approx(float(old["lp"]), rel=1e-9, abs=1e-9), row["arm"]
            if row["arm"] != "stan_newton":
                assert row["iterations"] == int(float(old["iterations"]))
        elif row["arm"] == "natural":
            assert 1.0 < row["shortfall"] < 10.0 and row["exact_zeros"] == 0
        else:
            assert row["lp"] == pytest.approx(float(old["lp"]), abs=1e-3), row["arm"]
    assert next(row for row in fresh if row["arm"] == "split")["matches_shipped"]
