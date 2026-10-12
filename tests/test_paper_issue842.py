"""
Issue #172: the prophet#842 reproducer, rerun on current software.

`paper/experiments/issue842.py` ports the thread's ten-point reprex and runs it
on current Prophet, on this implementation, and on the smoothed program. The
paper states the outcome in words -- it no longer stalls, the kink stops
L-BFGS, the flat direction prices the stop, and this implementation's own
Newton path ends at a saddle. Checked here:

  * **the port is exact** -- the regressor parsed from the thread's own
    twenty-digit literals, the dates, the values;
  * **every sentence the paper types about the outcome holds in the data**;
  * **the code reproduces it** through Prophet's bundled binary, exactly where
    the results were recorded and to the quoted precision elsewhere (#186).
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
RESULTS = PAPER / "results"

# prophet#842's reprex, as the thread prints it.
THREAD_XREG = """-0.76537379485893120012 -0.74533153677032704109 -0.62494542934172692127
-0.39778106465657125934 -0.06976829815746909968 0.34070346369915599504
0.80422096949408772292 1.28317620368430906907 1.73567381412022903042
2.12006230186317567998""".split()
THREAD_Y = [1, 1, 0, 1, 0, 1, 0, 2, 2, 0]


def _module():
    sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("issue842", EXPERIMENTS / "issue842.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


issue842 = _module()


@pytest.fixture(scope="module")
def points():
    with open(RESULTS / "issue842.csv", newline="") as handle:
        return {row["point"]: row for row in csv.DictReader(handle)}


@pytest.fixture(scope="module")
def variants():
    with open(RESULTS / "issue842_variants.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_port_is_the_threads_reprex():
    assert issue842.XREG == [float(text) for text in THREAD_XREG]
    assert issue842.Y == THREAD_Y
    df = issue842.frame()
    assert str(df["ds"].iloc[0].date()) == "2018-09-10" and len(df) == 10
    assert (df["ds"].diff().dropna().dt.days == 7).all()


def test_current_prophet_no_longer_stalls(variants):
    for row in variants:
        assert row["stalled"] == "False", row
        assert "timed out" not in row["stopped_by"], row
        assert float(row["seconds"]) < issue842.TIMEOUT, row


def test_the_default_rule_reaches_the_optimum_and_lbfgs_stops_short(points):
    assert float(points["prophet Newton"]["shortfall"]) < 1e-6
    lbfgs = points["prophet LBFGS"]
    assert float(lbfgs["shortfall"]) > 0.01
    assert "parameter change" in lbfgs["stopped_by"]
    assert float(lbfgs["residual_relaxed"]) > 0.1          # not stationary under any reading


def test_smoothing_alone_closes_the_gap(points):
    """'the kink is what stops it': the smoothed program's best L-BFGS fit is
    two orders closer than the l1 one, and at eps = 1e-8 it is the same stop."""
    lbfgs = float(points["prophet LBFGS"]["shortfall"])
    smoothed = [float(row["shortfall"]) for name, row in points.items() if name.startswith("smoothed")]
    assert min(smoothed) < lbfgs / 100
    assert float(points["smoothed eps=1e-08"]["shortfall"]) == pytest.approx(lbfgs, rel=0.01)


def test_the_shortfall_lies_along_the_softest_direction(points):
    """'the under-identification the maintainer diagnosed is what the stop costs'."""
    assert float(points["prophet LBFGS"]["cos_softest"]) > 0.95
    eigenvalues = json.loads((RESULTS / "issue842.meta.json").read_text())["summary"]["eigenvalues"]
    assert eigenvalues[0] > 0 and eigenvalues[-1] / eigenvalues[0] > 50


def test_the_model_is_not_convex_and_our_default_stops_at_a_saddle(points):
    summary = json.loads((RESULTS / "issue842.meta.json").read_text())["summary"]
    assert len(summary["local_optima"]) >= 2
    ours = points["ours default"]
    assert ours["algorithm"] == "Newton"
    assert float(ours["min_eigenvalue"]) < 0                # a direction of negative curvature
    assert float(ours["residual_relaxed"]) < 1e-2           # and stationary: a saddle
    assert float(ours["shortfall"]) > 1.0
    assert float(points["ours LBFGS"]["shortfall"]) < 1e-6
    assert float(points["prophet Newton"]["min_eigenvalue"]) > 0


def test_every_variant_stops_short_under_lbfgs_and_not_under_newton(variants):
    assert len({row["variant"] for row in variants}) == len(issue842.VARIANTS)
    for row in variants:
        if row["algorithm"] == "LBFGS":
            assert float(row["shortfall"]) > 0.05, row
        else:
            assert float(row["shortfall"]) < 1e-3, row


def _recorded_here():
    from importlib.metadata import version
    meta = json.loads((RESULTS / "issue842.meta.json").read_text())
    return (meta["platform"].split("-")[0] == platform.platform().split("-")[0]
            and meta["machine"] == platform.machine()
            and meta["versions"]["prophet"] == version("prophet"))


@pytest.mark.parametrize("algorithm, point", [(None, "prophet Newton"), ("LBFGS", "prophet LBFGS")])
def test_prophets_bundled_binary_reproduces_the_committed_fits(algorithm, point, points,
                                                               prophet_comparison):
    """Through Prophet's own fit and executable -- no CmdStan install needed."""
    import logging

    import _prophet_bridge as bridge

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    kwargs = {"sig_figs": 18} if algorithm is None else {"algorithm": algorithm, "sig_figs": 18}
    stan_model, stan_data, params = bridge.capture_stan_model(
        issue842.prophet_model(), issue842.frame(), **kwargs)
    lp = bridge.stan_log_prob(stan_model, stan_data, params["k"][0], params["m"][0],
                              params["delta"], params["sigma_obs"][0], params["beta"])
    committed = float(points[point]["lp"])
    if _recorded_here():
        assert lp == pytest.approx(committed, abs=1e-6)
    else:
        assert lp == pytest.approx(committed, abs=1e-2)
