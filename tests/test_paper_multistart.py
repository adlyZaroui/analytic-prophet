"""
Issue #169: Prophet from many random starts, where the optimum is unique.

The argument in section 4.3 is that with sigma fixed the objective is strictly
convex, so a converging method reaches one point from every start, and
Prophet's fits do not. Checked here, against the committed data and the code:

  * **the premise** -- the design has full column rank at every length, which
    is what makes the minimiser unique;
  * **the protocol** -- seeded starts, every length and setting complete;
  * **the reference** -- certified, and reached by this implementation from
    each of its starts;
  * **the sentences** -- "none of the 150 fits comes within 1e-3 nats",
    "Prophet's own default start is no exception";
  * **the code** -- one committed start refitted through Prophet's own fit.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import recorded_here

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"


def _module():
    sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("multistart", EXPERIMENTS / "multistart.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multistart = _module()


@pytest.fixture(scope="module")
def rows():
    return multistart.read()


def _cells(rows, size, setting, implementation):
    return [r for r in rows if r["observations"] == size and r["setting"] == setting
            and r["implementation"] == implementation]


def test_every_length_and_setting_is_complete(rows):
    for size in multistart.LENGTHS:
        for setting in ("fixed", "free"):
            prophet = _cells(rows, size, setting, "prophet")
            assert len(prophet) == multistart.STARTS + 1
            assert sum(1 for r in prophet if r["start"] == "default") == 1
            assert len(_cells(rows, size, setting, "analytic_prophet")) == multistart.OURS
            assert len(_cells(rows, size, setting, "reference")) == 1


def test_the_starts_are_seeded_and_shared_by_both_settings():
    first = multistart.starts(300, 25, 6)
    second = multistart.starts(300, 25, 6)
    assert all(np.array_equal(a["delta"], b["delta"]) and a["k"] == b["k"]
               for a, b in zip(first, second))
    assert multistart.starts(1000, 25, 26)[0]["k"] != first[0]["k"]


def test_the_reference_is_certified_and_reached_from_every_start(rows):
    for row in rows:
        if row["implementation"] == "reference":
            assert row["residual"] < 1e-5
        elif row["implementation"] == "analytic_prophet":
            assert abs(row["shortfall"]) < 1e-6


def test_no_prophet_fit_reaches_the_optimum(rows):
    """'None of the 150 fits comes within 1e-3 nats of the optimum', and the
    default start is no exception -- in both settings."""
    for row in rows:
        if row["implementation"] == "prophet":
            assert row["shortfall"] > 1e-3, row
    text = (PAPER / "sections" / "evidence.tex").read_text()
    assert "None of the \\MultiTotal{} fits" in text


def test_prophets_fits_scatter_in_lp_and_in_the_rates(rows):
    for size in multistart.LENGTHS:
        fixed = [r for r in _cells(rows, size, "fixed", "prophet") if r["start"] != "default"]
        lp = [r["lp"] for r in fixed]
        assert max(lp) - min(lp) > 0.1
        assert np.median([r["delta_distance"] for r in fixed]) > 0.05


def test_the_design_has_full_column_rank_at_every_length(prophet_comparison):
    """The premise of uniqueness: trend, changepoint ramps and seasonal columns
    linearly independent, so the sigma-fixed objective is strictly convex."""
    import logging

    import corpora
    import harness

    stan_runner = sys.modules.get("stan_runner") or importlib.import_module("stan_runner")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    Prophet = prophet_comparison[0]
    for size in multistart.LENGTHS:
        data, _ = stan_runner.prophet_inputs(Prophet(**harness.PROPHET_KWARGS),
                                             corpora.peyton_manning(size))
        t = np.asarray(data["t"])
        s = np.asarray(data["t_change"])
        ramps = np.maximum(t[:, None] - s[None, :], 0.0)
        design = np.column_stack([t, np.ones_like(t), ramps, np.asarray(data["X"], dtype=float)])
        assert np.linalg.matrix_rank(design) == design.shape[1], size


def test_prophets_own_fit_reproduces_a_committed_start(rows, prophet_comparison):
    """Start 0 at T = 300 with sigma free, through Prophet().fit(inits=...) and
    Prophet's bundled binary -- no CmdStan install needed.

    By conftest.recorded_here's rule. **Anywhere**: the fit falls more than
    1e-3 nats short of the certified optimum, which is the paper's claim of
    every start. On CI's Linux runner this start stops a full nat away from
    where it stopped on the recording machine, so its lp__ is not compared
    there. **Here**: the committed lp__ exactly."""
    import logging

    import _common
    import _prophet_bridge as bridge
    import kkt
    import corpora

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    start = multistart.starts(300, 25, 6)[0]
    inits = {"k": start["k"], "m": start["m"], "delta": np.asarray(start["delta"]),
             "beta": np.asarray(start["beta"]), "sigma_obs": start["sigma_obs"]}
    stan_model, stan_data, params = bridge.capture_stan_model(
        prophet_comparison[0](**_common.PROPHET_KWARGS), corpora.peyton_manning(300),
        inits=inits, sig_figs=18)
    # at full precision, as the committed point was scored: the bridge's
    # stan_log_prob returns lp__ as CmdStan prints it, to eight digits
    lp, _ = kkt.stan_gradient(stan_model, stan_data, params["k"][0], params["m"][0],
                              params["delta"], params["sigma_obs"][0], params["beta"])
    reference = _cells(rows, 300, "free", "reference")[0]
    assert reference["lp"] - lp > 1e-3
    if recorded_here(multistart.RESULTS.with_suffix(".meta.json")):
        committed = next(r for r in _cells(rows, 300, "free", "prophet") if r["start"] == "0")
        assert lp == pytest.approx(committed["lp"], abs=1e-6)
