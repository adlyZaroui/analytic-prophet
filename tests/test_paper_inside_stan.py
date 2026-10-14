"""
Issue #202: the fix restated inside Stan, on Stan's own L-BFGS.

Two reformulations of Prophet's program are smooth and keep the minimisers --
Hoff's Hadamard parametrisation, and the paper's split as Stan has to state it,
through `<lower=0>`. `paper/experiments/inside_stan.py` fits both with Prophet's
settings. What the committed result shows, and is held to here:

  * **neither reaches the optimum** -- every run stops short of the certified
    optimum, the closest by about 1e-3 nats;
  * **neither makes a rate exactly zero**, in any run;
  * **both depend on a starting choice Prophet does not make**, and the split
    by orders of magnitude;
  * **from random starts the Hadamard program falls far further short than
    Prophet's own**, from the same starts (#169's committed fits).

The programs themselves are checked in tests/test_paper_stan_variants.py.
"""
import csv
import importlib.util
import statistics
import sys
from pathlib import Path

import pytest

from conftest import recorded_here

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"


def _module(name):
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inside = _module("inside_stan")


@pytest.fixture(scope="module")
def rows():
    return inside.read()


def _runs(rows, **where):
    return [row for row in rows if row["variant"] != "reference"
            and all(row[key] == value for key, value in where.items())]


def test_every_length_setting_variant_and_start_is_there(rows):
    for size in inside.LENGTHS:
        for setting in ("fixed", "free"):
            cell = [row for row in rows if row["observations"] == size
                    and row["setting"] == setting]
            assert sum(1 for row in cell if row["variant"] == "reference") == 1
            for family, scales in inside.SCALES.items():
                runs = [row for row in cell if row["variant"] == family]
                assert sum(1 for row in runs if row["start"] == "default") == len(scales)
                assert sum(1 for row in runs if row["start"] != "default") == inside.STARTS


def test_the_reference_is_certified_and_no_run_beats_it(rows):
    """A run above the reference would mean the reference is not the optimum."""
    for row in rows:
        if row["variant"] == "reference":
            assert row["residual"] < 1e-5 and row["reference_zeros"] > 0
        else:
            assert row["shortfall"] > -1e-6, row


def test_no_run_reaches_the_optimum_and_none_makes_a_rate_exactly_zero(rows):
    runs = _runs(rows)
    assert min(row["shortfall"] for row in runs) > 5e-4
    assert all(row["exact_zeros"] == 0 for row in runs)


def test_both_depend_on_a_starting_choice_prophet_does_not_make(rows):
    """From Prophet's own start, where every rate is zero, each variant needs a
    value to start a zero rate from; the split's result moves with it by a
    factor of five or more at every length and setting, and of fifty at some."""
    spreads = []
    for size in inside.LENGTHS:
        for setting in ("fixed", "free"):
            split = [row["shortfall"] for row in _runs(rows, observations=size, setting=setting,
                                                       variant="split", start="default")]
            spreads.append(max(split) / min(split))
    assert min(spreads) > 5 and max(spreads) > 50, spreads


def test_from_random_starts_the_hadamard_program_falls_further_short_than_prophets(rows):
    """Prophet's own L-BFGS from the same seeded starts is #169's committed fit."""
    multistart = list(csv.DictReader(open(PAPER / "results" / "multistart.csv")))
    for setting in ("fixed", "free"):
        prophet = [float(row["shortfall"]) for row in multistart
                   if int(row["observations"]) == 2905 and row["setting"] == setting
                   and row["implementation"] == "prophet"
                   and row["start"] in {str(index) for index in range(inside.STARTS)}]
        hadamard = [row["shortfall"] for row in _runs(rows, observations=2905, setting=setting,
                                                      variant="hadamard")
                    if row["start"] != "default"]
        assert len(prophet) == len(hadamard) == inside.STARTS
        assert statistics.median(hadamard) > 10 * statistics.median(prophet)


def test_from_prophets_start_the_hadamard_program_gets_closer_than_prophet(rows):
    """The one place it does better, and the reason it is worth reporting."""
    multistart = list(csv.DictReader(open(PAPER / "results" / "multistart.csv")))
    for size in inside.LENGTHS:
        for setting in ("fixed", "free"):
            prophet = next(float(row["shortfall"]) for row in multistart
                           if int(row["observations"]) == size and row["setting"] == setting
                           and row["implementation"] == "prophet" and row["start"] == "default")
            hadamard = [row["shortfall"] for row in _runs(rows, observations=size,
                                                          setting=setting, variant="hadamard",
                                                          start="default")]
            assert min(hadamard) < prophet


def test_the_code_reproduces_a_committed_run(rows, prophet_comparison):
    """The Hadamard program from Prophet's start at T = 300, sigma fixed. Stan's
    L-BFGS stops where the arithmetic lets it, so the committed lp__ is compared
    only where it was recorded; anywhere, the run stops short of the optimum
    with no rate exactly zero."""
    runner = sys.modules.get("stan_runner") or _module("stan_runner")
    try:
        runner.cmdstan_path()
    except runner.CmdStanMissing as missing:
        pytest.skip(f"the Hadamard program needs CmdStan {runner.CMDSTAN_VERSION}: {missing}")
    import numpy as np

    import harness
    import multistart

    problem = multistart.Problem(300)
    scale = inside.SCALES["hadamard"][1](float(problem.data["tau"]))
    inits = {"k": problem.inits["k"], "m": problem.inits["m"],
             "beta": list(problem.inits["beta"]),
             "delta_u": [scale] * int(problem.data["S"]),
             "delta_v": [0.0] * int(problem.data["S"])}
    params, _ = runner.fit("fixed_sigma_hadamard", problem.data, inits,
                           sigma_obs=problem.sigma, sig_figs=18, seed=harness.SEED)
    lp, _ = problem.score(params)
    committed = next(row for row in _runs(rows, observations=300, setting="fixed",
                                          variant="hadamard", start="default")
                     if abs(row["scale"] - scale) < 1e-12)
    reference = next(row for row in rows if row["observations"] == 300
                     and row["setting"] == "fixed" and row["variant"] == "reference")
    assert reference["lp"] - lp > 5e-4
    assert int(np.sum(np.asarray(params["delta"]) == 0.0)) == 0
    if recorded_here(inside.RESULTS.with_suffix(".meta.json")):
        assert lp == pytest.approx(committed["lp"], abs=1e-6)
