"""
Issue #168: smoothing the prior inside Prophet's own Stan program.

`paper/experiments/smoothing.py` fits the smoothed program with Stan's L-BFGS
across eps and separates two quantities the naive reading mixes: how far Stan
stops from the smoothed problem's own optimum (the convergence gap) and how
far that optimum sits from Prophet's (the smoothing bias). Checked here:

  * **the protocol** -- the whole grid, L-BFGS in every run, one seed;
  * **that B is the smoothed optimum** -- every Newton decrement converged,
    and the three curves that are positive by construction are positive;
  * **the arithmetic** -- every derived column recomputes from the lp__ ones;
  * **the sentences the paper types** -- "comparable", "in proportion to eps
    up to 1e-2", "the two cross";
  * **the code** -- one cell refitted, exactly where it was recorded and to the
    quoted precision elsewhere (#186);
  * **the figure** -- redraws identically and refuses what it cannot show.
"""
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
RESULTS = PAPER / "results" / "smoothing.csv"


def _module(name):
    sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoothing = _module("smoothing")


@pytest.fixture(scope="module")
def rows():
    return smoothing.read()


def _by_length(rows):
    out = {}
    for row in rows:
        out.setdefault(int(row["observations"]), []).append(row)
    return {size: sorted(cells, key=lambda r: r["epsilon"]) for size, cells in out.items()}


def test_the_grid_is_complete_and_every_run_was_stans_lbfgs(rows):
    cells = {(int(r["observations"]), r["epsilon"]) for r in rows}
    expected = {(size, eps) for size in smoothing.LENGTHS for eps in smoothing.EPSILONS}
    assert len(cells) == len(expected) and all(
        any(size == s and np.isclose(eps, e, rtol=1e-12) for s, e in cells)
        for size, eps in expected)
    assert {r["stan_algorithm"] for r in rows} == {"lbfgs"}


def test_one_seed_for_every_run():
    meta = json.loads(RESULTS.with_suffix(".meta.json").read_text())
    source = (EXPERIMENTS / "smoothing.py").read_text()
    assert meta["seed"] and "seed=harness.SEED" in source


def test_every_b_is_the_smoothed_problems_optimum(rows):
    for row in rows:
        assert row["newton_decrement"] < smoothing.CONVERGED, row
        assert row["starts_converged"] >= 1, row


def test_the_curves_positive_by_construction_are_positive(rows):
    """B beats A under the smoothed density, and L1 beats B under Prophet's."""
    for row in rows:
        assert row["gap_smooth"] > 0 and row["bias_l1"] > 0 and row["bias_smooth"] > 0, row


def test_every_derived_column_recomputes(rows):
    for r in rows:
        assert r["gap_smooth"] == pytest.approx(r["lp_smooth_B"] - r["lp_smooth_A"], abs=1e-9)
        assert r["gap_l1"] == pytest.approx(r["lp_l1_B"] - r["lp_l1_A"], abs=1e-9)
        assert r["bias_l1"] == pytest.approx(r["lp_l1_L1"] - r["lp_l1_B"], abs=1e-9)
        assert r["bias_smooth"] == pytest.approx(r["lp_smooth_B"] - r["lp_smooth_L1"], abs=1e-9)
        assert r["total_l1"] == pytest.approx(r["gap_l1"] + r["bias_l1"], abs=1e-9)


def test_at_the_l1_end_stan_stops_about_as_short_as_prophet(rows):
    """'by amounts comparable to Prophet's shortfall on the l1 problem'."""
    for size, cells in _by_length(rows).items():
        first = cells[0]
        prophet = first["lp_l1_L1"] - first["lp_l1_prophet"]
        assert 0.5 < first["gap_smooth"] / prophet < 2.0, size


def test_the_bias_is_proportional_to_eps_up_to_one_hundredth(rows):
    for size, cells in _by_length(rows).items():
        ratios = [c["bias_l1"] / c["epsilon"] for c in cells if c["epsilon"] <= 1e-2 + 1e-15]
        assert max(ratios) / min(ratios) < 1.15, size
        biases = [c["bias_l1"] for c in cells]
        assert biases == sorted(biases), size


def test_the_two_curves_cross_at_every_length(rows):
    for size, cells in _by_length(rows).items():
        assert cells[0]["gap_smooth"] > cells[0]["bias_l1"], size
        assert cells[-1]["gap_smooth"] < cells[-1]["bias_l1"], size


def test_the_paper_quotes_the_drop_per_length_not_a_blanket_claim():
    """At T = 2905 the gap falls by about one order of magnitude, not several;
    the text gives each length's factor rather than 'orders of magnitude'."""
    text = (PAPER / "sections" / "evidence.tex").read_text()
    assert r"\SmoothDropLong" in text
    section = text[text.index(r"\label{sec:evidence:surrogate}"):text.index(r"\label{sec:evidence:flat}")]
    assert "orders of magnitude" not in section


def _recorded_here():
    from importlib.metadata import version
    meta = json.loads(RESULTS.with_suffix(".meta.json").read_text())
    return (meta["platform"].split("-")[0] == platform.platform().split("-")[0]
            and meta["machine"] == platform.machine()
            and meta["versions"]["prophet"] == version("prophet"))


def test_the_code_reproduces_a_committed_cell(rows, prophet_comparison):
    runner = _module("stan_runner")
    try:
        runner.cmdstan_path()
    except runner.CmdStanMissing as missing:
        pytest.skip(f"the smoothed program needs CmdStan {runner.CMDSTAN_VERSION}: {missing}")
    epsilon = 1e-4
    committed = next(r for r in rows if r["observations"] == 300
                     and np.isclose(r["epsilon"], epsilon, rtol=1e-12))
    fresh = smoothing.measure(300, epsilons=(epsilon,))[0]
    for key in ("lp_smooth_A", "lp_smooth_B", "lp_l1_B", "lp_l1_L1"):
        if _recorded_here():
            assert fresh[key] == pytest.approx(committed[key], rel=1e-9, abs=1e-9), key
        else:
            assert fresh[key] == pytest.approx(committed[key], abs=1e-3), key


def test_the_figure_redraws_identically_and_refuses_what_it_cannot_show(rows, tmp_path):
    pytest.importorskip("matplotlib")
    assert smoothing.FIGURE.exists() and smoothing.FIGURE.stat().st_size > 10_000
    first = smoothing.draw(rows, tmp_path / "a.pdf").read_bytes()
    assert first == smoothing.draw(rows, tmp_path / "b.pdf").read_bytes()
    broken = [dict(row) for row in rows]
    broken[0]["bias_l1"] = -1e-6
    with pytest.raises(ValueError, match="non-positive"):
        smoothing.draw(broken, tmp_path / "c.pdf")
