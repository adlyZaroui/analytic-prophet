"""
Issue #173: the lp__ margin against series length, as a figure.

`paper/experiments/margin.py` measures Tier 0's quantity on a grid of 79
lengths and draws it. What is checked here is what the figure and its caption
assert without a macro to carry them:

  * **where the boundaries are** -- T = 693 is where Peyton Manning first spans
    730 days, and T = 33 is where Prophet first places all 25 changepoints;
  * **which optimizer ran** on each side of T = 100, for both implementations;
  * **that the grid is Tier 0's measurement**, by agreeing with Tier 0's
    committed margins where the two overlap, and by reproducing committed
    cells from the code;
  * **the word "every"** in "positive in every one of them", which is typed.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"
RESULTS = PAPER / "results" / "margin.csv"
TIER0 = REPO / "evaluation" / "results" / "tier0_agreement.csv"


def _margin():
    sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("margin", EXPERIMENTS / "margin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


margin = _margin()


@pytest.fixture(scope="module")
def rows():
    return margin.read()


def test_yearly_switches_on_where_the_history_first_spans_730_days():
    """Prophet's auto rule enables yearly at 730 days of history; on Peyton
    Manning, which has gaps, that is row 693 and not row 730."""
    from conftest import DATA_PATH

    ds = pd.to_datetime(pd.read_csv(DATA_PATH)["ds"])
    span = (ds - ds.iloc[0]).dt.days.to_numpy()
    assert span[margin.YEARLY_FROM - 1] >= 730 > span[margin.YEARLY_FROM - 2]


def test_the_grid_is_complete_and_the_configurations_are_what_they_say(rows):
    measured = {(row["configuration"], row["observations"]) for row in rows}
    assert measured == set(margin.cells())
    for row in rows:
        if row["configuration"] == "yearly_only":
            assert row["seasonalities"] == "yearly"
        elif row["observations"] < margin.YEARLY_FROM:
            assert row["seasonalities"] == "weekly"
        else:
            assert row["seasonalities"] == "weekly+yearly"


def test_both_sides_run_newton_below_100_and_lbfgs_above(rows):
    """The figure labels the two sides of T = 100 by method; that label is
    only true if neither side fell back to Newton above it."""
    for row in rows:
        expected = "Newton" if row["observations"] < margin.NEWTON_BELOW else "LBFGS"
        assert row["prophet_algorithm"] == expected, row
        assert row["ours_algorithm"] == expected, row


def test_beta_crossed_between_identical_design_matrices(rows):
    assert max(row["design_gap"] for row in rows) < 1e-9


def test_the_margin_is_positive_in_every_cell(rows):
    """"Positive in every one of them" is typed in the paper; this is what
    keeps it true. It is also what lets the figure use a log scale at all."""
    assert min(row["margin"] for row in rows) > 0
    assert "positive in every one of them" in (PAPER / "sections" / "evidence.tex").read_text()


def test_the_grid_agrees_with_tier_0_where_they_overlap(rows):
    """Same measurement, two scripts. Tier 0 reads Prophet's optimum at
    CmdStan's default eight digits and this at eighteen, which is the only
    difference -- worth about 1e-4 nats here."""
    tier0 = {}
    with open(TIER0, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "lp___difference":
                size = int(row["series"].split("[:")[1].rstrip("]"))
                tier0[size] = float(row["value"])
    grid = {row["observations"]: row["margin"] for row in rows
            if row["configuration"] == "default"}
    assert set(tier0) <= set(grid)
    for size, value in tier0.items():
        assert grid[size] == pytest.approx(value, abs=2e-3), size


def test_prophet_places_all_25_changepoints_from_t_33(prophet_comparison):
    """The caption's "below T = 33": Prophet keeps n_changepoints + 1 within
    floor(0.8 T) observations, so 25 changepoints need T >= 33."""
    from conftest import DATA_PATH

    Prophet = prophet_comparison[0]
    df = pd.read_csv(DATA_PATH)
    counts = {}
    for size in (32, 33):
        model = Prophet()
        history = df.iloc[:size].copy()
        model.history = model.setup_dataframe(history.assign(ds=pd.to_datetime(history["ds"])),
                                              initialize_scales=True)
        model.set_changepoints()
        counts[size] = len(model.changepoints)
    assert counts == {32: 24, 33: 25}
    assert "Below $T = 33$ Prophet" in (PAPER / "sections" / "evidence.tex").read_text()


@pytest.mark.parametrize("cell", [("default", 99), ("default", 100)])
def test_the_code_reproduces_the_committed_cells(cell, rows, prophet_comparison,
                                                 compiled_optimizer_module):
    """The two cells either side of the method boundary, refitted from the
    code and compared with what is committed. Both fits are deterministic."""
    committed = next(row for row in rows
                     if (row["configuration"], row["observations"]) == cell)
    fresh = margin.measure(cell, compiled_optimizer_module)
    for key in ("lp_prophet", "lp_ours", "margin"):
        assert fresh[key] == pytest.approx(committed[key], rel=1e-9, abs=1e-9), key
    assert fresh["prophet_algorithm"] == committed["prophet_algorithm"]


def test_the_figure_is_committed_and_redraws_identically(rows, tmp_path):
    """Redrawing unchanged results must give an unchanged file, or a rerun is
    a binary diff nobody can review."""
    pytest.importorskip("matplotlib")
    assert margin.FIGURE.exists() and margin.FIGURE.stat().st_size > 10_000
    first = margin.draw(rows, tmp_path / "a.pdf").read_bytes()
    second = margin.draw(rows, tmp_path / "b.pdf").read_bytes()
    assert first == second


def test_a_non_positive_margin_is_refused_rather_than_dropped(rows, tmp_path):
    """A log axis cannot show it, and matplotlib would silently omit it."""
    pytest.importorskip("matplotlib")
    broken = [dict(row) for row in rows]
    broken[0]["margin"] = -0.01
    with pytest.raises(ValueError, match="non-positive"):
        margin.draw(broken, tmp_path / "c.pdf")
