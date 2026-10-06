"""
Issue #125: the memory comparison was attributed to the autodiff tape, and
measured more than that.

Three separate defects, and only one of them was a number:

  * `peak_rss_bytes()` took the **maximum** of `RUSAGE_SELF` and
    `RUSAGE_CHILDREN` while six places of prose -- the report caption, this
    tier's docstring, the benchmark README, `benchmark_memory.py` -- said it
    summed them. The sibling helper for *CPU* time really does sum, which is
    how the two drifted apart unnoticed.
  * The maximum was not merely a different convention, it was the wrong one
    here: the parent process is 75-120 MiB of interpreter, pandas and prophet
    against a cmdstan child peaking at 4-10 MiB, so it returned the parent at
    every size and never counted the child once -- defeating the only stated
    reason for looking at children at all.
  * The README read the resulting ratio as evidence of a retained tape. Peak
    RSS cannot establish that, because a constant difference in overhead would
    produce the same ratio.

What the tests below pin is the shape of the fix rather than the measurements:
that the function sums, that the prose describing it says so, that the two
sides are recorded separately, and that the tape claim is stated as what the
decomposition can support.
"""
import csv
import inspect
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
REPORT = REPO / "evaluation" / "results" / "report.md"
TIER3 = REPO / "evaluation" / "results" / "tier3_cost.csv"


@pytest.fixture(scope="module")
def report():
    return " ".join(REPORT.read_text().split())


@pytest.fixture(scope="module")
def tier3_rows():
    return list(csv.DictReader(open(TIER3)))


# -- the function does what it is described as doing ----------------------

def test_peak_rss_sums_the_two_sides_rather_than_taking_the_largest():
    """The claim, on a real child and independent of what ran before.

    Deliberately *not* phrased as "the parent exceeds the child", which is
    Prophet's case and was this test's first form: run under the full suite,
    an earlier test has already spawned a C++ compiler and left
    `RUSAGE_CHILDREN` near a gigabyte, so the premise fails for reasons that
    have nothing to do with the code under test. The identity below holds
    whatever ran first.
    """
    import subprocess
    import sys

    import _common

    subprocess.run([sys.executable, "-c",
                    "b = bytearray(48 * 1024 * 1024)\n"
                    "b[::4096] = b'x' * (len(b) // 4096)"], check=True)
    total = _common.peak_rss_bytes()
    own, children = _common.peak_rss_split()

    assert children > 0, "the child's peak RSS was not observed at all"
    assert total == pytest.approx(own + children, rel=1e-3), \
        "peak_rss_bytes is not the sum of the two sides"
    assert total > max(own, children), (
        "the sum does not exceed the maximum, so this measurement cannot "
        "distinguish the two conventions")


def test_the_convention_is_load_bearing_for_prophet_and_not_for_us(tier3_rows):
    """Why the choice had to be made deliberately rather than left implicit.

    On the committed numbers the two conventions give Prophet different answers
    and give us the same one -- so taking the maximum understates Prophet alone,
    which is the direction that matters for a claim in our favour.
    """
    sides = {}
    for row in tier3_rows:
        if row["metric"].startswith("fit_peak_rss_added_") and row["series"].startswith("peyton"):
            T = int(row["series"].split("[:")[1].rstrip("]"))
            side = row["metric"].rsplit("_", 1)[1]
            sides[(row["implementation"], T, side)] = float(row["value"])

    lengths = sorted({T for _, T, _ in sides})
    assert lengths, "no per-length memory rows"
    longest = lengths[-1]

    prophet_self = sides[("prophet", longest, "self")]
    prophet_child = sides[("prophet", longest, "children")]
    assert prophet_child > 0
    assert (prophet_self + prophet_child) > max(prophet_self, prophet_child), \
        "the convention makes no difference to Prophet, so #125 was about nothing"

    for T in lengths:
        own = sides[("compiled", T, "self")]
        child = sides[("compiled", T, "children")]
        assert child == 0, f"our fit forked something at T={T}"
        assert own + child == max(own, child)


def test_peak_rss_split_puts_the_child_on_the_child_side():
    """A child's memory must land on the child side and not on the parent's.

    Asserted as "the parent did not move" rather than "the child side rose",
    because `RUSAGE_CHILDREN` is a maximum over every child ever waited on:
    the test above already ran one, so a second child of the same size raises
    nothing and an ordering-dependent assertion fails. That is the same
    left-censoring the report warns about in Prophet's child column, met here
    in the test suite.
    """
    import subprocess
    import sys

    import _common

    before = _common.peak_rss_split()
    subprocess.run([sys.executable, "-c",
                    "b = bytearray(192 * 1024 * 1024)\n"
                    "b[::4096] = b'x' * (len(b) // 4096)"], check=True)
    after = _common.peak_rss_split()

    assert len(after) == 2
    assert after[0] == before[0], "the child's memory was charged to the parent"
    assert after[1] >= 150 * 1024 * 1024, (
        f"a 192 MiB child left the children side at {after[1]} bytes")


def test_the_prose_describing_the_measurement_matches_the_implementation():
    """What actually went wrong in #125: the code and its description drifted.

    Asserted mechanically because that drift is invisible to every other test
    in the suite -- the numbers were self-consistent, only the sentence
    explaining them was false.
    """
    import _common

    source = inspect.getsource(_common.peak_rss_bytes)
    assert "sum(" in source, "peak_rss_bytes no longer sums"

    caption = " ".join(Path(REPO / "evaluation" / "report.py").read_text().split())
    assert "**summing** `RUSAGE_SELF` and `RUSAGE_CHILDREN`" in caption, (
        "the Tier 3 caption no longer says the function sums; if the "
        "implementation changed, this is the sentence that has to change with it")


def test_the_report_states_which_direction_the_convention_biases(report):
    """Both conventions are one-sided, so a reader is owed the direction."""
    assert "upper bound on simultaneous residency" in report
    assert "lower" in report and "maximum" in report


# -- the attribution, and what it can support -----------------------------

def test_the_two_sides_of_the_fork_are_recorded_separately(tier3_rows):
    metrics = {row["metric"] for row in tier3_rows}

    assert {"fit_peak_rss_added", "fit_peak_rss_added_self",
            "fit_peak_rss_added_children"} <= metrics


def test_the_recorded_sides_add_up_to_the_recorded_total(tier3_rows):
    cells = {}
    for row in tier3_rows:
        cells[(row["series"], row["implementation"], row["metric"])] = float(row["value"])
    checked = 0
    for (series, implementation, metric), total in list(cells.items()):
        if metric != "fit_peak_rss_added":
            continue
        own = cells[(series, implementation, "fit_peak_rss_added_self")]
        child = cells[(series, implementation, "fit_peak_rss_added_children")]
        assert own + child == total, (series, implementation)
        checked += 1
    assert checked > 0, "no memory rows to check"


def test_prophets_growth_with_T_reaches_the_process_that_runs_the_autodiff(tier3_rows):
    """The closest this measurement gets to the mechanism (#125).

    If the gap were the tape, the growth has to appear in the cmdstan child,
    because that is where Stan's autodiff runs. Under the old maximum it could
    not appear there at all, which is what made the attribution unfalsifiable.
    """
    child = {}
    for row in tier3_rows:
        if (row["metric"] == "fit_peak_rss_added_children"
                and row["implementation"] == "prophet"
                and row["series"].startswith("peyton")):
            child[int(row["series"].split("[:")[1].rstrip("]"))] = float(row["value"])

    assert child, "no per-length child measurements for prophet"
    assert child[max(child)] > 0, "the cmdstan child's memory is still uncounted"
    assert child[max(child)] > child[min(child)], \
        "the child side does not grow with the series"


def test_the_decomposition_separates_fixed_cost_from_cost_per_observation(report):
    assert "memory = fixed + slope·T" in report
    assert "per 1000 obs" in report
    assert re.search(r"fixed costs are within [\d.]+ MiB of each other", report), \
        "the report no longer states the fixed-cost comparison"


def test_the_tape_is_offered_as_consistent_rather_than_established(report):
    """The claim #125 asked to be softened or supported. It is now both: the
    decomposition rules out fixed overhead, and the report says plainly that
    peak RSS cannot go further than that."""
    assert "peak RSS cannot tell those apart" in report
    assert "it does not isolate the tape" in report


def test_the_readme_no_longer_reads_the_ratio_as_proof_of_the_mechanism():
    readme = " ".join((REPO / "README.md").read_text().split())

    assert "rather than proving the mechanism" in readme
    assert "peak RSS cannot tell a tape from any other allocation" in readme
    # The old sentence, which asserted the mechanism from a ratio alone.
    assert "which is what a retained tape predicts. Predicting" not in readme


# -- the report's own arithmetic ------------------------------------------

def test_the_linear_fit_recovers_a_known_fixed_cost_and_slope():
    """`_linear_in_t` is the whole basis of the attribution, so it is tested
    against data whose answer is known rather than only on the real rows."""
    import report as report_module

    points = [(t, 4.0 + 0.5 * (t / 1000.0)) for t in (50, 100, 300, 1000, 2905)]

    fixed, slope, r_squared = report_module._linear_in_t(points)

    assert fixed == pytest.approx(4.0)
    assert slope == pytest.approx(0.5)
    assert r_squared == pytest.approx(1.0)
