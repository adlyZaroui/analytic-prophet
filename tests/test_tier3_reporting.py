"""
Issue #154: the speed claim in the README, against the results it comes from.

The README said "1.4–10× faster ... 10× at T = 50, falling to 1.4× by
T = 1000". Two things were wrong with that and only one of them was the range.

  * **The top end did not match the committed results.** `tier3_cost.csv` has
    0.170004 s against 0.020626 s at T = 50, which is 8.24×, not 10×. An
    overstatement in this project's own favour.
  * **It stopped at T = 1000, where memory is reported at T = 2905.** Given a
    stated trend of 10× → 1.4× a reader assumes it keeps falling, possibly
    through parity. It does not: the ratio is 1.48× at T = 2905, so T = 1000 is
    a trough rather than a waypoint on a decline, and the omission was costing
    the project a claim it had already measured.

Four README figures have now drifted from the committed results this way,
because the README is hand-written where `report.md` is generated. These tests
recompute from the CSV, so the drift fails the suite rather than waiting for a
reader to divide two numbers.
"""
import csv
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
TIER3 = REPO / "evaluation" / "results" / "tier3_cost.csv"


@pytest.fixture(scope="module")
def readme():
    return " ".join(README.read_text().split())


@pytest.fixture(scope="module")
def speedups():
    """`{T: prophet wall / our wall}` from the committed tier 3 results."""
    walls = {}
    for row in csv.DictReader(open(TIER3)):
        if (row["metric"] == "fit_wall" and row["configuration"] == "default"
                and row["series"].startswith("peyton")):
            size = int(row["series"].split("[:")[1].rstrip("]"))
            walls.setdefault(size, {})[row["implementation"]] = float(row["value"])
    return {size: cell["prophet"] / cell["compiled"]
            for size, cell in walls.items()
            if "prophet" in cell and "compiled" in cell}


def test_the_range_endpoints_are_the_measured_ones(readme, speedups):
    quoted = re.search(r"[Ff]itting is \*\*([\d.]+)–([\d.]+)× faster\*\*", readme)
    assert quoted, "the README no longer states the speed range in the form checked here"

    low, high = float(quoted.group(1)), float(quoted.group(2))
    assert low == pytest.approx(min(speedups.values()), abs=0.05), (
        f"README says the range starts at {low}×; the results say "
        f"{min(speedups.values()):.2f}×")
    assert high == pytest.approx(max(speedups.values()), abs=0.05), (
        f"README says the range reaches {high}×; the results say "
        f"{max(speedups.values()):.2f}× -- this was 10× against a measured 8.24×")


def test_every_speedup_the_readme_quotes_is_the_measured_one(readme, speedups):
    """Each `N× at T = M` in the claim, checked against the CSV."""
    quoted = re.findall(r"([\d.]+)× at T = (\d+)", readme)
    assert quoted, "the README quotes no speedup at a named T"

    for ratio, size in quoted:
        size = int(size)
        assert size in speedups, f"README quotes T = {size}, which the tier does not measure"
        assert float(ratio) == pytest.approx(speedups[size], abs=0.05), (
            f"README says {ratio}× at T = {size}; the results say "
            f"{speedups[size]:.2f}×")


def _speed_claim(readme):
    """The speed sentence alone.

    Scoped deliberately, and twice. `T = 2905` also appears in the *memory*
    claim a few lines later, so asserting it is somewhere in the README passes
    even when the speed claim has been pulled back to T = 1000 -- which is what
    the first version of this did, and a 600-character window did not fix it
    either, because the memory claim is inside 600 characters. The speed claim
    is everything before the memory claim, so that is what this returns.

    The anchor is case-insensitive because #152 made this sentence start a
    paragraph, which capitalised it. A case-sensitive anchor turned that
    restructure into three test failures that said nothing about the prose
    being wrong.
    """
    anchor = re.search(r"[Ff]itting is \*\*", readme)
    assert anchor, "the speed claim has moved; this test needs updating"
    end = readme.find("peak memory", anchor.start())
    return readme[anchor.start():end if end != -1 else anchor.start() + 400]


def test_the_claim_reaches_the_longest_series_memory_is_reported_at(readme, speedups):
    """Speed stopped at T = 1000 while memory was reported at T = 2905, which
    is what invited the assumption that it keeps falling."""
    longest = max(speedups)

    assert f"T = {longest}" in _speed_claim(readme), (
        f"the speed claim does not reach T = {longest}, where the memory claim "
        "is made -- which is the omission #154 was about")


def test_the_shape_is_described_as_a_trough_because_that_is_what_it_is(readme, speedups):
    """The correction only matters because the ratio turns back up. If it ever
    stops doing so, this prose becomes wrong and should fail here first."""
    longest = max(speedups)
    trough = min(speedups, key=speedups.get)

    assert trough != longest, (
        "the minimum speedup is now at the longest series, so the README's "
        "'trough rather than a decline' is no longer true")
    assert "trough rather than a decline" in readme
