"""
Issue #146: how the posterior margin is reported, in two places that disagreed.

`docs/non-smooth-objective.md` led with an **unlabelled** table -- 7794.9104
against 7797.2523, a margin of +2.34 nats -- while the README's `T = 2905` row
gave +0.361. A factor of 6.5 for what both described the same way: Stan's own
log density, on Prophet's own changepoints, only the optimizer differing.

The doc's table turned out to be `T = 2905` **with yearly seasonality only**,
reproduced to the last digit. So the margin depends on the width of the model
as well as the length of the series, and neither dependence is monotone:

    T       default     yearly only
    300      +1.99          -
    1000     +2.76        +1.26
    2905     +0.36        +2.34

Nothing said so, and a reader computing the relative margin off the README's
table found it falling 54x across the range with no acknowledgement. Both pages
now state it, and neither claims a mechanism for the size.

These tests recompute what the README quotes from `tier0_agreement.csv`. The
doc's yearly-only figures are not in a committed tier, so they are checked for
internal consistency only -- the grid has to agree with the single table above
it, which is the error that started this.
"""
import csv
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
DOC = REPO / "docs" / "non-smooth-objective.md"
TIER0 = REPO / "evaluation" / "results" / "tier0_agreement.csv"


@pytest.fixture(scope="module")
def readme():
    return " ".join(README.read_text().split())


@pytest.fixture(scope="module")
def doc():
    return " ".join(DOC.read_text().split())


@pytest.fixture(scope="module")
def margins():
    """`{T: (prophet lp__, our lp__)}` from the committed Tier 0 results."""
    rows = {}
    for row in csv.DictReader(open(TIER0)):
        if row["metric"] == "lp__" and row["series"].startswith("peyton"):
            size = int(row["series"].split("[:")[1].rstrip("]"))
            rows.setdefault(size, {})[row["implementation"]] = float(row["value"])
    return {size: (cell["prophet"], cell["analytic_prophet"])
            for size, cell in rows.items()
            if "prophet" in cell and "analytic_prophet" in cell}


def _readme_rows():
    """The README's Tier 0 table, as `{T: (prophet, ours, margin, percent)}`."""
    rows = {}
    for line in README.read_text().splitlines():
        cells = [c.strip().replace("*", "").replace("−", "-")
                 for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        try:
            rows[int(cells[0])] = (float(cells[1]), float(cells[2]),
                                   float(cells[3]), float(cells[4].rstrip("%")))
        except ValueError:
            continue
    return rows


def test_the_readme_margins_are_the_committed_ones(margins):
    table = _readme_rows()
    assert set(table) == set(margins), (
        f"the README's Tier 0 table covers {sorted(table)}, the results "
        f"{sorted(margins)}")

    for size, (prophet, ours, margin, percent) in table.items():
        theirs, mine = margins[size]
        assert prophet == pytest.approx(theirs, abs=5e-4), size
        assert ours == pytest.approx(mine, abs=5e-4), size
        assert margin == pytest.approx(mine - theirs, abs=5e-4), (
            f"T={size}: README says the margin is {margin}, the results give "
            f"{mine - theirs:.4f}")
        assert percent == pytest.approx((mine - theirs) / theirs * 100, rel=5e-3), (
            f"T={size}: README says {percent}% of lp__, the results give "
            f"{(mine - theirs) / theirs * 100:.4f}%")


def test_the_relative_decline_the_readme_claims_is_the_measured_one(readme, margins):
    relative = {size: (ours - theirs) / theirs
                for size, (theirs, ours) in margins.items()}
    shortest, longest = min(relative), max(relative)
    factor = relative[shortest] / relative[longest]

    quoted = re.search(r"factor of (\d+) across this range", readme)
    assert quoted, "the README no longer states the decline factor"
    assert int(quoted.group(1)) == pytest.approx(factor, abs=1.0), (
        f"README says a factor of {quoted.group(1)}; the results give {factor:.0f}")


def test_the_readme_says_the_absolute_margin_is_not_monotone(readme, margins):
    """It rises from T = 300 to T = 1000 before falling. A claim that the margin
    simply decays with T would be wrong, and was the shape the review assumed."""
    by_size = [margins[s][1] - margins[s][0] for s in sorted(margins)]
    rising_then_falling = by_size[1] > by_size[0] and by_size[-1] < by_size[-2]

    assert rising_then_falling, (
        "the absolute margin is now monotone, so the README's 'not even "
        "monotone, rising before it falls' is wrong")
    assert "not even monotone" in readme


def test_the_doc_labels_its_table(doc):
    """The fault this issue is about: the headline table carried no T and no
    configuration, while the one below it was labelled."""
    assert "with yearly seasonality only" in doc
    assert "`T = 2905`" in doc


def test_the_docs_grid_agrees_with_the_table_above_it(doc):
    """The grid's 2905 cells have to be the two numbers the file already
    states -- +2.34 from its own table, +0.36 from the README's."""
    assert "| 2905 | **+0.36** | **+2.34** |" in doc
    assert "+2.34 nats" in doc, "the single table and the grid disagree at T = 2905"


def test_neither_page_claims_a_mechanism_for_the_size(readme, doc):
    """Both state the direction and decline to explain the magnitude, which is
    what the data supports: the two configurations move opposite ways."""
    assert "No mechanism for the size is claimed" in readme
    assert "No mechanism for the *size* of the margin is claimed here" in doc
