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

#203 then removed the relative margin itself. lp__ is defined only up to an
additive constant -- Stan's `~` drops normalising constants, and Prophet fits
y / max|y| -- so the same margin was 0.0045%, 0.0068% or 0.061% of it depending
on which constants were kept. Both pages now give the margin per observation,
which no constant moves, and nothing may quote a share of lp__ again.

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
    """The README's Tier 0 table, as `{T: (prophet, ours, margin, per observation)}`."""
    rows = {}
    for line in README.read_text().splitlines():
        cells = [c.strip().replace("*", "").replace("−", "-")
                 for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        try:
            rows[int(cells[0])] = (float(cells[1]), float(cells[2]),
                                   float(cells[3]), float(cells[4]))
        except ValueError:
            continue
    return rows


def test_the_readme_margins_are_the_committed_ones(margins):
    table = _readme_rows()
    assert set(table) == set(margins), (
        f"the README's Tier 0 table covers {sorted(table)}, the results "
        f"{sorted(margins)}")

    for size, (prophet, ours, margin, per_observation) in table.items():
        theirs, mine = margins[size]
        assert prophet == pytest.approx(theirs, abs=5e-4), size
        assert ours == pytest.approx(mine, abs=5e-4), size
        assert margin == pytest.approx(mine - theirs, abs=5e-4), (
            f"T={size}: README says the margin is {margin}, the results give "
            f"{mine - theirs:.4f}")
        # quoted to two significant figures
        assert per_observation == pytest.approx((mine - theirs) / size, rel=5e-2), (
            f"T={size}: README says {per_observation} nats per observation, the "
            f"results give {(mine - theirs) / size:.2g}")


def test_the_decline_per_observation_the_readme_claims_is_the_measured_one(readme, margins):
    per_observation = {size: (ours - theirs) / size
                       for size, (theirs, ours) in margins.items()}
    shortest, longest = min(per_observation), max(per_observation)
    factor = per_observation[shortest] / per_observation[longest]

    quoted = re.search(r"factor of (\d+) across this range", readme)
    assert quoted, "the README no longer states the decline factor"
    assert int(quoted.group(1)) == round(factor), (
        f"README says a factor of {quoted.group(1)}; the results give {factor:.1f}")


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


# Affirmative forms only: the pages that explain why no share is given say so in
# other words ("never quoted as a share", "not given as a share").
SHARE_OF_LP = ("as % of", "% of lp", "% of `lp", "share of the posterior",
               "as a share of prophet", "fraction of `lp__`", "relative margin",
               "relative_margin_percent")


def test_no_page_quotes_a_share_of_lp():
    """#203: lp__ is defined only up to an additive constant, so a margin as a
    share of it is not a quantity."""
    pages = [README, DOC, REPO / "paper" / "README.md",
             REPO / "evaluation" / "results" / "report.md",
             *sorted((REPO / "paper" / "sections").glob("*.tex")), REPO / "paper" / "main.tex"]
    for page in pages:
        text = " ".join(page.read_text().split()).lower()
        for phrase in SHARE_OF_LP:
            assert phrase not in text, f"{page.name} quotes {phrase!r}"
