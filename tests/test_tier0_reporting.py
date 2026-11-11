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

#212 took the margins table out of the README, which now claims only the
direction -- higher at every series length measured -- and leaves the numbers
to the paper and the evaluation report. These tests hold that claim to
`tier0_agreement.csv`. #242 took the doc's tables out too, so neither page
carries a margin of its own any more, and the doc keeps the one sentence that
declines to explain its size.
"""
import csv
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


def test_the_readme_claims_the_direction_the_results_show(readme, margins):
    """The lede: "By Prophet's own posterior density, its fits score higher at
    every series length measured". True of every committed margin, and the
    README quotes no margin itself, so it cannot go stale on one.

    One figure in nats is allowed, and only one (#250): how far Prophet's own
    Newton stops short, for readers who stay on Prophet. It is a paper macro,
    held in its sentence by tests/test_readme_structure.py, so it cannot go
    stale either."""
    assert margins, "no committed margins to hold the claim to"
    assert all(ours > theirs for theirs, ours in margins.values()), (
        "a committed margin is no longer positive, so the lede's 'at every "
        "series length measured' is wrong")
    lede = readme[:readme.index("**Status: early development.**")]
    assert "score higher at every series length measured" in lede
    for size, (theirs, ours) in margins.items():
        for figure in (f"{ours - theirs:.3f}", f"{ours - theirs:.2f}"):
            assert figure not in readme, (
                f"the README quotes Tier 0's margin at T = {size} again; the paper owns them")
    one = " ".join(readme.split())
    assert one.count("nats") == 1, "a margin in nats beyond Prophet's own Newton's"
    staying = one.index("**Staying on Prophet?**")
    assert staying < one.index("nats") < one.index("([in the paper](#paper--citing))", staying)


def test_the_doc_carries_no_margin_table(doc):
    """The fault this issue was about -- a headline table with no T and no
    configuration -- cannot recur on a page with no table of margins. #242
    left the margins to the paper."""
    for figure in ("7797.2523", "7794.9104", "+2.34", "+0.36"):
        assert figure not in doc, f"the doc quotes a margin again: {figure}"


def test_the_doc_claims_no_mechanism_for_the_size(doc):
    """It states the direction and declines to explain the magnitude, which is
    what the data supports: the two configurations move opposite ways. The
    README states the direction only (#212)."""
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
