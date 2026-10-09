"""
Issue #100: the README's figure, and the claims its caption makes.

The figure is the most-looked-at thing in the repository and the easiest
place to overclaim, because the two implementations mostly agree: the median
RMSE advantage across Tier 2's 36 series is 0.45% and the largest is 35.3%.
A showcase of the largest alone would be true and misleading, so the figure
draws three panels chosen by rule -- largest, median, and one Prophet wins --
and the caption states the rule and the typical case.

What can go stale here is the arithmetic in the caption, which is the part a
reader checks nothing against. These tests recompute every number in it from
the committed results, so a regenerated Tier 2 that moves the median makes
the README fail rather than quietly become wrong.

Not tested here: that the figure *looks* right. It is checked in, and the
script that draws it is deterministic -- `evaluation/showcase.py` run twice
produces byte-identical output, because both implementations are seeded.
"""
import csv
import re
import statistics
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
FIGURE = REPO / "evaluation" / "results" / "figures" / "showcase.png"
TIER2 = REPO / "evaluation" / "results" / "tier2_accuracy.csv"


@pytest.fixture(scope="module")
def readme():
    return README.read_text()


def _paired(rows, metric):
    by_series = {}
    for row in rows:
        if row["metric"] == metric:
            by_series.setdefault(row["series"], {})[row["implementation"]] = \
                float(row["value"])
    return by_series


@pytest.fixture(scope="module")
def advantages():
    """Relative RMSE advantage per series, from the committed Tier 2 results.

    Every scored series, unfiltered -- this is what the caption's "across all
    36 series" is about.
    """
    rows = list(csv.DictReader(open(TIER2)))
    out = {}
    for series, values in _paired(rows, "rmse").items():
        ours, theirs = values.get("analytic_prophet"), values.get("prophet")
        if ours is not None and theirs:
            out[series] = (theirs - ours) / theirs
    return out


@pytest.fixture(scope="module")
def fitted_advantages(advantages):
    """The same, restricted to the series a Prophet-shaped model fits at all.

    The panels are selected from these, so the test re-derives the filter
    rather than trusting the script to have applied it.
    """
    from showcase import FITS_AT_ALL

    rows = list(csv.DictReader(open(TIER2)))
    smape = _paired(rows, "smape")
    keep = {}
    for series, advantage in advantages.items():
        fit = smape.get(series, {})
        if {"analytic_prophet", "prophet"} <= set(fit) and \
                max(fit.values()) <= FITS_AT_ALL:
            keep[series] = advantage
    return keep


def test_the_figure_exists_and_the_readme_shows_it(readme):
    assert FIGURE.exists(), "the showcase figure is not committed"
    assert FIGURE.stat().st_size > 10_000, "suspiciously small for a 3-panel figure"
    assert "figures/showcase.png" in readme

    # above the fold: before the feature list, as the issue asks
    assert readme.index("figures/showcase.png") < readme.index("## What it does")


def test_the_caption_states_the_selection_rule(readme):
    """A showcase that does not say it was selected is the overclaim this
    issue exists to avoid."""
    caption = readme[readme.index("figures/showcase.png"):]
    caption = caption[:caption.index("## What it does")]
    # one line, because markdown wraps and a phrase can straddle a newline
    caption = " ".join(caption.split())

    assert "chosen by rule" in caption
    for rule in ("beats Prophet's by the most", "median", "Prophet beats us by the most"):
        assert rule in caption, f"the caption does not name the {rule!r} panel"
    assert "held out" in caption or "neither model saw" in caption
    # the filter is part of the rule, so it has to be stated too
    assert "sMAPE" in caption, "the caption does not say the ranking was filtered"


def test_the_caption_reports_the_median_advantage_correctly(readme, advantages):
    """The number that stops the top panel being read as the typical case."""
    median = statistics.median(advantages.values())
    quoted = re.search(r"median RMSE advantage is \*\*([\d.]+)%\*\*", readme)
    assert quoted, "the caption no longer quotes the median advantage"
    assert float(quoted.group(1)) == pytest.approx(median * 100, abs=0.05), (
        f"the caption says {quoted.group(1)}%, the committed results say "
        f"{median:.2%}")


def test_the_caption_reports_the_series_count_correctly(readme, advantages):
    # The README writes 3,008 for a reader; the count is 3008.
    quoted = re.search(r"all ([\d,]+) series the median", readme)
    assert quoted and int(quoted.group(1).replace(",", "")) == len(advantages), (
        f"the caption gives {quoted.group(1) if quoted else 'no count'}, the "
        f"results score {len(advantages)} series")


def test_the_caption_reports_coverage_as_the_report_does(readme):
    """The intervals are drawn, so the caption carries the under-coverage --
    and carries the canonical report's numbers rather than an older run's."""
    report = (REPO / "evaluation" / "results" / "report.md").read_text()
    # one line: markdown wraps, and this phrase straddles a newline
    quoted = re.findall(r"coverage \*\*([\d.]+)\*\* for ours and \*\*([\d.]+)\*\*",
                        " ".join(readme.split()))
    assert quoted, "the caption no longer quotes coverage, though the figure draws bands"

    for ours, theirs in quoted:
        assert ours in report and theirs in report, (
            f"the caption's coverage ({ours}, {theirs}) is not what the "
            f"generated report says")


def test_the_three_panels_are_what_the_rule_selects(fitted_advantages):
    """The script picks by rule from the committed results; this is that rule,
    written again, so a change to either has to be deliberate."""
    from showcase import select

    chosen = [series for _label, _rule, _advantage, series, _o, _t in select()]
    ranked = sorted(fitted_advantages, key=fitted_advantages.get, reverse=True)

    assert chosen[0] == ranked[0], "panel 1 is not the largest advantage"
    assert chosen[1] == ranked[len(ranked) // 2], "panel 2 is not the median"
    assert chosen[2] == ranked[-1], "panel 3 is not the one Prophet wins by most"


def test_the_panels_are_series_the_model_actually_fits(fitted_advantages):
    """The point of the filter. A panel where both implementations miss badly
    shows the difficulty of the series, not the difference between two
    optimizers -- which is what the first version of this figure did."""
    from showcase import FITS_AT_ALL, select

    rows = list(csv.DictReader(open(TIER2)))
    smape = _paired(rows, "smape")
    for _label, _rule, _advantage, series, _o, _t in select():
        for implementation, value in smape[series].items():
            assert value <= FITS_AT_ALL, (
                f"{series} is drawn but {implementation} scores {value:.1%} "
                f"sMAPE on it, above the {FITS_AT_ALL:.0%} the caption claims")


def test_the_caption_states_the_size_of_the_filtered_set(readme, fitted_advantages):
    quoted = re.search(r"\*\*(\d+) where a Prophet-shaped model fits", readme)
    assert quoted, "the caption no longer says how many series the ranking covers"
    assert int(quoted.group(1)) == len(fitted_advantages)


def test_the_selection_spans_a_real_range(fitted_advantages):
    """If the best and worst series ever converge, three panels stop saying
    anything and the figure should be rethought rather than redrawn."""
    assert max(fitted_advantages.values()) > 0.05, "no series shows a visible advantage"
    assert min(fitted_advantages.values()) < 0, (
        "no series where Prophet wins; panel 3 is a lie")
