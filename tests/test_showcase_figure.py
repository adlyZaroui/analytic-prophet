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


@pytest.fixture(scope="module")
def advantages():
    """Relative RMSE advantage per series, from the committed Tier 2 results."""
    rows = list(csv.DictReader(open(TIER2)))
    by_series = {}
    for row in rows:
        if row["metric"] == "rmse":
            by_series.setdefault(row["series"], {})[row["implementation"]] = \
                float(row["value"])
    out = {}
    for series, values in by_series.items():
        ours, theirs = values.get("analytic_prophet"), values.get("prophet")
        if ours is not None and theirs:
            out[series] = (theirs - ours) / theirs
    return out


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


def test_the_caption_reports_the_median_advantage_correctly(readme, advantages):
    """The number that stops the top panel being read as the typical case."""
    median = statistics.median(advantages.values())
    quoted = re.search(r"median RMSE advantage is \*\*([\d.]+)%\*\*", readme)
    assert quoted, "the caption no longer quotes the median advantage"
    assert float(quoted.group(1)) == pytest.approx(median * 100, abs=0.05), (
        f"the caption says {quoted.group(1)}%, the committed results say "
        f"{median:.2%}")


def test_the_caption_reports_the_series_count_correctly(readme, advantages):
    quoted = re.search(r"all (\d+) series the median", readme)
    assert quoted and int(quoted.group(1)) == len(advantages)


def test_the_caption_reports_coverage_as_the_report_does(readme):
    """The intervals are drawn, so the caption carries the under-coverage --
    and carries the canonical report's numbers rather than an older run's."""
    report = (REPO / "evaluation" / "results" / "report.md").read_text()
    quoted = re.findall(r"coverage \*\*([\d.]+)\*\* for ours and \*\*([\d.]+)\*\*",
                        readme)
    assert quoted, "the caption no longer quotes coverage, though the figure draws bands"

    for ours, theirs in quoted:
        assert ours in report and theirs in report, (
            f"the caption's coverage ({ours}, {theirs}) is not what the "
            f"generated report says")


def test_the_three_panels_are_what_the_rule_selects(advantages):
    """The script picks by rule from the committed results; this is that rule,
    written again, so a change to either has to be deliberate."""
    from showcase import select

    chosen = [series for _label, _rule, _advantage, series, _o, _t in select()]
    ranked = sorted(advantages, key=advantages.get, reverse=True)

    assert chosen[0] == ranked[0], "panel 1 is not the largest advantage"
    assert chosen[1] == ranked[len(ranked) // 2], "panel 2 is not the median"
    assert chosen[2] == ranked[-1], "panel 3 is not the one Prophet wins by most"


def test_the_selection_spans_a_real_range(advantages):
    """If the best and worst series ever converge, three panels stop saying
    anything and the figure should be rethought rather than redrawn."""
    assert max(advantages.values()) > 0.05, "no series shows a visible advantage"
    assert min(advantages.values()) < 0, "no series where Prophet wins; panel 3 is a lie"
