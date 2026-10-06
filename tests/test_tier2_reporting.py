"""
Issue #124: how the Tier 2 result is reported.

The table is the project's headline evidence, and four things about the way it
was presented would have been raised by anyone reviewing it: nine p-values
with no multiplicity adjustment and the test unnamed, three rows pooling raw
units across series of wildly different scales, two coverage numbers on the
same page that looked inconsistent, and a corpus size stated but never framed
as a limitation.

None of it changed a number. All of it changed what a reader can conclude
from them, which is the same thing as changing the claim.
"""
import csv
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
REPORT = REPO / "evaluation" / "results" / "report.md"
TIER2 = REPO / "evaluation" / "results" / "tier2_accuracy.csv"


@pytest.fixture(scope="module")
def report():
    return " ".join(REPORT.read_text().split())


@pytest.fixture(scope="module")
def paired():
    return {row["metric"]: float(row["value"])
            for row in csv.DictReader(open(TIER2)) if row["series"] == "paired"}


# -- the test, and testing nine things at once -----------------------------

def test_the_test_is_named_where_its_p_values_are(report):
    """Tier 1 named it; Tier 2 printed nine p-values and left the reader to
    infer the test from the word "median"."""
    assert "Wilcoxon" in report
    assert "scipy.stats.wilcoxon" in report


def test_the_adjusted_p_values_are_holm_and_correct(paired):
    """Recomputed here rather than trusted: the report's arithmetic is the
    claim, so a second implementation of it is the check."""
    from report import TIER2_COMPARISONS, holm

    raw = {metric: paired[f"{metric}_p_value"] for metric in TIER2_COMPARISONS}
    adjusted = holm(raw)

    # Holm by hand: sort ascending, scale by (m - i), keep a running maximum
    ordered = sorted(raw.items(), key=lambda item: item[1])
    expected, running = {}, 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        expected[name] = running

    assert adjusted == pytest.approx(expected)
    # and it must never report a smaller p than the raw one
    for metric, value in raw.items():
        assert adjusted[metric] >= value


def test_holm_is_at_least_as_powerful_as_bonferroni(paired):
    """The reason for choosing it. If this ever fails, the implementation is
    wrong -- Holm dominates Bonferroni by construction."""
    from report import TIER2_COMPARISONS, holm

    raw = {metric: paired[f"{metric}_p_value"] for metric in TIER2_COMPARISONS}
    adjusted = holm(raw)
    for metric, value in raw.items():
        assert adjusted[metric] <= min(1.0, len(raw) * value) + 1e-12


def test_the_report_states_how_many_metrics_survive(report, paired):
    """And states the right number: recomputed from the committed results, so
    a regenerated tier that moves a p-value past 0.05 fails the report rather
    than quietly contradicting it."""
    from report import TIER2_COMPARISONS, holm

    adjusted = holm({m: paired[f"{m}_p_value"] for m in TIER2_COMPARISONS})
    surviving = sum(1 for m in ("mae", "rmse", "mape", "smape", "coverage")
                    if adjusted[m] < 0.05)

    quoted = re.search(r"\*\*(\d+) of the 5 comparative metrics survive", report)
    assert quoted, "the report no longer says how many survive adjustment"
    assert int(quoted.group(1)) == surviving


def test_the_sparsity_rows_are_excluded_from_the_family_and_it_says_so(report):
    """Counting descriptive readouts in a multiple-comparison family makes the
    correction harsher on the strength of tests nobody is using to claim
    anything. That is a choice, so it is stated."""
    assert "sparsity" in report
    assert "including the three sparsity rows in the family changes none of it" in report


# -- units -----------------------------------------------------------------

def test_the_raw_unit_rows_are_marked(report):
    """`mae = -1.9142` is a median across series whose levels differ by orders
    of magnitude. It ranks direction; it is not an effect size."""
    assert "⁑" in report, "the raw-unit rows carry no marker"
    assert "in the series' own units" in report
    assert "not* a pooled effect size" in report or "not a pooled effect size" in report


def test_the_reader_is_pointed_at_the_scale_free_rows(report):
    assert "mape" in report and "smape" in report
    assert "Read magnitude from the scale-free rows" in report


# -- the two coverage numbers ----------------------------------------------

def test_the_two_coverage_figures_are_reconciled(report):
    """+0.0026 in the table and a 0.015 gap in the prose, six times apart, on
    the same page with nothing joining them."""
    assert "different quantities" in report
    assert "median of paired differences is not the difference of means" in report


def test_the_skew_claim_is_the_one_the_data_supports():
    """The explanation says the per-series differences are skewed. That is a
    claim about the data and is checked against it, because a plausible
    explanation that happens to be wrong is worse than none.
    """
    by_series = {}
    for row in csv.DictReader(open(TIER2)):
        if row["metric"] == "coverage" and row["series"] != "paired":
            by_series.setdefault(row["series"], {})[row["implementation"]] = \
                float(row["value"])
    differences = sorted(pair["analytic_prophet"] - pair["prophet"]
                         for pair in by_series.values() if len(pair) == 2)

    near = sum(1 for d in differences if abs(d) < 0.02)
    above = sum(1 for d in differences if d > 0.05)
    below = sum(1 for d in differences if d < -0.05)

    text = " ".join(REPORT.read_text().split())
    assert f"**{near} of the {len(differences)} series differ by less than 0.02**" in text
    assert f"{above} differ by more than +0.05" in text
    assert below == 0, (
        "the report says none differs by more than 0.05 the other way, and now "
        f"{below} do")


# -- the corpus size -------------------------------------------------------

def test_the_readme_frames_the_corpus_size_as_a_limitation():
    """36 of M4's 100,000, and 11 for the showcase ranking. Both were stated
    where they are used and neither was framed as a limit on what the result
    means."""
    readme = " ".join((REPO / "README.md").read_text().split())
    section = readme[readme.index("## What this is not"):]
    assert "36 series" in section
    assert "100,000" in section
    assert "**11**" in section, "the showcase's effective sample is not mentioned"
