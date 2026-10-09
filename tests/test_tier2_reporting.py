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
    surviving = sum(1 for m in TIER2_COMPARISONS if adjusted[m] < 0.05)

    quoted = re.search(r"\*\*All (\d+) of the (\d+) survive adjustment\*\*", report)
    assert quoted, "the report no longer says how many survive adjustment"
    assert int(quoted.group(1)) == surviving
    assert int(quoted.group(2)) == len(TIER2_COMPARISONS), (
        "the family is the six comparisons; interval_width used not to survive "
        "and was counted separately, which stopped being true at census scale")


def test_the_sparsity_rows_are_excluded_from_the_family_and_it_says_so(report):
    """Counting descriptive readouts in a multiple-comparison family makes the
    correction harsher on the strength of tests nobody is using to claim
    anything. That is a choice, so it is stated."""
    assert "sparsity" in report
    assert ("including the three sparsity rows in the family changes none of it"
            in report.lower()), (
        "the report no longer says that adjusting over the descriptive rows "
        "would change nothing")


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
    # This used to assert `below == 0` -- true of the 36-series sample and a
    # property of that data, not of the report. At census scale 87 series fall
    # the other way, so what has to hold is that the report says so.
    assert f"{below} by more than 0.05 the other way" in text, (
        f"{below} series differ by more than 0.05 the other way and the report "
        "does not say it")


# -- the corpus size -------------------------------------------------------

def _plain(text):
    """The README with thousands separators removed.

    It writes `3,008` because a reader reads that; the results say `3008`. A
    guard that compares them has to agree on which, and stripping is the
    smaller intervention than making the prose unreadable.
    """
    return re.sub(r"(?<=\d),(?=\d\d\d)", "", text)


def test_the_readme_frames_the_corpus_size_as_a_limitation():
    """The corpus size and the showcase's narrower ranking pool both belong
    where the limits are stated, whatever those numbers currently are.

    They were 36 and 11; the census made them 3008 and 658. The test checks
    the README against the committed results rather than against either pair.
    """
    import csv as _csv

    rows = list(_csv.DictReader(open(TIER2)))
    measured = {r["series"] for r in rows if r["series"] not in ("paired", "corpus")}
    smape = {}
    for row in rows:
        if row["metric"] == "smape" and row["series"] not in ("paired", "corpus"):
            smape.setdefault(row["series"], {})[row["implementation"]] = float(row["value"])
    fits = sum(1 for v in smape.values() if len(v) == 2 and max(v.values()) < 0.10)

    readme = _plain(" ".join((REPO / "README.md").read_text().split()))
    section = readme[readme.index("## What this is not"):]
    assert f"{len(measured)} series" in section, (
        f"the limits section does not state the corpus size ({len(measured)})")
    assert "100,000" in _plain(" ".join((REPO / "README.md").read_text().split())) \
        or "100000" in section
    assert f"**{fits}**" in section, (
        f"the showcase's ranking pool ({fits} series) is not mentioned")


# -- the same table in the README (#148) -----------------------------------
#
# #124 fixed all of this in report.md and none of it reached the README, which
# carries the same five numbers. The two pages then disagreed about how to read
# them: the report named the test, marked the raw-unit rows, showed Holm and
# reconciled the two coverage figures, and the README did none of it. These
# recompute the README's table from the committed results, so it cannot drift
# from the report again -- which it has now done twice.

README = REPO / "README.md"


@pytest.fixture(scope="module")
def readme():
    return " ".join(README.read_text().split())


@pytest.fixture(scope="module")
def readme_table():
    """`{label: (median, p, holm)}` parsed out of the README's accuracy table."""
    rows = {}
    for line in README.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("", "---"):
            continue
        label = cells[0].replace("⁑", "").replace("*", "").strip()
        try:
            rows[label] = (float(cells[1].replace("−", "-").replace("*", "")),
                           cells[2], float(cells[3]), float(cells[4]))
        except ValueError:
            continue
    return rows


README_TO_METRIC = {"MAE": "mae", "RMSE": "rmse", "MAPE": "mape", "sMAPE": "smape",
                    "coverage": "coverage", "interval width": "interval_width"}


def test_the_readme_names_the_test_and_the_adjustment(readme):
    assert "Wilcoxon signed-rank" in readme
    assert "scipy.stats.wilcoxon" in readme
    assert "Holm" in readme, "the adjustment is claimed but not named"


def test_every_number_in_the_readme_table_is_the_committed_one(readme_table, paired):
    from report import TIER2_COMPARISONS, holm

    adjusted = holm({m: paired[f"{m}_p_value"] for m in TIER2_COMPARISONS})
    assert set(README_TO_METRIC) <= set(readme_table), (
        f"the README table no longer has every comparison: {sorted(readme_table)}")

    for label, metric in README_TO_METRIC.items():
        median, _wins, raw, adj = readme_table[label]
        assert median == pytest.approx(paired[f"{metric}_median"], abs=5e-4), label
        assert raw == pytest.approx(paired[f"{metric}_p_value"], abs=5e-5), label
        assert adj == pytest.approx(adjusted[metric], abs=5e-5), (
            f"{label}: README says Holm {adj}, the results give {adjusted[metric]:.4f}")


def test_the_readme_quotes_the_win_counts_it_can_check(readme_table, paired):
    """The scale-free summary the raw medians are not.

    Checked in each metric's own row, not as a substring of the file. `MAE` and
    `MAPE` both read 26/36, so "is 26/36 somewhere in the README" passes when
    one of them has been altered to something else -- which is what the first
    version of this did.
    """
    for label, metric in (("MAE", "mae"), ("RMSE", "rmse"), ("MAPE", "mape"),
                          ("sMAPE", "smape")):
        wins, n = int(paired[f"{metric}_wins"]), int(paired[f"{metric}_n"])
        assert readme_table[label][1] == f"{wins}/{n}", (
            f"README says {label} is lower on {readme_table[label][1]}; "
            f"the results say {wins}/{n}")


def test_the_readme_marks_its_raw_unit_rows(readme):
    """MAE, RMSE and interval width are in the series' own units, and the 36
    series differ in level by orders of magnitude."""
    assert readme.count("⁑") >= 4, "the raw-unit marker and its footnote"
    assert "in the series' own units" in readme
    assert "not* pooled effect sizes" in readme or "not pooled effect sizes" in readme


def test_the_readme_reconciles_its_two_coverage_figures(readme):
    """The median and the difference of means are different quantities and sat
    on one page unjoined. What the counts are is checked below; this checks the
    sentence that joins them exists."""
    assert "median of the per-series" in readme
    assert "difference of means" in readme


def test_the_readme_skew_claim_is_the_one_the_data_supports():
    """Recomputed, not copied from the report -- the point of this issue is
    that copying is how the README went stale."""
    import statistics

    by_series = {}
    for row in csv.DictReader(open(TIER2)):
        if row["metric"] == "coverage" and row["series"] != "paired":
            by_series.setdefault(row["series"], {})[row["implementation"]] = float(row["value"])
    diffs = [v["analytic_prophet"] - v["prophet"]
             for v in by_series.values() if len(v) == 2]

    near = sum(1 for d in diffs if abs(d) < 0.02)
    above = sum(1 for d in diffs if d > 0.05)
    below = sum(1 for d in diffs if d < -0.05)

    readme = _plain(" ".join(README.read_text().split()))
    assert f"{near} of the {len(diffs)}" in readme
    assert f"{above} by more than +0.05" in readme, (
        f"{above} series differ by more than +0.05 and the README does not say it")
    # `== 2` stood here, which was the 36-series data rather than a property of
    # the README. 87 series now fall the other way and the README has to say so.
    assert f"{below} by more than 0.05 the other way" in readme, (
        f"{below} series differ by more than 0.05 the other way, unstated")
    # `assert not [...]` stood here: the 36-series sample had no members in
    # that tail and this encoded it. 87 series do at census scale, and what
    # must hold is that the README states the count, which it does above.
    # The median itself was pinned to 0.0026 -- the 36-series value. What the
    # README has to do is quote whatever it currently is, to four places.
    assert f"{statistics.median(diffs):+.4f}" in readme, (
        f"the README does not quote the median coverage difference "
        f"({statistics.median(diffs):+.4f})")


# -- the figure the lede quotes (#153) -------------------------------------

def _relative_rmse_differences():
    """Per-series `(ours - prophet) / prophet` for RMSE, from committed results.

    Scale-free, which is why this is the figure the lede quotes: #148
    established that the raw medians rank direction and are not poolable across
    36 series differing in level by orders of magnitude.
    """
    by_series = {}
    for row in csv.DictReader(open(TIER2)):
        if row["metric"] == "rmse" and row["series"] != "paired":
            by_series.setdefault(row["series"], {})[row["implementation"]] = float(row["value"])
    return [(v["analytic_prophet"] - v["prophet"]) / v["prophet"]
            for v in by_series.values() if len(v) == 2]


def test_the_lede_carries_the_magnitude_and_the_scope(readme):
    """[#153] "on held-out M4 series our forecasts are more accurate" stood 69
    lines above the number that sizes it and 168 above the table. A reader who
    stopped at the lede took away "more accurate" with no sense that it is
    half a percent at the median."""
    import statistics

    relative = _relative_rmse_differences()
    advantage = -statistics.median(relative) * 100
    wins = sum(1 for d in relative if d < 0)

    lede = _plain(readme[:readme.index("**Status: early development.**")])

    assert f"{advantage:.2f}%" in lede, (
        f"the lede does not state the magnitude ({advantage:.2f}%)")
    assert f"{len(relative)} held-out M4" in lede, (
        f"the lede does not state the scope ({len(relative)} series)")
    assert str(wins) in lede, (
        f"the lede does not state how many series it wins on ({wins})")


def test_every_place_quoting_the_rmse_advantage_agrees_with_the_results(readme):
    """The lede and the showcase paragraph both quote it, so both are checked
    against the CSV rather than against each other."""
    import statistics

    relative = _relative_rmse_differences()
    advantage = -statistics.median(relative) * 100
    wins = sum(1 for d in relative if d < 0)

    quoted = f"{advantage:.2f}%"
    assert readme.count(quoted) >= 2, (
        f"the RMSE advantage ({quoted}) is quoted in fewer than the two places "
        "expected -- the lede and the showcase paragraph")
    assert _plain(readme).count(str(wins)) >= 1, (
        f"the README does not state the win count ({wins} of {len(relative)})")
