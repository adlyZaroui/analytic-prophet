"""
Issue #206: the forecast comparison, in an appendix a reader can check.

Section 6 quotes the census result in one paragraph; the protocol behind it was
in the code and nowhere in the paper. `paper/sections/appendix_forecast.tex`
states it: the census rule and what it excludes, the protocol, and the test.
Every count is a macro (tests/test_paper_numbers.py keeps them current). What
is checked here is that the rest is true of what ran:

  * **the exclusions add up to the census** -- each stage counted with the
    census's own code (`paper/experiments/m4_calendar.py`), so the counts
    cannot describe a rule other than the one that froze the manifest;
  * **the sentences the counts do not carry** -- "every other one yields a
    cutoff", "no series is lost to its starting date", a cutoff needing the
    initial window and the horizon -- hold, and would fail here if a rerun
    made them false;
  * **the protocol stated is the protocol that ran** -- one set of horizons
    across the manifest, the census and Tier 2, the seed, the rolling window,
    the six comparisons.
"""
import csv
import json
import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
APPENDIX = PAPER / "sections" / "appendix_forecast.tex"
CALENDAR = PAPER / "results" / "m4_calendar.csv"
MANIFEST = REPO / "evaluation" / "corpus" / "m4_census_v1.json"
CENSUS = ("Weekly", "Daily")
STAGES = ("below_min_length", "no_start_date", "without_cutoff")


def _calendar():
    with open(CALENDAR, newline="") as handle:
        return {row["frequency"]: row for row in csv.DictReader(handle)}


def _manifest():
    return json.loads(MANIFEST.read_text())


def _text():
    return re.sub(r"\s+", " ", APPENDIX.read_text())


@pytest.mark.parametrize("frequency", CENSUS)
def test_the_exclusions_add_up_to_the_frozen_census(frequency):
    row = _calendar()[frequency]
    kept = int(row["m4_series"]) - sum(int(row[stage]) for stage in STAGES)
    assert kept == int(row["census_series"]) == _manifest()["counts"][frequency]


def test_the_sentences_the_counts_do_not_carry_are_true():
    """The appendix says every weekly series that clears the floor yields a
    cutoff, and that no series is lost to its starting date. Both are typed;
    a rerun that made either false would fail here, not in the reader's hands."""
    calendar = _calendar()
    text = _text()
    assert "every other one yields a cutoff" in text
    assert int(calendar["Weekly"]["without_cutoff"]) == 0
    assert "No series is lost to its starting date" in text
    assert all(int(calendar[frequency]["no_start_date"]) == 0 for frequency in CENSUS)
    # the daily cutoff stage is the one that excludes, and the text names it
    assert int(calendar["Daily"]["without_cutoff"]) > 0
    assert "\\CensusDailyNoCutoff" in text


def test_a_cutoff_needs_the_initial_window_and_the_horizon(prophet_comparison):
    """'... to yield a cutoff, which needs \\CensusDailySpan days, the initial
    window and the horizon': Prophet's `generate_cutoffs` keeps its first
    cutoff only when the history spans both. Checked at the boundary."""
    import corpora
    from prophet.diagnostics import generate_cutoffs

    horizon, initial, period = (pd.Timedelta(text)
                                for text in corpora.CENSUS_HORIZONS["Daily"])
    span = (initial + horizon).days

    def cutoffs(days):
        frame = pd.DataFrame({"ds": pd.date_range("2000-01-01", periods=days + 1),
                              "y": 0.0})
        return generate_cutoffs(frame, horizon, initial, period)

    assert len(cutoffs(span)) == 1
    with pytest.raises(ValueError):
        cutoffs(span - 1)


def test_the_protocol_stated_is_the_one_that_ran():
    import corpora
    import harness
    import report
    from tiers import tier2

    manifest = _manifest()
    for frequency in CENSUS:
        assert (tuple(manifest["horizons"][frequency]) == corpora.CENSUS_HORIZONS[frequency]
                == tier2.HORIZONS[frequency])
    assert set(tier2.HORIZONS) == set(CENSUS)
    assert manifest["min_length"] == corpora.CENSUS_MIN_LENGTH
    assert manifest["fit_seed"] == harness.SEED
    # "scored by Prophet's own performance_metrics with a rolling window of one"
    assert "performance_metrics(cv_frame, rolling_window=1.0)" in (
        REPO / "evaluation" / "tiers" / "tier2.py").read_text()
    # "MAE, RMSE, MAPE, sMAPE and the coverage ..., beside which the mean width"
    assert report.TIER2_COMPARISONS == ("mae", "rmse", "mape", "smape", "coverage",
                                        "interval_width")


def test_the_appendix_describes_the_test_that_ran():
    """#200 replaced the Wilcoxon test with the sign test on relative
    differences; the appendix must not describe the old one."""
    text = _text()
    for phrase in ("sign test", "exact and two-sided", "Holm's procedure",
                   "this implementation's value over Prophet's minus one",
                   "rolling window of one", "\\LimHolmComparisons{} comparisons",
                   "\\LimHolmSurvivors{} of them survive"):
        assert phrase in text, phrase
    for stale in ("Wilcoxon", "t-test", "signed-rank"):
        assert stale not in text, stale


def test_the_appendix_is_in_the_paper_and_section_6_points_to_it():
    main = (PAPER / "main.tex").read_text()
    assert main.index("\\bibliography{references}") < main.index("\\appendix") \
        < main.index("\\input{sections/appendix_forecast}") < main.index("\\end{document}")
    assert "\\label{app:forecast}" in APPENDIX.read_text()
    consequences = re.sub(r"\s+", " ", (PAPER / "sections" / "consequences.tex").read_text())
    assert "\\Cref{app:forecast} gives the census rule, what it excludes, the protocol and the test" \
        in consequences


def test_the_census_rule_in_the_text_is_the_manifests():
    """The manifest's own statement of the rule, and the appendix's, name the
    same three conditions."""
    rule = _manifest()["rule"]
    text = _text()
    assert "at least 120 observations" in rule and "\\CensusMinLength{} observations" in text
    assert "parseable StartingDate" in rule and "a starting date that loads" in text
    assert "at least one rolling-origin cutoff" in rule
    assert "at least one rolling-origin cutoff" in text
