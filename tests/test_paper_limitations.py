"""
Issue #178: the limitations section, every item with its number and its source.

`paper/sections/limitations.tex` quotes nothing by hand: each figure is a
macro generated from committed results (tests/test_paper_numbers.py checks
they are current). What is checked here is the rest of the issue's acceptance:

  * **no item is softer than the README already says** -- where both quote a
    figure they quote the same one, and where the paper adds one it is the
    harsher (the census-wide worst case beside the showcase pool's);
  * **the sentences that are typed, not generated, are true of the data** --
    "ours is the longer" tail, "about a third", "the cost of calling CmdStan",
    "the rest is Stan's Newton", M4 recording "no location";
  * **no number is typed into the section** beyond the fixed ones it names.
"""
import csv
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
SECTION = PAPER / "sections" / "limitations.tex"
RESULTS = PAPER / "results"
README = REPO / "README.md"


@pytest.fixture(scope="module")
def macros():
    spec = importlib.util.spec_from_file_location(
        "quote_numbers", PAPER / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.macros()


def test_no_number_is_typed_into_the_section():
    """Every figure is a macro. The only literals allowed are the ones the
    section names rather than measures: the nominal 80%, Prophet's T = 100
    rule, and the two lengths that bound Tier 0's range."""
    text = re.sub(r"\\(?:prophet|project)issue\{\d+\}", " ", SECTION.read_text())
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    literals = set(re.findall(r"(?<![\w.])\d[\d,.]*", text))
    assert literals <= {"80", "100", "300", "2905"}, sorted(literals)


def test_where_the_readme_quotes_a_figure_the_paper_quotes_the_same(macros):
    """#212 extended this to every paper figure the README carries -- it had
    drifted on two, 8.2x at T = 50 against the paper's 7.5x and 0.0967% against
    0.0968%. tests/test_readme_structure.py checks each in its sentence, and
    that the README carries no number without a source."""
    readme = " ".join(README.read_text().split())
    for name in ("LimCoverageOurs", "LimCoverageProphet", "LimWorstPool", "LimPoolSize",
                 "LimWidthMedian", "LimWidthWider", "LimWorstCensus", "LimBestCensus",
                 "LimMfourSeries", "LimNewtonSpeedup", "LimOverheadShareMin",
                 "LimOverheadShareMax", "LimParityMin", "LimParityMax", "CensusSeries",
                 "RmseAdvantage", "RmseWins", "RmseWinRate", "MultiTotal",
                 "SmoothFactorLong", "SmoothFactorShort"):
        assert macros[name] in readme, f"README does not quote {name} = {macros[name]}"


def test_the_paper_states_the_census_wide_worst_case_not_only_the_pools(macros):
    """The README's 78.7% is the worst within the showcase pool. Across the
    census it is far worse, and the paper must not be the softer of the two."""
    assert float(macros["LimWorstCensus"]) > float(macros["LimWorstPool"])
    text = SECTION.read_text()
    assert r"\LimWorstCensus" in text and r"\LimWorstPool" in text


def test_ours_is_the_longer_tail(macros):
    assert float(macros["LimWorstCensus"]) > float(macros["LimBestCensus"])
    assert "ours is the longer" in SECTION.read_text()


def test_about_a_third_for_both(macros):
    for name in ("LimCoverageOurs", "LimCoverageProphet"):
        assert 0.28 < float(macros[name]) < 0.40, name
    assert "about a third, for both" in re.sub(r"\s+", " ", SECTION.read_text())


def _overhead():
    with open(RESULTS / "overhead.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def test_above_t_100_the_advantage_is_the_cost_of_calling_cmdstan():
    """'without it Prophet's fit takes between 0.9 and 1.1 times as long as this
    implementation's: the measured advantage there is the cost of calling CmdStan.'
    Held to the committed timings: once the one-iteration call is taken off,
    Prophet is within a quarter of this implementation at every L-BFGS length,
    and the call is a substantial share of its fit."""
    rows = [row for row in _overhead() if row["prophet_algorithm"] == "LBFGS"]
    assert {int(row["observations"]) for row in rows} == {100, 300, 1000, 2905}
    for row in rows:
        fit, call, ours = (float(row[key]) for key in
                           ("prophet_fit", "one_iteration_call", "ours_fit"))
        assert 0.8 < (fit - call) / ours < 1.25, row
        assert call / fit > 0.15, row


def test_below_t_100_the_rest_is_stans_newton():
    """At T = 50 the call is a small share of Prophet's fit, and taking it off
    leaves this implementation still several times faster."""
    row = next(row for row in _overhead() if row["observations"] == "50")
    fit, call, ours = (float(row[key]) for key in ("prophet_fit", "one_iteration_call",
                                                   "ours_fit"))
    assert row["prophet_algorithm"] == "Newton"
    assert call / fit < 0.25
    assert (fit - call) / ours > 3


def test_m4_records_no_location_and_the_census_is_two_frequencies():
    with open(RESULTS / "m4_calendar.csv", newline="") as handle:
        rows = {row["frequency"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"}
    assert sum(int(row["m4_series"]) for row in rows.values()) == 100_000
    columns = set(next(iter(rows.values()))["metadata_columns"].split("+"))
    assert columns == {"M4id", "category", "Frequency", "Horizon", "SP", "StartingDate"}
    measured = {name for name, row in rows.items() if int(row["census_series"]) > 0}
    assert measured == {"Weekly", "Daily"}


def test_the_dates_are_not_called_synthetic():
    """M4's organizers say the starting dates were withheld, not invented
    (paper/citations.md, m4methods); 'synthetic' was never supported."""
    assert "synthetic" not in SECTION.read_text()
