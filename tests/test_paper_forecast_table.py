"""
Issue #243: Appendix A reported one of its six comparisons.

The limitations section counts "all six of the six comparisons of Appendix A"
surviving Holm adjustment, and the appendix reported RMSE alone. The six are
`evaluation/report.py`'s TIER2_COMPARISONS: MAE, RMSE, MAPE, sMAPE, coverage
and interval width. They are one generated table now, `paper/generated/
forecast.tex`, written by `quote_numbers.py` the way Table 1 is (#209), with a
macro in every cell. Checked here:

  * **the table** is what the generator writes, every cell is a generated
    macro, and the appendix includes it and the limitations point to it;
  * **its figures are the committed results**: the Tier 2 paired summary, and
    report.py's own Holm adjustment;
  * **the figures agree** with the ones already quoted: the README's five
    rows, and the macros the paper quoted before the table existed.

The README check lives here rather than in test_readme_structure.py: the
README's forecast table is this issue's, and that file's pitch test is #248's.
"""
import csv
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
SECTIONS = PAPER / "sections"
TABLE = PAPER / "generated" / "forecast.tex"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "quote_numbers", PAPER / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def macros(generator):
    return generator.macros()


@pytest.fixture(scope="module")
def paired():
    rows = csv.DictReader(open(REPO / "evaluation" / "results" / "tier2_accuracy.csv"))
    return {row["metric"]: float(row["value"]) for row in rows
            if row["series"] == "paired" and row["configuration"] == "all"}


# -- the table ----------------------------------------------------------------

def test_the_committed_table_is_what_the_generator_writes(generator):
    assert TABLE.read_text() == generator.render_forecast_table(), (
        "paper/generated/forecast.tex is stale -- run `python paper/experiments/quote_numbers.py`")


def test_every_cell_is_a_generated_macro(macros):
    table = TABLE.read_text()
    named = set(re.findall(r"\\(Forecast[A-Za-z]+)", table))
    assert len(named) == 6 * 5
    assert named <= set(macros), f"not generated: {sorted(named - set(macros))}"
    for line in table.splitlines():
        if "&" in line and not line.lstrip().startswith("&"):
            for cell in line.rstrip("\\ ").split("&")[1:]:
                assert not re.search(r"\d", re.sub(r"\\[A-Za-z]+", "", cell)), (
                    f"a typed number in a cell: {cell!r}")


def test_one_row_per_comparison_the_report_makes(generator):
    for path in (REPO, REPO / "evaluation"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import report

    assert [metric for metric, _ in generator.FORECAST] == list(report.TIER2_COMPARISONS)
    rows = [line for line in TABLE.read_text().splitlines() if "\\Forecast" in line]
    assert [line.split(" & ")[0] for line in rows] == [
        "MAE", "RMSE", "MAPE", "sMAPE", "coverage", "interval width"]


def test_the_appendix_includes_it_and_the_limitations_point_to_it():
    appendix = (SECTIONS / "appendix_forecast.tex").read_text()
    float_ = appendix[appendix.index(r"\begin{table}"):appendix.index(r"\end{table}")]
    assert r"\input{generated/forecast}" in float_ and r"\label{tab:forecast}" in float_
    assert r"\Cref{tab:forecast} gives" in appendix
    limitations = " ".join((SECTIONS / "limitations.tex").read_text().split())
    assert (r"all \LimHolmSurvivors{} of the \LimHolmComparisons{} comparisons of "
            r"\cref{tab:forecast} survive Holm adjustment") in limitations


# -- its figures are the committed results ------------------------------------

def test_the_figures_are_the_paired_summary(generator, macros, paired):
    report, _ = generator._evaluation()
    adjusted = report.holm({metric: paired[f"{metric}_p_value"]
                            for metric in report.TIER2_COMPARISONS})
    for metric, tag in generator.FORECAST:
        median = paired[f"{metric}_median"]
        if metric == "coverage":
            assert float(macros[f"Forecast{tag}Median"]) == pytest.approx(median, abs=5e-5)
        else:
            assert float(macros[f"Forecast{tag}Median"]) == pytest.approx(median * 100, abs=5e-3)
        counts = [int(macros[f"Forecast{tag}{side}"].replace(",", ""))
                  for side in ("Lower", "Higher", "Tied")]
        assert counts == [paired[f"{metric}_wins"], paired[f"{metric}_losses"],
                          paired[f"{metric}_ties"]]
        assert sum(counts) == paired[f"{metric}_n"]
        assert macros[f"Forecast{tag}Holm"] == generator._scientific(adjusted[metric])


def test_it_agrees_with_what_the_paper_already_quoted(macros):
    assert macros["ForecastRmseLower"] == macros["RmseWins"]
    assert macros["ForecastRmseMedian"] == f"-{macros['RmseAdvantage']}"
    assert macros["ForecastWidthHigher"] == macros["LimWidthWider"]
    assert macros["ForecastWidthMedian"] == f"+{macros['LimWidthMedian']}"
    holm = {macros[f"Forecast{tag}Holm"] for tag in
            ("Mae", "Rmse", "Mape", "Smape", "Coverage", "Width")}
    assert {macros["LimHolmMin"], macros["LimHolmMax"]} <= holm


# -- the README's five rows are the table's figures ---------------------------

README_ROWS = {"RMSE": "Rmse", "MAE": "Mae", "MAPE": "Mape", "sMAPE": "Smape",
               "interval width": "Width"}


def test_the_readmes_five_rows_are_the_tables_figures(macros):
    readme = (REPO / "README.md").read_text()
    rows = dict(re.findall(r"^\| (RMSE|MAE|MAPE|sMAPE|interval width) \| (.+) \|$",
                           readme, flags=re.M))
    assert set(rows) == set(README_ROWS)
    total = macros["CensusSeries"].replace(",", "")
    for label, tag in README_ROWS.items():
        median, lower = [cell.strip() for cell in rows[label].split("|")]
        assert median == macros[f"Forecast{tag}Median"].replace("-", "−") + "%", label
        if tag == "Width":
            assert lower == "—"
        else:
            assert lower == f"{macros[f'Forecast{tag}Lower'].replace(',', '')}/{total}", label
