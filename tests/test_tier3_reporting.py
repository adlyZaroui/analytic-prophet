"""
Issue #154: the speed claim in the README, against the results it comes from.

The README said "1.4–10× faster ... 10× at T = 50, falling to 1.4× by
T = 1000". Two things were wrong with that and only one of them was the range.

  * **The top end did not match the committed results.** `tier3_cost.csv` has
    0.170004 s against 0.020626 s at T = 50, which is 8.24×, not 10×. An
    overstatement in this project's own favour.
  * **It stopped at T = 1000, where memory is reported at T = 2905.** Given a
    stated trend of 10× → 1.4× a reader assumes it keeps falling, possibly
    through parity. It does not: the ratio is 1.48× at T = 2905, so T = 1000 is
    a trough rather than a waypoint on a decline, and the omission was costing
    the project a claim it had already measured.

Four README figures have now drifted from the committed results this way,
because the README is hand-written where `report.md` is generated. These tests
recompute from the CSV, so the drift fails the suite rather than waiting for a
reader to divide two numbers.

#212 then found the README's 8.2× at T = 50 beside the paper's 7.5× for the
same comparison -- two measurements, both committed, quoted in two places. The
README's cost table now quotes the paper's fit ratios
(`paper/results/overhead.csv`, the measurement behind its limitations) and
this tier's for prediction, memory and import, and each figure is checked
against the result it comes from.
"""
import csv
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
TIER3 = REPO / "evaluation" / "results" / "tier3_cost.csv"


@pytest.fixture(scope="module")
def readme():
    return " ".join(README.read_text().split())


OVERHEAD = REPO / "paper" / "results" / "overhead.csv"
REPORT = REPO / "evaluation" / "results" / "report.md"


def _cost_table():
    """`{row label: cell}` from the README's two-column cost table."""
    rows = {}
    for line in README.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] and set(cells[0]) != {"-"}:
            rows[cells[0]] = cells[1]
    return rows


@pytest.fixture(scope="module")
def paper_fit_ratios():
    """`{T: Prophet's fit / ours}` from the paper's overhead measurement."""
    with open(OVERHEAD, newline="") as handle:
        return {int(row["observations"]): float(row["prophet_fit"]) / float(row["ours_fit"])
                for row in csv.DictReader(handle)}


def _quoted_fit_ratios():
    quoted = {}
    for label, cell in _cost_table().items():
        size = re.fullmatch(r"fit, T = (\d+)", label)
        if size:
            quoted[int(size.group(1))] = float(re.search(r"([\d.]+)× faster", cell).group(1))
    return quoted


def test_every_fit_ratio_the_readme_quotes_is_the_papers(paper_fit_ratios):
    quoted = _quoted_fit_ratios()
    assert quoted, "the README's cost table quotes no fit ratio in the form checked here"
    for size, ratio in quoted.items():
        assert size in paper_fit_ratios, f"README quotes T = {size}, which the paper does not measure"
        assert ratio == pytest.approx(paper_fit_ratios[size], abs=0.05), (
            f"README says {ratio}× at T = {size}; overhead.csv gives "
            f"{paper_fit_ratios[size]:.2f}×")


def test_the_short_series_ratio_is_the_one_the_paper_prints():
    """The figure #212 found quoted two ways: the README's T = 50 row is the
    paper's \\LimNewtonSpeedup, the generated value, not a second measurement."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "quote_numbers", REPO / "paper" / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert f"{_quoted_fit_ratios()[50]}" == module.macros()["LimNewtonSpeedup"]


def test_the_table_reaches_the_longest_series_memory_is_reported_at(paper_fit_ratios):
    """#154: speed stopped at T = 1000 while memory was reported at T = 2905,
    which invited the assumption that it keeps falling, possibly through parity."""
    longest = max(paper_fit_ratios)
    table = _cost_table()
    assert longest in _quoted_fit_ratios(), f"the fit rows do not reach T = {longest}"
    assert any(label.startswith("memory") and f"T = {longest}" in label for label in table)


def _tier3(series, implementation, metric):
    return next(float(row["value"]) for row in csv.DictReader(open(TIER3))
                if row["series"] == series and row["implementation"] == implementation
                and row["metric"] == metric)


def test_the_prediction_and_memory_rows_are_the_measured_ones():
    series = "peyton_manning[:2905]"
    table = _cost_table()
    predict = next(cell for label, cell in table.items() if label.startswith("predict, T = 2905"))
    approximate, exact = (float(v) for v in re.findall(r"([\d.]+)×", predict))
    assert approximate == pytest.approx(
        _tier3(series, "prophet", "predict_wall") / _tier3(series, "compiled", "predict_wall"),
        abs=0.05)
    assert exact == pytest.approx(
        _tier3(series, "prophet(exact)", "predict_wall")
        / _tier3(series, "compiled(exact)", "predict_wall"), abs=0.05)

    memory = next(cell for label, cell in table.items() if label.startswith("memory"))
    assert "about two fifths" in memory
    ratio = (_tier3(series, "compiled", "fit_peak_rss_added")
             / _tier3(series, "prophet", "fit_peak_rss_added"))
    assert 0.35 < ratio < 0.45, f"the fit's memory ratio is {ratio:.2f}, not about two fifths"


def test_the_import_and_decomposition_figures_are_the_reports(readme):
    """Rounded from the generated report, which computes them."""
    report = " ".join(REPORT.read_text().split())
    importing = next(cell for label, cell in _cost_table().items() if label.startswith("importing"))
    ours, theirs = re.findall(r"\+([\d.]+)", importing)
    assert f"+{ours} MiB" in report and f"+{theirs} MiB" in report
    fixed = re.search(r"within ([\d.]+) MiB of each other", readme).group(1)
    slopes = re.search(r"differ about ([\d.]+)×", readme).group(1)
    reported_fixed = re.search(r"within ([\d.]+) MiB of each other", report).group(1)
    assert float(fixed) == pytest.approx(float(reported_fixed), abs=0.05)
    assert f"differ {slopes}×" in report
