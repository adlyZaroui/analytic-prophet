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
same comparison, two measurements quoted in two places, and moved the README's
table to the paper's figures. #231 replaced the table with a figure drawn from
this tier at every length (`evaluation/cost_figure.py`, checked in
tests/test_cost_figure.py), so the README quotes no fit ratio in prose. What
it still quotes from this tier -- the import cost and the memory decomposition
-- is checked here against the generated report.
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


REPORT = REPO / "evaluation" / "results" / "report.md"


def test_the_import_and_decomposition_figures_are_the_reports(readme):
    """Rounded from the generated report, which computes them."""
    report = " ".join(REPORT.read_text().split())
    importing = re.search(r"Importing the library costs \+([\d.]+) MiB over numpy and "
                          r"pandas, against \+([\d.]+) for Prophet", readme)
    assert importing, "the README no longer states the import cost in the form checked here"
    ours, theirs = importing.groups()
    assert f"+{ours} MiB" in report and f"+{theirs} MiB" in report
    fixed = re.search(r"within ([\d.]+) MiB of each other", readme).group(1)
    slopes = re.search(r"differ about ([\d.]+)×", readme).group(1)
    reported_fixed = re.search(r"within ([\d.]+) MiB of each other", report).group(1)
    assert float(fixed) == pytest.approx(float(reported_fixed), abs=0.05)
    assert f"differ {slopes}×" in report
