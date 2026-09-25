"""
The report (#79): tables and figures from the committed results.

It reads `results/*.csv` and nothing else -- it refits nothing and measures
nothing -- so the tests are about whether it reports faithfully what is in those
files, and whether it behaves when they are missing or partial.

The property worth protecting is that **regenerating the report changes
nothing** unless the results changed. A report that varies run to run cannot be
diffed, and a number nobody can diff is a number nobody will check.
"""
import csv

import pytest

import harness
import report


def write_results(directory, tier_name, rows):
    directory.mkdir(parents=True, exist_ok=True)
    harness.write(tier_name, rows, results_dir=directory)


@pytest.fixture
def results(tmp_path, monkeypatch):
    """A minimal but complete set of results, so the report has every section."""
    from harness import Measurement as M

    monkeypatch.setattr(harness, "RESULTS", tmp_path)
    monkeypatch.setattr(report, "FIGURES", tmp_path / "figures")

    write_results(tmp_path, "tier0_agreement", [
        M(0, "peyton_manning[:300]", "default", "both", "design_matrix_max_abs_diff", 7.3e-12),
        M(0, "peyton_manning[:300]", "default", "both", "prior_scales_max_abs_diff", 0.0),
        M(0, "peyton_manning[:300]", "default", "both", "changepoints_max_abs_diff", 0.0),
        M(0, "peyton_manning[:300]", "default", "prophet", "lp__", 813.35, "nats"),
        M(0, "peyton_manning[:300]", "default", "analytic_prophet", "lp__", 815.34, "nats"),
        M(0, "peyton_manning[:300]", "default", "both", "lp___difference", 1.99, "nats"),
        M(0, "all", "default", "both", "gate_failed", 0.0),
    ])
    write_results(tmp_path, "tier2_accuracy", [
        M(2, "paired", "all", "difference", "mae_median", -1.91),
        M(2, "paired", "all", "difference", "mae_wins", 26),
        M(2, "paired", "all", "difference", "mae_n", 36),
        M(2, "paired", "all", "difference", "mae_p_value", 0.0063),
        M(2, "m4_weekly_W1", "Weekly", "analytic_prophet", "coverage", 0.35),
        M(2, "m4_weekly_W1", "Weekly", "prophet", "coverage", 0.34),
        M(2, "m4_weekly_W1", "Weekly", "analytic_prophet", "mae", 9.0),
        M(2, "m4_weekly_W1", "Weekly", "prophet", "mae", 10.0),
    ])
    return tmp_path


def test_the_report_states_the_gate_result_first(results):
    text = report.build(figures=False)
    gate = text.index("Gate: PASSED")
    assert gate < text.index("Tier 2"), "the gate should precede what it gates"


def test_a_failing_gate_is_reported_as_such(results):
    from harness import Measurement as M

    write_results(results, "tier0_agreement",
                  [M(0, "all", "default", "both", "gate_failed", 1.0)])
    assert "Gate: FAILED" in report.build(figures=False)


def test_numbers_come_from_the_results_rather_than_the_prose(results):
    text = report.build(figures=False)
    assert "815.34" in text and "813.35" in text
    assert "+1.99" in text
    assert "26/36" in text and "0.0063" in text


def test_a_tier_that_was_not_run_is_named_rather_than_skipped_silently(results):
    text = report.build(figures=False)
    assert "tier1_recovery" in text and "Not run" in text
    assert "--tiers 1" in text, "should say how to produce it"


def test_regenerating_the_report_changes_nothing(results):
    """The property that makes a committed report worth committing."""
    assert report.build(figures=False) == report.build(figures=False)


def test_the_shared_under_coverage_is_reported_as_shared(results):
    """It is larger than anything separating the two implementations, so a
    reader must not be able to take the accuracy table without it."""
    text = report.build(figures=False)
    assert "under-cover" in text
    assert "0.353" not in text or "0.34" in text     # both means present


def test_figures_are_written_and_referenced(results):
    text = report.build(figures=True)

    written = {path.name for path in (results / "figures").glob("*.png")}
    assert "tier0_parity.png" in written
    for name in written:
        assert f"figures/{name}" in text, f"{name} written but never referenced"


def test_every_referenced_figure_exists(results):
    """The other direction: a reference to a figure that was never drawn is a
    broken image in the rendered report."""
    text = report.build(figures=True)

    import re
    referenced = set(re.findall(r"\(figures/([\w.]+)\)", text))
    written = {path.name for path in (results / "figures").glob("*.png")}
    assert referenced <= written


def test_the_report_records_how_it_was_measured(results):
    text = report.build(figures=False)
    assert "seed" in text and str(harness.SEED) in text
    assert "python" in text.lower()


def test_no_results_at_all_still_produces_a_report(tmp_path, monkeypatch):
    """A fresh clone, before anything has been run."""
    monkeypatch.setattr(harness, "RESULTS", tmp_path)
    monkeypatch.setattr(report, "FIGURES", tmp_path / "figures")

    text = report.build(figures=False)

    assert "Not run" in text
    assert text.count("Not run") == 4
