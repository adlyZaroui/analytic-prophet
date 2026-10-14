"""
Issue #201: the objective and gradient against Stan's, measured at full precision.

`paper/experiments/agreement.py` scores this implementation's objective and
hand-derived gradient against Stan's `log_prob` at 21 points. Held here:

  * **the result** -- complete, the design matrices identical, the objective
    equal to minus `lp__` with no constant between them, the gradients equal;
  * **the quotations** -- the paper, the README and docs/model.md say what the
    committed file says, and nothing types the old "1e-12" or "10^{-7}";
  * **the code** -- T = 300 measured again, by conftest.recorded_here's rule.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from conftest import recorded_here

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"


def _module():
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("agreement", EXPERIMENTS / "agreement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agreement = _module()


@pytest.fixture(scope="module")
def rows():
    return agreement.read()


def _short(value):
    """1.4e-13, as prose writes it: no zero-padded exponent."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def test_every_length_has_both_solutions_and_the_random_points(rows):
    for size in agreement.LENGTHS:
        names = {row["point"] for row in rows if row["observations"] == size}
        assert names == {"prophet", "analytic_prophet",
                         *(f"random {i}" for i in range(agreement.RANDOM))}


def test_the_two_programs_see_the_same_design(rows):
    assert max(row["design_gap"] for row in rows) < 1e-9


def test_the_objective_is_minus_lp_with_no_constant_between_them(rows):
    """Not "equal up to a constant", which is what the README used to say:
    Stan's sampling statements drop exactly the constants the objective omits."""
    assert max(row["offset_relative"] for row in rows) < 1e-12
    # a constant would show as the same offset everywhere; there is none to find
    solutions = [row["offset"] for row in rows if not row["point"].startswith("random")]
    assert max(abs(offset) for offset in solutions) < 1e-9


def test_the_gradients_agree_including_at_exact_zeros(rows):
    assert max(row["gradient_max_relative"] for row in rows) < 1e-8
    ours = [row for row in rows if row["point"] == "analytic_prophet"]
    assert all(row["exact_zeros"] > 0 for row in ours), "sign(0) = 0 is not exercised"
    assert max(row["gradient_max_abs"] for row in ours) < 1e-6
    for row in rows:
        assert row["gradient_sigma_abs"] <= max(1e-9, 1e-12 * abs(row["lp_stan"])), row


def test_the_paper_quotes_the_measurement_and_types_neither_old_figure():
    evidence = (PAPER / "sections" / "evidence.tex").read_text()
    assert r"to $\KktStanGradientSolution$" in evidence
    assert "10^{-7}" not in evidence
    numbers = (REPO / "tests" / "test_paper_numbers.py").read_text()
    assert '"10^{-7}"' not in numbers, "10^{-7} is still listed as a typed constant"


def test_the_readme_and_the_doc_quote_the_committed_figures(rows):
    readme = " ".join((REPO / "README.md").read_text().split())
    assert "1e-12" not in readme
    objective = max(row["offset_relative"] for row in rows)
    gradient = max(row["gradient_max_relative"] for row in rows)
    # the README rounds to an order of magnitude
    assert "to about 1e-13 relative" in readme and 1e-14 <= objective < 1e-12
    assert "agree to about 1e-9 relative" in readme and 1e-10 <= gradient < 1e-8
    assert f"every one of {len(rows)} points" in readme

    doc = " ".join((REPO / "docs" / "model.md").read_text().split())
    solution = max(row["gradient_max_abs"] for row in rows if row["point"] == "analytic_prophet")
    for value in (objective, gradient, solution):
        assert _short(value) in doc, f"docs/model.md does not quote {_short(value)}"
    assert "days since the series start" not in doc, "the stale epoch caveat is back"


def test_the_code_reproduces_the_shortest_length(rows, prophet_comparison,
                                                 compiled_optimizer_module):
    """T = 300 measured again. The agreement is a property of two programs at a
    point, so it holds anywhere; the points include Prophet's L-BFGS endpoint,
    which is pinned by nothing, so the committed lp__ is compared only where it
    was recorded."""
    fresh = agreement.measure(300, compiled_optimizer_module)
    committed = {row["point"]: row for row in rows if row["observations"] == 300}
    assert {row["point"] for row in fresh} == set(committed)
    for row in fresh:
        assert row["offset_relative"] < 1e-12, row
        assert row["gradient_max_relative"] < 1e-8, row
        assert row["gradient_sigma_abs"] <= max(1e-9, 1e-12 * abs(row["lp_stan"])), row
    if recorded_here(agreement.RESULTS.with_suffix(".meta.json")):
        for row in fresh:
            assert row["lp_stan"] == pytest.approx(committed[row["point"]]["lp_stan"],
                                                   rel=1e-12, abs=1e-9), row["point"]
