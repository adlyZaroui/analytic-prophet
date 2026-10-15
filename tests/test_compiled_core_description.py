"""
Issue #236: the compiled core described as what it runs.

`benchmark/README.md` said the compiled core ran "liblbfgs OWL-QN", and the
binding's own help text said the same, long after #23 replaced both with the
split reformulation on LBFGSpp's L-BFGS-B. The README also described a
benchmark configuration and an epoch the code had left behind. Nothing tested
any of it, so nothing failed. Here each description is held to the code:

  * the benchmark README's compiled row to the solvers `optimize.cpp` builds;
  * its configuration paragraph to `PROPHET_KWARGS`, and that to Prophet's
    own defaults where Prophet is installed;
  * its epoch to `seasonality.FOURIER_EPOCH`, and the benchmarks' docstrings
    with it;
  * the binding's help text to what `optimize()` and `newton()` run, and its
    status codes to the ones the source sets.
"""
import inspect
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent.parent
README = REPO / "benchmark" / "README.md"
SOURCE = REPO / "analytic_prophet" / "optimize.cpp"


def _one_line(text):
    return " ".join(text.split())


def test_the_benchmark_readme_names_the_solvers_the_core_runs():
    readme = README.read_text()
    assert "liblbfgs" not in readme and "OWL-QN" not in readme

    row = next(line for line in readme.splitlines() if line.startswith("| `compiled` |"))
    for words in ("L-BFGS-B", "split reformulation", "Newton"):
        assert words in row, words

    source = SOURCE.read_text()
    assert "#include <LBFGSB.h>" in source and "LBFGSpp::LBFGSBSolver" in source
    assert "OptimizeResult newton(" in source
    assert "#include <lbfgs.h>" not in source


def test_the_benchmark_readme_describes_the_configuration_it_runs():
    from _common import PROPHET_KWARGS

    readme = _one_line(README.read_text())
    assert "restates Prophet's defaults" in readme
    assert "yearly seasonality only at order 10" not in readme
    for period in ("yearly", "weekly", "daily"):
        assert PROPHET_KWARGS[f"{period}_seasonality"] == "auto", period
    assert PROPHET_KWARGS["mcmc_samples"] == 0


def test_the_restated_defaults_are_prophets(prophet_comparison):
    """What "restates Prophet's defaults" promises, checked against Prophet."""
    from _common import PROPHET_KWARGS

    Prophet, _common, _bridge = prophet_comparison
    defaults = {name: parameter.default
                for name, parameter in inspect.signature(Prophet.__init__).parameters.items()}
    for name, value in PROPHET_KWARGS.items():
        assert defaults[name] == value, (name, defaults[name], value)


def test_the_agreement_benchmark_states_the_epoch_the_code_uses():
    from analytic_prophet.seasonality import FOURIER_EPOCH

    readme = _one_line(README.read_text())
    assert "Prophet's 1970 epoch (`seasonality.FOURIER_EPOCH`)" in readme
    assert FOURIER_EPOCH == pd.Timestamp("1970-01-01")
    # the same two reasons were repeated in the benchmarks' own docstrings
    for path in [README, *sorted((REPO / "benchmark").glob("*.py"))]:
        text = _one_line(path.read_text())
        for stale in ("series start", "rotated Fourier basis", "rotated basis",
                      "indexed against different changepoints"):
            assert stale not in text, (path.name, stale)


def test_the_bindings_help_text_names_what_runs(cpp_module):
    docs = {
        "module": cpp_module.__doc__,
        "optimize": cpp_module.optimize.__doc__,
        "newton": cpp_module.newton.__doc__,
        "minus_log_posterior_and_gradient": cpp_module.minus_log_posterior_and_gradient.__doc__,
        "OptimizeResult.status": cpp_module.OptimizeResult.status.__doc__,
    }
    for name, doc in docs.items():
        assert doc, name
        assert "liblbfgs" not in doc and "OWL-QN" not in doc, name
    assert "L-BFGS-B" in docs["module"] and "Newton" in docs["module"]
    assert "L-BFGS-B" in docs["optimize"] and "split reformulation" in docs["optimize"]


def test_the_status_codes_documented_are_the_ones_the_core_sets(cpp_module):
    doc = cpp_module.OptimizeResult.status.__doc__
    documented = {int(code) for code in re.findall(r"(?<![\w.])-?\d+(?![\w.])", doc)}
    assigned = {int(code) for code in re.findall(r"\bstatus = (-?\d+);", SOURCE.read_text())}
    assert documented == assigned == {0, 1, 2, -2}
