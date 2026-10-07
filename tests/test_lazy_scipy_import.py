"""
Issue #130: scipy was imported whatever backend you asked for.

scipy serves exactly one thing here -- `_fit_python`, the reference backend
that exists to be read and checked against the C++ core rather than to be
fast. It was imported at package import time regardless, and it is 57 MiB of
the 59 MiB that importing this package added on top of numpy and pandas. That
is the whole of the one cost comparison in the evaluation suite that Prophet
wins: Prophet adds about 40 MiB and does not pull scipy.

One of the two import lines, `from scipy.stats import halfcauchy`, was unused
by anything in the repository and was pulling in all of scipy.stats on its own.

The tests here are about the import graph rather than about megabytes. A
threshold in MiB would be a platform measurement; "did importing this package
import scipy" is a yes or no, and it is the thing the issue asked to change.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PACKAGE = REPO / "analytic_prophet"

FRAME = """
import numpy as np, pandas as pd
df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=180, freq="D"),
                   "y": np.sin(np.arange(180) / 9.0) + 10.0})
"""
SCIPY_MODULES = """
print("RESULT" + json.dumps(sorted(
    name for name in sys.modules
    if name == "scipy" or name.startswith("scipy."))))
"""


def _scipy_modules_after(body):
    """Which scipy modules a fresh interpreter ended up with, having run `body`.

    A subprocess because the question is what an import *graph* pulls in, and
    this process has scipy resident several times over -- pytest, the harness
    and every other test have seen to that.
    """
    script = ("import json, sys, warnings\n"
              'warnings.filterwarnings("ignore")\n'
              f"sys.path[:0] = [{str(REPO)!r}]\n"
              + body + SCIPY_MODULES)
    completed = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stderr[-2000:]
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    raise AssertionError(f"probe printed no result: {completed.stdout[-500:]}")


# -- the import graph -----------------------------------------------------

def test_importing_the_package_does_not_import_scipy():
    """The issue, in one assertion."""
    assert _scipy_modules_after("import analytic_prophet\n") == []


def test_importing_the_model_class_does_not_import_scipy():
    assert _scipy_modules_after(
        "from analytic_prophet import AnalyticProphet\n") == []


def test_a_compiled_fit_and_predict_never_import_scipy(compiled_optimizer_module):
    """The claim that matters: the fast path does not pay for the slow one.

    Fitting *and* predicting, because the uncertainty draws were the other
    plausible place for a scipy dependency to be hiding.
    """
    modules = _scipy_modules_after(
        FRAME
        + "from analytic_prophet import AnalyticProphet\n"
        + "m = AnalyticProphet(yearly_seasonality=3, weekly_seasonality=False,\n"
          "                    daily_seasonality=False)\n"
        + f"m.fit(df, lib_path={compiled_optimizer_module!r})\n"
        + "m.predict(m.make_future_dataframe(periods=30))\n")

    assert modules == [], f"the compiled path imported {modules}"


def test_the_python_backend_does_import_scipy_when_it_runs():
    """The deferral has to be a deferral and not a removal -- the reference
    backend is scipy over the split reformulation and cannot work without it."""
    modules = _scipy_modules_after(
        FRAME
        + "from analytic_prophet import AnalyticProphet\n"
        + "m = AnalyticProphet(yearly_seasonality=3, weekly_seasonality=False,\n"
          "                    daily_seasonality=False)\n"
        + 'm.fit(df, backend="python", analytic=True)\n')

    assert "scipy.optimize" in modules


# -- the source, so the import cannot creep back ---------------------------

def _module_scope_imports(tree):
    """Import statements that run when the module is imported.

    Walks into `if`/`try` at module scope, since an import guarded by either
    still executes, and stops at `def` and `class`, since those do not.
    """
    found = []

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Import):
                found.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.append(node.module or "")
            for field in ("body", "orelse", "finalbody"):
                visit(getattr(node, field, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                visit(handler.body)

    visit(tree.body)
    return found


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")),
                         ids=lambda p: p.name)
def test_no_module_scope_import_pulls_in_scipy(path):
    """The regression guard. Re-adding a top-level `from scipy... import ...`
    anywhere in the package undoes #130 silently -- the suite would still pass
    and only the evaluation tier would notice, a tier nobody runs per commit."""
    offenders = [name for name in _module_scope_imports(ast.parse(path.read_text()))
                 if name == "scipy" or name.startswith("scipy.")]

    assert offenders == [], f"{path.name} imports {offenders} at module scope"


def test_nothing_in_the_package_imports_or_uses_scipy_stats():
    """`halfcauchy` was imported and referenced nowhere, and `scipy.stats` is
    the heavier half of the two.

    Checked against the parse tree rather than the text, since both names are
    named in the comment explaining why they left.
    """
    imports, uses = [], []
    for path in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "") == "scipy.stats":
                imports.append(f"{path.name}: {[a.name for a in node.names]}")
            if isinstance(node, ast.Import):
                imports += [f"{path.name}: {a.name}" for a in node.names
                            if a.name == "scipy.stats"]
            if isinstance(node, ast.Name) and node.id == "halfcauchy":
                uses.append(path.name)

    assert imports == [], f"scipy.stats is imported in {imports}"
    assert uses == [], f"halfcauchy is referenced in {uses}"


# -- the lazy binding, which the tests of other modules depend on ----------

def test_reading_the_name_off_the_module_gives_the_real_function():
    """Four test modules do `real_minimize = forecaster.minimize` and patch
    around it. PEP 562 is what keeps that working now the import is deferred,
    and binding a `None` placeholder instead would have handed them `None`."""
    from analytic_prophet import forecaster

    assert forecaster.minimize.__name__ == "minimize"
    assert forecaster.approx_fprime.__name__ == "approx_fprime"


def test_a_name_assigned_on_the_module_wins_over_the_lazy_hook(monkeypatch):
    """Which is the behaviour the patching tests rely on: PEP 562 fires only
    for names absent from the module dict."""
    from analytic_prophet import forecaster

    sentinel = object()
    monkeypatch.setattr(forecaster, "minimize", sentinel)

    assert forecaster.minimize is sentinel


def test_the_binding_is_per_name_rather_than_all_or_nothing():
    """A module dict holding one of the two must not stop the other binding.
    All-or-nothing would raise `NameError` from whichever name a test did not
    happen to touch."""
    from analytic_prophet import forecaster

    saved = {name: forecaster.__dict__.get(name)
             for name in forecaster.LAZY_SCIPY_NAMES}
    try:
        forecaster.__dict__.pop("approx_fprime", None)
        forecaster.__dict__["minimize"] = "patched"

        forecaster._ensure_scipy()

        assert forecaster.__dict__["minimize"] == "patched", "overwrote a patch"
        assert forecaster.__dict__["approx_fprime"].__name__ == "approx_fprime"
    finally:
        for name, value in saved.items():
            if value is None:
                forecaster.__dict__.pop(name, None)
            else:
                forecaster.__dict__[name] = value


def test_an_unknown_attribute_still_raises():
    """The hook must not turn every typo into an import attempt."""
    from analytic_prophet import forecaster

    with pytest.raises(AttributeError):
        forecaster.no_such_name
