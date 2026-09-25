"""
The packaging metadata, checked rather than assumed.

`pyproject.toml` exists so the package can be installed -- `pip install -e .`
-- instead of reached by `sys.path` surgery in every entry point. Three things
have to hold for that to be true rather than merely declared, and each has bitten
somewhere:

  * the version in pyproject and the one on the package cannot drift, since
    pyproject reads it from `analytic_prophet.__version__` and a typo there
    would surface as a build error rather than a wrong number;
  * `optimize.cpp` must ship *with* the package, because it is compiled on
    demand and an install without it can never build the C++ core;
  * an editable install has to actually import. That one is not hypothetical:
    on macOS with Python 3.13+ it silently does not, which
    test_an_editable_install_actually_imports exists to catch.
"""
import os
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

import analytic_prophet

REPO = Path(__file__).parent.parent
PYPROJECT = REPO / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject():
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)


def test_the_package_version_is_the_one_pyproject_publishes(pyproject):
    """One source of truth: pyproject declares `version` dynamic and reads the
    attribute, so this checks the wiring rather than two hand-kept copies."""
    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "analytic_prophet.__version__"}
    assert analytic_prophet.__version__

    try:
        installed = version("analytic-prophet")
    except PackageNotFoundError:
        pytest.skip("not installed -- run `pip install -e .` to check this")
    assert installed == analytic_prophet.__version__


def test_the_cpp_core_ships_with_the_package(pyproject):
    """It is compiled on demand rather than at install time, so the *source*
    is what has to be in the distribution. Leaving it out would produce an
    install where the C++ path can never work."""
    assert pyproject["tool"]["setuptools"]["package-data"] == {
        "analytic_prophet": ["optimize.cpp"]}
    assert (REPO / "analytic_prophet" / "optimize.cpp").exists()


def test_the_package_is_named_rather_than_discovered(pyproject):
    """Auto-discovery would sweep up whatever top-level directory is added
    next -- the evaluation suite, for one -- into the distribution."""
    assert pyproject["tool"]["setuptools"]["packages"] == ["analytic_prophet"]


def test_tests_run_without_an_install(pyproject):
    """`pythonpath` is what lets a fresh clone run `pytest` with no install and
    no PYTHONPATH.

    `benchmark/` is on it because the Prophet-comparison plumbing lives there
    and conftest imports it; `evaluation/` because its harness is tested like
    any other code. Neither is a package -- they are directories of scripts, so
    they are reached this way rather than by import."""
    assert pyproject["tool"]["pytest"]["ini_options"]["pythonpath"] == [
        ".", "benchmark", "evaluation"]


def test_an_editable_install_actually_imports():
    """The one that is not paperwork.

    `pip install -e .` writes a `.pth` file that installs an import hook. On
    macOS, setuptools creates that file with the `UF_HIDDEN` flag set, and
    Python 3.13+ **skips hidden .pth files** -- a hardening change in
    `site.addpackage`. The install then reports success, registers its metadata,
    and the import never works. Nothing warns: `pip show` is happy, the version
    query below passes, and only an actual import from another directory fails.

        chflags nohidden .venv/lib/python3.*/site-packages/__editable__*

    is the remedy. This test is how that gets noticed.

    Run out of process, from a directory that is not the repo, with PYTHONPATH
    cleared -- otherwise the repo root on sys.path would answer the import and
    the test would pass while proving nothing.
    """
    try:
        version("analytic-prophet")
    except PackageNotFoundError:
        pytest.skip("not installed -- nothing to check")

    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "-c", "import analytic_prophet; print(analytic_prophet.__file__)"],
        cwd=REPO.parent, env=environment, capture_output=True, text=True, timeout=120)

    assert completed.returncode == 0, (
        "analytic-prophet is installed but does not import from outside the "
        "repo. On macOS with Python 3.13+ this is the hidden-.pth problem in "
        "the docstring above; `chflags nohidden` on the __editable__* files in "
        f"site-packages fixes it.\n{completed.stderr}")
