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

`tomllib` is imported with a fallback because this module would otherwise be
the one thing in the suite that cannot run on 3.9 or 3.10 -- two of the
versions `requires-python` claims. Found by building the matrix (#103), which
is exactly what a matrix is for.
"""
import os
import re
import subprocess
import sys
from importlib.metadata import (PackageNotFoundError, distributions,
                                version)
from pathlib import Path

import pytest

import analytic_prophet

REPO = Path(__file__).parent.parent
PYPROJECT = REPO / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject():
    from conftest import load_pyproject

    return load_pyproject()


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


def test_the_citation_file_is_valid_and_cites_this_version():
    """[#137] `CITATION.cff` is the one place that carries a *literal* version.

    `pyproject.toml` declares `version` dynamic and reads the attribute, so
    until this file there was exactly one copy of the string in the repository
    and nothing to keep in sync. A citation without a version is less use to
    whoever is citing, so the literal is allowed and this is the assertion that
    pays for it.

    The required keys are the four the Citation File Format 1.2.0 schema marks
    required; `date-released` is a string there rather than a YAML date, which
    is why it is quoted in the file.
    """
    import yaml

    text = (REPO / "CITATION.cff").read_text()
    citation = yaml.safe_load(text)

    for key in ("cff-version", "message", "title", "authors"):
        assert key in citation, f"CITATION.cff is missing the required {key!r}"
    assert citation["cff-version"] == "1.2.0"
    assert citation["authors"], "a citation with no authors cites nobody"

    assert citation["version"] == analytic_prophet.__version__, (
        "CITATION.cff cites a version this package is not; it is the one "
        "literal copy of the string and has to be updated with a release")
    assert isinstance(citation["date-released"], str), (
        "date-released must be quoted: the format schema types it as a string, "
        "and an unquoted value parses as a YAML date object")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", citation["date-released"])


def test_the_cpp_core_ships_with_the_package(pyproject):
    """It is compiled on demand rather than at install time, so the *source*
    is what has to be in the distribution. Leaving it out would produce an
    install where the C++ path can never work."""
    shipped = pyproject["tool"]["setuptools"]["package-data"]["analytic_prophet"]
    assert "optimize.cpp" in shipped
    assert (REPO / "analytic_prophet" / "optimize.cpp").exists()
    # the third-party licences joined it in #97: a binary wheel contains code
    # compiled from Eigen and LBFGSpp and has to carry their terms
    assert any("licences" in entry for entry in shipped)


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

    # An `analytic_prophet.egg-info` left in the source tree by a wheel build
    # answers that version query from the repo root, so the package looks
    # installed when nothing is (#97 added a setup.py, which is what started
    # leaving one). A real install puts its metadata in site-packages.
    if not any("site-packages" in str(dist._path)
               for dist in distributions()
               if dist.metadata["Name"] == "analytic-prophet"):
        pytest.skip("only a source-tree egg-info, not an install -- "
                    "nothing to check")

    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "-c", "import analytic_prophet; print(analytic_prophet.__file__)"],
        cwd=REPO.parent, env=environment, capture_output=True, text=True, timeout=120)

    assert completed.returncode == 0, (
        "analytic-prophet is installed but does not import from outside the "
        "repo. On macOS with Python 3.13+ this is the hidden-.pth problem in "
        "the docstring above; `chflags nohidden` on the __editable__* files in "
        f"site-packages fixes it.\n{completed.stderr}")
