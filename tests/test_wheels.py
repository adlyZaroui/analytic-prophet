"""
Issue #97: the distribution, and the claims it makes.

`pip install analytic-prophet` has to work on a machine with no compiler. The
package compiles its C++ core on first use otherwise (#108), and moving a
toolchain requirement from install time to first-fit time is not removing it --
so a binary wheel is the only shape that delivers what the install promises.

What can be checked here is the configuration and its coherence: that the
wheel matrix covers every Python the test matrix runs, that the platforms
built are the platforms tested, that the licences of the two libraries
compiled into the wheel are shipped, and that the publish step cannot run
unattended. Whether a wheel actually works is checked by the release
workflow's `verify` job, on the built artefact, because that is the only place
the question can be asked honestly.
"""
import re
from pathlib import Path

import pytest
import yaml

from conftest import load_pyproject

REPO = Path(__file__).parent.parent
WORKFLOWS = REPO / ".github" / "workflows"


@pytest.fixture(scope="module")
def pyproject():
    return load_pyproject()


@pytest.fixture(scope="module")
def cibuildwheel(pyproject):
    return pyproject["tool"]["cibuildwheel"]


@pytest.fixture(scope="module")
def release():
    with open(WORKFLOWS / "release.yml") as handle:
        loaded = yaml.safe_load(handle)
    if True in loaded:                       # `on:` is a YAML 1.1 boolean
        loaded["on"] = loaded.pop(True)
    return loaded


@pytest.fixture(scope="module")
def tested_versions():
    """The Pythons the test matrix actually runs."""
    with open(WORKFLOWS / "tests.yml") as handle:
        loaded = yaml.safe_load(handle)
    return set(loaded["jobs"]["suite"]["strategy"]["matrix"]["python"])


# -- the matrix says what requires-python claims --------------------------

def test_a_wheel_is_built_for_every_python_that_is_tested(cibuildwheel,
                                                          tested_versions):
    """`requires-python` is load-bearing the moment there is a package: it
    tells pip which interpreters may install this. A version in the test
    matrix with no wheel is a version that falls back to compiling from the
    sdist, which is the friction the wheels exist to remove."""
    built = {f"3.{m.group(1)}" for m in
             re.finditer(r"cp3(\d+)-", cibuildwheel["build"])}
    missing = tested_versions - built
    assert missing == set(), f"tested but no wheel built: {sorted(missing)}"


def test_requires_python_is_covered_by_the_wheels(pyproject, cibuildwheel):
    floor = pyproject["project"]["requires-python"]
    assert floor.startswith(">="), floor
    lowest_built = min(int(m.group(1)) for m in
                       re.finditer(r"cp3(\d+)-", cibuildwheel["build"]))
    assert floor == f">=3.{lowest_built}", (
        f"requires-python says {floor} and the lowest wheel is 3.{lowest_built}")


def test_windows_is_not_built(cibuildwheel):
    """Deliberate: the suite has never run there and `analytic_prophet.build`
    only knows Unix compilers, so a Windows wheel would ship for a platform
    nothing here has been tested on."""
    assert "win" not in cibuildwheel.get("skip", "") + cibuildwheel["build"] or \
        "win_amd64" not in cibuildwheel["build"]
    assert "windows" not in str(cibuildwheel).lower()


# -- the headers, and their licences --------------------------------------

def test_the_header_versions_are_pinned(cibuildwheel):
    """An unpinned fetch makes a wheel's contents depend on the day it was
    built."""
    environment = cibuildwheel["environment"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", environment["EIGEN_VERSION"])
    assert re.fullmatch(r"v\d+\.\d+\.\d+", environment["LBFGSPP_VERSION"])


def test_the_build_looks_where_the_fetch_puts_them(cibuildwheel):
    """`analytic_prophet.build`'s finders read exactly these two variables, so
    the wheel build and the on-demand build search by the same rule."""
    environment = cibuildwheel["environment"]
    assert environment["EIGEN_INCLUDE_DIR"].startswith(environment["AP_HEADERS"])
    assert environment["LBFGSPP_INCLUDE_DIR"].startswith(environment["AP_HEADERS"])

    script = (REPO / "tools" / "fetch_headers.sh").read_text()
    assert "AP_HEADERS" in script
    assert "Eigen/Dense" in script and "LBFGSB.h" in script, (
        "the fetch does not verify it got the files the build needs")


def test_the_licences_are_shipped(pyproject):
    """A binary wheel contains code compiled from Eigen (MPL-2.0) and LBFGSpp
    (MIT), so it has to carry their terms. Neither is vendored, so nothing in
    the repository holds the text -- it is collected at build time."""
    package_data = pyproject["tool"]["setuptools"]["package-data"]["analytic_prophet"]
    assert any("licences" in entry for entry in package_data)

    collector = (REPO / "tools" / "collect_licences.py").read_text()
    assert "MPL-2.0" in collector and "MIT" in collector
    assert "unmodified" in collector, (
        "the NOTICE does not state that the sources are used unmodified, which "
        "is what MPL-2.0 turns on")


def test_the_build_is_run_before_anything_is_compiled(cibuildwheel):
    before = cibuildwheel["before-all"]
    assert "fetch_headers.sh" in before
    assert "collect_licences.py" in before
    assert before.index("fetch_headers") < before.index("collect_licences"), (
        "the licences are collected out of the fetched trees, so the fetch "
        "has to happen first")


# -- the extension is optional --------------------------------------------

def test_the_extension_does_not_break_a_toolchain_free_build():
    """The old contract, which still matters for the sdist and for a clone: a
    build without a compiler produces an installable package, and the core is
    compiled on first use as it always was."""
    setup = (REPO / "setup.py").read_text()
    assert "class OptionalBuildExt" in setup
    assert "return []" in setup, "a missing toolchain must yield no extension"
    assert "backend='python'" in setup, (
        "the message does not mention the path that needs no compiler at all")


def test_setup_py_does_not_import_the_package():
    """It needs the compile flags, and importing the package to get them pulls
    numpy into the isolated build environment, where it is not installed. This
    failed exactly that way the first time."""
    setup = (REPO / "setup.py").read_text()
    assert "spec_from_file_location" in setup
    assert not re.search(r"^from analytic_prophet", setup, re.MULTILINE)
    assert not re.search(r"^import analytic_prophet", setup, re.MULTILINE)


def test_the_wheel_and_the_on_demand_build_use_the_same_flags():
    """A wheel compiled differently from the on-demand path would make every
    published timing describe a binary nobody has."""
    setup = (REPO / "setup.py").read_text()
    assert "COMPILE_FLAGS" in setup, "setup.py has its own copy of the flags"


# -- publishing ------------------------------------------------------------

def test_the_release_builds_from_a_tag(release):
    assert "v*" in release["on"]["push"]["tags"]


def test_publishing_is_gated_on_an_environment(release):
    """Irreversible and outward-facing: an environment with a required
    reviewer does not run until a human approves it, however the workflow was
    triggered."""
    publish = release["jobs"]["publish"]
    assert publish["environment"]["name"] == "pypi"
    assert publish["needs"] == "verify" or "verify" in publish["needs"]


def test_publishing_uses_trusted_publishing_rather_than_a_token(release):
    publish = release["jobs"]["publish"]
    assert publish["permissions"]["id-token"] == "write"
    assert "password" not in str(publish), "an API token in the workflow"
    assert "PYPI_TOKEN" not in str(release)


def test_the_release_verifies_the_wheel_runs_the_compiled_core(release):
    """The acceptance criterion, and the one a plain import test would pass
    while the wheel quietly compiled its core on first use."""
    verify = release["jobs"]["verify"]
    shell = " ".join(step.get("run", "") for step in verify["steps"])
    assert "ANALYTIC_PROPHET_CACHE" in str(verify), (
        "nothing checks that the runtime builder stayed idle")
    assert "load_cpp_module" in shell
    assert "licences/LICENSE.eigen" in shell
