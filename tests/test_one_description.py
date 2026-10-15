"""
Issue #213: one description everywhere, the badges, and what 0.1.1 publishes.

The project described itself four ways -- the README's first line, the PyPI
summary, CITATION.cff and GitHub's About -- one of them in capitals as an
"EARLY DEVELOPMENT STAGE", and two of them crediting the gradient with a speed
that section 1 of the README shows is mostly the cost of calling CmdStan.
Three of the four are files, held here to the README's pitch (#212); the About
is a repository setting, set to the same sentence with `gh repo edit`.

PyPI still shows the 0.1.0 README until a release replaces it, so the version
is 0.1.1 here and CITATION.cff cites it (`tests/test_packaging.py` holds the
two together). Publishing is the author's: a `v0.1.1` tag starts release.yml.
"""
import re
from pathlib import Path

import yaml

from conftest import load_pyproject

REPO = Path(__file__).parent.parent
README = REPO / "README.md"


def _pitch():
    """The README's first line of prose, without its markdown."""
    first = next(line for line in README.read_text().splitlines()[1:]
                 if line.strip() and not line.startswith("[!["))
    return first.strip("*")


def test_the_pypi_summary_is_the_readmes_pitch():
    assert load_pyproject()["project"]["description"] == _pitch()


def test_the_citation_says_the_same_thing():
    citation = yaml.safe_load((REPO / "CITATION.cff").read_text())
    assert citation["title"] == "analytic-prophet: " + _pitch().rstrip(".")
    assert " ".join(citation["abstract"].split()).startswith(_pitch())


def test_the_readme_carries_the_badges_a_pypi_reader_looks_for():
    targets = " ".join(line for line in README.read_text().splitlines()
                       if line.startswith("[!["))
    assert "img.shields.io/pypi/v/analytic-prophet" in targets
    assert "img.shields.io/pypi/pyversions/analytic-prophet" in targets
    assert "img.shields.io/badge/licence-MIT" in targets
    # and the licence badge says what the licence file says
    assert (REPO / "LICENSE").read_text().startswith("MIT License")


def test_every_python_ci_tests_is_a_classifier():
    """The Python-versions badge reads the classifiers, and pyproject declared
    only `Python :: 3`: the matrix's claim was invisible on PyPI."""
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "tests.yml").read_text())
    tested = set(workflow["jobs"]["suite"]["strategy"]["matrix"]["python"])
    declared = {classifier.rsplit(" :: ", 1)[1]
                for classifier in load_pyproject()["project"]["classifiers"]
                if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", classifier)}
    assert declared == tested


def test_the_status_is_stated_the_same_way_where_it_is_stated():
    """"EARLY DEVELOPMENT STAGE" in the About, against `Development Status ::
    3 - Alpha` in the classifiers. The classifier and the README's status line
    say it; the About is the pitch alone."""
    assert "Development Status :: 3 - Alpha" in load_pyproject()["project"]["classifiers"]
    assert "**Status: early development.**" in README.read_text()
