"""
Issue #213: one description everywhere, the badges, and what 0.1.1 publishes.

The project described itself four ways -- the README's first line, the PyPI
summary, CITATION.cff and GitHub's About. Three of the four are files, held
here to the README's pitch; the About is a repository setting, set to the same
sentence with `gh repo edit`.

#230 chose the sentence by what the package is for: a faster, lighter Prophet,
whose forecasts are slightly better as well. The paper keeps the fix as its
subject. The lede's speed and memory claim is checked against the cost tier.

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


def test_the_lede_claim_holds_at_every_length_measured():
    """"Fits are faster and lighter on memory at every series length
    measured" -- Tier 3's fit time and the memory a fit adds, at each length,
    against Prophet's."""
    import csv

    rows = list(csv.DictReader(open(REPO / "evaluation" / "results" / "tier3_cost.csv")))

    def value(series, implementation, metric):
        return next(float(row["value"]) for row in rows if row["series"] == series
                    and row["implementation"] == implementation and row["metric"] == metric)

    lengths = sorted({row["series"] for row in rows if row["series"].startswith("peyton")})
    assert len(lengths) >= 5
    for series in lengths:
        assert value(series, "compiled", "fit_wall") < value(series, "prophet", "fit_wall"), series
        assert (value(series, "compiled", "fit_peak_rss_added")
                < value(series, "prophet", "fit_peak_rss_added")), series
    assert "faster and lighter on memory at every series length measured" in \
        " ".join(README.read_text().split())
