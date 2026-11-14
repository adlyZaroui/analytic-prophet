"""
Issue #245: the abstract states the two results a Prophet user can act on.

#208 measured both, and section 7 reports them, but the abstract did not:
Prophet's own `algorithm="Newton"` comes close to the optimum at a price in
time, and the fix runs with Prophet's Stan model unchanged, through its
`log_prob` alone. Checked here:

  * **the abstract states both**, each figure a generated macro, and the
    macros are the ones section 7 quotes for the same results;
  * **the committed results bear them out**;
  * **it stays under arXiv's 1,920 characters**, measured as it renders, with
    every macro expanded.
"""
import csv
import importlib.util
import re
import statistics
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
RESULTS = PAPER / "results"

ARXIV_LIMIT = 1920   # characters, arXiv's limit on an abstract


@pytest.fixture(scope="module")
def macros():
    spec = importlib.util.spec_from_file_location(
        "quote_numbers", PAPER / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.macros()


@pytest.fixture(scope="module")
def abstract():
    text = (PAPER / "main.tex").read_text()
    return text[text.index(r"\begin{abstract}") + len(r"\begin{abstract}"):
                text.index(r"\end{abstract}")]


def _one(text):
    return " ".join(text.split())


def rendered(abstract, macros):
    """The abstract as a reader gets it: macros expanded, markup gone."""
    text = re.sub(r"\\code\{([^}]*)\}", r"\1", abstract)
    text = re.sub(r"\\([A-Za-z]+)(\{\})?",
                  lambda match: macros.get(match.group(1), match.group(0)), text)
    text = text.replace("~", " ").replace(r"\%", "%")
    text = re.sub(r"\$([^$]*)\$",
                  lambda match: match.group(1).replace(r"\sigma", "σ"), text)
    return _one(text)


# -- it states both, as section 7 does ---------------------------------------

NEWTON = (r'Users who stay on Prophet can come close: its own \code{algorithm="Newton"} '
          r"stops a median of \NewtonCensusNewtonMedian{} nats short of the optimum on "
          r"\NewtonCensusSeries{} census series, against \NewtonCensusLbfgsMedian{} for the "
          r"default L-BFGS, at a median of \NewtonCensusTimeRatio{} times the default's fit "
          r"time.")
FROZEN = (r"The fix itself runs with Prophet's Stan model unchanged: from Prophet's estimate, "
          r"a solver given only the model's \code{log_prob} reaches the optimum.")


def test_the_abstract_states_both_results(abstract):
    one = _one(abstract)
    assert NEWTON in one
    assert FROZEN in one


def test_its_figures_are_the_ones_section_7_quotes(abstract):
    discussion = _one((PAPER / "sections" / "discussion.tex").read_text())
    today = discussion[discussion.index(r"\paragraph{What a Prophet user can do today.}"):
                       discussion.index(r"\paragraph{Adopting the fix")]
    for name in ("NewtonCensusSeries", "NewtonCensusLbfgsMedian",
                 "NewtonCensusNewtonMedian", "NewtonCensusTimeRatio"):
        assert f"\\{name}" in abstract and f"\\{name}" in today, name
    frozen = discussion[discussion.index(r"\paragraph{Adopting the fix with Prophet's Stan "
                                         r"model frozen.}"):]
    assert r"\code{log_prob} of Prophet's unmodified program" in frozen


def test_no_figure_in_the_abstract_is_typed(abstract):
    typed = re.findall(r"\d", re.sub(r"\\[A-Za-z]+|\bM4\b", "", abstract))
    assert not typed, f"typed digits in the abstract: {typed}"


# -- the committed results bear them out ----------------------------------------

def test_prophets_newton_comes_close_at_a_price():
    rows = list(csv.DictReader(open(RESULTS / "newton_census.csv")))
    assert rows and not any(row["newton_error"] for row in rows), "every series ran"
    newton = [float(row["shortfall_newton"]) for row in rows]
    lbfgs = [float(row["shortfall_lbfgs"]) for row in rows]
    assert statistics.median(newton) < statistics.median(lbfgs) / 100
    assert statistics.median(float(row["seconds_newton"]) / float(row["seconds_lbfgs"])
                             for row in rows) > 1


def test_log_prob_alone_reaches_the_optimum():
    """From Prophet's estimate, at every length it ran: at the optimum or just
    above it, with as many rates exactly zero."""
    rows = list(csv.DictReader(open(RESULTS / "polish.csv")))
    assert len(rows) == 3
    for row in rows:
        assert float(row["shortfall_prophet"]) > 0.1, "it starts short"
        assert float(row["shortfall_polished"]) <= 1e-6, row["observations"]
        assert row["exact_zeros_polished"] == row["exact_zeros_optimum"]


# -- the length ------------------------------------------------------------------

def test_the_renderer_leaves_no_markup(abstract, macros):
    text = rendered(abstract, macros)
    assert "\\" not in text and "$" not in text and "{" not in text, text
    assert "fails the first-order optimality conditions" in text


def test_the_abstract_fits_arxivs_limit(abstract, macros):
    length = len(rendered(abstract, macros))
    assert length <= ARXIV_LIMIT, f"the abstract renders at {length} characters"
