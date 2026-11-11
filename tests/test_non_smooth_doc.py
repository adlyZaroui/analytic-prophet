"""
Issue #242: docs/non-smooth-objective.md, as a short explainer.

The README links the page twice, and it still made the argument the project
had stopped making: that the kink "costs it real accuracy", that this
implementation "is more accurate" on a 36-series sample, and the liblbfgs
history the paper and the README had dropped. It is now a short explainer that
points to the paper and the evaluation report, and its history is in
CHANGELOG.md. Checked here:

  * **no claim the paper or the report contradicts** is back;
  * **every number it keeps is generated or tested**: the paper's macros, as
    the README's are, or a stated constant;
  * **every figure is in its sentence**, so a number cannot drift to a claim
    it was never measured for;
  * **what it points at exists**: the paper's sections by title, the scripts,
    and the anchors it links;
  * **the README's two links describe what the page is now.**
"""
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
DOC = REPO / "docs" / "non-smooth-objective.md"
PAPER = REPO / "paper"


@pytest.fixture(scope="module")
def doc():
    return DOC.read_text()


@pytest.fixture(scope="module")
def one(doc):
    return " ".join(doc.split())


@pytest.fixture(scope="module")
def macros():
    spec = importlib.util.spec_from_file_location(
        "quote_numbers", PAPER / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.macros()


def _slug(heading):
    """GitHub's anchor for a markdown heading."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path):
    return {_slug(line.lstrip("#")) for line in path.read_text().splitlines()
            if line.startswith("#")}


def test_no_claim_the_paper_or_the_report_contradicts(one):
    for phrase in ("costs it real accuracy", "more accurate", "liblbfgs", "17.8%",
                   "36 M4", "26 of 36", "p = "):
        assert phrase not in one, f"the page says {phrase!r} again"
    assert "a real improvement, and usually a small one" in one


# -- every number is generated, or a stated constant ------------------------

def _prose_numbers(text):
    """Every number a reader meets, outside code, link targets, URLs and
    issue references."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"#\d+", " ", text)
    return {token.rstrip(",") for token in
            re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:e-?\d+)?", text)}


FROM_THE_PAPER = ("KktZerosPeyton", "KktZerosCensusMedian", "KktSeries", "CensusSeries",
                  "SmoothFactorLong", "SmoothFactorShort", "NewtonShortMin", "NewtonShortMax",
                  "NewtonCensusSeries", "NewtonCensusNewtonMedian", "NewtonCensusLbfgsMedian",
                  "NewtonCensusTimeRatio", "RmseWins", "RmseAdvantage", "LimWidthMedian",
                  "LimWidthWider", "LimCoverageOurs", "LimCoverageProphet")

CONSTANTS = {
    "25": "Prophet's default n_changepoints",
    "80": "Prophet's default interval_width, in percent",
    "100": "Prophet's rule: Newton below 100 observations",
    "2": "Tier 2, the evaluation's forecast tier",
}


def test_every_number_is_generated_or_a_constant(doc, macros):
    paper = {macros[name] for name in FROM_THE_PAPER}
    unexplained = sorted(_prose_numbers(doc) - paper - set(CONSTANTS))
    assert not unexplained, f"numbers with no generated source or stated reason: {unexplained}"


def test_each_figure_is_in_its_sentence(one, macros):
    for sentence in (
            f"{macros['KktZerosPeyton']} of its 25 rates are exactly zero",
            f"Across the {macros['KktSeries']} series of the census, the median is "
            f"{macros['KktZerosCensusMedian']}",
            f"fails the first-order optimality conditions** on every one of the "
            f"{macros['KktSeries']} census series",
            f"brings the same L-BFGS {macros['SmoothFactorLong']} to "
            f"{macros['SmoothFactorShort']} times closer",
            f"stops {macros['NewtonShortMin']} to {macros['NewtonShortMax']} nats short",
            f"On {macros['NewtonCensusSeries']} census series it stops a median of "
            f"{macros['NewtonCensusNewtonMedian']} nats short, against "
            f"{macros['NewtonCensusLbfgsMedian']} for its L-BFGS, and takes a median of "
            f"{macros['NewtonCensusTimeRatio']} times as long",
            f"On {macros['CensusSeries']} held-out M4 series the RMSE is lower on "
            f"{macros['RmseWins']} of them, by a median of {macros['RmseAdvantage']}%",
            f"wider, by a median of {macros['LimWidthMedian']}% on {macros['LimWidthWider']} "
            f"of the {macros['CensusSeries']} series",
            f"mean coverage is {macros['LimCoverageOurs']} for this package and "
            f"{macros['LimCoverageProphet']} for Prophet"):
        assert sentence in one, f"not found: {sentence}"


# -- what it points at exists -------------------------------------------------

def _paper_titles():
    titles = set()
    for path in [PAPER / "main.tex", *(PAPER / "sections").glob("*.tex")]:
        titles |= set(re.findall(r"\\(?:sub)?section\{([^}]*)\}", path.read_text()))
    return titles


def test_the_paper_sections_it_names_exist(doc):
    titles = _paper_titles()
    named = {" ".join(name.split()) for name in re.findall(r'\("([^"]+)"\)', doc)}
    table = [line.split("|")[2].strip() for line in doc.splitlines()
             if line.startswith("| ") and line.count("|") == 4][2:]
    for cell in table:
        named |= {part for part in cell.split(", and ") if part != "Appendix A"}
    assert len(named) >= 7, named
    missing = sorted(named - titles)
    assert not missing, f"no paper section has these titles: {missing}"
    # Appendix A is the forecast comparison
    appendix = PAPER / "sections" / "appendix_forecast.tex"
    assert "\\section{The forecast comparison}" in appendix.read_text()


def test_its_links_land(doc):
    for target in re.findall(r"\]\((\.\./[^)#]+)(#[^)]*)?\)", doc):
        path, anchor = target
        assert (DOC.parent / path).resolve().exists(), path
        if anchor:
            assert anchor[1:] in _anchors((DOC.parent / path).resolve()), (path, anchor)


def test_the_readme_describes_what_the_page_is_now():
    readme = " ".join((REPO / "README.md").read_text().split())
    assert "and the argument in brief: [the non-smooth objective]" in readme
    assert ("| the argument in brief: the kink, why Prophet's optimizer stops short there, "
            "the fix, and where the paper measures each part |") in readme
    assert "the central argument" not in readme and "argument in prose" not in readme


def test_the_comments_that_cite_it_name_a_section_it_has(doc):
    """Two comments in forecaster.py cite the short-series section by title;
    they used to say it was in the README."""
    source = " ".join((REPO / "analytic_prophet" / "forecaster.py").read_text()
                      .replace("#", " ").split())
    assert source.count('See docs/non-smooth-objective.md, "Prophet\'s rule for short series, '
                        'and what it costs"') == 2
    assert "### Prophet's rule for short series, and what it costs" in doc
