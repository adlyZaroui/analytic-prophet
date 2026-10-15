"""
Issue #212: the README, restructured around three stories.

It had grown into one list that held the API, the caches, the gradient, the
kink, the optimum and the forecast statistics, with its own revision history
woven through. The structure now is a pitch, how to install and start, what a
Prophet user needs to know, then three stories, and the limitations, each once.

#230 set the order of the stories by what the package is for: a faster, lighter
Prophet first, the slightly better forecasts second, and why -- the optimum
Prophet's optimizer misses -- third. The paper keeps the fix as its subject.

The acceptance, checked here:

  * **the structure**, in order;
  * **every number is generated and tested, or a stated constant** -- the
    guard below lists each one the prose may carry and where it comes from,
    as `tests/test_paper_numbers.py` does for the paper;
  * **no history talk**: how the README used to read is in CHANGELOG.md;
  * **every term a new reader meets is defined or not used** -- Tier 2, lp__,
    the census, T;
  * **the framing is the package's** (#230): faster and lighter first, the
    forecast gain slight, and not "more accurate";
  * the two figures where the stories need them, the paper and how to cite
    it, and why this is a separate package.
"""
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
PAPER = REPO / "paper"
RAW = "https://raw.githubusercontent.com/adlyZaroui/analytic-prophet/main/"


@pytest.fixture(scope="module")
def readme():
    return README.read_text()


@pytest.fixture(scope="module")
def macros():
    spec = importlib.util.spec_from_file_location(
        "quote_numbers", PAPER / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.macros()


def _one_line(text):
    return " ".join(text.split())


def _section(readme, heading):
    start = readme.index(f"\n## {heading}\n")
    end = readme.find("\n## ", start + 1)
    return readme[start:end if end != -1 else len(readme)]


SECTIONS = ["Install", "Quickstart", "Coming from Prophet", "1. A faster, lighter Prophet",
            "2. Slightly better forecasts", "3. Why: the optimum Prophet misses", "Limitations",
            "Documentation", "Paper & citing", "Licence"]


def test_the_structure_is_the_one_agreed(readme):
    headings = re.findall(r"^## (.+)$", readme, flags=re.M)
    assert headings == SECTIONS


def test_the_pitch_says_what_this_is_in_one_sentence(readme):
    """#213 repeats this sentence in the PyPI summary, CITATION.cff and the
    GitHub About, so it is the README's first line of prose. #230 chose it: the
    package is a faster, lighter Prophet."""
    first = next(line for line in readme.splitlines()[1:]
                 if line.strip() and not line.startswith("[!["))
    assert first == ("**Stan-free Prophet forecasting model using an analytical gradient "
                     "instead of AutoDiff: faster and lighter on memory.**")


# -- every number is generated and tested, or a stated constant -------------

def _prose_numbers(readme):
    """Every number a reader meets, outside code, link targets and badges."""
    text = re.sub(r"```.*?```", " ", readme, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return {token.rstrip(",") for token in
            re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:e-?\d+)?", text)}


# Generated: the paper's macros, so the README and the paper cannot disagree.
FROM_THE_PAPER = ("CensusSeries", "RmseAdvantage", "RmseWins", "RmseWinRate", "MultiTotal",
                  "SmoothFactorLong", "SmoothFactorShort", "LimNewtonSpeedup",
                  "LimOverheadShareMin", "LimOverheadShareMax", "LimParityMin", "LimParityMax",
                  "LimPoolSize", "LimBestCensus", "LimWorstPool", "LimWorstCensus",
                  "LimWidthMedian", "LimWidthWider", "LimCoverageOurs", "LimCoverageProphet",
                  "LimMfourSeries")

# Measured, and held to the committed results by the test named beside each.
TESTED_ELSEWHERE = {
    "40": "test_parity_surface: Prophet's public methods", "15": "test_parity_surface",
    "1e-13": "test_paper_agreement", "1e-9": "test_paper_agreement",
    "21": "test_paper_agreement: points measured",
    "1.4": "test_tier3_reporting: fit ratios at T = 1000 and 2905; and the samplers' "
           "disagreement, which the report quotes (below)",
    "1.7": "test_tier3_reporting", "2.6": "test_tier3_reporting",
    "1.6": "test_tier3_reporting", "40.2": "test_tier3_reporting",
    "0.3": "test_tier3_reporting", "3.6": "test_tier3_reporting",
    "0.52": "test_tier2_reporting", "0.43": "test_tier2_reporting",
    "0.48": "test_tier2_reporting", "1894": "test_tier2_reporting",
    "1807": "test_tier2_reporting", "1838": "test_tier2_reporting",
    "1929": "test_tier2_reporting", "3008": "test_tier2_reporting",
    "1e-8": "rounded: tests/test_backend_parity.py records 1.5e-8 at T = 2905",
}

# Not results: settings, structure, and dates.
CONSTANTS = {
    "1": "a section number, and ours / Prophet's - 1", "2": "a section number",
    "3": "a section number",
    "0": "T = 0", "17": "C++17; and Prophet's methods that are the same (test_parity_surface)",
    "8": "Prophet's methods that are module-level here (test_parity_surface); and "
         "\\SmoothFactorLong",
    "3.9": "the oldest Python CI runs", "3.14": "the newest Python CI runs",
    "50": "a series length measured", "100": "Prophet's rule: Newton below 100 observations",
    "1000": "a series length measured", "2905": "the full Peyton Manning series",
    "80": "Prophet's default interval_width, in percent",
    "2023": "the date of Prophet's post",
    "10": "the showcase's sMAPE threshold, in percent (showcase.FITS_AT_ALL)",
}


def test_every_number_in_the_readme_is_generated_tested_or_a_constant(readme, macros):
    paper = {macros[name] for name in FROM_THE_PAPER}
    unexplained = sorted(_prose_numbers(readme) - paper - set(TESTED_ELSEWHERE) - set(CONSTANTS))
    assert not unexplained, (
        f"numbers in the README with no generated source, test or stated reason: {unexplained}")


def test_the_constants_are_what_they_say(readme):
    """The few constants that are not structure, checked at their source."""
    from showcase import FITS_AT_ALL

    assert FITS_AT_ALL == 0.10 and "within 10% sMAPE" in _one_line(readme)
    report = _one_line((REPO / "evaluation" / "results" / "report.md").read_text())
    assert "about 1.4% on the interval bounds" in report
    assert "about 1.4% on the interval bounds" in _one_line(readme)


def test_the_paper_figures_are_quoted_where_they_belong(readme, macros):
    """Each macro the README carries, in the sentence that carries it rather
    than merely somewhere in the file."""
    one = _one_line(readme)
    for sentence in (
            f"On {macros['CensusSeries']} held-out M4 series its RMSE is lower on "
            f"{macros['RmseWins']} of them, by a median of {macros['RmseAdvantage']}%",
            f"lower on {macros['RmseWins']} of them ({macros['RmseWinRate']}%)",
            f"From {macros['MultiTotal']} random starts",
            f"brings the same L-BFGS {macros['SmoothFactorLong']} to "
            f"{macros['SmoothFactorShort']} times closer",
            f"| fit, T = 50 | **{macros['LimNewtonSpeedup']}× faster** |",
            f"takes {macros['LimOverheadShareMin']}–{macros['LimOverheadShareMax']}% of "
            f"Prophet's whole fit",
            f"takes {macros['LimParityMin']}–{macros['LimParityMax']} times as long",
            f"its RMSE is {macros['LimWorstCensus']}% above Prophet's, and on its best "
            f"{macros['LimBestCensus']}% below",
            f"ours is {macros['LimWorstPool']}% higher",
            f"wider by a median {macros['LimWidthMedian']}%, on {macros['LimWidthWider']} "
            f"of the {macros['CensusSeries']} series",
            f"mean coverage **{macros['LimCoverageOurs']}** for ours and "
            f"**{macros['LimCoverageProphet']}** for Prophet's",
            f"3,008 series of M4's {macros['LimMfourSeries']}",
            f"**{macros['LimPoolSize']} where a Prophet-shaped model fits at all**"):
        assert sentence in one, sentence


# -- no history, and no private vocabulary ----------------------------------

HISTORY = ("README claimed before", "this README", "flattering by accident", "At 36 series",
           "at 36 series", "36-series", "the README said so", "used to", "no longer",
           "since #", "before #", "Until [#")


@pytest.mark.parametrize("phrase", HISTORY)
def test_the_readme_does_not_talk_about_its_past(readme, phrase):
    """A new reader never saw the earlier versions; CHANGELOG.md has them."""
    assert phrase not in _one_line(readme)


def test_the_history_is_in_the_changelog():
    changelog = _one_line((REPO / "CHANGELOG.md").read_text())
    assert "## What the README used to say" in changelog
    for subject in ("flattering for that reason alone", "36-series sample",
                    "17.8% above the minimum", '"the mechanism"', "8.2× at T = 50"):
        assert subject in changelog, subject
    # and its introduction, which appeared twice, appears once
    assert changelog.count("Every item below was found and fixed with a regression test") == 1


def test_no_private_term_is_left_undefined(readme):
    """Tier 2, lp__, the census and T meant something only to whoever wrote
    the evaluation suite. Each is either gone or defined where it is first met."""
    prose = re.sub(r"\]\([^)]*\)", "]", readme)              # link targets carry tier anchors
    for tier in ("Tier 0", "Tier 1", "Tier 2", "Tier 3"):
        assert tier not in prose
    assert "lp__" not in prose
    one = _one_line(prose)
    first_census = one.index("census")
    assert "every weekly and daily series the evaluation protocol can measure" in \
        one[first_census:first_census + 160]
    first_t = re.search(r"\bT = \d", one).start()
    assert "T is the number of observations" in one[:first_t]


# -- the framing, the figures, the paper ------------------------------------

def test_the_framing_is_the_packages(readme):
    """#230: the package is a faster, lighter Prophet, and its forecasts are
    slightly better -- in that order, and sized as the paper sizes the gain.
    The pitch names the analytical gradient; section 1 still says how little of
    the speed it accounts for, which is mostly not calling CmdStan."""
    one = _one_line(readme)
    lede = one[one.index("**Stan-free"):one.index("**Status: early development.**")]
    faster = lede.index("faster and lighter on memory at every series length measured")
    better = lede.index("The forecasts are slightly better too")
    assert faster < better
    assert "a real improvement, and usually a small one" in lede
    assert "more accurate" not in one
    assert "analytic gradient was for" not in one
    assert "Very little of the speed is the gradient" in one


def test_the_build_story_is_told_once(readme):
    """It was in Install, in 'What it does', in 'Building from source' and in
    a Quickstart comment, and the last two predated the wheels."""
    one = _one_line(readme)
    assert one.count("compiles `optimize.cpp`") == 1
    assert "built on first use" not in one
    assert "compiled on demand from one source file" not in one


def test_each_figure_is_where_its_story_is(readme):
    optimum = _section(readme, "3. Why: the optimum Prophet misses")
    forecasts = _section(readme, "2. Slightly better forecasts")
    assert f"]({RAW}paper/figures/tau_path_readme.png)" in optimum
    assert (REPO / "paper" / "figures" / "tau_path_readme.png").exists()
    assert f"]({RAW}evaluation/results/figures/showcase.png)" in forecasts


def test_the_evidence_leads_with_what_a_reader_can_reproduce(readme):
    """The README quoted liblbfgs and a scipy run the docs mark as historical
    and not kept. Section 3 now leads with three results the paper
    regenerates, each linked into it."""
    bullets = [line for line in _section(readme, "3. Why: the optimum Prophet misses").splitlines()
               if line.startswith("- **")]
    assert len(bullets) == 3
    assert "liblbfgs" not in readme and "17.8%" not in readme
    # each links into the paper by its file, not by a section number, which the
    # paper's restructure (#204) and its discussion (#208) both move
    section = _section(readme, "3. Why: the optimum Prophet misses")
    assert section.count("paper/sections/evidence.tex)") == len(bullets)
    assert not re.search(r"\[§\s*\d", readme)


def test_the_paper_and_how_to_cite_it(readme):
    section = _one_line(_section(readme, "Paper & citing"))
    title = re.search(r"\\title\{(.*?)\}\n\\author", (PAPER / "main.tex").read_text(), re.S).group(1)
    title = _one_line(title.replace("\\\\", " "))
    assert f"*{title}*" in section
    assert "CITATION.cff" in section and "tree/main/paper/" in section
    # #230: the package is about speed and memory, the paper about the fix
    assert "The paper's subject is the fix, not the package's speed" in section


def test_why_a_separate_package_quotes_prophets_plan_verbatim(readme):
    """The sentence is recorded in paper/citations.md (#207); the quotation has
    to be a run of it, and keep "at this moment" (tests/test_paper_related.py)."""
    quoted = re.search(r'"(at this moment [^"]+)"', _one_line(readme)).group(1)
    record = _one_line((PAPER / "citations.md").read_text())
    assert quoted in record
    assert "Why a separate package" in _section(readme, "Coming from Prophet")
