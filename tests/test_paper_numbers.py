"""
Issue #167: the paper's quoted numbers are generated, never typed.

The census landing on `main` turned the suite red because the README quoted the
36-series sample after the corpus had become 3,008 series. The paper is going
to quote many of the same figures, so it does not get the chance:
`paper/experiments/quote_numbers.py` writes them as LaTeX macros from the
committed results, and this checks the committed `generated/numbers.tex` is
what that script produces now.

`generated/numbers.tex` is committed rather than built on demand because arXiv
compiles the source as uploaded -- it does not run Python -- so the macros have
to be in the tarball.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
NUMBERS = PAPER / "generated" / "numbers.tex"


def _macros(text):
    # one level of nested braces, for a value such as 5.9\times10^{-8}
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", text))


def _generator():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "quote_numbers", PAPER / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_numbers_are_what_the_results_give():
    """Regenerating must reproduce the committed file. A difference means an
    evaluation result moved and the paper still quotes the old value."""
    generator = _generator()
    current = generator.macros()
    committed = _macros(NUMBERS.read_text())

    stale = {name: (committed.get(name), value)
             for name, value in current.items() if committed.get(name) != value}
    assert not stale, (
        "paper/generated/numbers.tex is stale -- run "
        f"`python paper/experiments/quote_numbers.py`. Changed: {stale}")
    assert set(committed) == set(current), "macros added or removed"
    assert NUMBERS.read_text() == generator.render(current), (
        "the committed file is not byte-identical to what the generator writes")


def test_every_macro_the_paper_uses_is_generated():
    """A macro the prose uses but the generator does not write would fail the
    LaTeX build; this says which, faster and by name."""
    defined = set(_macros(NUMBERS.read_text()))
    sources = [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]
    generated_style = re.compile(r"\\((?:Census|Rmse|Lp|Kkt|Margin|Newton|Lim|Smooth|Stall|Tau|Multi|Flat|Tol|Strength|Env)\w+)")
    used = {name for path in sources
            for name in generated_style.findall(path.read_text())}

    assert used <= defined, f"used but not generated: {sorted(used - defined)}"


def test_no_section_types_a_census_figure_by_hand():
    """The specific figures that went stale in the README must reach the paper
    only through a macro."""
    for path in sorted((PAPER / "sections").glob("*.tex")):
        text = path.read_text()
        for literal in ("3,008", "3008", "0.40\\%", "64\\%", "1,929"):
            assert literal not in text, (
                f"{path.name} types {literal!r} by hand; use the generated macro")


# Numbers the prose may type, because they are not results: Prophet's and
# Stan's settings, the design of an experiment, a version, a date in a thread's
# history. Anything else typed into a section is a result that has not been
# generated (#175) -- add a macro in quote_numbers.py, or, if it really is a
# constant, add it here with the reason.
TYPED_CONSTANTS = {
    "0": "structure: zero rates, (0, 1), sign(0)",
    "1": "structure: exponents, sigma = 1, 1/tau",
    "2": "structure: squares, factors of two, delta_j^2",
    "20": "1/tau at Prophet's default tau = 0.05",
    "25": "Prophet's default n_changepoints",
    "33": "the T below which Prophet places fewer than 25 changepoints",
    "99": "the last T at which Prophet runs Newton",
    "100": "Prophet's rule: Newton below 100 observations",
    "300": "a length of the lp tables",
    "1000": "a length of the lp tables",
    "2905": "the full Peyton Manning series",
    "365": "the forecast horizon in days",
    "730": "Prophet's rule: yearly seasonality from 730 days",
    "10,000": "Prophet's iteration budget, int(1e4)",
    "0.01": "Prophet's plotting threshold; prophet#1280's tau",
    "0.05": "Prophet's default changepoint_prior_scale",
    "80": "Prophet's default interval_width, in percent",
    "1950": "a decade, in related work",
    "2019": "dates in prophet#842's history, recorded in citations.md",
    "04": "2019-04-22", "22": "2019-04-22", "05": "2019-05-21", "21": "2019-05-21",
    "10^{-2}": "a point of the smoothing grid",
    "10^{-3}": "a threshold chosen for the text; Stan's Newton step",
    "10^{-4}": "a point of the smoothing grid",
    "10^{-6}": "the smallest lambda of the strength grid",
    "2\\times10^{-3}": "twice Stan's Newton step",
}


def _prose(text):
    """A section with what is not the authors' own figures removed."""
    text = re.sub(r"(?<!\\)%.*", "", text)                      # comments
    text = re.sub(r"``.*?''", " ", text, flags=re.S)            # a source's words
    for command in (r"cite\w*", "label", "ref", "cref", "Cref", "eqref", "prophetissue",
                    "projectissue", "code", "includegraphics", "input", "begin", "end",
                    "pending", "href"):
        text = re.sub(r"\\" + command + r"\s*(?:\[[^\]]*\])*\{[^}]*\}", " ", text)
    return text.replace("\\,", ",")


def test_every_number_typed_into_the_prose_is_a_constant():
    """#175: every result the paper quotes is generated. What a section may
    still type is listed above with its reason."""
    token = re.compile(r"\d+(?:\.\d+)?\\times10\^\{-?\d+\}|10\^\{-?\d+\}"
                       r"|(?<![\w\\^{])\d[\d,]*(?:\.\d+)?")
    typed = {}
    for path in sorted((PAPER / "sections").glob("*.tex")):
        for match in token.finditer(_prose(path.read_text())):
            typed.setdefault(match.group(0).rstrip(","), set()).add(path.name)
    unexplained = {number: sorted(files) for number, files in typed.items()
                   if number not in TYPED_CONSTANTS}
    assert not unexplained, (
        f"numbers typed into the paper that are not listed constants: {unexplained}")
