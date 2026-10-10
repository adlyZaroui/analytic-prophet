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
    generated_style = re.compile(r"\\((?:Census|Rmse|Lp|Kkt|Margin)\w+)")
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
