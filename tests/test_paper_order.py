"""
Issue #204: the paper is about the kink, the fix comes before the evidence,
and the implementation is one subsection of the fix.

The evidence used "the corrected optimum" and "the reformulation" a section
before defining them, and "this implementation" first appeared in a figure
caption. Checked here:

  * **the order** -- introduction, model, why smooth optimizers fail, the fix,
    the evidence, consequences, discussion (#208), limitations, related work,
    conclusion (#208), reproducibility, and the forecast protocol as an
    appendix;
  * **every term is defined before its first use**, read in that order: the
    terms the evidence leans on, the units and names it quotes in, and the
    error measures the forecast comparison reports;
  * **the implementation paragraph** says what the issue asked: it fits Stan's
    model (#201's figures), it never grades itself, why its solver runs outside
    Stan (#202's result), and that its hand-derived gradient is a convenience,
    with Stan's own reachable through `log_prob` or BridgeStan;
  * **the two cuts** -- no item on speed in the limitations, no liblbfgs in the
    reproducibility section -- and the overhead figures still generated for the
    README, which quotes them.
"""
import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
SECTIONS = PAPER / "sections"

ORDER = ["introduction", "model", "smooth", "reformulation", "evidence", "consequences",
         "discussion", "limitations", "related", "conclusion", "reproducibility"]
APPENDICES = ["appendix_forecast"]


def _inputs():
    main = (PAPER / "main.tex").read_text()
    body, appendix = main.split("\\appendix")
    return (re.findall(r"\\input\{sections/(\w+)\}", body),
            re.findall(r"\\input\{sections/(\w+)\}", appendix))


def _reading(names):
    """The sections as a reader meets them: comments gone, whitespace one space,
    and references and labels reduced to nothing, since `\\cref{sec:reformulation}`
    names a section rather than using its term."""
    text = " ".join((SECTIONS / f"{name}.tex").read_text() for name in names)
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(?:label|ref|cref|Cref|eqref|cite\w*)\s*(?:\[[^\]]*\]\s*)*\{[^}]*\}",
                  " ", text)
    return re.sub(r"\s+", " ", text)


def test_the_sections_are_in_the_agreed_order():
    body, appendix = _inputs()
    assert body == ORDER
    assert appendix == APPENDICES


# (what a reader meets, the passage that defines it). No occurrence of a term,
# in reading order, may come before its definition. The abstract is a summary
# and is not read here; the introduction is, so a term it uses before the body
# defines it has to be defined there.
TERMS = [
    (r"\bMAP\b", "maximum a~posteriori (MAP)"),
    (r"L-BFGS", "the limited-memory quasi-Newton method L-BFGS"),
    (r"corrected optimum", r"\emph{corrected optimum}"),
    (r"[Tt]his implementation", r"\emph{this implementation}"),
    (r"census", "a frozen census of"),
    (r"\bRMSE\b", "root mean squared error (RMSE)"),
    (r"default configuration", r"\emph{default configuration}"),
    (r"Peyton Manning", r"\emph{Peyton Manning series}"),
    (r"L-BFGS-B", "L-BFGS-B, the bound-constrained form of L-BFGS"),
    (r"[Ss]plit", "The split problem is"),
    (r"natural parameteri", r"is the \emph{natural} one"),
    (r"initialisation", "Prophet's own initialisation, in which every rate is zero"),
    (r"\\lp\b", r"written $\lp$ here"),
    (r"\bnats?\b", "given in nats, the unit of a natural logarithm"),
    (r"relaxed residual", "The relaxed residual grants that to every rate at once"),
    (r"convergence gap", r"The \emph{convergence gap}"),
    (r"smoothing bias", r"the \emph{smoothing bias}"),
    (r"exact path", "the exact path is computed by"),
    (r"\bsMAPE\b", "symmetric mean absolute percentage error (sMAPE)"),
    (r"\bMAE\b", "mean absolute error (MAE)"),
    (r"(?<![A-Za-z])MAPE\b", "mean absolute percentage error (MAPE)"),
]


def test_every_term_is_defined_before_its_first_use():
    text = _reading(ORDER + APPENDICES)
    late = []
    for pattern, definition in TERMS:
        defined = text.find(definition)
        assert defined >= 0, f"no definition {definition!r}"
        first = re.search(pattern, text).start()
        if first < defined:
            late.append((pattern, text[max(0, first - 60):first + 40]))
    assert not late, f"used before it is defined: {late}"


def test_the_reformulation_is_defined_before_the_evidence_uses_it():
    """The introduction announces it; in the body, its first mention is the fix's."""
    body = _reading(ORDER[1:])
    fix = _reading(["reformulation"]).strip()
    assert body.index(fix) <= body.index("reformulation")
    assert body.index(fix) < body.index(_reading(["evidence"]).strip())


def _implementation():
    text = re.sub(r"\s+", " ", (SECTIONS / "reformulation.tex").read_text())
    return text[text.index(r"\subsection{The implementation}"):]


def test_the_implementation_is_part_of_the_fix():
    text = (SECTIONS / "reformulation.tex").read_text()
    assert text.startswith("\\section{The fix: an exact reformulation}")
    assert text.index("\\begin{proposition}") < text.index("\\label{sec:implementation}")


def test_the_implementation_paragraph_says_what_the_issue_asks():
    paragraph = _implementation()
    # it fits Stan's model: #201's figures, generated
    assert "It fits Stan's model, not a transcription of it" in paragraph
    for macro in ("KktStanPoints", "KktStanObjectiveRelative", "KktStanGradientRelative"):
        assert f"\\{macro}" in paragraph, macro
    # it never grades itself
    assert "Nor does it grade itself" in paragraph
    assert "no claim depends on this implementation's gradient being right" in paragraph
    # why a solver outside Stan: #202's result for the split, generated
    assert "The solver runs outside Stan because the fix cannot run inside it" in paragraph
    for macro in ("InsideSplitStartMin", "InsideSplitStartMax", "InsideSplitFloorWorst"):
        assert f"\\{macro}" in paragraph, macro
    assert "\\code{paper/experiments/inside_stan.py}" in paragraph
    # a gradient outside Stan need not be analytic
    assert "\\code{log_prob}" in paragraph and "\\citep{roualdes2023bridgestan}" in paragraph
    assert "that is a convenience, and nothing in this paper depends on it" in paragraph


def test_the_gradient_formulas_are_not_in_the_paper():
    """The issue's last item: the formulas go in an appendix "if kept". They were
    never in the paper, and the paper does not claim the gradient, so they stay
    where they are written out and tested: `_gradient` and `_split_gradient` in
    forecaster.py, and the growth modes' derivatives in trend.py."""
    _body, appendix = _inputs()
    assert appendix == ["appendix_forecast"]
    package = REPO / "analytic_prophet"
    forecaster = (package / "forecaster.py").read_text()
    assert "def _gradient(" in forecaster and "def _split_gradient(" in forecaster
    assert "derivatives the analytic gradient needs" in (package / "trend.py").read_text()


def test_the_item_on_speed_is_cut():
    """The paper makes no claim about speed, so it carries no caveat about it."""
    limitations = (SECTIONS / "limitations.tex").read_text()
    assert "speed" not in limitations.lower()
    tex = "\n".join(p.read_text() for p in [PAPER / "main.tex", *SECTIONS.glob("*.tex")])
    for word in ("faster", "speed-up", "speedup", "speed"):
        assert word not in tex.lower(), word


def test_the_overhead_figures_are_still_generated_for_the_readme():
    """overhead.py stays because the README's cost section quotes it
    (tests/test_readme_structure.py holds the README to these macros)."""
    spec = importlib.util.spec_from_file_location(
        "quote_numbers", PAPER / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    macros = module.macros()
    tex = "\n".join(p.read_text() for p in SECTIONS.glob("*.tex"))
    for name in ("LimOverheadShareMin", "LimOverheadShareMax", "LimParityMin", "LimParityMax"):
        assert name in macros, name
        assert f"\\{name}" not in tex, f"the paper quotes {name} again"
    readme = " ".join((REPO / "README.md").read_text().split())
    assert f"takes {macros['LimParityMin']}–{macros['LimParityMax']} times as long" in readme
    # the speed-up at T = 50 went once nothing quoted it (#241)
    assert "LimNewtonSpeedup" not in macros


def test_the_liblbfgs_paragraph_is_cut():
    """Project history the paper does not rely on; the record stays in
    docs/non-smooth-objective.md."""
    for path in [PAPER / "main.tex", *SECTIONS.glob("*.tex")]:
        assert "liblbfgs" not in path.read_text(), path.name
    assert "liblbfgs" in (REPO / "docs" / "non-smooth-objective.md").read_text()
