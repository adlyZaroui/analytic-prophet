"""
Issue #209: one generated table for the per-length results, and one symbol per
meaning.

Section 5 quoted its results as "X, Y and Z at the three lengths" about a
dozen times. They are one table now, `paper/generated/lengths.tex`, written by
`quote_numbers.py` beside the macros, with a macro in every cell so that each
number keeps one source. The notation had four symbols with two or three
meanings each, and two more turned up on the way. Checked here:

  * **the table** is what the generator writes, every cell is a generated
    macro, the paper includes it, and no section quotes a per-length triple;
  * **the per-length Newton figures** it adds are the committed results;
  * **one symbol per meaning**: s is the changepoint times, F the smooth part,
    theta the full parameter vector and psi the rest, phi the split map,
    epsilon the smoothing parameter, g the trend, and p a p-value.
"""
import csv
import importlib.util
import re
from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
SECTIONS = PAPER / "sections"
TABLE = PAPER / "generated" / "lengths.tex"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "quote_numbers", PAPER / "experiments" / "quote_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def macros(generator):
    return generator.macros()


def _section(name):
    return (SECTIONS / f"{name}.tex").read_text()


def _math(text):
    """The math in a section: what is between dollar signs, and equations."""
    inline = re.findall(r"(?<!\\)\$(.+?)(?<!\\)\$", text, flags=re.S)
    display = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", text, flags=re.S)
    return " ".join(inline + display)


# -- the table ----------------------------------------------------------------

def test_the_committed_table_is_what_the_generator_writes(generator):
    assert TABLE.read_text() == generator.render_table(), (
        "paper/generated/lengths.tex is stale -- run `python paper/experiments/quote_numbers.py`")


def test_every_cell_is_a_generated_macro(macros):
    table = TABLE.read_text()
    cells = set(re.findall(r"\\([A-Z][A-Za-z]+)", table))
    structural = {"toprule", "midrule", "bottomrule", "multicolumn", "textit", "quad"}
    named = {name for name in cells if name not in structural}
    assert named, "the table has no macros in it"
    assert named <= set(macros), f"not generated: {sorted(named - set(macros))}"
    for line in table.splitlines():
        if "&" in line and not line.lstrip().startswith("&"):
            for cell in line.rstrip("\\ ").split("&")[1:]:
                assert not re.search(r"(?<![\\\w{^-])\d", re.sub(r"\\[A-Za-z]+", "", cell)), (
                    f"a typed number in a cell: {cell!r}")


def test_the_paper_includes_the_table_and_points_to_it():
    evidence = _section("evidence")
    float_ = evidence[evidence.index(r"\begin{table}"):evidence.index(r"\end{table}")]
    assert r"\input{generated/lengths}" in float_ and r"\label{tab:lengths}" in float_
    assert evidence.count(r"\cref{tab:lengths}") + evidence.count(r"\Cref{tab:lengths}") >= 6


def test_no_section_quotes_a_per_length_triple():
    """What the table replaced: one quantity at all three lengths, in prose."""
    for path in sorted(SECTIONS.glob("*.tex")):
        families = defaultdict(set)
        for prefix, tag in re.findall(r"\\([A-Z][A-Za-z]*?)(Short|Mid|Long)\b", path.read_text()):
            families[prefix].add(tag)
        triples = sorted(prefix for prefix, tags in families.items()
                         if tags == {"Short", "Mid", "Long"})
        assert not triples, f"{path.name} quotes {triples} at all three lengths"


def test_the_range_the_prose_quotes_is_a_range(macros):
    """'between \\SmoothFactorLong and \\SmoothFactorShort times closer', and the
    README's '8 to 1,592': the long series gains least, the short most."""
    factor = {tag: float(macros[f"SmoothFactor{tag}"].replace(",", ""))
              for tag in ("Short", "Mid", "Long")}
    assert factor["Long"] == min(factor.values())
    assert factor["Short"] == max(factor.values())


def test_the_per_length_newton_figures_are_the_committed_results(macros):
    with open(PAPER / "results" / "newton.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for size, tag in ((300, "Short"), (1000, "Mid"), (2905, "Long")):
        shortfall = {row["arm"]: float(row["shortfall"]) for row in rows
                     if int(row["observations"]) == size}
        assert macros[f"NewtonNatural{tag}"] == f"{shortfall['natural']:.1f}"
        assert macros[f"NewtonStan{tag}"] == f"{shortfall['stan_newton']:.2g}"
        # Stan's comes closer at every length, which the prose says of all six
        assert shortfall["stan_newton"] < shortfall["natural"]


# -- one symbol per meaning ---------------------------------------------------

def test_epsilon_is_the_smoothing_parameter_only():
    """It was also the prior scale of sigma, and a small step in a proof."""
    for name in ("model", "reformulation"):
        assert r"\varepsilon" not in _math(_section(name)), name
    assert r"\sigma_\sigma" in _math(_section("model"))
    assert r"\varepsilon" in _math(_section("evidence"))


def test_s_is_the_changepoint_times_only():
    """It was also the smooth part (now F) and the split map (now phi)."""
    for path in sorted(SECTIONS.glob("*.tex")):
        math = _math(path.read_text())
        assert not re.search(r"(?<![\\\w])s\(", math), path.name
        assert not re.search(r"\\partial s\b", math), path.name
    assert r"\varphi(\delta" in _math(_section("reformulation"))
    assert r"\partial F/\partial" in _math(_section("evidence"))


def test_theta_is_the_full_parameter_vector_only():
    """Section 2's (k, m, delta, beta, sigma); the rest of the parameters are psi."""
    for path in sorted(SECTIONS.glob("*.tex")):
        if path.name != "model.tex":
            assert r"\theta" not in path.read_text() and r"\Theta" not in path.read_text(), path.name
    assert r"\psi = (k, m, \beta, \sigma)" in _section("reformulation")


def test_g_is_the_trend_and_p_a_p_value():
    reformulation = _math(_section("reformulation"))
    assert not re.search(r"(?<![\\\w])g(_j)?\b", reformulation)
    assert r"\mathbb{R}^p" not in reformulation
