"""
Issue #244: the margins in nats, given a scale.

Section 5.1 quoted Prophet's density as a share of the optimum's (6% at
T = 1000) and nothing said how many nats is a lot. The scale is the posterior's
own spread: a typical draw sits at least d/2 nats below the mode, d being the
number of parameters. Checked here:

  * **d is the fitted model's**, read from its layout at each of Table 1's
    lengths: the changepoint rates, the regressor columns, and k, m and sigma.
    The columns are the ones Tier 0 recorded both implementations fitting;
  * **the figures** follow from d and Tier 0's margins. d and d/2 are rows of
    Table 1, which #209 made the one place a per-length figure is quoted;
  * **the bound it states**, E f >= d/2 with the kinks included, holds where
    it can be integrated directly;
  * **the paragraph is in section 5.1**, every figure a macro.
"""
import csv
import importlib.util
import math
import re
from pathlib import Path

import pytest
from scipy import integrate

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EVIDENCE = PAPER / "sections" / "evidence.tex"
LENGTHS = {300: "Short", 1000: "Mid", 2905: "Long"}


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


@pytest.fixture(scope="module")
def models(generator):
    return {size: generator.dimensions(size) for size in LENGTHS}


@pytest.fixture(scope="module")
def paragraph():
    text = EVIDENCE.read_text()
    section = text[text.index(r"\label{sec:evidence:lp}"):text.index(r"\label{sec:evidence:kkt}")]
    start = section.index("The posterior's own spread gives these margins a scale.")
    return " ".join(section[start:section.index("\n\n", start)].split())


# -- d is the fitted model's ---------------------------------------------------

def test_d_is_the_layout_of_the_model_at_each_length(models, macros):
    for size, tag in LENGTHS.items():
        layout = models[size].layout
        assert layout.n_changepoints == 25, "Prophet's default, placed in full at every length"
        assert layout.size == layout.n_changepoints + layout.n_regressor_columns + 3
        assert macros[f"LpParams{tag}"] == f"{layout.size}"


def test_the_columns_are_the_ones_tier_0_fitted(models):
    """Tier 0 checks its design against Prophet's Stan data before it records a
    width, so d is the dimension of both fits, not only of this one."""
    recorded = {}
    for row in csv.DictReader(open(REPO / "evaluation" / "results" / "tier0_agreement.csv")):
        if row["metric"] == "design_columns" and row["implementation"] == "both":
            recorded[int(row["series"].split("[:")[1].rstrip("]"))] = float(row["value"])
    for size in LENGTHS:
        assert models[size].layout.n_regressor_columns == recorded[size]


def test_only_the_shortest_has_no_yearly_term(models):
    """What the paragraph gives as the reason d is smaller at T = 300."""
    assert "yearly" not in models[300].seasonalities
    assert all("yearly" in models[size].seasonalities for size in (1000, 2905))


# -- the figures the paragraph quotes ----------------------------------------

def test_the_scale_is_half_of_d(macros):
    for tag in LENGTHS.values():
        assert float(macros[f"LpTypical{tag}"]) == int(macros[f"LpParams{tag}"]) / 2


def test_the_margins_and_their_share_of_the_scale(generator, macros):
    margins = generator._margins()
    for size, tag in LENGTHS.items():
        assert float(macros[f"LpMargin{tag}"]) == pytest.approx(margins[size], abs=5e-4)
    assert macros["LpMarginLeast"] == f"{min(margins[size] for size in LENGTHS):.2g}"
    assert macros["LpMarginMost"] == f"{max(margins[size] for size in LENGTHS):.2g}"
    shares = [margins[size] / float(macros[f"LpTypical{tag}"]) for size, tag in LENGTHS.items()]
    assert macros["LpTypicalShare"] == f"{100 * max(shares):.0f}"
    # what "small on the posterior's own scale" claims
    assert max(shares) < 0.25


# -- the bound it states -------------------------------------------------------

def _mean_drop(curvature, kink, slope):
    """E f under e^{-f}, f(u) = curvature u^2 / 2 + kink |u| + slope u: one
    coordinate of the paragraph's local model, a zero rate when kink > 0."""
    def f(u):
        return curvature * u * u / 2 + kink * abs(u) + slope * u
    mass = integrate.quad(lambda u: math.exp(-f(u)), -math.inf, 0)[0] + \
        integrate.quad(lambda u: math.exp(-f(u)), 0, math.inf)[0]
    first = integrate.quad(lambda u: f(u) * math.exp(-f(u)), -math.inf, 0)[0] + \
        integrate.quad(lambda u: f(u) * math.exp(-f(u)), 0, math.inf)[0]
    return first / mass


@pytest.mark.parametrize("curvature, kink, slope", [
    (1.0, 0.0, 0.0), (4.0, 0.0, 0.0),            # Gaussian: exactly 1/2
    (1.0, 0.5, 0.0), (1.0, 3.0, 0.0), (0.2, 1.0, 0.0),
    (1.0, 2.0, 1.5), (1.0, 2.0, -2.0),           # the first-order term, |slope| <= kink
])
def test_the_kinks_only_raise_the_average_drop(curvature, kink, slope):
    """Per coordinate the paragraph's E f = d - E q >= d/2 reads E f >= 1/2,
    with equality only without a kink."""
    drop = _mean_drop(curvature, kink, slope)
    if kink == 0:
        assert drop == pytest.approx(0.5, abs=1e-8)
    else:
        assert drop > 0.5 + 1e-3
        assert drop < 1.0 + 1e-8, "the pure-kink limit, one nat per coordinate"


# -- the paragraph -------------------------------------------------------------

def test_d_and_its_half_are_rows_of_table_1(generator):
    """Under Prophet's estimate against the corrected optimum, beside the
    density ratios, every cell a macro (tests/test_paper_table.py)."""
    labels = [label for label, _ in generator.TABLE_ROWS]
    first_group = labels[:labels.index(next(label for label, cell in generator.TABLE_ROWS[1:]
                                            if cell is None))]
    cells = dict(generator.TABLE_ROWS)
    assert cells["parameters, $d$"] == r"\LpParams{t}" and "parameters, $d$" in first_group
    row = r"\quad a typical posterior draw below the mode, at least $d/2$"
    assert cells[row] == r"\LpTypical{t}" and row in first_group
    table = (PAPER / "generated" / "lengths.tex").read_text()
    assert r"\LpParamsShort & \LpParamsMid & \LpParamsLong" in table
    assert r"\LpTypicalShort & \LpTypicalMid & \LpTypicalLong" in table


def test_the_paragraph_is_in_section_5_1_with_its_figures_as_macros(paragraph):
    for name in ("LpMarginLeast", "LpMarginMost", "LpTypicalShare"):
        assert f"\\{name}" in paragraph, name
    assert r"\cref{tab:lengths} gives $d$ and $d/2$ at each length" in paragraph
    assert r"\mathbb{E} f = d - \mathbb{E} q \ge d/2" in paragraph
    assert r"\cref{sec:consequences}" in paragraph
    # nothing typed but the mathematics: the half of the quadratic, the
    # chi-squared's square, d/2, 2q and ell >= 0. No figure.
    typed = re.findall(r"(?<![\\\w])\d+(?:\.\d+)?", re.sub(r"\\[A-Za-z]+", "", paragraph))
    assert sorted(set(typed)) == ["0", "1", "2"], typed
