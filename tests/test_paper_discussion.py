"""
Issue #208: a discussion and a short conclusion, with every number generated.

The paper ended on reproducibility. It now has a discussion after the
consequences, covering what a Prophet user can do today, how Prophet could
adopt the fix with its Stan model frozen, how far the result reaches, and what
is out of scope. A short conclusion follows related work. Two experiments back
the discussion:

  * `newton_census.py`: Prophet's own `algorithm="Newton"` on the census sample
    `rounding.py` draws, beside its default L-BFGS and the corrected optimum;
  * `polish.py`: the split problem solved from Prophet's own solution with
    nothing but `log_prob` on Prophet's unmodified program.

Checked here: each result is complete, says what the text says of it, and
reproduces where it was recorded; and the sources the discussion leans on --
Prophet's R package and its Python sampling path -- are recorded as read.
"""
import csv
import importlib.util
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import recorded_here

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"
RESULTS = PAPER / "results"
RECORD = PAPER / "citations.md"


def _module(name):
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discussion():
    return " ".join((PAPER / "sections" / "discussion.tex").read_text().split())


@pytest.fixture(scope="module")
def census():
    return _module("newton_census").read()


@pytest.fixture(scope="module")
def polished():
    return _module("polish").read()


# -- Prophet's own Newton, on the census sample ---------------------------------

def test_the_census_run_is_the_rounding_sample_and_complete(census):
    with open(RESULTS / "rounding.csv", newline="") as handle:
        rounding = {row["series"] for row in csv.DictReader(handle)}
    assert {row["series"] for row in census} == rounding
    assert len(census) == _module("newton_census").SAMPLE
    assert not [row["series"] for row in census if row["newton_error"]]


def test_what_the_discussion_says_of_prophets_newton_holds(census):
    text = _discussion()
    # "closer on every one of them"
    assert "closer on every one of them" in text
    assert all(row["shortfall_newton"] < row["shortfall_lbfgs"] for row in census)
    # "no rate comes out exactly zero", and the residual says it is not the optimum
    assert "no rate comes out exactly zero" in text
    assert all(row["exact_zeros_newton"] == 0 for row in census)
    assert statistics.median(row["relaxed_newton"] for row in census) > 0.01
    # it is slower: the ratio the text quotes is a slowdown on most series
    ratios = [row["seconds_newton"] / row["seconds_lbfgs"] for row in census]
    assert statistics.median(ratios) > 2
    # #240: measured from the converged optimum, Newton stays below it everywhere.
    # From the default fit it scored above on 15 series, which is what that
    # issue found: the default fit stops by Stan's tests a little short.
    optimum = {row["series"]: row["lp_converged"] for row in _module("converged").read()}
    assert all(optimum[row["series"]] > row["lp_newton"] for row in census)
    assert "it stays below it on every series" in text
    assert "even scores above the corrected optimum" not in text


def test_the_census_macros_are_these_results(census):
    macros = _module("quote_numbers").macros()
    optimum = {row["series"]: row["lp_converged"] for row in _module("converged").read()}
    newton = [optimum[row["series"]] - row["lp_newton"] for row in census]
    # closer than the corrected optimum, this implementation's default fit
    above = sum(1 for row in census if row["shortfall_newton"] < 0)
    assert macros["NewtonCensusSeries"] == f"{len(census)}"
    assert macros["NewtonCensusAbove"] == f"{above}"
    assert macros["NewtonCensusNewtonMedian"] == f"{statistics.median(newton):.2g}"
    assert float(macros["NewtonCensusLbfgsMedian"]) > 10 * float(macros["NewtonCensusNewtonMedian"])


# -- the fix from outside Prophet's frozen Stan model ---------------------------

def test_the_polish_reaches_the_optimum_through_log_prob_alone(polished):
    assert [row["observations"] for row in polished] == list(_module("polish").LENGTHS)
    for row in polished:
        assert row["shortfall_prophet"] > 0.1, row                 # it started short
        # and finished at the corrected optimum or just above it (a tighter tolerance)
        assert -1e-3 < row["shortfall_polished"] <= 1e-6, row
        assert row["exact_zeros_polished"] == row["exact_zeros_optimum"] > 0, row
        assert row["relaxed_polished"] < 0.1 * row["relaxed_prophet"], row
        assert row["calls"] > 0 and row["seconds"] > 0
    text = _discussion()
    assert "with as many rates exactly zero as the optimum has" in text
    assert "ends at the corrected optimum or just above it" in text
    assert r"\code{paper/experiments/polish.py}" in text


def test_the_frozen_models_gradient_is_the_split_problems(prophet_comparison):
    """The step's whole claim is that log_prob alone gives the split problem's
    gradient: checked against central differences at an interior point."""
    polish = _module("polish")              # sets up the paths kkt needs
    import corpora
    import harness
    import pandas as pd

    Prophet, _common, _bridge = prophet_comparison
    df = corpora.peyton_manning(300)
    df = df.assign(ds=pd.to_datetime(df["ds"]))
    stan_model, stan_data, theirs = harness.capture_stan_model(
        Prophet(**harness.PROPHET_KWARGS), df, sig_figs=18)
    S, K = len(theirs["delta"]), len(theirs["beta"])
    model = polish.FrozenModel(stan_model, stan_data, S, K)
    z = model.pack(theirs["k"][0], theirs["m"][0], theirs["delta"], theirs["sigma_obs"][0],
                   theirs["beta"])
    rng = np.random.default_rng(208)
    z = z + rng.normal(scale=1e-3, size=z.size)
    z[2:2 + 2 * S] = np.abs(z[2:2 + 2 * S]) + 1e-3                # inside the bounds
    _, gradient = model(z)
    for i in (0, 1, 2, 2 + S, 2 + 2 * S, z.size - 1):            # k, m, d+, d-, zeta, beta
        h = 1e-6 * max(1.0, abs(z[i]))
        up, down = z.copy(), z.copy()
        up[i] += h
        down[i] -= h
        numeric = (model(up)[0] - model(down)[0]) / (2 * h)
        assert numeric == pytest.approx(gradient[i], rel=1e-5, abs=1e-5), i


def test_the_code_reproduces_the_shortest_polish(polished, prophet_comparison,
                                                compiled_optimizer_module):
    """T = 300 again. The polish starts from Prophet's L-BFGS endpoint, which is
    pinned by nothing across platforms, so it is compared only where it was
    recorded (conftest.recorded_here); anywhere, it must reach the optimum."""
    polish = _module("polish")
    fresh = polish.measure(300, compiled_optimizer_module)
    assert -1e-3 < fresh["shortfall_polished"] <= 1e-6
    assert fresh["exact_zeros_polished"] == fresh["exact_zeros_optimum"]
    committed = polished[0]
    if recorded_here(polish.RESULTS.with_suffix(".meta.json")):
        assert fresh["lp_prophet"] == pytest.approx(committed["lp_prophet"], abs=1e-9)
        assert fresh["calls"] == committed["calls"]


# -- the sources the discussion leans on -----------------------------------------

def test_prophets_r_package_is_recorded_as_read():
    """'fits the same program the same way': the lines that say so, verbatim."""
    record = " ".join(RECORD.read_text().split())
    for line in ("algorithm = if(dat$T < 100) {'Newton'} else {'LBFGS'},",
                 "iter = 1e4,",
                 "Optimization terminated abnormally. Falling back to Newton optimizer.",
                 "if (m$mcmc.samples > 0) {",
                 "only by three blank lines"):
        assert line in record, line
    macros = _module("quote_numbers").macros()
    assert f"tag `v{macros['EnvProphet']}`" in record
    assert r"In version \EnvProphet{} of Prophet's repository" in _discussion()


def test_the_sampling_path_is_recorded_as_read():
    record = " ".join(RECORD.read_text().split())
    assert "elif self.mcmc_samples > 0:" in record
    assert "self.stan_fit = self.model.sample(**args)" in record
    assert "No-U-Turn sampler" in _discussion()


def test_the_conclusion_restates_and_adds_nothing_new():
    """Every macro the conclusion uses is one an earlier section uses."""
    sections = PAPER / "sections"
    conclusion = (sections / "conclusion.tex").read_text()
    earlier = "\n".join((sections / f"{name}.tex").read_text()
                        for name in ("introduction", "evidence", "consequences"))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", conclusion)) - {"Cref"}
    assert used and used <= set(re.findall(r"\\([A-Z][A-Za-z]+)", earlier)), used
