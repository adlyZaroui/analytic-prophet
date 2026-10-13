"""
Issue #175: every result the paper quotes is regenerable from paper/.

Four results the paper stated without a script behind them now have one, and
each is held here to what the paper says of it:

  * **flat growth** (`flat.py`) -- the negative control's tie, measured
    rather than asserted: every rate exactly zero on both sides, lp within a
    stated bound;
  * **the tolerances** (`tolerance.py`) -- Prophet's L-BFGS with all five
    stopping tests at machine precision still stops short, the run the notes
    reported reproduces, and Prophet itself cannot make it;
  * **the prior's strength** (`strength.py`) -- the replacement for the lost
    liblbfgs sweep: SciPy's L-BFGS-B reaches the optimum where it has no zero
    rate and stops short as the zeros arrive, against a reference that is
    exact and certified;
  * **the rounding** (`rounding.py`) -- what CmdStan's eight printed digits
    do to Prophet's residual, on the same run printed twice.

And the exact zeros the model section quotes come from the certificate's
committed results, with Prophet's count checked to be the "none" the text says.
"""
import csv
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import recorded_here

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"
RESULTS = PAPER / "results"


def _load(name):
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flat = _load("flat")
tolerance = _load("tolerance")
strength = _load("strength")
rounding = _load("rounding")


def _text(name):
    return " ".join((PAPER / "sections" / f"{name}.tex").read_text().split())


# -- flat growth --------------------------------------------------------------

def test_flat_growth_is_complete_and_every_rate_is_exactly_zero():
    rows = flat.read()
    assert {(r["observations"], r["implementation"]) for r in rows} == {
        (size, side) for size in flat.LENGTHS for side in ("prophet", "analytic_prophet")}
    for row in rows:
        assert row["exact_zeros"] == row["changepoints"] == 25, row
        assert row["max_abs_delta"] == 0.0
    assert "every one of the 25 rates is exactly zero" in _text("evidence")


def test_flat_growth_agrees_far_inside_the_margins():
    """The tie is a bound, and the bound is orders of magnitude below the
    linear-growth margins it is set against (#173's smallest is tenths of a nat)."""
    rows = flat.read()
    assert max(r["lp_gap"] for r in rows) < 1e-4
    assert max(max(r["m_gap"], r["beta_gap"]) for r in rows) < 1e-4


def test_flat_growth_refits_as_committed(compiled_optimizer_module, prophet_comparison):
    """T = 300 again, through both implementations, by conftest.recorded_here's
    rule. Under flat growth both reach the optimum, every rate exactly zero, so
    the point is pinned by the problem and compared anywhere; here, exactly."""
    committed = {r["implementation"]: r for r in flat.read() if r["observations"] == 300}
    for row in flat.measure(300, compiled_optimizer_module):
        assert row["exact_zeros"] == 25
        old = committed[row["implementation"]]["lp"]
        assert row["lp"] == pytest.approx(old, abs=1e-4)
        if recorded_here(flat.RESULTS.with_suffix(".meta.json")):
            assert row["lp"] == pytest.approx(old, abs=1e-9)


# -- the tolerances -----------------------------------------------------------

@pytest.fixture(scope="module")
def tolerance_rows():
    return tolerance.read()


def test_the_tolerance_runs_are_complete_and_certified(tolerance_rows):
    assert {(r["configuration"], r["observations"], r["tolerances"]) for r in tolerance_rows} == {
        (*cell, name) for cell in tolerance.CELLS for name in tolerance.SETTINGS}
    assert all(r["reference_residual"] < 1e-5 for r in tolerance_rows)


def test_removing_every_tolerance_still_stops_short(tolerance_rows):
    """'Nor is the margin a matter of Prophet's stopping tolerances.'"""
    for row in tolerance_rows:
        assert row["shortfall"] > 0.1, row
    for cell in tolerance.CELLS:
        runs = {r["tolerances"]: r for r in tolerance_rows
                if (r["configuration"], r["observations"]) == cell}
        assert runs["five"]["iterations"] > runs["default"]["iterations"], cell


def test_four_tolerances_are_not_enough(tolerance_rows):
    """The notes once listed four; with tol_rel_grad at its default the run
    is the default run, step for step."""
    for cell in tolerance.CELLS:
        runs = {r["tolerances"]: r for r in tolerance_rows
                if (r["configuration"], r["observations"]) == cell}
        assert runs["four"]["iterations"] == runs["default"]["iterations"]
        assert runs["four"]["lp"] == runs["default"]["lp"]
        assert "relative gradient" in runs["default"]["stopped_by"]


def test_the_notes_run_reproduces(tolerance_rows):
    """docs/non-smooth-objective.md's figures are the yearly-only cell's."""
    row = next(r for r in tolerance_rows if r["configuration"] == "yearly_only"
               and r["tolerances"] == "five")
    doc = (REPO / "docs" / "non-smooth-objective.md").read_text()
    assert f"it runs {row['iterations']} iterations" in doc
    assert f"**{row['lp']:.2f}**" in doc
    assert f"still {row['shortfall']:.2f} nats short" in doc
    assert "Line search failed" in row["stopped_by"]
    assert f"{row['reference_lp']:.4f}" in doc


def test_prophet_cannot_make_the_run(tolerance_rows):
    """Where the line search fails, Prophet's fall-back hands Newton the
    L-BFGS tolerances, and cmdstanpy refuses them."""
    failed = [r for r in tolerance_rows if "Line search failed" in r["stopped_by"]]
    assert failed
    for row in failed:
        assert row["prophet_outcome"].startswith("raised ValueError"), row
    for row in tolerance_rows:
        if row not in failed:
            assert row["prophet_outcome"] == "LBFGS"
            assert row["prophet_lp"] == pytest.approx(row["lp"], abs=1e-6)
    assert "cmdstanpy rejects them for Newton" in _text("evidence")


def test_the_fallback_mechanism_is_prophets_own(prophet_comparison):
    """The mechanism, deterministically: Prophet's fall-back keeps the caller's
    arguments, and cmdstanpy rejects a tolerance for Newton."""
    from prophet.models import CmdStanPyBackend

    import corpora
    import harness

    source = inspect.getsource(CmdStanPyBackend.fit)
    assert source.index("args.update(kwargs)") < source.index("args['algorithm'] = 'Newton'")
    with pytest.raises(ValueError, match="tol_rel_obj requires"):
        prophet_comparison[0](**harness.PROPHET_KWARGS).fit(corpora.peyton_manning(50),
                                                            tol_rel_obj=1)


# -- the prior's strength -----------------------------------------------------

@pytest.fixture(scope="module")
def strength_rows():
    return strength.read()


def test_the_strength_sweep_is_complete_and_its_references_certified(strength_rows):
    assert len(strength_rows) == len(strength.LENGTHS) * len(strength.LAMBDAS) * len(strength.RUNS)
    assert strength.DEFAULT_LAMBDA in strength.LAMBDAS
    assert min(strength.LAMBDAS) == 1e-6 and max(strength.LAMBDAS) == 100.0
    for row in strength_rows:
        assert row["reference_residual"] < 1e-6, row


def test_lbfgs_reaches_the_optimum_off_the_kink_and_not_on_it(strength_rows):
    run = [r for r in strength_rows if r["run"] == "none"]
    for row in run:
        assert row["iterations"] < strength.RUNS["none"]["maxiter"], row   # ended on progress
        if row["reference_zeros"] == 0:
            assert abs(row["shortfall"]) < 1e-6, row
        if row["lam"] == strength.DEFAULT_LAMBDA:
            assert row["shortfall"] > 0.1, row
            assert row["reference_zeros"] >= 10
    assert sum(1 for r in run if r["reference_zeros"] == 0) >= 9 * len(strength.LENGTHS)


def test_the_shortfall_tracks_the_zeros(strength_rows):
    from scipy.stats import spearmanr

    qn = _load("quote_numbers")
    run = [r for r in strength_rows if r["run"] == "none"]
    zeros, shortfall = [r["reference_zeros"] for r in run], [r["shortfall"] for r in run]
    rho = spearmanr(zeros, shortfall).statistic
    assert rho > 0.8
    assert qn._rank_correlation(zeros, shortfall) == pytest.approx(rho, abs=1e-12)


def _random_lasso(seed, n=12, penalized=8):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(40, n))
    Q = Z.T @ Z + np.eye(n) * 0.1
    c = rng.normal(size=n) * 5
    mask = np.zeros(n, dtype=bool)
    mask[2:2 + penalized] = True
    return Q, c, mask


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [1e-6, 0.5, 3.0, 50.0])
def test_the_active_set_solver_meets_the_optimality_conditions(seed, lam):
    """Convex, so these conditions are the definition of the minimiser."""
    Q, c, mask = _random_lasso(seed)
    x, _ = strength.exact(Q, c, mask, lam)
    gradient = Q @ x - c
    scale = 1e-8 * (1 + np.abs(c).max())
    assert np.all(np.abs(gradient[~mask]) < scale)
    on = mask & (x != 0)
    assert np.all(np.abs(gradient[on] + lam * np.sign(x[on])) < scale)
    off = mask & (x == 0)
    assert np.all(np.abs(gradient[off]) <= lam * (1 + 1e-9))


def test_the_quadratic_is_the_librarys_objective(prophet_comparison):
    """Q and c reproduce AnalyticProphet's objective at fixed sigma up to a
    constant, so the reference minimises what L-BFGS was given."""
    problem = strength.Problem(300)
    rng = np.random.default_rng(1)
    for lam in (20.0, 1e-3):
        model = problem.model(lam)
        layout, design = model.layout, model._design_matrices()
        Q, c, mask = strength.quadratic(model, problem.sigma)
        keep = np.ones(layout.size, dtype=bool)
        keep[layout.sigma_obs_idx] = False

        def library(x):
            full = np.empty(layout.size)
            full[keep], full[layout.sigma_obs_idx] = x, problem.sigma
            return model._minus_log_posterior(full, design=design)

        def ours(x):
            return 0.5 * x @ Q @ x - c @ x + lam * np.abs(x[mask]).sum()

        a, b = rng.normal(size=c.size) * 0.1, rng.normal(size=c.size) * 0.1
        assert ours(a) - ours(b) == pytest.approx(library(a) - library(b), rel=1e-9)


def test_the_paper_says_which_option_replaced_liblbfgs():
    """#175: the lost sweep is cited as not reproduced, and its replacement named."""
    text = _text("reproducibility")
    assert "One earlier measurement is not reproduced, and nothing here rests on it" in text
    assert "liblbfgs" in text and "L-BFGS-B" in text
    for number in ("3528", "3,528"):
        assert number not in " ".join(_text(name) for name in
                                      ("evidence", "smooth", "reproducibility"))


# -- the rounding -------------------------------------------------------------

def test_the_rounding_sample_is_the_seeded_census_sample():
    import corpora
    import kkt

    rows = rounding.read()
    sample = kkt.sample_manifest(corpora.load_manifest(), rounding.SAMPLE)
    # rows are named as corpora names them, m4_daily_D1015 for D1015
    assert sorted(r["series"].rsplit("_", 1)[-1] for r in rows) == sorted(
        name for names in sample["series"].values() for name in names)


def test_the_default_printout_is_the_full_point_rounded():
    """The premise: two runs, one printed at eight digits, are the same run."""
    assert all(r["printed_is_rounded"] for r in rounding.read())


def test_rounding_is_small_beside_the_residual():
    rows = rounding.read()
    assert max(r["strict_change"] for r in rows) < 1.0
    assert max(r["strict_change"] for r in rows) < min(r["strict_full"] for r in rows) / 10


# -- exact zeros at the optimum -----------------------------------------------

def test_prophets_solutions_have_no_exact_zero():
    """'Prophet's solutions have none, on any of the series.'"""
    census = list(csv.DictReader(open(RESULTS / "kkt_census.csv")))
    assert all(int(r["exact_zeros"]) == 0 for r in census if r["implementation"] == "prophet")
    detail = list(csv.DictReader(open(RESULTS / "kkt_peyton_manning.csv")))
    assert all(float(r["value"]) != 0.0 for r in detail
               if r["implementation"] == "prophet" and r["coordinate"].startswith("delta"))
    assert "Prophet's solutions have none" in _text("model")
    assert "\\pending" not in (PAPER / "sections" / "model.tex").read_text()
