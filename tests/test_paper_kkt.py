"""
Issue #170: a first-order optimality certificate at both solutions, per series.

`paper/experiments/kkt.py` grades each point by the minimum-norm subgradient of
Prophet's negative log posterior, computed from Stan's own gradient. A
certificate is only as good as its instrument, so three things are checked
before the committed numbers are:

  * **the arithmetic** -- the subgradient on hand-built gradients, where the
    answer is known, and the relaxed reading never exceeding the strict one;
  * **Stan's convention at the kink** -- the exact-zero branch assumes Stan
    differentiates |delta| as sign(delta) with sign(0) = 0. If it chose a side
    instead, every exact zero of ours would be graded against the wrong value;
  * **a second instrument** -- the same quantity from this implementation's
    analytic gradient, which shares no code with Stan's autodiff.

Then the committed results: every frozen series measured, by both, on
identical design matrices.
"""
import csv
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
RESULTS = PAPER / "results"
MANIFEST = REPO / "evaluation" / "corpus" / "m4_census_v1.json"
TAU = 0.05


def _kkt():
    spec = importlib.util.spec_from_file_location("kkt", PAPER / "experiments" / "kkt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kkt = _kkt()


def _gradient(delta_lp, k=0.0, m=0.0, beta=()):
    return {"k": k, "m": m, "delta": np.asarray(delta_lp, float),
            "beta": np.asarray(beta, float)}


# ---------------------------------------------------------------------------
# The arithmetic.

def test_a_nonzero_rate_must_balance_the_prior_exactly():
    """At delta_j != 0 the condition is ds/ddelta_j + sign(delta_j)/tau = 0,
    which is minus Stan's gradient of lp__ there (it carries the prior)."""
    cert = kkt.certificate(_gradient([0.0, 3.0, -2.5]), [0.4, -1e-9, 2.0], TAU)
    np.testing.assert_allclose(cert["strict"], [0.0, -3.0, 2.5])
    # and the smooth part has the prior's -sign/tau taken back out
    np.testing.assert_allclose(cert["smooth_delta"], [-20.0, 17.0, -17.5])


def test_an_exact_zero_is_soft_thresholded():
    """At delta_j == 0 anything within [-1/tau, 1/tau] is stationary."""
    cert = kkt.certificate(_gradient([12.0, -19.9, 25.0, -23.0]), np.zeros(4), TAU)
    np.testing.assert_allclose(cert["strict"], [0.0, 0.0, -5.0, 3.0])
    np.testing.assert_allclose(cert["relaxed"], cert["strict"])


def test_a_tiny_rate_left_short_of_the_kink_costs_up_to_one_over_tau():
    """The issue's prediction, in miniature: a rate stopped at 1e-9 with the
    smooth gradient of an inactive coordinate leaves a strict residual of order
    1/tau, which the relaxed reading forgives entirely."""
    smooth = 5.0                                  # well inside [-20, 20]
    stan = -smooth - np.sign(1e-9) / TAU          # what Stan reports at 1e-9
    cert = kkt.certificate(_gradient([stan]), [1e-9], TAU)
    assert cert["strict"][0] == pytest.approx(smooth + 1 / TAU)
    assert cert["relaxed"][0] == 0.0


def test_the_relaxed_reading_never_exceeds_the_strict_one():
    """sign(delta)/tau is an endpoint of [-1/tau, 1/tau], so granting the zero
    reading can only shrink a residual. Checked on random points, both signs
    and exact zeros."""
    rng = np.random.default_rng(170)
    for _ in range(200):
        delta = rng.choice([-1.0, 0.0, 1.0], size=25) * rng.lognormal(-5, 4, size=25)
        cert = kkt.certificate(_gradient(rng.normal(scale=30, size=25)), delta, TAU)
        assert np.all(np.abs(cert["relaxed"]) <= np.abs(cert["strict"]) + 1e-12)


def test_the_smooth_coordinates_are_the_plain_gradient_of_minus_lp():
    cert = kkt.certificate(_gradient([0.0], k=2.0, m=-3.0, beta=[0.5, -0.25]), [0.0], TAU)
    assert cert["k"][0] == -2.0 and cert["m"][0] == 3.0
    np.testing.assert_allclose(cert["beta"], [-0.5, 0.25])


def test_the_summary_locates_the_violation():
    """The acceptance asks for the coordinates, not only the size."""
    stan = np.zeros(5)
    stan[2] = -7.0 - 1 / TAU          # delta[3] = 1e-8: strict residual 27
    delta = np.array([0.0, 0.0, 1e-8, 0.0, 0.0])
    summary = kkt.summarise(kkt.certificate(_gradient(stan, k=-4.0, beta=[0.1]), delta, TAU), delta)
    assert summary["argmax_strict"] == "delta[3]"
    assert summary["delta_at_argmax_strict"] == 1e-8
    assert summary["residual_delta_strict"] == pytest.approx(27.0)
    # granted the zero reading, delta[3] is fine and k carries what is left
    assert summary["argmax_relaxed"] == "k"
    assert summary["residual_k"] == 4.0
    assert summary["exact_zeros"] == 4


# ---------------------------------------------------------------------------
# The instrument.

@pytest.fixture(scope="module")
def peyton_manning_300(prophet_comparison, compiled_optimizer_module):
    """Both fits on Peyton Manning's first 300 rows, through kkt's own path."""
    import logging

    import corpora

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = corpora.peyton_manning(300)
    stan_model, stan_data, points, design_gap = kkt._fit_both(df, compiled_optimizer_module)
    return df, stan_model, stan_data, points, design_gap


def test_stan_takes_sign_zero_as_zero_at_the_kink(peyton_manning_300):
    """At delta_j = 0 exactly, Stan's gradient is the smooth part's: midway
    between its one-sided values, which differ by 2/tau. The exact-zero branch
    of the certificate depends on this."""
    _, stan_model, stan_data, points, _ = peyton_manning_300
    k, m, delta, sigma, beta = points["prophet"]
    j, h = 2, 1e-9
    values = {}
    for offset in (-h, 0.0, h):
        moved = np.array(delta, dtype=float)
        moved[j] = offset
        values[offset] = kkt.stan_gradient(stan_model, stan_data, k, m, moved, sigma, beta)[1]["delta"][j]
    # the smooth part moves too over 2h, by its curvature times 2e-9 -- about
    # 1e-4 here -- so the jump is 2/tau to that, and the midpoint to far better
    assert values[-h] - values[h] == pytest.approx(2 / TAU, abs=1e-3)
    assert values[0.0] == pytest.approx((values[-h] + values[h]) / 2, abs=1e-6)


def test_stans_gradient_agrees_with_this_implementations_own(peyton_manning_300,
                                                             compiled_optimizer_module):
    """Two instruments that share no code: Stan's autodiff of Prophet's program
    and the hand-derived gradient of `AnalyticProphet._gradient`. They agree on
    the smooth part at our solution to rounding, so the certificate is not one
    implementation grading itself."""
    import harness
    from analytic_prophet import AnalyticProphet

    df, stan_model, stan_data, points, _ = peyton_manning_300
    k, m, delta, sigma, beta = points["analytic_prophet"]
    _, gradient = kkt.stan_gradient(stan_model, stan_data, k, m, delta, sigma, beta)
    cert = kkt.certificate(gradient, delta, float(stan_data["tau"]))

    ours = AnalyticProphet(**harness.PROPHET_KWARGS)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=compiled_optimizer_module)
    S = len(delta)
    params = np.r_[k, m, delta, sigma, beta]
    analytic = ours._gradient(params, include_l1_prior=False)

    assert analytic[0] == pytest.approx(cert["k"][0], abs=1e-7)
    assert analytic[1] == pytest.approx(cert["m"][0], abs=1e-7)
    np.testing.assert_allclose(analytic[2:2 + S], cert["smooth_delta"], atol=1e-7)
    np.testing.assert_allclose(analytic[3 + S:], cert["beta"], atol=1e-7)


def test_our_solution_is_certified_and_prophets_is_not(peyton_manning_300):
    """The finding at the smallest scale it holds. The bounds are loose on
    purpose -- the census is where the sizes are measured."""
    _, stan_model, stan_data, points, design_gap = peyton_manning_300
    assert design_gap < kkt.DESIGN_TOLERANCE
    tau = float(stan_data["tau"])
    summaries = {}
    for implementation, (k, m, delta, sigma, beta) in points.items():
        _, gradient = kkt.stan_gradient(stan_model, stan_data, k, m, delta, sigma, beta)
        summaries[implementation] = kkt.summarise(kkt.certificate(gradient, delta, tau), delta)
    assert summaries["analytic_prophet"]["norm_strict"] < 0.05 / tau
    assert summaries["prophet"]["norm_relaxed"] > 10 * summaries["analytic_prophet"]["norm_strict"]
    assert summaries["prophet"]["exact_zeros"] == 0


# ---------------------------------------------------------------------------
# The committed results.

def _census_rows():
    with open(RESULTS / "kkt_census.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_frozen_series_was_certified_by_both():
    manifest = json.loads(MANIFEST.read_text())
    frozen = {f"m4_{frequency.lower()}_{identifier}"
              for frequency, identifiers in manifest["series"].items()
              for identifier in identifiers}
    rows = _census_rows()
    by_series = {}
    for row in rows:
        by_series.setdefault(row["series"], set()).add(row["implementation"])
    assert set(by_series) == frozen
    assert all(found == set(kkt.IMPLEMENTATIONS) for found in by_series.values())
    meta = json.loads((RESULTS / "kkt_census.meta.json").read_text())
    assert meta["digest"] == manifest["digest"]


def test_beta_crossed_between_identical_design_matrices_on_every_series():
    """beta goes from one implementation to the other untransformed, which is
    only legitimate where the design matrices agree. Recorded per series rather
    than assumed from Peyton Manning."""
    assert max(float(row["design_gap"]) for row in _census_rows()) < kkt.DESIGN_TOLERANCE


def test_the_committed_results_respect_the_certificates_own_algebra():
    for row in _census_rows():
        assert float(row["residual_delta_relaxed"]) <= float(row["residual_delta_strict"]) + 1e-12
        assert float(row["norm_relaxed"]) <= float(row["norm_strict"]) + 1e-12


def test_the_peyton_manning_detail_is_complete():
    with open(RESULTS / "kkt_peyton_manning.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for implementation in kkt.IMPLEMENTATIONS:
        names = [row["coordinate"] for row in rows if row["implementation"] == implementation]
        assert names[:2] == ["k", "m"]
        deltas = [name for name in names if name.startswith("delta")]
        assert deltas == [f"delta[{j + 1}]" for j in range(len(deltas))] and len(deltas) == 25
    for row in rows:
        assert abs(float(row["residual_relaxed"])) <= abs(float(row["residual_strict"])) + 1e-12


def test_the_papers_every_is_true():
    """The abstract says Prophet's solution fails the conditions on *every*
    series, under the reading most favourable to it. That word is typed, not
    generated, so it is held to the data here: the relaxed residual must clear
    the two instruments' disagreement (1e-7) by orders of magnitude on each
    series, and the census must still be complete."""
    rows = [row for row in _census_rows() if row["implementation"] == "prophet"]
    abstract = (PAPER / "main.tex").read_text()
    assert "On every one of \\KktSeries{} M4 series" in abstract
    assert min(float(row["norm_relaxed"]) for row in rows) > 1e-3
