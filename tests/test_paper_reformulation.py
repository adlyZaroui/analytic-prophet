"""
Issue #176: why the split reformulation is exact, stated for the full model.

Proposition 1 in `paper/sections/reformulation.tex` claims, for *any*
F(theta, delta) and tau > 0: equal optimal values; complementarity at every
local or global minimiser of the split problem, which carries back to
Prophet's problem; and that the split problem's KKT points are exactly the
complementary splits of points where 0 is in Prophet's subdifferential. It
claims all of that without convexity, and the paper leans on that for the
configurations that are not convex.

A proof is not a test, but its claims can be exercised:

  * **the identity the proof rests on**, Q = P + (2/tau) sum min(d+, d-), at
    random points;
  * **the proposition where convexity fails**, on an F built to have many
    local minima: every local solution a solver returns is complementary,
    satisfies the KKT conditions, and is stationary for Prophet's problem;
  * **the shipped fit**, whose split solution is complementary exactly.
"""
import numpy as np
import pytest

TAU = 0.5
P_THETA, S = 3, 6


@pytest.fixture(scope="module")
def nonconvex():
    """F(theta, delta) with sines, cosines and a bilinear coupling: smooth,
    bounded below, and nowhere near convex."""
    rng = np.random.default_rng(176)
    A = rng.normal(size=(8, S))
    b = rng.normal(size=8)
    C = 0.3 * rng.normal(size=(P_THETA, S))

    def F(theta, delta):
        return (np.sum(np.sin(3 * theta)) + 0.5 * theta @ theta
                + 0.5 * np.sum((A @ delta - b) ** 2) + np.sum(np.cos(2 * delta))
                + theta @ C @ delta)

    def gradient(theta, delta):
        return (3 * np.cos(3 * theta) + theta + C @ delta,
                A.T @ (A @ delta - b) - 2 * np.sin(2 * delta) + C.T @ theta)

    return F, gradient


def _parts(z):
    return z[:P_THETA], z[P_THETA:P_THETA + S], z[P_THETA + S:]


def test_the_identity_the_proof_rests_on(nonconvex):
    """delta+ + delta- = |delta| + 2 min(delta+, delta-), so Q exceeds P at the
    corresponding point by exactly (2/tau) sum min -- zero on complementary
    pairs, positive otherwise."""
    F, _ = nonconvex
    rng = np.random.default_rng(0)
    for _ in range(200):
        theta = rng.normal(size=P_THETA)
        plus, minus = rng.exponential(size=S), rng.exponential(size=S)
        minus[rng.random(S) < 0.4] = 0.0
        delta = plus - minus
        Q = F(theta, delta) + np.sum(plus + minus) / TAU
        P = F(theta, delta) + np.sum(np.abs(delta)) / TAU
        assert Q - P == pytest.approx(2 / TAU * np.sum(np.minimum(plus, minus)),
                                      rel=1e-12, abs=1e-12)


@pytest.fixture(scope="module")
def local_solutions(nonconvex):
    """L-BFGS-B on the split problem from 30 random starts."""
    from scipy.optimize import minimize

    F, gradient = nonconvex

    def Q(z):
        theta, plus, minus = _parts(z)
        return F(theta, plus - minus) + np.sum(plus + minus) / TAU

    def gQ(z):
        theta, plus, minus = _parts(z)
        g_theta, g_delta = gradient(theta, plus - minus)
        return np.concatenate([g_theta, g_delta + 1 / TAU, -g_delta + 1 / TAU])

    rng = np.random.default_rng(1)
    bounds = [(None, None)] * P_THETA + [(0, None)] * (2 * S)
    solutions = []
    for _ in range(30):
        start = np.concatenate([rng.normal(scale=2, size=P_THETA),
                                rng.exponential(size=2 * S)])
        result = minimize(Q, start, jac=gQ, method="L-BFGS-B", bounds=bounds,
                          options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000})
        solutions.append(result)
    return solutions


def test_the_problem_really_is_not_convex(local_solutions):
    """Otherwise the next test would say nothing about the non-convex case."""
    values = {round(result.fun, 6) for result in local_solutions}
    assert len(values) >= 5


def test_every_local_solution_is_complementary_and_stationary_for_prophets_problem(
        nonconvex, local_solutions):
    """Parts (ii) and (iii), where convexity fails: complementarity at every
    local solution, the KKT multipliers non-negative and summing to 2/tau,
    and the corresponding delta stationary for P = F + |delta|/tau."""
    _, gradient = nonconvex
    for result in local_solutions:
        theta, plus, minus = _parts(result.x)
        assert np.all(np.minimum(plus, minus) == 0.0)

        delta = plus - minus
        g_theta, g_delta = gradient(theta, delta)
        mu_plus, mu_minus = g_delta + 1 / TAU, -g_delta + 1 / TAU
        assert np.all(mu_plus >= -1e-6) and np.all(mu_minus >= -1e-6)
        assert np.max(np.abs(mu_plus * plus)) < 1e-6
        assert np.max(np.abs(mu_minus * minus)) < 1e-6

        residual = np.where(delta != 0, g_delta + np.sign(delta) / TAU,
                            np.sign(g_delta) * np.maximum(np.abs(g_delta) - 1 / TAU, 0))
        assert np.max(np.abs(g_theta)) < 1e-6
        assert np.max(np.abs(residual)) < 1e-6


def test_the_shipped_split_fit_is_exactly_complementary(peyton_manning_df):
    """Prophet's own F, the reference path's L-BFGS-B: no pair has both members
    positive, so the split solution is the complementary split of its delta."""
    from analytic_prophet import AnalyticProphet

    model = AnalyticProphet().fit(peyton_manning_df.iloc[:300].reset_index(drop=True),
                                  backend="python")
    n = model.layout.n_changepoints
    plus, minus = model.opt.x[2:2 + n], model.opt.x[2 + n:2 + 2 * n]
    assert np.all(np.minimum(plus, minus) == 0.0)
