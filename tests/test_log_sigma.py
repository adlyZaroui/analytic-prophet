"""
Issue #229: on logistic growth the two backends' default fits ended in
different local minima.

The compiled core has always optimized zeta = log(sigma_obs); the Python
backend optimized sigma_obs itself, floored at 1e-6. On a convex model the two
coordinate systems reach the same optimum. On logistic growth -- non-convex --
the Newton twins took different steps from the same start and ended 8.36 nats
apart at T = 60 and 0.50 at T = 99 on Peyton Manning, both at minima. Run in
zeta, nothing else changed, the Python Newton lands exactly where the C++ one
does. L-BFGS-B moved too: in sigma_obs it exits abnormally on logistic growth
at T = 300 where the compiled L-BFGS does not, and Prophet's rule then sends it
to Newton, a different basin.

Checked here: the twins agree on those cases, and `layout.LogSigma` is the
calculus it claims to be.
"""
import numpy as np
import pytest

from analytic_prophet import AnalyticProphet
from analytic_prophet.layout import LogSigma


def _logistic(peyton_manning_df, n_rows):
    df = peyton_manning_df.head(n_rows).reset_index(drop=True).copy()
    df["cap"] = df["y"].max() * 1.2
    return df


@pytest.mark.parametrize("n_rows", [60, 99, 300])
def test_the_backends_agree_on_logistic_growth(peyton_manning_df, compiled_optimizer_module,
                                               n_rows):
    """The default rule at each length: Newton at 60 and 99, L-BFGS at 300.
    Before #229 these were 8.36, 0.50 and -- through the Newton fallback once
    the zeta-only C++ L-BFGS converged and the sigma_obs scipy one did not --
    13 nats apart."""
    df = _logistic(peyton_manning_df, n_rows)
    compiled = AnalyticProphet(growth="logistic")
    compiled.fit(df, lib_path=compiled_optimizer_module)
    python = AnalyticProphet(growth="logistic")
    python.fit(df, backend="python")

    assert python.optimizer_used == compiled.optimizer_used
    gap = python.loss_over_iterations[-1] - compiled.loss_over_iterations[-1]
    # the Newton twins agree to the last few bits; the L-BFGS pair are two
    # libraries under #24's tolerance, which is why the bound is looser there
    assert abs(gap) < (1e-9 if compiled.optimizer_used == "Newton" else 1e-6), gap


def test_the_fit_still_reports_sigma_obs(peyton_manning_df):
    """`opt.x` and `sigma_obs` are in sigma_obs, as every reader of them
    expects; only the optimizers see the logarithm."""
    df = peyton_manning_df.head(50).reset_index(drop=True)
    model = AnalyticProphet()
    model.fit(df, backend="python")
    n_delta = model.layout.n_changepoints
    assert model.opt.x[2 + 2 * n_delta] == pytest.approx(model.sigma_obs)
    assert 0 < model.sigma_obs < 1


# -- the calculus ------------------------------------------------------------

def _quadratic():
    """f(z) = sum((z - c)^2 * w), with sigma at index 1."""
    centre = np.array([0.3, 0.7, -1.2])
    weight = np.array([1.0, 3.0, 0.5])
    f = lambda z: float(np.sum(weight * (z - centre) ** 2))
    g = lambda z: 2 * weight * (z - centre)
    return f, g


def test_the_coordinates_round_trip():
    coordinates = LogSigma(1)
    z = np.array([0.3, 0.42, -1.2])
    w = coordinates.to_log(z)
    assert w[1] == pytest.approx(np.log(0.42)) and w[0] == z[0] and w[2] == z[2]
    np.testing.assert_allclose(coordinates.from_log(w), z, rtol=1e-15)


def test_the_gradient_is_the_chain_rule():
    f, g = _quadratic()
    coordinates = LogSigma(1)
    fw, gw = coordinates.objective(f), coordinates.gradient(g)
    w = coordinates.to_log(np.array([0.1, 0.9, -1.0]))
    numeric = np.array([(fw(w + h) - fw(w - h)) / 2e-6
                        for h in np.eye(3) * 1e-6])
    np.testing.assert_allclose(gw(w), numeric, rtol=1e-6)
    value, grad = coordinates.objective_and_gradient(lambda z: (f(z), g(z)))(w)
    assert value == fw(w)
    np.testing.assert_array_equal(grad, gw(w))
    # and back
    np.testing.assert_allclose(coordinates.gradient_from_log(w, gw(w)),
                               g(coordinates.from_log(w)), rtol=1e-15)


def test_the_floor_goes_and_a_pin_stays():
    """zeta needs no bound, as in the C++ core; a pinned sigma_obs stays pinned."""
    coordinates = LogSigma(1)
    free = coordinates.bounds([(None, None), (1e-6, None), (0, None)])
    assert free == [(None, None), (None, None), (0, None)]
    pinned = coordinates.bounds([(None, None), (0.25, 0.25), (0, None)])
    assert pinned[1] == (pytest.approx(np.log(0.25)), pytest.approx(np.log(0.25)))
