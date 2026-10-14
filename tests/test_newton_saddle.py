"""
Issue #193: the Newton path could stop at a saddle point, and report success.

On a non-convex model -- prophet#842's reprex: ten weekly points,
multiplicative seasonality, an extra regressor, two changepoints -- the damped
Newton method stopped from Prophet's own start at a stationary point with a
direction of negative curvature, 3.22 nats below the optimum. Stan's Newton
reflects negative eigenvalues of the Hessian and does not stop there. Both
twins now check the free Hessian where they stop and, at a saddle, step off
along the negative curvature and continue (`negative_curvature` and
`leave_saddle` in analytic_prophet/optimizer.py, and their twins in
optimize.cpp).

Checked here:

  * **the default rule reaches the optimum on the reproducer**, on both
    backends, and stops at a minimum;
  * **started at the saddle itself, both twins leave it** -- a test that does
    not depend on the path into the saddle, and that failed on both before;
  * **a fit that stops at a minimum is untouched**: no escape, so it takes
    exactly the steps it took before -- which is why the check runs where a
    run stops rather than inside every step, as Stan's does;
  * the helpers, on functions whose curvature is known, and the two twins'
    constants.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet
from analytic_prophet.forecaster import canonical_to_split, from_dict_to_array
from analytic_prophet.optimizer import (MAX_SADDLE_ESCAPES, SADDLE_TOLERANCE, leave_saddle,
                                        negative_curvature)

REPO = Path(__file__).parent.parent

# prophet#842's reprex, as paper/experiments/issue842.py ports it.
XREG = [-0.76537379485893120012, -0.74533153677032704109, -0.62494542934172692127,
        -0.39778106465657125934, -0.06976829815746909968, 0.34070346369915599504,
        0.80422096949408772292, 1.28317620368430906907, 1.73567381412022903042,
        2.12006230186317567998]
Y = [1, 1, 0, 1, 0, 1, 0, 2, 2, 0]

# Where the compiled Newton stopped before #193, from Prophet's own start.
SADDLE = {"k": -0.29145376004907364, "m": 0.23628082799675193, "delta": [0.0, 0.0],
          "beta": [-1.0274399368510634], "sigma_obs": 0.4671846254815927}


def _frame():
    return pd.DataFrame({"ds": pd.date_range("2018-09-10", periods=10, freq="7D"),
                         "y": Y, "xreg": XREG})


def _model():
    model = AnalyticProphet(seasonality_mode="multiplicative", n_changepoints=2)
    model.add_regressor("xreg")
    return model


def _fit(df, model, backend, lib_path, **kwargs):
    if backend == "cpp":
        kwargs["lib_path"] = lib_path
    model.fit(df, backend=backend, **kwargs)
    return model


def _start(rng):
    return {"k": float(rng.normal(0, 1)), "m": float(rng.normal(0.5, 0.5)),
            "delta": rng.normal(0, 0.1, 2), "beta": rng.normal(0, 1, 1),
            "sigma_obs": float(rng.uniform(0.1, 1.0))}


@pytest.fixture(scope="module")
def best(compiled_optimizer_module):
    """The best objective L-BFGS finds on the reproducer, from Prophet's start
    and from ten seeded random starts: the model is not convex, so this is the
    reference the issue asks for."""
    rng = np.random.default_rng(193)
    values = []
    for start in [None] + [_start(rng) for _ in range(10)]:
        kwargs = {"algorithm": "LBFGS"}
        if start is not None:
            kwargs["initial_params"] = start
        model = _fit(_frame(), _model(), "cpp", compiled_optimizer_module, **kwargs)
        values.append(model.loss_over_iterations[-1])
    return min(values)


def _curves_downwards(model, df):
    """The free coordinates' negative curvature at `model`'s fitted point, under
    the Python objective -- None at a minimum. Bounds as _fit_python sets them."""
    reference = _model()
    reference.fit(df, backend="python", algorithm="LBFGS")
    design = reference._design_matrices()
    gradient = lambda z: reference._split_gradient(z, design=design)
    point = {"k": float(model.params["k"][0][0]), "m": float(model.params["m"][0][0]),
             "delta": np.asarray(model.params["delta"][0], dtype=float),
             "beta": np.ravel(model.params["beta"][0]),
             "sigma_obs": float(model.sigma_obs)}
    z = canonical_to_split(from_dict_to_array(point, reference.layout), reference.layout)
    S = reference.layout.n_changepoints
    lower = np.array([-np.inf] * 2 + [0.0] * (2 * S) + [1e-6]
                     + [-np.inf] * reference.layout.n_regressor_columns)
    return negative_curvature(gradient, z, gradient(z), lower, np.full(z.size, np.inf))


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_the_default_rule_reaches_the_optimum_on_the_842_reproducer(
        backend, best, compiled_optimizer_module):
    """The issue's acceptance: the default fit (Newton, since T = 10) within
    1e-6 nats of the best start, and not a saddle. Before the fix the compiled
    path stopped 3.22 nats short."""
    model = _fit(_frame(), _model(), backend, compiled_optimizer_module)
    assert model.optimizer_used == "Newton"
    assert model.loss_over_iterations[-1] <= best + 1e-6
    assert _curves_downwards(model, _frame()) is None


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_newton_started_at_the_saddle_leaves_it(backend, best, compiled_optimizer_module):
    """From the saddle itself both twins used to stay put and report
    convergence; it does not depend on how a run gets there."""
    model = _fit(_frame(), _model(), backend, compiled_optimizer_module,
                 algorithm="Newton", initial_params=dict(SADDLE))
    assert model.opt.saddle_escapes >= 1
    assert model.loss_over_iterations[-1] <= best + 1e-6
    assert _curves_downwards(model, _frame()) is None


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("n_rows", [20, 50, 99])
def test_a_fit_that_stops_at_a_minimum_is_untouched(backend, n_rows, peyton_manning_df,
                                                     compiled_optimizer_module):
    """Linear growth and additive seasonality: no saddle to find, so no escape,
    and the run takes exactly the steps it took before the check existed."""
    df = peyton_manning_df.head(n_rows).reset_index(drop=True)
    model = AnalyticProphet()
    model = _fit(df, model, backend, compiled_optimizer_module)
    assert model.optimizer_used == "Newton"
    assert model.opt.saddle_escapes == 0


def test_lbfgs_reports_no_escapes(compiled_optimizer_module):
    model = _fit(_frame(), _model(), "cpp", compiled_optimizer_module, algorithm="LBFGS")
    assert model.opt.saddle_escapes == 0


# -- the helpers, where the curvature is known -------------------------------

def _saddle_function():
    """x^2 - y^2 + y^4: a saddle at the origin, minima at y = +-1/sqrt(2)."""
    f = lambda z: z[0] ** 2 - z[1] ** 2 + z[1] ** 4
    g = lambda z: np.array([2 * z[0], -2 * z[1] + 4 * z[1] ** 3])
    return f, g


def test_negative_curvature_finds_the_descending_direction():
    _, g = _saddle_function()
    z = np.zeros(2)
    direction = negative_curvature(g, z, g(z), np.full(2, -np.inf), np.full(2, np.inf))
    assert direction is not None
    assert abs(abs(direction[1]) - 1.0) < 1e-9 and abs(direction[0]) < 1e-9


def test_negative_curvature_is_none_at_a_minimum():
    g = lambda z: 2 * z
    z = np.zeros(3)
    assert negative_curvature(g, z, g(z), np.full(3, -np.inf), np.full(3, np.inf)) is None


@pytest.mark.parametrize("smallest", [-1e-9, 1e-9])
def test_curvature_within_the_tolerance_is_not_a_saddle(smallest):
    """A curvature this small beside the largest is differencing noise: the
    Hessian's eigenvalues move by about 3e-10 of the largest between steps h
    and 4h. A quadratic's gradient is linear, so here the differences are
    exact and the eigenvalues are the ones written down."""
    g = lambda z: np.array([smallest * z[0], z[1]])
    z = np.zeros(2)
    assert smallest > -SADDLE_TOLERANCE
    assert negative_curvature(g, z, g(z), np.full(2, -np.inf), np.full(2, np.inf)) is None


def test_a_coordinate_pinned_at_its_bound_is_not_a_direction():
    """Negative curvature along y, but y sits at its bound with the gradient
    pushing into it -- held there by Newton's active set, so not a saddle. The
    same holds for a changepoint rate pinned at zero by the L1 term's kink."""
    g = lambda z: np.array([2 * z[0], -2 * z[1] + 3.0])
    z = np.zeros(2)
    lower = np.array([-np.inf, 0.0])
    assert negative_curvature(g, z, g(z), lower, np.full(2, np.inf)) is None


def test_leave_saddle_lowers_the_objective_or_says_it_cannot():
    f, g = _saddle_function()
    z = np.zeros(2)
    free = np.full(2, np.inf)
    direction = negative_curvature(g, z, g(z), -free, free)
    point, value = leave_saddle(f, z, f(z), direction, -free, free)
    assert value < f(z) and abs(point[1]) > 0
    # along a direction of positive curvature from a minimum nothing is lower
    assert leave_saddle(lambda z: float(z @ z), np.zeros(2), 0.0,
                        np.array([1.0, 0.0]), -free, free) is None


def test_the_twins_share_their_saddle_constants():
    source = (REPO / "analytic_prophet" / "optimize.cpp").read_text()
    block = re.search(r"namespace newton_saddle \{(.*?)\}", source, flags=re.S).group(1)
    tolerance = float(re.search(r"TOLERANCE = ([^;]+);", block).group(1))
    escapes = int(re.search(r"MAX_ESCAPES = ([^;]+);", block).group(1))
    assert tolerance == SADDLE_TOLERANCE
    assert escapes == MAX_SADDLE_ESCAPES
