"""
Issue #8: the compiled optimizer terminated after ~2 iterations with
LBFGSERR_ROUNDING_ERROR, nowhere near converged.

The issue proposed two directions. Both were investigated:

1. Mistuned L-BFGS parameters. Partly -- `xtol` was set to 1e-10, *looser*
   than liblbfgs's own 1e-16 default, and xtol is precisely the knob whose
   documentation says the line search "will terminate with the status code
   (LBFGSERR_ROUNDING_ERROR) if the relative width of the interval of
   uncertainty is less than this parameter"; `gtol` was 1e-5, below
   `ftol` = 1e-4, violating the documented constraint that gtol exceed ftol.
   But retuning those constants alone does not fix it.

2. A subtle gradient bug. Ruled out -- test_cpp_gradient_matches_python_analytic
   below pins the C++ objective and gradient against the Python analytic
   reference to ~1e-9, including at the delta=0 kink.

The actual root cause is neither: the Laplace (double-exponential) prior on
delta, which vanilla Prophet uses to keep changepoint rates sparse, puts a
|delta|/changepoint_prior_scale term in the posterior. That makes the objective non-differentiable
at delta = 0 -- and the optimum sits exactly on those kinks, since the prior is
what drives most changepoint rates to zero. More-Thuente, liblbfgs's default
line search, assumes a smooth objective and cannot bracket a step across a
kink, so it gave up almost immediately. Sweeping changepoint_prior_scale showed the failure
tracking the L1 strength exactly: 1/changepoint_prior_scale = 20 died at 2 iterations, while
1/changepoint_prior_scale = 1e-6 (an effectively smooth objective) ran 3528.

The fix is OWL-QN, liblbfgs's built-in support for exactly this class of
objective, which takes the L1 coefficient itself and handles the kink by
projecting each step onto the current orthant.

The same root cause independently broke fit(): plain L-BFGS-B stalls on the
same kinks, at a loss 17.8% above the optimum, while reporting success=True.
fit() now optimizes over the standard smooth reformulation instead
(delta = delta_pos - delta_neg, both non-negative). These tests cover both.
"""
import numpy as np
import pytest

from analytic_prophet import (AnalyticProphet, N_CHANGE_POINTS, n_yearly, K_IDX, M_IDX,
                           DELTA_SLICE, SIGMA_OBS_IDX, BETA_SLICE, canonical_to_cpp)
from conftest import pin_yearly_only

LBFGSERR_ROUNDING_ERROR = -1001

SIGMA_OBS = 1.0
MATCHED_INIT = {
    "k": 0.0,
    "m": 0.0,
    "delta": np.zeros(N_CHANGE_POINTS),
    "beta": np.zeros(2 * n_yearly),
}


def fit_both(df):
    """Run both fit paths from the same start with sigma_obs pinned the same."""
    python_model = pin_yearly_only(AnalyticProphet())
    python_model.fit(df, backend="python", analytic=True, initial_params=MATCHED_INIT, fixed_sigma_obs=SIGMA_OBS)

    cpp_model = pin_yearly_only(AnalyticProphet())
    cpp_model.sigma_obs = SIGMA_OBS
    return python_model, cpp_model


def first_order_residuals(model, params):
    """Subgradient optimality residuals for  f_smooth(x) + sum_j|delta_j|/changepoint_prior_scale.

    At a minimum of a convex objective of that form:
        delta_j != 0  ->  d f_smooth/d delta_j + sign(delta_j)/changepoint_prior_scale == 0
        delta_j == 0  ->  |d f_smooth/d delta_j| <= 1/changepoint_prior_scale
    and every smooth coordinate (k, m, beta) has a vanishing derivative.

    Returns (max |grad| over smooth coords, max violation over delta). This
    certifies optimality on its own -- no second optimizer to agree with, and
    no assumption that either solver is right.

    sigma_obs is excluded: these checks run with it pinned by fixed_sigma_obs,
    so it sits at a bound where the derivative need not vanish.
    """
    gradient = model._gradient(params, include_l1_prior=False)
    delta = np.asarray(params)[DELTA_SLICE]
    ddelta = gradient[DELTA_SLICE]
    threshold = 1.0 / model.changepoint_prior_scale

    smooth = np.concatenate(([gradient[K_IDX], gradient[M_IDX]], gradient[BETA_SLICE]))

    active = np.abs(delta) > 1e-8
    violation = np.zeros_like(delta)
    violation[active] = np.abs(ddelta[active] + np.sign(delta[active]) * threshold)
    violation[~active] = np.maximum(np.abs(ddelta[~active]) - threshold, 0.0)

    return np.max(np.abs(smooth)), np.max(violation)


@pytest.fixture
def small_df(peyton_manning_df):
    """A few hundred rows -- enough to be a real fit, small enough to stay fast."""
    return peyton_manning_df.iloc[:300].reset_index(drop=True)


@pytest.mark.parametrize(
    "case",
    ["zeros", "k_only", "random_small", "random_wide"],
)
def test_cpp_gradient_matches_python_analytic(prepared_model, cpp_mlp_and_gradient, param_size, case):
    """Direction 2 from the issue: is the C++ gradient subtly wrong?

    It is not. The C++ objective and gradient agree with the Python analytic
    reference at every point tried -- including "zeros", which sits exactly on
    the delta=0 kink where the two could most plausibly disagree.

    The objectives now agree *exactly*, with no constant offset: both carry the
    T*log(sigma_obs) normalization and the half-normal prior, since both
    estimate sigma_obs (#18).
    """
    rng = np.random.default_rng(seed=0)
    canonical, sigma_obs = {
        "zeros": (np.zeros(param_size), 1.0),
        "k_only": (np.zeros(param_size), 1.0),
        "random_small": (rng.normal(scale=0.5, size=param_size), 0.4),
        "random_wide": (rng.normal(scale=2.0, size=param_size), 2.5),
    }[case]
    canonical = canonical.copy()
    if case == "k_only":
        canonical[K_IDX] = 0.4
    canonical[SIGMA_OBS_IDX] = sigma_obs  # must be positive: zeta = log(sigma_obs)

    cpp_mlp, cpp_grad = cpp_mlp_and_gradient(prepared_model, canonical_to_cpp(canonical))
    py_mlp, py_grad = prepared_model._minus_log_posteriorAndGradient(canonical)

    assert cpp_mlp == pytest.approx(py_mlp, rel=1e-9, abs=1e-9)

    # Repack the gradient, which does NOT transform like a parameter vector:
    # the blocks are reordered, and the sigma_obs slot becomes d/d_zeta via the
    # chain rule, d(sigma_obs)/d(zeta) = sigma_obs.
    expected = np.concatenate((
        py_grad[:2],
        py_grad[DELTA_SLICE],
        py_grad[BETA_SLICE],
        [py_grad[SIGMA_OBS_IDX] * sigma_obs],
    ))
    np.testing.assert_allclose(cpp_grad, expected, rtol=1e-9, atol=1e-8)


def test_cpp_optimizer_no_longer_bails_out_early(small_df, compiled_optimizer_module):
    """The regression test for the bug as reported: it used to stop after 2
    iterations with LBFGSERR_ROUNDING_ERROR."""
    model = pin_yearly_only(AnalyticProphet())
    model.fit(small_df, initial_params=MATCHED_INIT, lib_path=compiled_optimizer_module)

    assert model.opt_status != LBFGSERR_ROUNDING_ERROR
    assert len(model.loss_over_iterations) > 50


def test_cpp_optimizer_reaches_first_order_optimum(small_df, compiled_optimizer_module):
    """The fitted point satisfies the subgradient optimality conditions, so it
    is the optimum of a convex objective -- not merely 'where the optimizer
    happened to stop'."""
    reference, model = fit_both(small_df)
    model.fit(small_df, initial_params=MATCHED_INIT, lib_path=compiled_optimizer_module)

    smooth_residual, delta_violation = first_order_residuals(reference, model.get_parameters())

    # The bound is 5e-2, not the 1e-3 this asserted before #21. That is a real
    # weakening and worth stating plainly: Stan stops on objective *progress*
    # (tol_rel_obj) rather than on gradient size, so a Prophet-faithful run
    # halts once improvement stalls, leaving a larger gradient than a run
    # driven to gradient tolerance would. Measured residual here is ~1.5e-2,
    # against ~1.7e-4 when the optimizer was allowed to grind 33k further
    # iterations for 4e-5 of relative loss.
    #
    # It still discriminates: the L1 subdifferential is [-1/changepoint_prior_scale, 1/changepoint_prior_scale] =
    # [-20, 20], and perturbing k by 0.05 drives the residual to ~7 (see
    # test_first_order_residuals_reject_a_near_miss), so this keeps two orders
    # of headroom over a near-miss.
    assert smooth_residual < 5e-2
    assert delta_violation < 1e-6


def test_fit_reaches_first_order_optimum(small_df):
    """Same certificate for the Python path, which the same non-smoothness
    used to leave stranded 17.8% above the optimum while reporting success."""
    model = pin_yearly_only(AnalyticProphet())
    model.fit(small_df, backend="python", analytic=True, initial_params=MATCHED_INIT, fixed_sigma_obs=SIGMA_OBS)

    smooth_residual, delta_violation = first_order_residuals(model, model.get_parameters())

    assert smooth_residual < 1e-3
    assert delta_violation < 1e-6


def test_first_order_residuals_reject_a_near_miss(small_df):
    """Guards the certificate itself: a point a hair off the optimum has to
    fail it, otherwise the two tests above would pass on anything."""
    model = pin_yearly_only(AnalyticProphet())
    model.fit(small_df, backend="python", analytic=True, initial_params=MATCHED_INIT, fixed_sigma_obs=SIGMA_OBS)

    perturbed = model.get_parameters().copy()
    perturbed[K_IDX] += 0.05

    smooth_residual, _ = first_order_residuals(model, perturbed)
    assert smooth_residual > 1.0


def test_objective_is_not_convex_once_sigma_obs_is_free(prepared_model):
    """Freeing sigma_obs costs the convexity that issue #5 leaned on.

    Along the sigma_obs axis the objective carries T*log(sigma_obs), which is
    concave; between the 1/(2*sigma_obs^2) term below and the sigma_obs^2 prior
    above there is a window where it dominates. This exhibits a concrete
    violation of the midpoint inequality, deterministically -- no optimizer
    involved, so it cannot go flaky.

    It replaces a multi-start test that asserted every starting point reaches
    the same optimum. That was true while the problem was convex; it is now
    false, and measurably so -- 2 of 5 perturbed starts land on local optima
    1206 and 912 nats worse than Prophet's deterministic initialization does.
    Which is precisely why that initialization matters (#12).
    """
    def objective_at(sigma_obs):
        params = np.zeros(2 + N_CHANGE_POINTS + 1 + 2 * n_yearly)
        params[SIGMA_OBS_IDX] = sigma_obs
        return prepared_model._minus_log_posterior(params)

    low, high = 1.0, 3.0
    chord_midpoint = (objective_at(low) + objective_at(high)) / 2
    at_midpoint = objective_at((low + high) / 2)

    # convexity would require f(mid) <= (f(low) + f(high)) / 2
    assert at_midpoint > chord_midpoint


def test_both_fit_paths_record_a_monotone_decreasing_trajectory(small_df, compiled_optimizer_module):
    """Both optimizers now expose a per-iteration loss, and neither should ever
    move uphill on a convex objective."""
    python_model, cpp_model = fit_both(small_df)
    cpp_model.fit(small_df, initial_params=MATCHED_INIT, lib_path=compiled_optimizer_module)

    for name, trajectory in (("fit", python_model.loss_over_iterations),
                             ("compiled", cpp_model.loss_over_iterations)):
        trajectory = np.asarray(trajectory)
        assert len(trajectory) > 1, f"{name} recorded no trajectory"
        assert np.all(np.diff(trajectory) <= 1e-9), f"{name} loss increased between iterations"
