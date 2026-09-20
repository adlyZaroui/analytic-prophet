"""
Issue #8: fit_cpp()'s compiled optimizer terminated after ~2 iterations with
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
|delta|/tau term in the posterior. That makes the objective non-differentiable
at delta = 0 -- and the optimum sits exactly on those kinks, since the prior is
what drives most changepoint rates to zero. More-Thuente, liblbfgs's default
line search, assumes a smooth objective and cannot bracket a step across a
kink, so it gave up almost immediately. Sweeping tau showed the failure
tracking the L1 strength exactly: 1/tau = 20 died at 2 iterations, while
1/tau = 1e-6 (an effectively smooth objective) ran 3528.

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

from customProphet import (CustomProphet, N_CHANGE_POINTS, n_yearly, K_IDX, M_IDX,
                           DELTA_SLICE, SIGMA_OBS_IDX, BETA_SLICE)

LBFGSERR_ROUNDING_ERROR = -1001

SIGMA_OBS = 1.0
MATCHED_INIT = {
    "k": 0.0,
    "m": 0.0,
    "delta": np.zeros(N_CHANGE_POINTS),
    "beta": np.zeros(2 * n_yearly),
}


def canonical_to_cpp(params):
    """Drop the sigma_obs slot: the C++ core never estimates it."""
    return np.delete(np.asarray(params, dtype=np.float64), SIGMA_OBS_IDX)


def fit_both(df):
    """Run both fit paths from the same start with sigma_obs pinned the same."""
    python_model = CustomProphet()
    python_model.fit(df, analytic=True, initial_params=MATCHED_INIT, fixed_sigma_obs=SIGMA_OBS)

    cpp_model = CustomProphet()
    cpp_model.sigma_obs = SIGMA_OBS
    return python_model, cpp_model


def first_order_residuals(model, params):
    """Subgradient optimality residuals for  f_smooth(x) + sum_j|delta_j|/tau.

    At a minimum of a convex objective of that form:
        delta_j != 0  ->  d f_smooth/d delta_j + sign(delta_j)/tau == 0
        delta_j == 0  ->  |d f_smooth/d delta_j| <= 1/tau
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
    threshold = 1.0 / model.tau

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
def test_cpp_gradient_matches_python_analytic(prepared_model, cpp_mlp_and_gradient, cpp_loss_offset, case):
    """Direction 2 from the issue: is the C++ gradient subtly wrong?

    It is not. The C++ objective and gradient agree with the Python analytic
    reference to ~1e-9 at every point tried -- including "zeros", which sits
    exactly on the delta=0 kink where the two could most plausibly disagree.
    This is what rules out a gradient bug and points at the line search.
    """
    rng = np.random.default_rng(seed=0)
    cpp_params = {
        "zeros": np.zeros(47),
        "k_only": np.concatenate(([0.4, 0.0], np.zeros(N_CHANGE_POINTS), np.zeros(2 * n_yearly))),
        "random_small": rng.normal(scale=0.5, size=47),
        "random_wide": rng.normal(scale=2.0, size=47),  # fit_cpp's own STAN-style init
    }[case]

    cpp_mlp, cpp_grad = cpp_mlp_and_gradient(prepared_model, cpp_params, SIGMA_OBS)

    canonical = np.insert(cpp_params, SIGMA_OBS_IDX, SIGMA_OBS)
    py_mlp, py_grad = prepared_model._minus_log_posteriorAndGradient(canonical)

    # The Python objective additionally carries the two sigma_obs terms the C++
    # side omits (it never estimates sigma_obs); with sigma_obs fixed they are
    # an additive constant, so subtract it to compare like for like.
    offset = cpp_loss_offset(prepared_model, SIGMA_OBS)

    assert cpp_mlp == pytest.approx(py_mlp - offset, rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(cpp_grad, canonical_to_cpp(py_grad), rtol=1e-9, atol=1e-8)


def test_cpp_optimizer_no_longer_bails_out_early(small_df, compiled_optimizer_lib):
    """The regression test for the bug as reported: it used to stop after 2
    iterations with LBFGSERR_ROUNDING_ERROR."""
    model = CustomProphet()
    model.sigma_obs = SIGMA_OBS
    model.fit_cpp(small_df, initial_params=MATCHED_INIT, lib_path=compiled_optimizer_lib)

    assert model.opt_status != LBFGSERR_ROUNDING_ERROR
    assert len(model.loss_over_iterations) > 50


def test_cpp_optimizer_reaches_first_order_optimum(small_df, compiled_optimizer_lib):
    """The fitted point satisfies the subgradient optimality conditions, so it
    is the optimum of a convex objective -- not merely 'where the optimizer
    happened to stop'."""
    reference, model = fit_both(small_df)
    model.fit_cpp(small_df, initial_params=MATCHED_INIT, lib_path=compiled_optimizer_lib)

    smooth_residual, delta_violation = first_order_residuals(reference, model.opt_params)

    # Scale: the L1 subdifferential is [-1/tau, 1/tau] = [-20, 20], and
    # perturbing k by 0.05 pushes the smooth residual to ~7, so 1e-3 is a
    # demanding bound with four orders of headroom over a near-miss.
    assert smooth_residual < 1e-3
    assert delta_violation < 1e-6


def test_fit_reaches_first_order_optimum(small_df):
    """Same certificate for the Python path, which the same non-smoothness
    used to leave stranded 17.8% above the optimum while reporting success."""
    model = CustomProphet()
    model.fit(small_df, analytic=True, initial_params=MATCHED_INIT, fixed_sigma_obs=SIGMA_OBS)

    smooth_residual, delta_violation = first_order_residuals(model, model.opt_params)

    assert smooth_residual < 1e-3
    assert delta_violation < 1e-6


def test_first_order_residuals_reject_a_near_miss(small_df):
    """Guards the certificate itself: a point a hair off the optimum has to
    fail it, otherwise the two tests above would pass on anything."""
    model = CustomProphet()
    model.fit(small_df, analytic=True, initial_params=MATCHED_INIT, fixed_sigma_obs=SIGMA_OBS)

    perturbed = model.opt_params.copy()
    perturbed[K_IDX] += 0.05

    smooth_residual, _ = first_order_residuals(model, perturbed)
    assert smooth_residual > 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cpp_optimizer_converges_from_any_start(small_df, compiled_optimizer_lib, seed):
    """The objective is convex in (k, m, delta, beta) for fixed sigma_obs, so
    the optimal value does not depend on where the search starts. Before the
    fix the loss varied wildly with the starting point, because the line search
    quit at whatever point it happened to reach."""
    reference, _ = fit_both(small_df)
    target = reference._minus_log_posterior(reference.opt_params)

    rng = np.random.default_rng(seed)
    init = {
        "k": rng.normal(),
        "m": rng.normal(),
        "delta": rng.normal(scale=0.5, size=N_CHANGE_POINTS),
        "beta": rng.normal(scale=0.5, size=2 * n_yearly),
    }

    model = CustomProphet()
    model.sigma_obs = SIGMA_OBS
    model.fit_cpp(small_df, initial_params=init, lib_path=compiled_optimizer_lib)

    assert reference._minus_log_posterior(model.opt_params) == pytest.approx(target, rel=1e-6)


def test_both_fit_paths_record_a_monotone_decreasing_trajectory(small_df, compiled_optimizer_lib):
    """Both optimizers now expose a per-iteration loss, and neither should ever
    move uphill on a convex objective."""
    python_model, cpp_model = fit_both(small_df)
    cpp_model.fit_cpp(small_df, initial_params=MATCHED_INIT, lib_path=compiled_optimizer_lib)

    for name, trajectory in (("fit", python_model.loss_over_iterations),
                             ("fit_cpp", cpp_model.loss_over_iterations)):
        trajectory = np.asarray(trajectory)
        assert len(trajectory) > 1, f"{name} recorded no trajectory"
        assert np.all(np.diff(trajectory) <= 1e-9), f"{name} loss increased between iterations"
