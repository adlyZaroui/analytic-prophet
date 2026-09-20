"""
Issue #5: fit() initializes every parameter at exactly zero; fit_cpp() draws
a wide random Gaussian ("STAN initialization"). The underlying objective is
convex in (k, m, delta, beta) for any fixed sigma_obs -- trend and
seasonality are both linear in the parameters, and every remaining
likelihood/prior term is a convex function of an affine map, scaled by a
positive constant -- so both optimizers should converge to (about) the same
loss regardless of starting point. Running them from matched initial values
turns that claim into a check: a real mismatch points at an actual bug, not
"found a different local optimum," since convexity says there shouldn't be
one.

fit_cpp()'s compiled optimizer never estimates sigma_obs (see optimize.cpp's
extract_params -- it hard-codes a (k, m, delta, beta) layout with no sigma_obs
slot); fit() now does (issue #4). For a fair comparison this test pins
sigma_obs to the same fixed value on both sides via fit()'s fixed_sigma_obs
parameter, so both sides are optimizing the exact same (k, m, delta, beta)
objective.

Comparing final loss (not raw parameter vectors) is deliberate: with 25
changepoints against a few hundred data points and an L1 prior on delta, the
minimizer need not be unique even though the objective is convex, so two
correct optimizers can land on different delta vectors that are equally
good. Loss is the quantity convexity actually promises will agree.

Both sides were broken when this test was written, by the same root cause:
the Laplace prior on delta makes the objective non-smooth, and neither
optimizer handled that. fit_cpp() failed loudly (LBFGSERR_ROUNDING_ERROR
after ~2 iterations); fit() failed silently, stalling 17.8% above the
optimum while reporting success=True. Issue #8 fixed both, and this test
went from xfail to asserting real equality.

Per-iteration trajectories are not compared elementwise: the two sides run
genuinely different algorithms (scipy's Fortran L-BFGS-B over the smooth
split reformulation vs liblbfgs's OWL-QN), so their iterates differ even
though the objective, data and starting point are identical. What convexity
actually promises -- and what is asserted here -- is that both descend
monotonically to the same optimal value, and that they agree exactly on the
objective itself at any shared point (see test_cpp_optimizer_convergence.py).
"""
import ctypes

import numpy as np
import pytest

from customProphet import CustomProphet, N_CHANGE_POINTS, n_yearly


def test_fit_and_fit_cpp_converge_to_same_loss_from_matched_init(peyton_manning_df, compiled_optimizer_lib, cpp_loss_offset):
    small_df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    matched_init = {
        "k": 0.0,
        "m": 0.0,
        "delta": np.zeros(N_CHANGE_POINTS),
        "beta": np.zeros(2 * n_yearly),
    }
    fixed_sigma_obs = 1.0

    python_model = CustomProphet()
    python_model.fit(small_df, analytic=True, initial_params=matched_init, fixed_sigma_obs=fixed_sigma_obs)
    assert python_model.opt.success

    cpp_model = CustomProphet()
    cpp_model.sigma_obs = fixed_sigma_obs
    cpp_model.fit_cpp(small_df, initial_params=matched_init, lib_path=compiled_optimizer_lib)

    # Both sides share the exact same (k, m, delta, beta) objective for a
    # fixed sigma_obs, so evaluate both parameter vectors on the Python
    # reference posterior to get a loss that's directly comparable.
    python_loss = python_model._minus_log_posterior(python_model.opt_params)
    cpp_loss = python_model._minus_log_posterior(cpp_model.opt_params)

    assert cpp_loss == pytest.approx(python_loss, rel=1e-6)

    # Same start, same objective: both trajectories descend to that same value.
    # The C++ trajectory is in its own objective's units, hence the offset.
    offset = cpp_loss_offset(python_model, fixed_sigma_obs)
    trajectories = (
        np.asarray(python_model.loss_over_iterations),
        np.asarray(cpp_model.loss_over_iterations) + offset,
    )
    for trajectory in trajectories:
        assert np.all(np.diff(trajectory) <= 1e-9)
        assert trajectory[-1] == pytest.approx(cpp_loss, rel=1e-6)


def test_compiled_lib_exposes_optimize_symbol(compiled_optimizer_lib):
    """Sanity check on the build step itself, independent of convergence:
    the shared library builds and exposes the `optimize` entry point
    fit_cpp() calls through ctypes."""
    lib = ctypes.CDLL(compiled_optimizer_lib)
    assert hasattr(lib, "optimize")
