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

Both sides now estimate sigma_obs (#4 for fit(), #18 for fit_cpp()), so no
pinning is needed any more: the two objectives are identical term for term,
and the comparison is over the whole parameter vector.

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

Per-iteration trajectories are not compared elementwise. Since #23 both sides
solve the same split reformulation with a bound-constrained L-BFGS, but they
are still different implementations (scipy's Fortran L-BFGS-B vs LBFGSpp),
and the C++ trace samples objective evaluations rather than iterations. What convexity
actually promises -- and what is asserted here -- is that both descend
monotonically to the same optimal value, and that they agree exactly on the
objective itself at any shared point (see test_cpp_optimizer_convergence.py).
"""

import numpy as np
import pytest

from customProphet import CustomProphet, N_CHANGE_POINTS, n_yearly


def test_fit_and_fit_cpp_converge_to_same_loss_from_matched_init(peyton_manning_df, compiled_optimizer_module):
    small_df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    matched_init = {
        "k": 0.0,
        "m": 0.0,
        "delta": np.zeros(N_CHANGE_POINTS),
        "beta": np.zeros(2 * n_yearly),
    }
    python_model = CustomProphet()
    python_model.fit(small_df, analytic=True, initial_params=matched_init)
    assert python_model.opt.success

    cpp_model = CustomProphet()
    cpp_model.fit_cpp(small_df, initial_params=matched_init, lib_path=compiled_optimizer_module)

    # Both sides share the exact same (k, m, delta, beta) objective for a
    # fixed sigma_obs, so evaluate both parameter vectors on the Python
    # reference posterior to get a loss that's directly comparable.
    python_loss = python_model._minus_log_posterior(python_model.opt_params)
    cpp_loss = python_model._minus_log_posterior(cpp_model.opt_params)

    # Back to 1e-6, the value #21 had to loosen to 1e-4. That loosening was
    # forced by the paths optimizing *different* problems -- scipy on the split
    # reformulation against OWL-QN on the natural one -- which left them 3.8e-5
    # apart on the full series. #23 put the split reformulation in the C++ too,
    # so both now solve the same problem with a bound-constrained L-BFGS, and
    # the measured gap is 4.2e-9 / 8.2e-9 / 1.5e-8 at T = 300 / 1000 / 2905.
    # 1e-6 keeps roughly two orders of headroom over the worst of those.
    assert cpp_loss == pytest.approx(python_loss, rel=1e-6)

    # Same start, same objective -- and since #18 the two objectives agree
    # exactly, with no constant offset between them, so the trajectories are
    # directly comparable.
    trajectories = (
        np.asarray(python_model.loss_over_iterations),
        np.asarray(cpp_model.loss_over_iterations),
    )
    for trajectory in trajectories:
        assert np.all(np.diff(trajectory) <= 1e-9)
        assert trajectory[-1] == pytest.approx(cpp_loss, rel=1e-6)


def test_compiled_extension_exposes_its_entry_points(cpp_module):
    """Sanity check on the build step itself, independent of convergence: the
    extension imports and exposes the entry points fit_cpp() calls."""
    assert callable(cpp_module.optimize)
    assert callable(cpp_module.minus_log_posterior_and_gradient)
