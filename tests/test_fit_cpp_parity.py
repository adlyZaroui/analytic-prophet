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

xfail, not a hard failure: building legacy/optimize.cpp and running it from
matched init surfaced a real bug this way, as the issue anticipated --
fit_cpp()'s compiled optimizer terminates after only a couple of iterations
with LBFGSERR_ROUNDING_ERROR, nowhere near converged (loss ~18.5 vs fit()'s
~2.57 on a 300-row slice from zero init). Tracked in issue #8. This test
stays xfail(strict=True) until that's fixed, so it flips to an error (not a
silent pass) the moment someone tightens the tolerance below without
actually fixing optimize.cpp, and to a clear "XPASS" the moment the real fix
lands.
"""
import ctypes

import numpy as np
import pytest

from customProphet import CustomProphet, N_CHANGE_POINTS, n_yearly


@pytest.mark.xfail(
    reason="fit_cpp()'s compiled optimizer terminates early with LBFGSERR_ROUNDING_ERROR, "
           "far from converged -- see issue #8",
    strict=True,
)
def test_fit_and_fit_cpp_converge_to_same_loss_from_matched_init(peyton_manning_df, compiled_optimizer_lib):
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

    assert cpp_loss == pytest.approx(python_loss, rel=0.05)


def test_compiled_lib_exposes_optimize_symbol(compiled_optimizer_lib):
    """Sanity check on the build step itself, independent of convergence:
    the shared library builds and exposes the `optimize` entry point
    fit_cpp() calls through ctypes."""
    lib = ctypes.CDLL(compiled_optimizer_lib)
    assert hasattr(lib, "optimize")
