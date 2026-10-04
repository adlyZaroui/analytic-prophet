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

from analytic_prophet import AnalyticProphet, N_CHANGE_POINTS, n_yearly
from conftest import pin_yearly_only


def test_fit_and_fit_cpp_converge_to_same_loss_from_matched_init(peyton_manning_df, compiled_optimizer_module):
    small_df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    matched_init = {
        "k": 0.0,
        "m": 0.0,
        "delta": np.zeros(N_CHANGE_POINTS),
        "beta": np.zeros(2 * n_yearly),
    }
    python_model = pin_yearly_only(AnalyticProphet())
    python_model.fit(small_df, analytic=True, initial_params=matched_init)
    # This configuration -- yearly forced at order 10 on 328 days, from an
    # all-zero start, neither of which the defaults would produce -- is the one
    # place scipy reports ABNORMAL, which is #24. Since #25 the Newton fallback
    # catches it: the retry converges on the same objective to 1e-13, so the run
    # now reports success and the fitted point is unchanged.
    #
    # *Whether* it reports ABNORMAL is a property of the scipy build and not of
    # this code. On macOS/arm64 the line search fails here and Newton takes
    # over; in every CI job on Linux the same configuration converges under
    # L-BFGS and the fallback never fires. Both land on the same optimum, which
    # is what this test is about, so the path taken is no longer asserted (#103).
    assert python_model.optimizer_used in ("Newton", "LBFGS")
    assert np.all(np.isfinite(python_model.get_parameters()))

    cpp_model = pin_yearly_only(AnalyticProphet())
    cpp_model.fit_cpp(small_df, initial_params=matched_init, lib_path=compiled_optimizer_module)

    # Both sides share the exact same (k, m, delta, beta) objective for a
    # fixed sigma_obs, so evaluate both parameter vectors on the Python
    # reference posterior to get a loss that's directly comparable.
    python_loss = python_model._minus_log_posterior(python_model.get_parameters())
    cpp_loss = python_model._minus_log_posterior(cpp_model.get_parameters())

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
    for name, trajectory in zip(("python", "cpp"), trajectories):
        # Both traces are monotone by construction here: each records a value
        # only when one improves. The Python side reached this configuration
        # through the Newton fallback, whose line search accepts a step only on
        # a decrease; before #25 it was scipy's per-iteration trace, which rose
        # once by 8e-5 at step 12 of 170 on the run that reported ABNORMAL. The
        # bound keeps the claim the test is about -- the trajectory descends --
        # rather than pinning which optimizer produced it.
        rises = np.diff(trajectory)
        assert rises.max() <= 1e-9, name
        assert trajectory[-1] == pytest.approx(cpp_loss, rel=1e-6)
        assert trajectory[-1] < trajectory[0]


def test_compiled_extension_exposes_its_entry_points(cpp_module):
    """Sanity check on the build step itself, independent of convergence: the
    extension imports and exposes the entry points fit_cpp() calls."""
    assert callable(cpp_module.optimize)
    assert callable(cpp_module.minus_log_posterior_and_gradient)


# -- the preprocessing both paths share -----------------------------------

# Every attribute preprocess() is responsible for. Listed rather than derived,
# because the point is to notice when one path stops setting something the
# other does.
PREPROCESSED = ("T", "t", "t_seasonality", "y", "y_scaled", "y_scale",
                "changepoints_t", "layout", "sigmas", "s_a", "s_m",
                "condition_masks", "_fit_design_matrix", "_data_columns",
                "_data_column_count", "_holiday_columns",
                "train_holiday_names", "train_holiday_column_names")


def preprocessed_state(model):
    return {name: getattr(model, name) for name in PREPROCESSED}


def same(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        return left.shape == right.shape and (left.size == 0 or np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    try:
        return bool(left == right)
    except ValueError:
        return repr(left) == repr(right)


@pytest.mark.parametrize("n_rows", [300, 1000])
def test_both_paths_preprocess_the_history_identically(peyton_manning_df,
                                                       compiled_optimizer_module, n_rows):
    """The invariant `preprocess` exists to guarantee.

    It used to hold by coincidence -- 45 lines copied into each of `fit` and
    `fit_cpp`, which every change had to be made to twice. It now holds by
    construction, and this is what notices if the two ever diverge again.
    """
    df = peyton_manning_df.iloc[:n_rows].reset_index(drop=True)

    python_model = AnalyticProphet()
    python_model.fit(df, analytic=True)
    cpp_model = AnalyticProphet()
    cpp_model.fit_cpp(df, lib_path=compiled_optimizer_module)

    differing = [name for name in PREPROCESSED
                 if not same(getattr(python_model, name), getattr(cpp_model, name))]
    assert differing == []


def test_preprocess_leaves_the_same_state_a_fit_would(peyton_manning_df):
    """Called on its own, without optimizing. That it can be is the point: the
    history and the design matrix do not depend on which optimizer runs next."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)

    prepared = AnalyticProphet()
    prepared.preprocess(df)

    fitted = AnalyticProphet()
    fitted.fit(df, analytic=True)

    differing = [name for name in PREPROCESSED
                 if not same(getattr(prepared, name), getattr(fitted, name))]
    assert differing == []


def test_both_paths_start_from_the_same_initial_params(peyton_manning_df):
    """[fc] calculate_initial_params. Prophet passes these to Stan explicitly,
    so Stan's random init is never reached; here they are what both optimizers
    are handed, which is what makes the two paths comparable at all."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    model = AnalyticProphet()
    model.preprocess(df)

    defaults = model.calculate_initial_params()

    assert set(defaults) == {"k", "m", "delta", "sigma_obs", "beta"}
    assert defaults["sigma_obs"] == 1.0          # [fc] Stan's init, not a draw
    np.testing.assert_array_equal(defaults["delta"],
                                  np.zeros(model.layout.n_changepoints))
    np.testing.assert_array_equal(defaults["beta"],
                                  np.zeros(model.layout.n_regressor_columns))
    assert np.isfinite(defaults["k"]) and np.isfinite(defaults["m"])


def test_caller_supplied_initial_params_override_the_defaults(peyton_manning_df):
    """Including `sigma_obs`, which is the one key the two paths used to handle
    differently: `fit` seeded it and `fit_cpp` did not, reading it back with a
    `.get(..., SIGMA_OBS_INIT)` instead. The shared helper always sets it, so
    both now read the same key -- equivalent in every case, and the one place
    this refactor was not a literal move."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    model = AnalyticProphet()
    model.preprocess(df)

    given = model.calculate_initial_params({"k": 0.5, "sigma_obs": 0.25})

    assert given["k"] == 0.5
    assert given["sigma_obs"] == 0.25
    assert model.calculate_initial_params()["sigma_obs"] == 1.0   # not mutated
