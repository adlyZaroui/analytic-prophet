"""
Issue #12: the model's constants and initialization, checked against
facebook/prophet (additive mode, linear growth, MAP).

Sources these are pinned against:
  [stan] https://github.com/facebook/prophet/blob/main/python/stan/prophet.stan
  [fc]   https://github.com/facebook/prophet/blob/main/python/prophet/forecaster.py

The point of pinning constants in a test is not that a literal equals a
literal -- it's that each one carries its category and its provenance, so a
later change has to be deliberate. The categories matter because the failure
mode they guard against is collapsing a SCALE (the scale of a prior on a fitted
parameter) into a FITTED value.
"""
import numpy as np
import pytest

import customProphet
from customProphet import (CustomProphet, CHANGEPOINT_RANGE, N_CHANGE_POINTS, SIGMA,
                           SIGMA_OBS_INIT, SIGMA_OBS_PRIOR_SCALE, TAU, linear_growth_init,
                           n_yearly, sigma_k, sigma_m)


def test_prior_scales_match_prophet_stan():
    """SCALE constants. [stan] model block:
        k ~ normal(0, 5); m ~ normal(0, 5);
        delta ~ double_exponential(0, changepoint_prior_scale);
        sigma_obs ~ normal(0, 0.5); beta ~ normal(0, sigmas)
    changepoint_prior_scale and sigmas are Prophet's changepoint_prior_scale / seasonality_prior_scale
    defaults; the other three are hardcoded in the Stan file.
    """
    assert sigma_k == 5
    assert sigma_m == 5
    assert SIGMA_OBS_PRIOR_SCALE == 0.5
    assert TAU == 0.05      # [fc] changepoint_prior_scale=0.05
    assert SIGMA == 10.0    # [fc] seasonality_prior_scale=10.0


def test_structural_constants_match_prophet():
    """FIXED constants. [fc] Prophet.__init__ and set_auto_seasonalities."""
    assert N_CHANGE_POINTS == 25
    assert CHANGEPOINT_RANGE == 0.8
    assert n_yearly == 10                 # K = 2 * 10 = 20 seasonality columns
    assert 2 + N_CHANGE_POINTS + 1 + 2 * n_yearly == 48  # Stan's 2 + S + 1 + K


def test_scaling_conventions(prepared_model):
    """FIXED. y by absmax, t to [0, 1]."""
    assert prepared_model.y_scale == np.max(np.abs(prepared_model.y))
    np.testing.assert_allclose(prepared_model.y_scaled,
                               prepared_model.y / prepared_model.y_scale)
    assert prepared_model.t.min() == pytest.approx(0.0)
    assert prepared_model.t.max() == pytest.approx(1.0)


def test_linear_growth_init_passes_through_first_and_last_points(prepared_model):
    """[fc] linear_growth_init: the line through the first and last points of
    the scaled series, so it reproduces both exactly."""
    t, y = prepared_model.t, prepared_model.y_scaled
    k, m = linear_growth_init(t, y)

    i0, i1 = int(np.argmin(t)), int(np.argmax(t))
    assert k * t[i0] + m == pytest.approx(y[i0])
    assert k * t[i1] + m == pytest.approx(y[i1])


def test_fit_starts_from_prophets_deterministic_initialization(prepared_model, peyton_manning_df, monkeypatch):
    """[fc] calculate_initial_params: k/m from linear_growth_init, delta and
    beta zeros, sigma_obs 1.0 -- not Stan's random init, which Prophet never
    reaches because it always passes explicit values."""
    seen = {}
    original = customProphet.from_dict_to_array

    def capture(params, layout=None):
        seen.update(params)
        return original(params) if layout is None else original(params, layout)

    monkeypatch.setattr(customProphet, "from_dict_to_array", capture)

    model = CustomProphet()
    model.fit(peyton_manning_df.iloc[:200].reset_index(drop=True), analytic=True)

    expected_k, expected_m = linear_growth_init(model.t, model.y_scaled)
    assert seen["k"] == pytest.approx(expected_k)
    assert seen["m"] == pytest.approx(expected_m)
    np.testing.assert_array_equal(seen["delta"], np.zeros(N_CHANGE_POINTS))
    assert seen["sigma_obs"] == SIGMA_OBS_INIT
    # beta is zeros at whatever width the selected components need -- 200 rows
    # of daily data gives weekly only, so 2 * 3 rather than 2 * n_yearly
    np.testing.assert_array_equal(seen["beta"],
                                  np.zeros(model.layout.n_seasonality_columns))
    assert list(model.seasonalities) == ["weekly"]


def test_fit_cpp_is_deterministic(peyton_manning_df, compiled_optimizer_module):
    """fit_cpp() used to draw an unseeded 2.0 * N(0, 1) init, so repeated fits
    on identical data disagreed. It now starts where fit() starts."""
    small_df = peyton_manning_df.iloc[:300].reset_index(drop=True)

    runs = []
    for _ in range(2):
        model = CustomProphet()
        model.fit_cpp(small_df, lib_path=compiled_optimizer_module)
        runs.append(model.get_parameters())

    np.testing.assert_array_equal(runs[0], runs[1])


def test_both_fit_paths_start_from_the_same_point(peyton_manning_df, compiled_optimizer_module):
    """The two entry points must initialize identically, or "same model" is
    only true asymptotically."""
    small_df = peyton_manning_df.iloc[:300].reset_index(drop=True)

    python_model = CustomProphet()
    python_model.fit(small_df, analytic=True)

    cpp_model = CustomProphet()
    cpp_model.fit_cpp(small_df, lib_path=compiled_optimizer_module)

    expected = linear_growth_init(python_model.t, python_model.y_scaled)
    assert linear_growth_init(cpp_model.t, cpp_model.y_scaled) == expected

    # and they land on the same optimum from there
    python_loss = python_model._minus_log_posterior(python_model.get_parameters())
    cpp_loss = python_model._minus_log_posterior(cpp_model.get_parameters())
    assert cpp_loss == pytest.approx(python_loss, rel=1e-6)


def test_trend_is_continuous_at_changepoints():
    """The code now uses `>=` to match [stan] get_changepoint_matrix
    (`t[i] >= changepoints_t[cp_idx]`). That alignment provably changes nothing
    numerically, and this pins down why.

    Changepoint j contributes `a_j(t) * delta_j * (t - s_j)` to the trend --
    delta_j enters the slope and gamma_j = -s_j * delta_j cancels it in the
    offset. At t == s_j that contribution is zero whichever way the indicator
    goes, so the trend is continuous there and `>=` and `>` agree exactly.

    If someone changes the trend parameterization so continuity no longer
    holds, this fails and the choice of comparison starts to matter.
    """
    changepoints_t = np.array([0.25, 0.5, 0.75])
    delta = np.array([1.3, -0.7, 2.0])
    t_at_changepoints = changepoints_t.copy()

    inclusive = customProphet.predict_trend(
        k=0.4, m=0.1, delta=delta, changepoints_t=changepoints_t,
        t=t_at_changepoints, y_scale=1.0)

    # The same trend under the exclusive convention.
    A = (t_at_changepoints[:, None] > changepoints_t) * 1
    gamma = -changepoints_t * delta
    exclusive = ((0.4 + customProphet.det_dot(A, delta)) * t_at_changepoints
                 + (0.1 + customProphet.det_dot(A, gamma)))

    # Equal mathematically, but not bit-for-bit: including a changepoint makes
    # the arithmetic go (k + delta) * s_j - s_j * delta instead of k * s_j, and
    # those round differently in the last ulp. The agreement is exact in real
    # arithmetic and ~1e-16 in floating point.
    np.testing.assert_allclose(inclusive, exclusive, rtol=0, atol=1e-15)

    # and the trend really is continuous across a changepoint
    eps = 1e-9
    around = customProphet.predict_trend(
        k=0.4, m=0.1, delta=delta, changepoints_t=changepoints_t,
        t=np.array([0.5 - eps, 0.5, 0.5 + eps]), y_scale=1.0)
    assert around[1] == pytest.approx(around[0], abs=1e-6)
    assert around[1] == pytest.approx(around[2], abs=1e-6)


def test_stan_convergence_constants_match_cmdstan_defaults():
    """Issue #21. Prophet's CmdStanPyBackend.fit calls
    optimize(algorithm='LBFGS', iter=int(1e4)) and sets no tolerances or
    history_size, so CmdStan's defaults are what the original runs under.
    [stan] src/stan/optimization/bfgs.hpp, ConvergenceOptions.
    """
    assert customProphet.STAN_EPS == np.finfo(float).eps   # Stan's machine epsilon
    assert customProphet.STAN_TOL_REL_OBJ == 1e+4          # tol_rel_obj
    assert customProphet.STAN_TOL_GRAD == 1e-8             # tol_grad
    assert customProphet.STAN_MAX_ITERATIONS == 10000      # Prophet's iter=int(1e4)


def test_cpp_converges_within_prophets_iteration_cap(peyton_manning_df, compiled_optimizer_module):
    """Before #21 the C++ core had no reachable stopping test -- `past = 0`
    disabled the objective-change test and the gradient threshold was
    unattainable -- so every run ended on line-search exhaustion tens of
    thousands of iterations past convergence.

    #23 then replaced OWL-QN with L-BFGS-B on the split reformulation, which
    converges in a fraction of the iterations. The criterion that fires is no
    longer named: LBFGSpp reports an iteration count, not which of its tests
    stopped the run.
    """
    model = CustomProphet()
    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.opt_status == 0
    assert model.opt_status_message == "converged"
    # Prophet passes iter=int(1e4); a converged run lands well inside it
    assert 0 < model.opt.n_iterations < customProphet.STAN_MAX_ITERATIONS


@pytest.mark.parametrize("n_changepoints,fourier_order", [(25, 10), (5, 10), (25, 3), (40, 6)])
def test_cpp_core_accepts_any_dimensions(prepared_model, cpp_module, n_changepoints, fourier_order):
    """#3: S and the Fourier order are arguments now, not compiled-in literals.

    The C++ previously carried `params.segment(2, 25)` and
    `fourier_series(..., 10)`, so any other shape either mis-sliced
    silently or aborted the process with an Eigen "invalid matrix product".
    """
    from customProphet import SIGMA_OBS_PRIOR_SCALE, YEARLY_PERIOD

    model = prepared_model
    model.n_changepoints = n_changepoints
    model.set_changepoints()

    n_params = 2 + n_changepoints + 2 * fourier_order + 1
    params = np.zeros(n_params)
    sigmas = np.full(2 * fourier_order, SIGMA, dtype=float)

    value, gradient = cpp_module.minus_log_posterior_and_gradient(
        params=params, t=model.t, changepoints_t=model.changepoints_t,
        t_seasonality=model.t_seasonality, y_scaled=model.y_scaled,
        sigma_obs_prior_scale=SIGMA_OBS_PRIOR_SCALE, sigma_k=model.sigma_k,
        sigma_m=model.sigma_m, sigmas=sigmas, changepoint_prior_scale=model.changepoint_prior_scale,
        fourier_orders=[fourier_order], seasonality_periods=[YEARLY_PERIOD])

    assert np.isfinite(value)
    assert gradient.shape == (n_params,)


def test_mismatched_parameter_length_is_rejected(prepared_model, cpp_module):
    """A length inconsistent with S and K names both rather than aborting."""
    from customProphet import SIGMA_OBS_PRIOR_SCALE, YEARLY_PERIOD

    with pytest.raises(ValueError, match="params has length"):
        cpp_module.minus_log_posterior_and_gradient(
            params=np.zeros(47), t=prepared_model.t,
            changepoints_t=prepared_model.changepoints_t,
            t_seasonality=prepared_model.t_seasonality,
            y_scaled=prepared_model.y_scaled,
            sigma_obs_prior_scale=SIGMA_OBS_PRIOR_SCALE, sigma_k=prepared_model.sigma_k,
            sigma_m=prepared_model.sigma_m, sigmas=prepared_model.sigmas,
            changepoint_prior_scale=prepared_model.changepoint_prior_scale, fourier_orders=[10], seasonality_periods=[YEARLY_PERIOD])


@pytest.mark.parametrize("n_changepoints", [5, 10, 40])
def test_non_default_changepoint_count_fits(peyton_manning_df, compiled_optimizer_module,
                                            n_changepoints):
    """#16 task 2: the Python layout follows the model rather than the module
    constants, so S is no longer pinned at 25.

    This used to raise NotImplementedError by design -- the parameter vector
    was sliced with compiled-in offsets, and anything else mis-sliced silently.
    """
    model = CustomProphet()
    model.n_changepoints = n_changepoints
    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.layout.n_changepoints == n_changepoints
    assert model.get_parameters().shape == (model.layout.size,)
    assert np.all(np.isfinite(model.get_parameters()))
    assert model.sigma_obs > 0
