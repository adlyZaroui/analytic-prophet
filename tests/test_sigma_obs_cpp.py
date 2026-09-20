"""
Issue #18: sigma_obs estimated in the C++ core, as in prophet.stan's
`parameters` block (`real<lower=0> sigma_obs`, prior `normal(0, 0.5)`).

The C++ vector runs [k, m, delta(S), beta(K), zeta] with zeta = log(sigma_obs)
last. The log parameterization is what keeps sigma_obs positive: liblbfgs has
no box constraints, so there is nothing else holding it above zero.

Three things are worth pinning here, and the middle one is the load-bearing
check: the T*log(sigma_obs) normalization term is droppable only while
sigma_obs is a constant, and if it is missing nothing penalizes sigma_obs
growing -- the residual term just shrinks monotonically as it rises.
test_fitted_sigma_obs_approaches_the_gaussian_mle is what catches that, since
the MLE it converges to is exactly the value that term balances against.
"""
import numpy as np
import pytest

from customProphet import (CustomProphet, BETA_SLICE, DELTA_SLICE, N_CHANGE_POINTS,
                           SIGMA_OBS_IDX, canonical_to_cpp, cpp_to_canonical,
                           fourier_components, n_yearly)

CPP_PARAM_SIZE = 2 + N_CHANGE_POINTS + 2 * n_yearly + 1


def test_zeta_gradient_matches_finite_differences(prepared_model, cpp_mlp_and_gradient):
    """The new component, checked against central differences of the C++
    objective itself -- the same treatment test_gradient_numerical.py gives the
    Python gradient.

    d/d_zeta = T - sum(r^2)/sigma_obs^2 + sigma_obs^2/scale^2, by chain rule
    from d/d_sigma_obs using d(sigma_obs)/d(zeta) = sigma_obs.
    """
    rng = np.random.default_rng(seed=0)
    params = rng.normal(scale=0.4, size=CPP_PARAM_SIZE)
    params[-1] = 0.3  # zeta, i.e. sigma_obs = exp(0.3)

    _, gradient = cpp_mlp_and_gradient(prepared_model, params)

    eps = 1e-6
    forward, backward = params.copy(), params.copy()
    forward[-1] += eps
    backward[-1] -= eps
    numerical = (cpp_mlp_and_gradient(prepared_model, forward)[0]
                 - cpp_mlp_and_gradient(prepared_model, backward)[0]) / (2 * eps)

    assert gradient[-1] == pytest.approx(numerical, rel=1e-6)


def test_fitted_sigma_obs_approaches_the_gaussian_mle(peyton_manning_df, compiled_optimizer_module):
    """With the priors on k, m, delta and beta made negligible, the posterior
    is essentially the Gaussian likelihood, whose stationary point in sigma_obs
    is the MLE sqrt(sum(r^2)/T).

    This is the check that catches a missing T*log(sigma_obs) term: without it
    the objective decreases monotonically in sigma_obs and the fit runs away
    instead of settling on the MLE.

    The sigma_obs prior stays at its real 0.5. It does not need widening: at a
    fitted sigma_obs of ~0.035 its contribution to the stationarity condition
    is sigma_obs^2/0.25 ~ 5e-3 against T = 150, five orders down.
    """
    model = CustomProphet()
    model.tau = model.sigma = model.sigma_k = model.sigma_m = 1e4

    model.fit_cpp(peyton_manning_df.iloc[:150].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    k, m = model.opt_params[0], model.opt_params[1]
    delta = model.opt_params[DELTA_SLICE]
    beta = model.opt_params[BETA_SLICE]

    A = (model.t_scaled[:, None] >= model.change_points) * 1
    gamma = -model.change_points * delta
    trend = (k + A.dot(delta)) * model.t_scaled + (m + A.dot(gamma))
    seasonality = fourier_components(
        model.t_scaled, 365.25 / model.scale_period, n_yearly).dot(beta)
    residuals = model.normalized_y - trend - seasonality

    mle = np.sqrt(np.sum(residuals**2) / model.T)
    assert model.sigma_obs == pytest.approx(mle, rel=1e-3)


def test_sigma_obs_is_positive_by_construction():
    """sigma_obs = exp(zeta), so no value of the optimized vector can produce a
    non-positive one -- which is what replaces the box constraint liblbfgs does
    not have."""
    for zeta in (-500.0, -1.0, 0.0, 1.0, 50.0):
        params = np.zeros(CPP_PARAM_SIZE)
        params[-1] = zeta
        assert cpp_to_canonical(params)[SIGMA_OBS_IDX] > 0


def test_fit_cpp_estimates_sigma_obs_rather_than_taking_it_as_given(peyton_manning_df, compiled_optimizer_module):
    """The behaviour change itself: sigma_obs comes out of the fit now, and
    whatever was sitting on the instance beforehand does not steer it."""
    small_df = peyton_manning_df.iloc[:300].reset_index(drop=True)

    fits = []
    for preset in (1.0, 0.5):
        model = CustomProphet()
        model.sigma_obs = preset   # used to be the value the C++ optimized against
        model.fit_cpp(small_df, lib_path=compiled_optimizer_module)
        fits.append(model.sigma_obs)

    assert fits[0] == pytest.approx(fits[1], rel=1e-9)
    assert 0 < fits[0] < 1.0        # data says ~0.034, far from either preset


def test_canonical_and_cpp_layouts_round_trip():
    """The two packings describe the same model; only the ordering and the log
    differ. [k, m, delta, sigma_obs, beta] <-> [k, m, delta, beta, zeta]."""
    rng = np.random.default_rng(seed=1)
    canonical = rng.normal(size=2 + N_CHANGE_POINTS + 1 + 2 * n_yearly)
    canonical[SIGMA_OBS_IDX] = 0.37

    round_tripped = cpp_to_canonical(canonical_to_cpp(canonical))

    np.testing.assert_allclose(round_tripped, canonical, rtol=1e-12)
