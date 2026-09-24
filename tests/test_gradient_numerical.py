"""
Validates the hand-derived analytic gradient against finite differences
of the same objective (_minus_log_posterior) -- independent of L-BFGS,
the C++ layer, or anything else. This checks the calculus itself.
"""
import numpy as np
import pytest

from analytic_prophet import SIGMA_OBS_IDX


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


def test_analytic_gradient_matches_numerical(prepared_model, random_params):
    analytic = prepared_model._gradient(random_params)
    numerical = numerical_gradient(prepared_model._minus_log_posterior, random_params)
    np.testing.assert_allclose(analytic, numerical, atol=1e-4, rtol=1e-4)


def test_combined_call_matches_separate_calls(prepared_model, random_params):
    """_minus_log_posteriorAndGradient is meant to be a fused version of
    _minus_log_posterior + _gradient, not a third independent
    implementation -- so it must agree with calling the two separately."""
    mlp_combined, grad_combined = prepared_model._minus_log_posteriorAndGradient(random_params)
    mlp_separate = prepared_model._minus_log_posterior(random_params)
    grad_separate = prepared_model._gradient(random_params)

    assert mlp_combined == pytest.approx(mlp_separate)
    np.testing.assert_allclose(grad_combined, grad_separate)


def test_gradient_finite_at_delta_equals_zero(prepared_model, param_size):
    """
    Regression test for the Laplace-prior kink: d|delta|/d(delta) is
    undefined exactly at delta=0. This isn't a theoretical edge case --
    your own fit() initializes k=m=0 and every delta/beta at 0
    (initial_params_dict), so plain .fit() starts every run at exactly
    this point. np.sign(0) == 0 (confirmed, not NaN), which is what
    _gradient relies on; this pins that down explicitly so it can't
    silently regress if the implementation ever changes.

    sigma_obs is set to fit()'s actual init value (1.0), not left at 0 --
    it appears as log(sigma_obs) and 1/sigma_obs**3 in the objective, so a
    literal zero there is a genuine domain violation, not a kink to probe.
    """
    params_at_kink = np.zeros(param_size)
    params_at_kink[SIGMA_OBS_IDX] = 1.0
    grad = prepared_model._gradient(params_at_kink)
    assert np.all(np.isfinite(grad)), "gradient must be finite at delta=0, got NaN/inf"
