"""
Issue #4: sigma_obs was drawn once in __init__ (self.sigma_obs = rng.normal(...))
and never entered the optimized parameter vector -- extract_params only ever
unpacked (k, m, delta, beta), so the analytic posterior/gradient always read
the noise level off self, not off the point scipy was actually searching over.

Vanilla Prophet (see "Forecasting at Scale") estimates sigma_obs as part of
MAP fitting: it's a Stan parameter with prior sigma_obs ~ normal(0, 0.5),
constrained positive, appearing in the Gaussian likelihood
y ~ normal(trend + seasonality, sigma_obs). That likelihood's normalizing
constant is T*log(sigma_obs) -- droppable when sigma_obs is a fixed constant,
but not once it's a variable being optimized over.

These tests target the fix: sigma_obs now lives in the parameter vector
(layout: k, m, delta, sigma_obs, beta -- see SIGMA_OBS_IDX in analytic_prophet/forecaster.py,
matching the order of Prophet's own Stan parameters block), the analytic
posterior includes the T*log(sigma_obs) term and the sigma_obs prior, and the
gradient has a matching dsigma_obs term. test_analytic_gradient_matches_numerical
in test_gradient_numerical.py already covers dsigma_obs's correctness generically
(random_params spans the full vector including that index); the tests here
target the specific behavior described in the issue instead.
"""
import numpy as np
import pytest

from analytic_prophet import SIGMA_OBS_IDX, SIGMA_OBS_PRIOR_SCALE, SIGMA_OBS_INIT, extract_params


def test_extract_params_unpacks_sigma_obs_at_its_own_slot(prepared_model, param_size):
    params = np.arange(param_size, dtype=float)
    k, m, delta, sigma_obs, beta = extract_params(params)

    assert sigma_obs == params[SIGMA_OBS_IDX]
    # sigma_obs must sit strictly between delta and beta, not be swallowed by either
    assert sigma_obs not in delta
    assert sigma_obs not in beta


def test_minus_log_posterior_matches_closed_form_gaussian_nll(prepared_model, param_size):
    """With k=m=delta=beta=0 the residual is just y_scaled, so the
    posterior reduces to a closed form that can be checked by hand: the
    Gaussian NLL (including its T*log(sigma_obs) normalizing term) plus the
    normal(0, SIGMA_OBS_PRIOR_SCALE) prior on sigma_obs."""
    model = prepared_model
    sigma_obs = 0.3
    params = np.zeros(param_size)
    params[SIGMA_OBS_IDX] = sigma_obs

    mlp = model._minus_log_posterior(params)

    r = model.y_scaled
    expected = (
        model.T * np.log(sigma_obs)
        + np.sum(r**2) / (2 * sigma_obs**2)
        + sigma_obs**2 / (2 * SIGMA_OBS_PRIOR_SCALE**2)
    )
    assert mlp == pytest.approx(expected)


def test_minus_log_posterior_depends_on_params_sigma_obs_not_self_attribute(prepared_model, random_params):
    """The bug: _minus_log_posterior read self.sigma_obs, ignoring whatever
    value scipy had placed at the sigma_obs slot of the vector it passed in.
    Setting self.sigma_obs to an absurd, unmistakable value and perturbing
    only the vector's sigma_obs entry should still change the result -- if it
    doesn't, the function is silently falling back to self.sigma_obs again."""
    model = prepared_model
    model.sigma_obs = 999.0

    baseline = model._minus_log_posterior(random_params)

    perturbed = random_params.copy()
    perturbed[SIGMA_OBS_IDX] *= 2.0
    result = model._minus_log_posterior(perturbed)

    assert result != pytest.approx(baseline)


def test_fit_estimates_sigma_obs_away_from_init(prepared_model, peyton_manning_df):
    """End-to-end: fit() should move sigma_obs away from its initial value
    (SIGMA_OBS_INIT) and land somewhere positive -- confirming it's a fitted
    parameter, not the fixed draw the issue describes. Uses a small slice of
    the data and default settings so the test stays fast."""
    model = prepared_model
    small_df = peyton_manning_df.iloc[:300].reset_index(drop=True)

    model.fit(small_df)

    assert len(model.get_parameters()) == model.layout.size
    fitted_sigma_obs = model.params["sigma_obs"][0][0]

    assert fitted_sigma_obs > 0
    assert fitted_sigma_obs == pytest.approx(model.sigma_obs)
    assert fitted_sigma_obs != pytest.approx(SIGMA_OBS_INIT, rel=0.1)


def test_fit_keeps_sigma_obs_positive(prepared_model, peyton_manning_df):
    """Mirrors Stan's `real<lower=0> sigma_obs` constraint: L-BFGS-B is given
    an explicit lower bound on the sigma_obs slot so the optimizer can't wander
    into the domain where log(sigma_obs) and 1/sigma_obs**3 blow up."""
    model = prepared_model
    small_df = peyton_manning_df.iloc[:300].reset_index(drop=True)

    model.fit(small_df)

    assert model.params["sigma_obs"][0][0] > 0
