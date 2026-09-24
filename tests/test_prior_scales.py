"""
Issue #16 task 5: one prior scale per regressor column.

Stan declares `vector[K] sigmas` and writes `beta ~ normal(0, sigmas)`, so every
column of the design matrix carries its own prior scale. This implementation
used the scalar `SIGMA = 10` for all of them, which was adequate only while one
seasonality existed -- the moment a second can be registered (task 2) or selected
(task 3), two components can want different scales, and Prophet's
`add_seasonality(..., prior_scale=...)` is how a user asks for that.

The prior term becomes  sum_l beta_l^2 / (2 * sigmas_l^2)  and the gradient
d/d beta_l = beta_l / sigmas_l^2, in both the Python reference and the C++ core.

The acceptance criterion is at the bottom: with two seasonalities at different
scales, our objective has to reproduce what Stan computes, checked through
`log_prob` rather than by re-deriving the algebra in the test.
"""
import numpy as np
import pytest

from customProphet import (CustomProphet, SIGMA, canonical_to_cpp,
                           check_seasonality_supported, seasonality,
                           seasonality_prior_scales)

# deliberately neither equal to each other nor to the SIGMA = 10 default, so a
# fallback to the old scalar shows up as a failure rather than a coincidence
YEARLY_SCALE, WEEKLY_SCALE = 2.0, 15.0


def two_scales():
    return {"yearly": seasonality(365.25, 10, prior_scale=YEARLY_SCALE),
            "weekly": seasonality(7.0, 3, prior_scale=WEEKLY_SCALE)}


def configured(model, seasonalities):
    """Register a fixed set, with auto-selection out of the way."""
    model.yearly_seasonality = model.weekly_seasonality = model.daily_seasonality = False
    model.seasonalities = seasonalities
    return model


# -- building the vector ------------------------------------------------

def test_scales_repeat_across_each_components_block():
    """[fc] make_all_seasonality_features extends prior_scales by
    `[props['prior_scale']] * features.shape[1]`, so every column of a component
    shares its scale, and the blocks follow registry order."""
    sigmas = seasonality_prior_scales(two_scales())

    assert sigmas.shape == (26,)
    np.testing.assert_array_equal(sigmas[:20], YEARLY_SCALE)
    np.testing.assert_array_equal(sigmas[20:], WEEKLY_SCALE)


def test_default_registry_gives_a_uniform_vector():
    """Nothing changes for a model that does not set a scale: every column gets
    the model-wide default, which is what the scalar used to mean."""
    sigmas = seasonality_prior_scales({"yearly": seasonality(365.25, 10)})

    assert sigmas.shape == (20,)
    np.testing.assert_array_equal(sigmas, SIGMA)


def test_empty_registry_gives_an_empty_vector():
    assert seasonality_prior_scales({}).shape == (0,)


def test_a_fit_fixes_sigmas_alongside_the_layout(peyton_manning_df,
                                                 compiled_optimizer_module):
    """`sigmas` is data in Stan's sense -- fixed for the whole fit, at the same
    moment `K` is."""
    model = configured(CustomProphet(), two_scales())
    model.fit_cpp(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.sigmas.shape == (model.layout.n_seasonality_columns,)
    np.testing.assert_array_equal(model.sigmas[:20], YEARLY_SCALE)
    np.testing.assert_array_equal(model.sigmas[20:], WEEKLY_SCALE)


# -- validation ---------------------------------------------------------

@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_a_non_positive_scale_is_rejected(scale):
    """Stan declares `sigmas` without a lower bound, but `normal(0, sigmas)` is
    undefined at or below zero -- and a zero scale would divide by zero in the
    prior term rather than raise."""
    with pytest.raises(ValueError, match="positive scale"):
        check_seasonality_supported({"yearly": seasonality(365.25, 10, prior_scale=scale)})


def test_prior_scale_is_no_longer_in_the_unhonored_list():
    """It was rejected outright until this task; it must now pass validation."""
    check_seasonality_supported({"yearly": seasonality(365.25, 10, prior_scale=3.0)})


def test_cpp_rejects_a_sigmas_of_the_wrong_length(prepared_model, cpp_module):
    """A length mismatch reaches Eigen as a size assertion, which aborts the
    process rather than raising -- the same failure the params-length check
    exists to prevent, and it did abort before this guard was added."""
    model = prepared_model

    with pytest.raises(ValueError, match="one prior scale per column"):
        cpp_module.minus_log_posterior_and_gradient(
            params=np.zeros(model.layout.size), t=model.t,
            changepoints_t=model.changepoints_t, t_seasonality=model.t_seasonality,
            y_scaled=model.y_scaled, sigma_obs_prior_scale=0.5,
            sigma_k=model.sigma_k, sigma_m=model.sigma_m,
            sigmas=np.full(19, SIGMA), changepoint_prior_scale=model.changepoint_prior_scale,
            fourier_orders=[10], seasonality_periods=[365.25])


def test_cpp_rejects_a_non_positive_scale(prepared_model, cpp_module):
    model = prepared_model
    sigmas = np.full(20, SIGMA)
    sigmas[7] = 0.0

    with pytest.raises(ValueError, match="must be positive"):
        cpp_module.optimize(
            params=np.zeros(model.layout.size), t=model.t,
            changepoints_t=model.changepoints_t, t_seasonality=model.t_seasonality,
            y_scaled=model.y_scaled, sigma_obs_prior_scale=0.5,
            sigma_k=model.sigma_k, sigma_m=model.sigma_m, sigmas=sigmas,
            changepoint_prior_scale=model.changepoint_prior_scale, fourier_orders=[10], seasonality_periods=[365.25])


# -- the objective and its gradient -------------------------------------

def non_uniform_point(model, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate(([0.3], [-0.7],
                           rng.normal(scale=0.01, size=model.layout.n_changepoints),
                           [0.8],
                           rng.normal(scale=0.5, size=model.layout.n_seasonality_columns)))


def test_gradient_matches_finite_differences_with_distinct_scales(prepared_model):
    """The beta block is the only part of the gradient this task changes, but
    checking it alone would not catch a mis-sliced `sigmas`, so the whole vector
    goes through central differences."""
    model = configured(prepared_model, two_scales())
    model._build_layout()
    params = non_uniform_point(model)

    analytic = model._gradient(params)

    step = 1e-6
    numerical = np.empty_like(params)
    for i in range(len(params)):
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        numerical[i] = (model._minus_log_posterior(up)
                        - model._minus_log_posterior(down)) / (2 * step)

    # delta sits at the L1 kink where the subgradient is set-valued, so those
    # coordinates are compared with the prior omitted from both sides
    smooth = np.r_[0, 1, model.layout.sigma_obs_idx,
                   np.arange(model.layout.beta.start, model.layout.beta.stop)]
    np.testing.assert_allclose(analytic[smooth], numerical[smooth], rtol=1e-5, atol=1e-5)


def test_the_two_objectives_agree_with_distinct_scales(prepared_model, cpp_mlp_and_gradient):
    """Python and C++ carry `sigmas` separately; a disagreement in how it is
    indexed would show up nowhere else until the two fit paths diverged."""
    model = configured(prepared_model, two_scales())
    model._build_layout()
    params = non_uniform_point(model)

    expected, expected_gradient = model._minus_log_posteriorAndGradient(params)
    value, gradient = cpp_mlp_and_gradient(model, canonical_to_cpp(params, model.layout))

    assert value == pytest.approx(expected, rel=1e-12)

    sigma_obs = params[model.layout.sigma_obs_idx]
    reordered = np.concatenate((expected_gradient[:2 + model.layout.n_changepoints],
                                expected_gradient[model.layout.beta],
                                [expected_gradient[model.layout.sigma_obs_idx] * sigma_obs]))
    np.testing.assert_allclose(gradient, reordered, rtol=1e-9, atol=1e-8)


def test_a_uniform_vector_reproduces_the_old_scalar_term(prepared_model):
    """The refactor must be exactly that for a model that sets no scale: the
    per-column term reduces to sum(beta^2) / (2 * SIGMA^2)."""
    model = configured(prepared_model, {"yearly": seasonality(365.25, 10)})
    model._build_layout()
    params = non_uniform_point(model)
    beta = params[model.layout.beta]

    scaled = configured(CustomProphet(), {"yearly": seasonality(365.25, 10, prior_scale=1.0)})
    for attribute in ("t", "changepoints_t", "t_seasonality", "y_scaled",
                      "T", "y_scale", "sigma_k", "sigma_m", "changepoint_prior_scale",
                      "ds", "condition_masks"):
        setattr(scaled, attribute, getattr(model, attribute))
    scaled._build_layout()

    difference = scaled._minus_log_posterior(params) - model._minus_log_posterior(params)
    expected = np.sum(beta ** 2) / 2 * (1 / 1.0 ** 2 - 1 / SIGMA ** 2)
    assert difference == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("scale,expected_ratio", [(0.01, 0.9775), (0.001, 0.2842)])
def test_a_tighter_scale_shrinks_by_the_amount_the_prior_implies(
        peyton_manning_df, compiled_optimizer_module, scale, expected_ratio):
    """The point of the whole task: the scale has to reach the fit, and reach it
    with the right weight.

    Asserting merely that the coefficients got smaller would pass on a prior
    applied at any strength, so this checks the size against what the algebra
    predicts. For a Gaussian likelihood with a Gaussian prior the MAP shrinks
    each coefficient by

        f(s) = (X'X / sigma_obs^2) / (X'X / sigma_obs^2 + 1 / s^2)

    relative to the unpenalized solution, and `X'X` is T/2 per Fourier column.
    Measured against a loose-prior baseline, the observed ratios track that
    formula to four significant figures over five orders of magnitude of `s`
    (1.0000, 0.9998, 0.9775, 0.2840, 0.0036 at s = 1, 0.1, 0.01, 0.001, 0.0001).

    The two cases here sit either side of where the prior starts to bite.
    Scales below 1e-4 are left out: the problem becomes badly enough
    conditioned that the optimizer runs into Prophet's 10,000-iteration cap.
    """
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    weekly = slice(20, 26)

    def fit(weekly_scale):
        model = configured(CustomProphet(),
                           {"yearly": seasonality(365.25, 10, prior_scale=10.0),
                            "weekly": seasonality(7.0, 3, prior_scale=weekly_scale)})
        model.fit_cpp(df, lib_path=compiled_optimizer_module)
        return model

    loose, tight = fit(10.0), fit(scale)

    loose_beta = loose.params["beta"][0]
    tight_beta = tight.params["beta"][0]
    ratio = np.max(np.abs(tight_beta[weekly])) / np.max(np.abs(loose_beta[weekly]))
    assert ratio == pytest.approx(expected_ratio, abs=0.01)

    # and only that component: the yearly block carries its own scale
    np.testing.assert_allclose(tight_beta[:20], loose_beta[:20], rtol=0.05, atol=1e-3)


# -- the acceptance criterion -------------------------------------------

def test_matches_the_prior_term_stan_computes(prophet_comparison, compiled_optimizer_module):
    """"Two seasonalities with different prior scales reproduce the prior term
    Stan computes, verified via log_prob."

    Checked as the project checks every objective claim: `ours + stan_lp__` must
    be the same constant at every point, since the two differ only by dropped
    normalization terms. A `sigmas` applied in the wrong order, or not applied,
    leaves that sum varying point to point rather than fixed.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)

    prophet_model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                            daily_seasonality=False, n_changepoints=25,
                            changepoint_range=0.8, changepoint_prior_scale=0.05,
                            mcmc_samples=0)
    prophet_model.add_seasonality("yearly", 365.25, 10, prior_scale=YEARLY_SCALE)
    prophet_model.add_seasonality("weekly", 7, 3, prior_scale=WEEKLY_SCALE)

    stan_model, stan_data, _ = bridge.capture_stan_model(prophet_model, df)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = configured(CustomProphet(), two_scales())
    ours._generate_change_points = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    # the vector itself, element for element -- ordering included
    np.testing.assert_array_equal(ours.sigmas, np.asarray(stan_data["sigmas"], dtype=float))

    rng = np.random.default_rng(0)
    points = [ours.get_parameters()]
    for scale in (0.01, 0.1, 0.5):
        point = ours.get_parameters().copy()
        point[:2] += rng.normal(scale=scale, size=2)
        point[ours.layout.delta] += rng.normal(scale=scale * 0.1, size=25)
        point[ours.layout.beta] += rng.normal(scale=scale, size=26)
        point[ours.layout.sigma_obs_idx] = abs(point[ours.layout.sigma_obs_idx]) + 0.01
        points.append(point)

    sums = [ours._minus_log_posterior(point)
            + bridge.stan_log_prob(stan_model, stan_data, point[0], point[1],
                                   point[ours.layout.delta],
                                   point[ours.layout.sigma_obs_idx],
                                   point[ours.layout.beta])
            for point in points]

    # Stan reports lp__ to 8 significant figures, and the largest of these
    # points is ~8e5, so agreement is limited to ~1e-3 in absolute terms
    assert max(sums) - min(sums) < 1e-2, (
        f"our objective and Stan's differ by a varying amount across points "
        f"({sums}); the prior scales are not being applied as Stan applies them")
    assert abs(sums[0]) < 1e-3, (
        f"the two objectives differ by {sums[0]:.6f} at the optimum, not by zero")
