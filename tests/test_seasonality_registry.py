"""
Issue #16 task 2: seasonality as a registry rather than a compiled-in yearly term.

The model used to hardwire one order-10 yearly component in four separate
places -- the module constants that sliced the parameter vector, the Python
design matrix, the C++ design matrix, and predict(). Adding weekly seasonality
meant editing all four consistently, and nothing checked that they agreed.

A `ParameterLayout` now derives the slices from (S, K), and K comes from the
registry. These tests fix that contract: the layout follows the registry, both
language's design matrices are built from it, and -- the acceptance criterion
for the task -- a model with only yearly registered fits exactly as before.
"""
import numpy as np
import pandas as pd
import pytest

from customProphet import (BUILT_IN_SEASONALITIES, check_seasonality_supported,
                           CustomProphet, DEFAULT_LAYOUT, ParameterLayout,
                           extract_params, fourier_components,
                           from_dict_to_array, n_yearly, N_CHANGE_POINTS, seasonality,
                           seasonality_columns, seasonality_design_matrix, SIGMA,
                           SIGMA_OBS_PRIOR_SCALE, YEARLY_PERIOD)


def yearly_and_weekly():
    """Prophet's own defaults on daily data spanning more than two years."""
    return {"yearly": dict(BUILT_IN_SEASONALITIES["yearly"]),
            "weekly": dict(BUILT_IN_SEASONALITIES["weekly"])}


# -- the layout ---------------------------------------------------------

def test_layout_slices_partition_the_vector():
    """k, m, delta, sigma_obs, beta must tile the vector with no gap and no
    overlap, or some parameter is silently read as another."""
    layout = ParameterLayout(n_changepoints=7, n_seasonality_columns=12)

    assert layout.size == 2 + 7 + 1 + 12
    covered = np.zeros(layout.size, dtype=int)
    covered[layout.k_idx] += 1
    covered[layout.m_idx] += 1
    covered[layout.delta] += 1
    covered[layout.sigma_obs_idx] += 1
    covered[layout.beta] += 1

    np.testing.assert_array_equal(covered, np.ones(layout.size, dtype=int))


def test_layout_round_trips_a_parameter_dict():
    layout = ParameterLayout(n_changepoints=3, n_seasonality_columns=4)
    original = {"k": 0.5, "m": -1.25, "delta": np.array([0.1, -0.2, 0.3]),
                "sigma_obs": 0.75, "beta": np.array([1.0, 2.0, 3.0, 4.0])}

    k, m, delta, sigma_obs, beta = extract_params(from_dict_to_array(original, layout), layout)

    assert (k, m, sigma_obs) == (original["k"], original["m"], original["sigma_obs"])
    np.testing.assert_array_equal(delta, original["delta"])
    # beta used to come back as zeros regardless of what went in (#34)
    np.testing.assert_array_equal(beta, original["beta"])


def test_from_dict_to_array_rejects_a_beta_of_the_wrong_width():
    layout = ParameterLayout(n_changepoints=3, n_seasonality_columns=4)

    with pytest.raises(ValueError, match="beta has"):
        from_dict_to_array({"k": 0.0, "m": 0.0, "delta": np.zeros(3),
                            "sigma_obs": 1.0, "beta": np.zeros(6)}, layout)


# -- the registry -------------------------------------------------------

def test_default_layout_describes_the_yearly_only_shape():
    """DEFAULT_LAYOUT is no longer "what a fit produces" -- the history decides
    that (#16 task 3). It remains the 25-changepoint, order-10-yearly shape
    that the objective tests pin their vectors to."""
    assert seasonality_columns({"yearly": seasonality(365.25, n_yearly)}) == 2 * n_yearly
    assert DEFAULT_LAYOUT.n_seasonality_columns == 2 * n_yearly


def test_built_in_periods_and_orders_match_prophet():
    """[fc] Prophet.set_auto_seasonalities: yearly 365.25/10, weekly 7/3,
    daily 1/4. Pinned here so task 3 has something to build against."""
    for name, period, order in [("yearly", 365.25, 10), ("weekly", 7.0, 3), ("daily", 1.0, 4)]:
        assert BUILT_IN_SEASONALITIES[name]["period"] == period
        assert BUILT_IN_SEASONALITIES[name]["fourier_order"] == order


def test_entries_carry_prophets_full_field_set():
    """[fc] add_seasonality stores {period, fourier_order, prior_scale, mode,
    condition_name}. The shape matches now so later tasks fill fields in rather
    than reshaping every construction site."""
    entry = seasonality(7.0, 3)

    assert set(entry) == {"period", "fourier_order", "prior_scale", "mode", "condition_name"}
    assert entry["prior_scale"] == SIGMA
    assert entry["mode"] == "additive"
    assert entry["condition_name"] is None


@pytest.mark.parametrize("field,value", [
    ("prior_scale", 0.0),
    ("prior_scale", -1.0),
    ("mode", "sideways"),
])
def test_invalid_field_values_are_rejected(field, value):
    """Nothing is refused as unimplemented any more -- prior_scale became
    honored in task 5, condition_name in task 7 and mode in task 11. The check
    remains as a validity test: an unrecognized mode would otherwise reach the
    design matrix as "not multiplicative", i.e. silently additive.
    """
    entry = seasonality(365.25, 10, **{field: value})

    with pytest.raises(ValueError):
        check_seasonality_supported({"yearly": entry})


def test_design_matrix_concatenates_blocks_in_registry_order(prepared_model):
    """One block per seasonality, left to right in registry order -- which is
    the order `beta` is laid out in, so getting it wrong misassigns every
    coefficient."""
    t = prepared_model.t_seasonality
    x = seasonality_design_matrix(t, yearly_and_weekly())

    assert x.shape == (len(t), 2 * 10 + 2 * 3)
    np.testing.assert_array_equal(x[:, :20], fourier_components(t, 365.25, 10))
    np.testing.assert_array_equal(x[:, 20:], fourier_components(t, 7.0, 3))


def test_empty_registry_gives_a_zero_width_block(prepared_model):
    """A model with no seasonality is trend plus noise, not an error."""
    x = seasonality_design_matrix(prepared_model.t_seasonality, {})
    assert x.shape == (len(prepared_model.t_seasonality), 0)


def test_layout_follows_the_registered_seasonalities(peyton_manning_df,
                                                     compiled_optimizer_module):
    model = CustomProphet()
    model.fit_cpp(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert list(model.seasonalities) == ["yearly", "weekly"]
    assert model.layout.n_seasonality_columns == 26
    assert model.opt_params.shape == (2 + N_CHANGE_POINTS + 1 + 26,)
    assert np.all(np.isfinite(model.opt_params))


# -- the two design matrices agreeing -----------------------------------

def test_cpp_and_python_objectives_agree_with_two_seasonalities(prepared_model,
                                                                cpp_mlp_and_gradient):
    """The design matrix is built twice -- once in Python, once in C++ -- and
    a disagreement in block order or period would show up nowhere else until
    the two fit paths quietly diverged.

    Checked away from the optimum, where every term is active.
    """
    model = prepared_model
    model.seasonalities = yearly_and_weekly()
    model._build_layout()

    rng = np.random.default_rng(0)
    canonical = np.concatenate(([0.3], [-0.7],
                                rng.normal(scale=0.01, size=model.layout.n_changepoints),
                                [0.8],
                                rng.normal(scale=0.5, size=26)))

    expected = model._minus_log_posterior(canonical)
    expected_gradient = model._gradient(canonical)

    from customProphet import canonical_to_cpp
    value, gradient = cpp_mlp_and_gradient(model, canonical_to_cpp(canonical, model.layout))

    assert value == pytest.approx(expected, rel=1e-12)
    # C++ order is (k, m, delta, beta, zeta): the blocks are reordered, and the
    # sigma_obs slot becomes d/d_zeta by the chain rule d(sigma_obs)/d(zeta) =
    # sigma_obs.
    sigma_obs = canonical[model.layout.sigma_obs_idx]
    reordered = np.concatenate((expected_gradient[:2 + model.layout.n_changepoints],
                                expected_gradient[model.layout.beta],
                                [expected_gradient[model.layout.sigma_obs_idx] * sigma_obs]))
    np.testing.assert_allclose(gradient, reordered, rtol=1e-9, atol=1e-8)


def test_cpp_rejects_a_params_vector_the_registry_cannot_fill(prepared_model, cpp_module):
    """A 20-column beta against a 26-column design matrix used to abort the
    process with an Eigen "invalid matrix product"."""
    model = prepared_model

    with pytest.raises(ValueError, match="seasonality coefficients"):
        cpp_module.optimize(
            params=np.zeros(DEFAULT_LAYOUT.size), t_scaled=model.t_scaled,
            change_points=model.change_points, t_seasonality=model.t_seasonality,
            normalized_y=model.normalized_y,
            sigma_obs_prior_scale=SIGMA_OBS_PRIOR_SCALE, sigma_k=model.sigma_k,
            sigma_m=model.sigma_m, sigmas=np.full(26, SIGMA), tau=model.tau,
            fourier_orders=[10, 3], seasonality_periods=[365.25, 7.0])


def test_cpp_rejects_mismatched_orders_and_periods(prepared_model, cpp_module):
    with pytest.raises(ValueError, match="one period per seasonality"):
        cpp_module.minus_log_posterior_and_gradient(
            params=np.zeros(DEFAULT_LAYOUT.size), t_scaled=prepared_model.t_scaled,
            change_points=prepared_model.change_points,
            t_seasonality=prepared_model.t_seasonality,
            normalized_y=prepared_model.normalized_y,
            sigma_obs_prior_scale=SIGMA_OBS_PRIOR_SCALE, sigma_k=prepared_model.sigma_k,
            sigma_m=prepared_model.sigma_m, sigmas=np.full(26, SIGMA),
            tau=prepared_model.tau, fourier_orders=[10, 3], seasonality_periods=[365.25])



# -- the acceptance criterion -------------------------------------------

def test_registering_yearly_explicitly_reproduces_the_default_fit(
        peyton_manning_df, compiled_optimizer_module):
    """"A model with only yearly registered fits identically to today's."

    Bit-for-bit, not to a tolerance: the registry is meant to be a refactor of
    how the yearly term is described, not a change to what is fit.
    """
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)

    registered = CustomProphet()
    registered.seasonalities = {"yearly": seasonality(YEARLY_PERIOD, n_yearly)}
    registered.weekly_seasonality = False
    registered.fit_cpp(df, lib_path=compiled_optimizer_module)

    # the same model reached the other way: let the built-in yearly be selected
    # automatically, and only suppress the weekly the history also supports
    automatic = CustomProphet()
    automatic.weekly_seasonality = False
    automatic.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert list(automatic.seasonalities) == ["yearly"]
    np.testing.assert_array_equal(registered.opt_params, automatic.opt_params)
    assert registered.opt.n_iterations == automatic.opt.n_iterations


def test_second_seasonality_reaches_predict(peyton_manning_df, compiled_optimizer_module):
    """predict() built its own yearly-only Fourier matrix, so a registered
    weekly term would have been fit and then dropped from the forecast."""
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)

    model = CustomProphet()
    model.fit_cpp(df, lib_path=compiled_optimizer_module)
    forecast = model.predict(model.make_future_dataframe(periods=30))

    assert len(forecast) == len(df) + 30
    assert np.all(np.isfinite(forecast["yhat"].values))

    # the weekly term is what distinguishes this fit: the fitted seasonality
    # must carry a day-of-week signal the yearly-only model cannot express
    history = forecast.iloc[:len(df)]
    by_weekday = pd.Series(history["seasonality"].values).groupby(
        pd.to_datetime(history["ds"]).dt.dayofweek.values).mean()

    yearly_only = CustomProphet()
    yearly_only.weekly_seasonality = False
    yearly_only.fit_cpp(df, lib_path=compiled_optimizer_module)
    yearly_forecast = yearly_only.predict(yearly_only.make_future_dataframe(periods=30))
    yearly_by_weekday = pd.Series(yearly_forecast["seasonality"].values[:len(df)]).groupby(
        pd.to_datetime(history["ds"]).dt.dayofweek.values).mean()

    assert by_weekday.max() - by_weekday.min() > 5 * (yearly_by_weekday.max()
                                                      - yearly_by_weekday.min())
