"""
Issue #16 task 6: add_seasonality, the public way into the registry.

Tasks 2-5 built a registry, a rule for filling it from the history, and a prior
scale per column -- but left no supported way to use any of it. Anything other
than what the auto rule selects meant assigning to `model.seasonalities`
directly, which is what these tests used to do.

The validation is most of the work, and it is not guessable: Prophet exempts
built-in names from the collision check, reserves a list of column names that
`predict()` also emits, and treats `prior_scale=0` and `fourier_order=0` as
errors rather than "off". Each branch below is checked against the installed
Prophet rather than a reading of its source.

Two arguments diverge deliberately. `mode='multiplicative'` needs the s_a/s_m
split of task 11 and `condition_name` needs task 7; Prophet accepts both. This
implementation validates them exactly as Prophet does and then refuses them,
because accepting an argument it will not fit is the failure mode this whole
issue exists to avoid -- and refusing at the call site beats refusing three
steps later at fit time.
"""
import numpy as np
import pytest

from customProphet import (BUILT_IN_NAMES, CustomProphet, RESERVED_COLUMN_NAMES,
                           SIGMA, seasonality_design_matrix)


def registered(model, name):
    return model.seasonalities[name]


# -- validation, branch for branch against Prophet ----------------------

@pytest.mark.parametrize("args,kwargs", [
    (("monthly", 30.5, 5), {}),
    (("weekly", 7, 10), {}),                          # overwriting a built-in
    (("monthly", 30.5, 5), {"prior_scale": 2.0}),
    (("monthly", 30.5, 5), {"prior_scale": 0.0}),     # 0 is an error, not "off"
    (("monthly", 30.5, 5), {"prior_scale": -1.0}),
    (("monthly", 30.5, 0), {}),
    (("monthly", 30.5, -2), {}),
    (("monthly", 30.5, 5), {"mode": "additive"}),
    (("monthly", 30.5, 5), {"mode": "sideways"}),
    (("trend", 30.5, 5), {}),
    (("yhat_lower", 30.5, 5), {}),
    (("ds", 30.5, 5), {}),
    (("a_delim_b", 30.5, 5), {}),
    (("monthly", 30.5, 5), {"condition_name": "trend"}),   # reserved condition
])
def test_accepts_and_rejects_what_prophet_does(prophet_comparison, args, kwargs):
    """Same outcome, and for rejections the same exception class.

    Not the same message: ours name the field and point at #16 where Prophet's
    do not, and matching prose would pin us to their wording.
    """
    Prophet, _, _ = prophet_comparison

    def outcome(model):
        try:
            model.add_seasonality(*args, **kwargs)
        except Exception as exc:                      # noqa: BLE001 -- the class is the assertion
            return type(exc)
        return None

    assert outcome(CustomProphet()) is outcome(Prophet())


def test_reserved_names_match_prophets_list(prophet_comparison):
    """Derived rather than transcribed -- the _lower/_upper variants are built
    from the stems, so a typo in one would not be visible by reading."""
    Prophet, _, _ = prophet_comparison
    prophet_model = Prophet()

    for name in RESERVED_COLUMN_NAMES:
        with pytest.raises(ValueError, match="reserved"):
            prophet_model.validate_column_name(name)
        with pytest.raises(ValueError, match="reserved"):
            CustomProphet().validate_column_name(name)


def test_built_in_names_are_exempt_from_the_collision_check():
    """[fc] `if name not in ['daily', 'weekly', 'yearly']` -- registering
    `weekly` twice is how a user raises its Fourier order, so it must not trip
    the "already used for a seasonality" check."""
    model = CustomProphet()
    model.add_seasonality("weekly", 7, 10)
    model.add_seasonality("weekly", 7, 12)

    assert registered(model, "weekly")["fourier_order"] == 12
    assert set(BUILT_IN_NAMES) == {"daily", "weekly", "yearly"}


def test_a_name_colliding_with_a_regressor_is_rejected():
    """The check is a no-op today -- extra_regressors is always empty until
    task 10 -- so it is exercised directly rather than through add_regressor."""
    model = CustomProphet()
    model.extra_regressors = {"temperature": {}}

    with pytest.raises(ValueError, match="already used for an added regressor"):
        model.add_seasonality("temperature", 30.5, 5)


# -- what gets registered -----------------------------------------------

def test_returns_self_so_calls_chain():
    """[fc] returns the Prophet object."""
    model = CustomProphet()
    assert model.add_seasonality("monthly", 30.5, 5) is model
    assert model.add_seasonality("a", 2, 1).add_seasonality("b", 3, 1) is model


def test_the_entry_carries_prophets_full_field_set():
    model = CustomProphet().add_seasonality("monthly", 30.5, 5, prior_scale=2.0)
    entry = registered(model, "monthly")

    assert entry == {"period": 30.5, "fourier_order": 5, "prior_scale": 2.0,
                     "mode": "additive", "condition_name": None}


def test_prior_scale_falls_back_to_the_model_wide_default():
    """[fc] `ps = self.seasonality_prior_scale` when none is given."""
    default = CustomProphet().add_seasonality("monthly", 30.5, 5)
    assert registered(default, "monthly")["prior_scale"] == SIGMA

    overridden = CustomProphet()
    overridden.seasonality_prior_scale = 4.0
    overridden.add_seasonality("monthly", 30.5, 5)
    assert registered(overridden, "monthly")["prior_scale"] == 4.0


def test_the_model_wide_default_also_reaches_the_auto_selected_components(
        peyton_manning_df, compiled_optimizer_module):
    """[fc] set_auto_seasonalities registers with self.seasonality_prior_scale
    too, so an override is not silently confined to hand-registered ones."""
    model = CustomProphet()
    model.seasonality_prior_scale = 4.0
    model.fit_cpp(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert list(model.seasonalities) == ["yearly", "weekly"]
    assert all(props["prior_scale"] == 4.0 for props in model.seasonalities.values())
    np.testing.assert_array_equal(model.sigmas, 4.0)


def test_adding_after_a_fit_is_refused(peyton_manning_df, compiled_optimizer_module):
    """[fc] "Seasonality must be added prior to model fitting." The registry is
    read when the layout is built, so a later addition would be accepted and
    then ignored."""
    model = CustomProphet()
    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    with pytest.raises(RuntimeError, match="before fitting"):
        model.add_seasonality("monthly", 30.5, 5)


# -- the deliberate divergences -----------------------------------------

def test_every_argument_prophet_accepts_is_now_accepted():
    """This test used to assert the opposite for `mode` and `condition_name`.
    Both are fitted as of tasks 11 and 7, so the refusals are gone and the
    signature matches Prophet's in what it accepts as well as what it rejects.
    """
    model = (CustomProphet()
             .add_seasonality("monthly", 30.5, 5, mode="multiplicative")
             .add_seasonality("quarterly", 91.0, 3, condition_name="in_season"))

    assert registered(model, "monthly")["mode"] == "multiplicative"
    assert registered(model, "quarterly")["condition_name"] == "in_season"


def test_invalid_values_are_still_rejected():
    """Accepting the argument is not accepting anything for it."""
    with pytest.raises(ValueError, match="additive"):
        CustomProphet().add_seasonality("monthly", 30.5, 5, mode="sideways")
    with pytest.raises(ValueError, match="reserved"):
        CustomProphet().add_seasonality("monthly", 30.5, 5, condition_name="trend")

    model = CustomProphet()
    with pytest.raises(ValueError):
        model.add_seasonality("monthly", 30.5, 5, mode="sideways")
    assert model.seasonalities == {}


# -- reaching the fit ---------------------------------------------------

def test_a_custom_component_reaches_the_parameter_vector(peyton_manning_df,
                                                         compiled_optimizer_module):
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)

    model = CustomProphet().add_seasonality("monthly", 30.5, 5)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    # registered first, so its block comes first -- auto-selection appends
    assert list(model.seasonalities) == ["monthly", "yearly", "weekly"]
    assert model.layout.n_seasonality_columns == 2 * (5 + 10 + 3)
    assert model.opt_params.shape == (model.layout.size,)
    assert np.any(np.abs(model.opt_params[model.layout.beta][:10]) > 1e-6)


def test_a_custom_component_reaches_predict(peyton_manning_df, compiled_optimizer_module):
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)

    model = CustomProphet().add_seasonality("monthly", 30.5, 5)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)
    forecast = model.predict(model.make_future_dataframe(periods=30))

    assert len(forecast) == len(df) + 30
    assert np.all(np.isfinite(forecast["yhat"].values))

    # the design matrix predict() builds has to be the one that was fitted
    x = seasonality_design_matrix(model.t_seasonality, model.seasonalities)
    assert x.shape[1] == model.layout.n_seasonality_columns


def test_overwriting_a_built_in_survives_auto_selection(peyton_manning_df,
                                                        compiled_optimizer_module):
    """The point of the exemption: a hand-registered `weekly` must not be
    replaced by the order-3 built-in at fit time."""
    model = CustomProphet().add_seasonality("weekly", 7, 10, prior_scale=3.0)
    model.fit_cpp(peyton_manning_df.iloc[:1000].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert registered(model, "weekly") == {"period": 7.0, "fourier_order": 10,
                                           "prior_scale": 3.0, "mode": "additive",
                                           "condition_name": None}
    assert list(model.seasonalities) == ["weekly", "yearly"]
    np.testing.assert_array_equal(model.sigmas[:20], 3.0)     # weekly block, order 10
    np.testing.assert_array_equal(model.sigmas[20:], SIGMA)   # yearly, auto-registered


def test_a_hand_registered_component_survives_a_refit(peyton_manning_df,
                                                      compiled_optimizer_module):
    """#41: only what the auto rule registered is cleared between fits. A
    component the user added is theirs."""
    model = CustomProphet().add_seasonality("monthly", 30.5, 5)
    model.fit_cpp(peyton_manning_df, lib_path=compiled_optimizer_module)
    assert list(model.seasonalities) == ["monthly", "yearly", "weekly"]

    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    assert list(model.seasonalities) == ["monthly", "weekly"]


# -- against Prophet ----------------------------------------------------

@pytest.mark.parametrize("configure", [
    lambda model: model.add_seasonality("monthly", 30.5, 5),
    lambda model: model.add_seasonality("weekly", 7, 10, prior_scale=3.0),
])
def test_registry_and_posterior_agree_with_prophet(prophet_comparison,
                                                   compiled_optimizer_module, configure):
    """The registry has to match entry for entry *and in order*, because the
    order is `beta`'s layout -- and then our optimum has to be at least as good
    under Stan's own density, as it is for the auto-selected model.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    configure(prophet_model)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, prophet_params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    t_change = np.asarray(stan_data["t_change"], dtype=float)

    ours = CustomProphet()
    configure(ours)
    ours._generate_change_points = lambda: setattr(ours, "t_change", t_change.copy())
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert list(ours.seasonalities) == list(prophet_model.seasonalities)
    for name, props in ours.seasonalities.items():
        theirs = prophet_model.seasonalities[name]
        assert props["fourier_order"] == theirs["fourier_order"]
        assert props["prior_scale"] == theirs["prior_scale"]
        assert float(props["period"]) == float(theirs["period"])
    np.testing.assert_array_equal(ours.sigmas, np.asarray(stan_data["sigmas"], dtype=float))

    X_ours = seasonality_design_matrix(ours.t_seasonality, ours.seasonalities)
    beta_in_stan, residual = bridge.transfer_seasonality(
        ours.opt_params[ours.layout.beta], X_ours, np.asarray(stan_data["X"], dtype=float))
    assert residual < 1e-8

    lp_ours = bridge.stan_log_prob(stan_model, stan_data, ours.opt_params[0],
                                   ours.opt_params[1], ours.opt_params[2:2 + len(t_change)],
                                   ours.sigma_obs, beta_in_stan)
    assert lp_ours >= lp_prophet - 1e-6
