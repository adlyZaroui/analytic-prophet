"""
Issue #41: refitting is allowed here and forbidden in Prophet, and until now
neither behaviour was written down.

[fc] `Prophet.fit` refuses a second call outright:

    if self.history is not None:
        raise Exception('Prophet object can only be fit once. '
                        'Instantiate a new object.')

So there is no original behaviour to copy -- there is a refusal, and a
reimplementation that allows the call owes a contract in its place. The one
implemented is the one the refusal implies:

    **A refit is equivalent to a fresh instance carrying the same user
    configuration, fit on the new data.**

What the user set survives. What the previous history produced does not.

`FIT_DERIVED_ATTRIBUTES` in customProphet.py is the list that makes it true, and
this module is what keeps the list honest: it fits a fresh instance and a refit
on the same data across a matrix of configurations and compares *every*
attribute, so a stateful feature that forgets to reset shows up as a failing
test rather than as a wrong number.

It is not hypothetical tidiness. Two bugs of exactly that shape were live when
this was written, and neither raised:

  * a model fit on twenty rows kept `n_changepoints` capped at 15, and fitted 15
    rather than 25 changepoints on every later history however long;
  * a model fit across one date range kept that range's holiday names and forced
    them onto the next as all-zero columns, while dropping holidays the new
    history actually had.

Both are pinned below.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from customProphet import FIT_DERIVED_ATTRIBUTES, CustomProphet

# Two attributes the sweep cannot compare by value, for reasons that have
# nothing to do with refitting:
#
#   rng   unseeded by design, since uncertainty sampling draws from it -- two
#         *fresh* instances differ in it too
#   opt   the solver's own result object, from pybind11 or from scipy depending
#         on the path, and neither defines equality. `solver_summary` below
#         compares what it carries instead, and its contents also reach the
#         sweep through `params`, `_params_vector` and `loss_over_iterations`.
NOT_STATE = {"rng", "opt"}


def solver_summary(model):
    """The comparable content of `model.opt`, whichever solver produced it."""
    opt = model.opt
    iterations = getattr(opt, "n_iterations", None)
    if iterations is None:
        iterations = getattr(opt, "nit", None)
    return (iterations, opt.status,
            np.asarray(getattr(opt, "params", getattr(opt, "x", []))))

HOLIDAYS = pd.DataFrame({"holiday": "playoff",
                         "ds": pd.to_datetime(["2008-01-13", "2009-01-03", "2010-01-16"]),
                         "lower_window": 0, "upper_window": 1})


def with_regressor(df):
    frame = df.copy()
    frame["temp"] = np.arange(len(frame), dtype=float) % 17
    return frame


@pytest.fixture
def histories(peyton_manning_df):
    """Three histories that differ in length *and* in date range, since a
    stale attribute can be one the new range cannot produce rather than only
    one the new length cannot."""
    return {
        "long": peyton_manning_df.iloc[:1200].reset_index(drop=True),
        "short": peyton_manning_df.iloc[:300].reset_index(drop=True),
        "late": peyton_manning_df.iloc[1500:2100].reset_index(drop=True),
        "tiny": peyton_manning_df.iloc[:20].reset_index(drop=True),
    }


CONFIGURATIONS = {
    "plain": lambda: CustomProphet(),
    "custom seasonality": lambda: CustomProphet().add_seasonality("monthly", 30.5, 5),
    "hand-written holidays": lambda: CustomProphet(holidays=HOLIDAYS),
    "country holidays": lambda: CustomProphet().add_country_holidays("US"),
    "extra regressor": lambda: CustomProphet().add_regressor("temp"),
    "multiplicative": lambda: CustomProphet(seasonality_mode="multiplicative"),
    "flat growth": lambda: CustomProphet(growth="flat"),
    "explicit changepoints": lambda: CustomProphet(
        changepoints=pd.to_datetime(["2008-06-01", "2008-09-01"])),
}


def equivalent(left, right):
    """Value equality across the container types these attributes use."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        return left.shape == right.shape and (
            left.size == 0 or np.array_equal(left, right))
    if isinstance(left, (pd.DataFrame, pd.Series)) or \
            isinstance(right, (pd.DataFrame, pd.Series)):
        return type(left) is type(right) and left.equals(right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and \
            all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and \
            all(equivalent(a, b) for a, b in zip(left, right))
    try:
        return bool(left == right)
    except ValueError:
        return repr(left) == repr(right)


def differing_attributes(fresh, refit):
    """Every attribute on which the two models disagree, `rng` aside."""
    names = (set(fresh.__dict__) | set(refit.__dict__)) - NOT_STATE
    return sorted(
        name for name in names
        if not equivalent(fresh.__dict__.get(name, "<missing>"),
                          refit.__dict__.get(name, "<missing>")))


def fit_fresh_and_refit(build, first, second, lib_path, path="cpp"):
    """(a fresh instance fit on `second`, one fit on `first` then on `second`)."""
    def run(model, frame):
        if path == "cpp":
            model.fit_cpp(frame, lib_path=lib_path)
        else:
            model.fit(frame, analytic=True)
        return model

    fresh = run(build(), second)
    refit = run(run(build(), first), second)
    return fresh, refit


# -- what Prophet actually does ------------------------------------------

def test_prophet_refuses_a_second_fit(prophet_comparison):
    """Verified rather than recalled, because the whole contract below exists
    to stand in for this refusal."""
    Prophet, common, _ = prophet_comparison
    df = common.load_data(300)

    model = Prophet(**common.PROPHET_KWARGS)
    model.fit(df)

    with pytest.raises(Exception, match="can only be fit once"):
        model.fit(df)


def test_this_implementation_allows_one(peyton_manning_df, compiled_optimizer_module):
    """The divergence itself, stated as a test so it is a decision on the record
    rather than something that fell out."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    model = CustomProphet()

    model.fit_cpp(df, lib_path=compiled_optimizer_module)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert np.all(np.isfinite(model.get_parameters()))


# -- the contract ---------------------------------------------------------

@pytest.mark.parametrize("configuration", list(CONFIGURATIONS))
@pytest.mark.parametrize("first,second", [("long", "short"), ("long", "late")])
def test_a_refit_matches_a_fresh_instance(histories, compiled_optimizer_module,
                                          configuration, first, second):
    """The contract, over the whole attribute surface.

    Not a spot check of the attributes known to have been wrong: every one of
    them, so that the next stateful feature is covered before it is written.
    """
    build = CONFIGURATIONS[configuration]
    frames = histories
    if configuration == "extra regressor":
        frames = {name: with_regressor(frame) for name, frame in histories.items()}
    if configuration == "explicit changepoints" and second == "late":
        # The given dates are outside the late window. What matters is that the
        # refit fails the same way a fresh instance does, rather than silently
        # carrying the previous history's changepoints into a range that cannot
        # hold them -- so that is asserted instead of skipped.
        message = "Changepoints must fall within training data"
        with pytest.raises(ValueError, match=message):
            build().fit_cpp(frames[second], lib_path=compiled_optimizer_module)
        reused = build()
        reused.fit_cpp(frames[first], lib_path=compiled_optimizer_module)
        with pytest.raises(ValueError, match=message):
            reused.fit_cpp(frames[second], lib_path=compiled_optimizer_module)
        return

    fresh, refit = fit_fresh_and_refit(
        build, frames[first], frames[second], compiled_optimizer_module)

    assert differing_attributes(fresh, refit) == []
    fresh_solver, refit_solver = solver_summary(fresh), solver_summary(refit)
    assert fresh_solver[:2] == refit_solver[:2]
    np.testing.assert_array_equal(fresh_solver[2], refit_solver[2])


@pytest.mark.parametrize("configuration", ["plain", "country holidays", "extra regressor"])
def test_the_python_path_keeps_the_same_contract(histories, configuration):
    """Both paths, or a script that switches between them gets different state."""
    build = CONFIGURATIONS[configuration]
    frames = histories
    if configuration == "extra regressor":
        frames = {name: with_regressor(frame) for name, frame in histories.items()}

    fresh, refit = fit_fresh_and_refit(
        build, frames["long"], frames["short"], None, path="python")

    assert differing_attributes(fresh, refit) == []
    fresh_solver, refit_solver = solver_summary(fresh), solver_summary(refit)
    assert fresh_solver[:2] == refit_solver[:2]
    np.testing.assert_array_equal(fresh_solver[2], refit_solver[2])


def test_switching_paths_between_fits_leaves_no_stale_status(histories,
                                                             compiled_optimizer_module):
    """`opt_status` is written by fit_cpp and not by fit, so a fit() following a
    fit_cpp() would otherwise report the compiled run's status as its own."""
    df = histories["short"]
    model = CustomProphet()

    model.fit_cpp(df, lib_path=compiled_optimizer_module)
    assert model.opt_status_message == "converged"

    model.fit(df, analytic=True)
    assert not hasattr(model, "opt_status")
    assert not hasattr(model, "opt_status_message")

    model.fit_cpp(df, lib_path=compiled_optimizer_module)
    assert model.opt_status_message == "converged"


@pytest.mark.parametrize("configuration", list(CONFIGURATIONS))
def test_a_refit_forecasts_what_a_fresh_instance_forecasts(histories,
                                                           compiled_optimizer_module,
                                                           configuration):
    """The contract as a user meets it. Attribute equality should imply this,
    but it is the forecast that is the product."""
    build = CONFIGURATIONS[configuration]
    frames = histories
    if configuration == "extra regressor":
        frames = {name: with_regressor(frame) for name, frame in histories.items()}

    fresh, refit = fit_fresh_and_refit(
        build, frames["long"], frames["short"], compiled_optimizer_module)

    future = fresh.make_future_dataframe(periods=30)
    if configuration == "extra regressor":
        future["temp"] = np.arange(len(future), dtype=float) % 17

    for column in ("yhat", "trend"):
        np.testing.assert_array_equal(
            fresh.predict(future)[column].values,
            refit.predict(future)[column].values, err_msg=column)


def test_a_refit_agrees_with_prophet_exactly_as_a_fresh_fit_does(prophet_comparison,
                                                                 compiled_optimizer_module):
    """Against the original, which is what "matching Prophet's behaviour" can
    mean when Prophet has no behaviour to match.

    Prophet cannot refit, so the comparison is: a fresh Prophet on the new data
    against *both* of ours. If the refit is equivalent to a fresh instance, it
    stands in the same relation to Prophet that a fresh instance does -- same
    posterior, to the last digit compared.
    """
    Prophet, common, bridge = prophet_comparison
    first, second = common.load_data(1200), common.load_data(300)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    stan_model, stan_data, params = bridge.capture_stan_model(prophet_model, second)
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    def build():
        model = CustomProphet(n_changepoints=len(changepoints_t))
        model.set_changepoints = lambda: setattr(
            model, "changepoints_t", changepoints_t.copy())
        return model

    fresh, refit = fit_fresh_and_refit(build, first, second, compiled_optimizer_module)

    def score(model):
        return bridge.stan_log_prob(
            stan_model, stan_data, model.params["k"][0][0], model.params["m"][0][0],
            model.params["delta"][0], model.sigma_obs, model.params["beta"][0])

    assert score(refit) == score(fresh)
    assert score(refit) >= lp_prophet - 1e-6


# -- the two bugs this closed --------------------------------------------

def test_a_short_first_fit_does_not_cap_the_changepoints_forever(histories,
                                                                 compiled_optimizer_module):
    """[fc] `set_changepoints` caps `n_changepoints` at `floor(T * range) - 1`
    and overwrites the attribute, which Prophet can do because it never fits
    twice. Here the cap has to be undone."""
    fresh = CustomProphet()
    fresh.fit_cpp(histories["long"], lib_path=compiled_optimizer_module)

    refit = CustomProphet()
    refit.fit_cpp(histories["tiny"], lib_path=compiled_optimizer_module)
    assert refit.n_changepoints == 15, "the cap should bite on twenty rows"

    refit.fit_cpp(histories["long"], lib_path=compiled_optimizer_module)

    assert refit.n_changepoints == fresh.n_changepoints == 25
    assert refit.layout == fresh.layout
    np.testing.assert_array_equal(refit.get_parameters(), fresh.get_parameters())


def test_setting_the_changepoint_count_between_fits_is_respected(histories,
                                                                 compiled_optimizer_module):
    """The other side of undoing the cap: `n_changepoints` is configuration the
    user can assign after construction, so the reset must not overwrite a value
    they set. Undoing the cap only while the capped value still stands is what
    tells the two apart."""
    model = CustomProphet()
    model.fit_cpp(histories["tiny"], lib_path=compiled_optimizer_module)
    assert model.n_changepoints == 15

    model.n_changepoints = 8
    model.fit_cpp(histories["long"], lib_path=compiled_optimizer_module)

    assert model.n_changepoints == 8
    assert model.layout.n_changepoints == 8


def test_holiday_names_come_from_the_history_being_fitted(histories,
                                                          compiled_optimizer_module):
    """[fc] `construct_holiday_dataframe` pins the training holiday set so that
    predict produces the same columns as fit. On a refit the pinned set has to
    be the new history's, or the old one is forced onto data that never had it.

    The two histories here are different *date ranges*, which is what makes the
    holiday sets differ: observed-day holidays land in some years and not others.
    """
    fresh = CustomProphet().add_country_holidays("US")
    fresh.fit_cpp(histories["late"], lib_path=compiled_optimizer_module)

    refit = CustomProphet().add_country_holidays("US")
    refit.fit_cpp(histories["long"], lib_path=compiled_optimizer_module)
    stale = set(refit.train_holiday_names)
    refit.fit_cpp(histories["late"], lib_path=compiled_optimizer_module)

    assert stale != set(fresh.train_holiday_names), (
        "the two histories no longer disagree, so this test proves nothing")
    assert set(refit.train_holiday_names) == set(fresh.train_holiday_names)
    assert refit.layout == fresh.layout


# -- what survives --------------------------------------------------------

def test_user_configuration_survives_a_refit(histories, compiled_optimizer_module):
    """The other half of the contract. A refit resets what the data produced,
    not what the caller asked for."""
    model = (CustomProphet(seasonality_mode="multiplicative",
                           changepoint_prior_scale=0.123, interval_width=0.5)
             .add_seasonality("monthly", 30.5, 5)
             .add_country_holidays("US")
             .add_regressor("temp"))
    frames = {name: with_regressor(frame) for name, frame in histories.items()}

    model.fit_cpp(frames["long"], lib_path=compiled_optimizer_module)
    model.fit_cpp(frames["short"], lib_path=compiled_optimizer_module)

    assert model.seasonality_mode == "multiplicative"
    assert model.changepoint_prior_scale == 0.123
    assert model.interval_width == 0.5
    assert model.seasonalities["monthly"]["fourier_order"] == 5
    assert model.country_holidays == "US"
    assert "temp" in model.extra_regressors


def test_an_explicit_changepoint_list_survives_a_refit(histories,
                                                       compiled_optimizer_module):
    """`changepoints` is in FIT_DERIVED_ATTRIBUTES, which looks wrong until you
    see what restoring the *constructed* value does: it hands back the user's
    list where one was given and clears the generated dates where one was not.
    One rule, both cases."""
    given = pd.to_datetime(["2008-06-01", "2008-09-01", "2009-02-01"])
    model = CustomProphet(changepoints=given)

    model.fit_cpp(histories["long"], lib_path=compiled_optimizer_module)
    model.fit_cpp(histories["long"], lib_path=compiled_optimizer_module)

    assert model.specified_changepoints
    pd.testing.assert_series_equal(
        model.changepoints, pd.Series(given, name="ds"), check_names=False)
    assert model.layout.n_changepoints == 3


def test_generated_changepoints_do_not_become_a_specified_list(histories,
                                                               compiled_optimizer_module):
    """The mirror case: dates this implementation generated must not be read
    back on the next fit as though the user had supplied them."""
    model = CustomProphet()
    model.fit_cpp(histories["long"], lib_path=compiled_optimizer_module)
    first = np.asarray(model.changepoints_t)

    model.fit_cpp(histories["late"], lib_path=compiled_optimizer_module)

    assert not model.specified_changepoints
    assert not np.array_equal(first, np.asarray(model.changepoints_t))


# -- the list itself ------------------------------------------------------

def test_every_listed_attribute_exists_on_a_constructed_model():
    """The list is written by hand, so a typo in it would silently reset
    nothing."""
    model = CustomProphet()
    missing = [name for name in FIT_DERIVED_ATTRIBUTES if not hasattr(model, name)]
    assert missing == []


def test_the_reset_restores_the_constructed_values(histories,
                                                   compiled_optimizer_module):
    """`_reset_fit_state` on its own, without a second fit after it."""
    constructed = CustomProphet()
    baseline = {name: copy.deepcopy(getattr(constructed, name))
                for name in FIT_DERIVED_ATTRIBUTES}

    model = CustomProphet()
    model.fit_cpp(histories["short"], lib_path=compiled_optimizer_module)
    model._reset_fit_state()

    for name, value in baseline.items():
        assert equivalent(getattr(model, name), value), name
