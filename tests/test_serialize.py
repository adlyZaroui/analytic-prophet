"""
Issue #78: a fitted model can be saved and loaded.

[fc] prophet/serialize.py, down to the four entry points:

    from analytic_prophet.serialize import model_to_json, model_from_json

    with open('model.json', 'w') as handle:
        handle.write(model_to_json(model))

    with open('model.json') as handle:
        model = model_from_json(handle.read())

`pickle` is deliberately not supported, and the reason is sharper here than in
Prophet: its pickles carry a Stan object and do not survive a version change;
ours would carry a handle to whatever build of `optimize.cpp` was loaded, tying
the file to one machine.

The round trip is asserted **bit-identical**, not close. Nothing is refitted, so
any difference at all is a dropped attribute rather than numerical drift.

Three of the tests below are regressions against #41 rather than against #78.
Serialization is the second way to walk a model's configuration forward into a
history that cannot support it, and all three of #41's failure modes came back
through the file before they were fixed here -- one of them by crashing.
"""
import json

import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet, models
from analytic_prophet import serialize
from analytic_prophet.serialize import (model_from_dict, model_from_json,
                                        model_to_dict, model_to_json)

HOLIDAYS = pd.DataFrame({"holiday": "playoff",
                         "ds": pd.to_datetime(["2008-01-13", "2009-01-03", "2010-01-16"]),
                         "lower_window": 0, "upper_window": 1})


def with_regressor(df):
    return df.assign(temp=np.arange(len(df), dtype=float) % 17)


def with_condition(df):
    return df.assign(is_weekend=pd.to_datetime(df["ds"]).dt.dayofweek >= 5)


# name -> (build model, prepare frame, prepare future frame)
CONFIGURATIONS = {
    "plain": (lambda: AnalyticProphet(), lambda df: df, lambda f: f),
    "flat": (lambda: AnalyticProphet(growth="flat"), lambda df: df, lambda f: f),
    "logistic": (lambda: AnalyticProphet(growth="logistic"),
                 lambda df: df.assign(cap=df["y"].max() * 1.5),
                 lambda f: f.assign(cap=f["cap"].max() if "cap" in f else 12.0)),
    "multiplicative": (lambda: AnalyticProphet(seasonality_mode="multiplicative"),
                       lambda df: df, lambda f: f),
    "holidays": (lambda: AnalyticProphet(holidays=HOLIDAYS), lambda df: df, lambda f: f),
    "country": (lambda: AnalyticProphet().add_country_holidays("US"),
                lambda df: df, lambda f: f),
    "custom seasonality": (lambda: AnalyticProphet().add_seasonality("monthly", 30.5, 5),
                           lambda df: df, lambda f: f),
    "conditional seasonality": (
        lambda: AnalyticProphet().add_seasonality("weekend", 7, 3,
                                                  condition_name="is_weekend"),
        with_condition, with_condition),
    "regressor": (lambda: AnalyticProphet().add_regressor("temp"),
                  with_regressor, with_regressor),
    "regressor predictor": (
        lambda: AnalyticProphet().add_regressor("temp", regressor_predictor=True),
        with_regressor, with_regressor),
    "explicit changepoints": (
        lambda: AnalyticProphet(changepoints=pd.to_datetime(["2008-06-01", "2008-09-01"])),
        lambda df: df, lambda f: f),
}


def seed_all(model, value=0):
    """Seed the model and any nested regressor predictors.

    A predictor is a whole model with its own generator, and the outer model's
    interval draws on it -- seeding only the outer one leaves the regressor's
    contribution random.
    """
    model.rng = np.random.default_rng(value)
    for props in model.extra_regressors.values():
        predictor = props.get("predictor")
        if predictor is not None:
            seed_all(predictor, value)
    return model


@pytest.fixture
def fit_one(peyton_manning_df, compiled_optimizer_module):
    """(fitted model, future frame) for a named configuration."""
    def run(configuration, n_rows=500, periods=60):
        build, prepare, prepare_future = CONFIGURATIONS[configuration]
        df = prepare(peyton_manning_df.iloc[:n_rows].reset_index(drop=True))
        model = build()
        model.fit(df, lib_path=compiled_optimizer_module)
        future = prepare_future(model.make_future_dataframe(periods=periods))
        if "cap" in df and "cap" not in future:
            future = future.assign(cap=df["cap"].iloc[0])
        return model, future
    return run


# -- the round trip -------------------------------------------------------

@pytest.mark.parametrize("configuration", list(CONFIGURATIONS))
def test_a_loaded_model_predicts_identically(fit_one, configuration):
    """Bit-identical, not close. Nothing is refitted on the way back, so any
    difference is an attribute that was not written.

    Both generators are seeded, and that is what makes this the strong version
    of the test rather than the weak one. `rng` is unseeded by design (#41), so
    the interval columns differ between two calls on the *same* model -- compare
    without seeding and they have to be skipped, which would leave the whole
    uncertainty path unchecked. Seeded, the intervals must match too, and they
    only can if `params`, `sigma_obs`, `changepoints_t` and `y_scale` all came
    back exactly.
    """
    model, future = fit_one(configuration)
    restored = model_from_json(model_to_json(model))

    before = seed_all(model).predict(future)
    after = seed_all(restored).predict(future)

    assert list(before.columns) == list(after.columns)
    compared = 0
    for column in before.columns:
        if before[column].dtype.kind in "ifu":
            np.testing.assert_array_equal(before[column].values,
                                          after[column].values, err_msg=column)
            compared += 1
    assert compared >= 4, "too few numeric columns compared to mean anything"


def test_the_intervals_are_sampled_which_is_why_the_test_above_seeds(fit_one):
    """Guards the guard: if `rng` ever became seeded by default, the test above
    would still pass while checking something weaker than it claims to."""
    model, future = fit_one("plain")

    first = model.predict(future)["yhat_upper"].values
    second = model.predict(future)["yhat_upper"].values

    assert not np.array_equal(first, second), (
        "yhat_upper is now reproducible without seeding -- the round-trip test "
        "no longer needs to seed, and its docstring is out of date")


def test_the_dict_form_round_trips_too(fit_one):
    """[fc] both pairs exist there, and the json ones are thin wrappers."""
    model, future = fit_one("country")

    restored = model_from_dict(model_to_dict(model))

    np.testing.assert_array_equal(model.predict(future)["yhat"].values,
                                  restored.predict(future)["yhat"].values)


def test_what_is_written_is_json(fit_one):
    model, _ = fit_one("plain")
    blob = model_to_json(model)

    assert isinstance(blob, str)
    payload = json.loads(blob)
    assert payload["__analytic_prophet_version"]
    assert payload["growth"] == "linear"


def test_serializing_an_unfitted_model_is_refused():
    """[fc] the wording, so a user who has met Prophet's message meets the
    same one here."""
    with pytest.raises(ValueError, match="already been fit"):
        model_to_json(AnalyticProphet())


# -- what must not be written --------------------------------------------

def test_no_handle_on_the_compiled_extension_is_written(fit_one):
    """The reason pickle is not the route. A model fitted with the compiled path must
    load on a machine that has never built optimize.cpp, so nothing may carry a
    path to one machine's build."""
    model, _ = fit_one("plain")
    assert model._fit_lib_path is not None, "fixture no longer exercises the C++ path"

    payload = json.loads(model_to_json(model))

    assert "_fit_lib_path" not in payload
    blob = json.dumps(payload)
    assert ".so" not in blob and "analytic_prophet_cpp" not in blob
    assert model_from_dict(payload)._fit_lib_path is None


def test_a_loaded_model_predicts_without_the_extension(fit_one, monkeypatch):
    """The same claim, from the other side: predict never loads the core, so a
    restored model works where it cannot even be built.

    Patched on `models`, where load_cpp_module reads its own globals -- patching
    the copy `forecaster` imported would leave the loader untouched.
    """
    model, future = fit_one("plain")
    restored = model_from_json(model_to_json(model))

    def refuse(*args, **kwargs):
        raise AssertionError("predict must not load the compiled extension")

    monkeypatch.setattr(models, "load_cpp_module", refuse)
    assert np.all(np.isfinite(restored.predict(future)["yhat"].values))


def test_the_optimizers_own_record_is_dropped(fit_one):
    """[fc] Prophet skips stan_fit and stan_backend for the same reason: a
    prediction never consults them."""
    model, _ = fit_one("plain")
    payload = json.loads(model_to_json(model))

    for attribute in ("opt", "opt_status", "opt_status_message", "optimizer_used",
                      "loss_over_iterations", "_fit_design_matrix", "_data_columns"):
        assert attribute not in payload, attribute


# -- the two attribute lists have to account for the model ----------------

def test_the_two_attribute_lists_account_for_the_model(peyton_manning_df,
                                                       compiled_optimizer_module):
    """Every attribute is either written or deliberately skipped.

    #41's `FIT_DERIVED_ATTRIBUTES` names what a refit must forget; the lists
    here name what a load must restore. An attribute in neither is one nobody
    has thought about, which is how the three regressions below got in.
    """
    df = with_regressor(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = AnalyticProphet().add_country_holidays("US").add_regressor("temp")
    model.fit(df, lib_path=compiled_optimizer_module)

    written = set(serialize.SIMPLE_ATTRIBUTES + serialize.PD_SERIES
                  + serialize.PD_DATAFRAME + serialize.NP_ARRAY
                  + serialize.ORDEREDDICT + serialize.SET_ATTRIBUTES
                  + ["params", "layout"])
    unaccounted = set(model.__dict__) - written - set(serialize.SKIPPED)

    assert unaccounted == set()
    assert written & set(serialize.SKIPPED) == set(), "an attribute both written and skipped"


# -- #41's contract has to survive the file ------------------------------

def test_a_loaded_model_still_undoes_the_changepoint_cap(peyton_manning_df,
                                                         compiled_optimizer_module):
    """[fc] set_changepoints caps `n_changepoints` and overwrites it; #41 undoes
    the cap before the next fit. Without `_n_changepoints_before_cap` in the
    file, a model fitted on twenty rows, saved, loaded and refitted on 1200 kept
    15 changepoints instead of 25 -- a different model, silently."""
    tiny = peyton_manning_df.iloc[:20].reset_index(drop=True)
    long = peyton_manning_df.iloc[:1200].reset_index(drop=True)

    model = AnalyticProphet()
    model.fit(tiny, lib_path=compiled_optimizer_module)
    assert model.n_changepoints == 15, "the cap should bite on twenty rows"

    restored = model_from_json(model_to_json(model))
    restored.fit(long, lib_path=compiled_optimizer_module)

    assert restored.n_changepoints == 25


def test_a_loaded_model_still_drops_auto_components_it_cannot_identify(
        peyton_manning_df, compiled_optimizer_module):
    """The bug #41 was filed about, and the one that came back through the file.

    Without `_auto_registered`, a model fitted on two years (yearly + weekly),
    saved, loaded and refitted on 300 days kept `yearly` -- which that history
    cannot identify and which Prophet's own rule would have disabled.
    """
    long = peyton_manning_df.iloc[:1200].reset_index(drop=True)
    short = peyton_manning_df.iloc[:300].reset_index(drop=True)

    fresh = AnalyticProphet()
    fresh.fit(short, lib_path=compiled_optimizer_module)

    model = AnalyticProphet()
    model.fit(long, lib_path=compiled_optimizer_module)
    assert "yearly" in model.seasonalities
    restored = model_from_json(model_to_json(model))
    restored.fit(short, lib_path=compiled_optimizer_module)

    assert sorted(restored.seasonalities) == sorted(fresh.seasonalities)
    assert "yearly" not in restored.seasonalities


def test_a_loaded_model_can_be_refitted_with_its_explicit_changepoints(
        peyton_manning_df, compiled_optimizer_module):
    """This one did not fit a different model -- it raised.

    `_reset_fit_state` restores `changepoints` from the snapshot __init__ takes,
    and a plainly-constructed instance records None there. Refitting a restored
    model whose changepoints were given raised `TypeError: object of type
    'NoneType' has no len()`, which is why model_from_dict constructs with them.
    """
    given = pd.to_datetime(["2008-06-01", "2008-09-01", "2009-02-01"])
    df = peyton_manning_df.iloc[:1200].reset_index(drop=True)

    model = AnalyticProphet(changepoints=given)
    model.fit(df, lib_path=compiled_optimizer_module)
    restored = model_from_json(model_to_json(model))

    restored.fit(df, lib_path=compiled_optimizer_module)

    assert restored.specified_changepoints
    assert len(restored.changepoints) == 3
    assert restored.layout.n_changepoints == 3


# -- the nested predictors, which Prophet has no equivalent of -----------

def test_a_regressors_own_model_survives_the_round_trip(fit_one):
    """Since #33 a regressor may carry a whole AnalyticProphet of its own.
    Serialization recurses, or a restored model can fit a future frame's trend
    and then have nothing to put in its regressor column."""
    model, future = fit_one("regressor predictor")
    bare = future.drop(columns=["temp"])

    restored = model_from_json(model_to_json(model))
    predictor = restored.extra_regressors["temp"]["predictor"]

    assert isinstance(predictor, AnalyticProphet)
    assert predictor._regressor_name == "temp"
    np.testing.assert_array_equal(
        model._ensure_regressor_values(bare)["temp"].values,
        restored._ensure_regressor_values(bare)["temp"].values)
