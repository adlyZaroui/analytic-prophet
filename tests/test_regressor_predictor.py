"""
Issue #16 task 14: forecasting a regressor with its own model.

`add_regressor('driver', regressor_predictor=True)` fits a second model on the
regressor column and uses it to supply the future values, so the caller does not
have to invent them.

This changes nothing about the design matrix, the objective or the gradient --
which is why it is a task of its own rather than part of task 10. What it adds
is a second fitted model per regressor, a rule for which rows it fills, and a
question about uncertainty.

The rule is narrower than it first looks. The nested model fills **only rows
past the end of the history**. Rows inside the history that the caller left out
come back from the fit's own values, because the model is there to forecast, not
to re-explain what was already observed. Rows the caller did supply are left
alone, wherever they are.

Uncertainty is not propagated. Prophet draws `predictive_samples` from the
nested model and widens `yhat`'s interval with them; this implementation has no
sampling path to draw from, so the regressor's forecast enters as a point
estimate and the interval is the trend's alone. That is a real gap, and it is
stated here rather than left to be discovered.
"""
import numpy as np
import pandas as pd
import pytest

from customProphet import CustomProphet


def driven(df):
    """A regressor with a trend and a weekly shape, so a nested model has
    something real to forecast rather than noise."""
    t = np.arange(len(df))
    return df.assign(driver=3.0 + 0.002 * t + 0.8 * np.sin(2 * np.pi * t / 7))


# -- registering --------------------------------------------------------

def test_no_predictor_by_default():
    model = CustomProphet().add_regressor("driver")
    assert model.extra_regressors["driver"]["predictor_spec"] is None
    assert model.extra_regressors["driver"]["predictor"] is None


@pytest.mark.parametrize("argument,expected", [
    (True, {}),
    ({"n_changepoints": 5}, {"n_changepoints": 5}),
    (False, None),
    (None, None),
])
def test_a_truthy_non_dict_means_default_settings(argument, expected):
    """[fc] `if isinstance(regressor_predictor, dict)` ... `else {}`."""
    model = CustomProphet().add_regressor("driver", regressor_predictor=argument)
    assert model.extra_regressors["driver"]["predictor_spec"] == expected


def test_an_unsupported_setting_is_rejected():
    """[fc] the spec goes to the constructor, which rejects what it does not
    know. Until #52 this ran against a hand-kept whitelist, which was a second
    list of settable arguments maintained alongside the first."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        CustomProphet().add_regressor("driver",
                                      regressor_predictor={"not_a_setting": 1})


def test_a_setting_prophet_has_and_this_rejects_still_rejects():
    """`mcmc_samples` is a real Prophet argument this implementation refuses.
    The refusal has to reach the spec too, rather than being accepted there
    and failing later inside the nested fit."""
    with pytest.raises(NotImplementedError, match="MAP only"):
        CustomProphet().add_regressor("driver",
                                      regressor_predictor={"mcmc_samples": 100})


# -- fitting the nested model -------------------------------------------

def test_the_nested_model_is_fitted_at_fit_time(peyton_manning_df,
                                                compiled_optimizer_module):
    df = driven(peyton_manning_df.iloc[:1000].reset_index(drop=True))
    model = CustomProphet().add_regressor("driver", regressor_predictor=True)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    predictor = model.extra_regressors["driver"]["predictor"]
    assert isinstance(predictor, CustomProphet)
    assert predictor.opt_params is not None
    assert predictor._regressor_name == "driver"     # [fc] marker


def test_the_nested_model_uses_the_same_fit_path(peyton_manning_df,
                                                 compiled_optimizer_module):
    """A regressor predictor must not quietly drop the compiled path -- it
    would turn one fit into one fast and one slow."""
    df = driven(peyton_manning_df.iloc[:400].reset_index(drop=True))

    compiled = CustomProphet().add_regressor("driver", regressor_predictor=True)
    compiled.fit_cpp(df, lib_path=compiled_optimizer_module)
    assert compiled.extra_regressors["driver"]["predictor"]._fitted_with_cpp

    python_path = CustomProphet().add_regressor("driver", regressor_predictor=True)
    python_path.fit(df, analytic=True)
    assert not python_path.extra_regressors["driver"]["predictor"]._fitted_with_cpp


def test_the_spec_reaches_the_nested_model(peyton_manning_df,
                                           compiled_optimizer_module):
    df = driven(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = CustomProphet().add_regressor(
        "driver", regressor_predictor={"n_changepoints": 5, "weekly_seasonality": False})
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    predictor = model.extra_regressors["driver"]["predictor"]
    assert predictor.n_changepoints == 5
    assert predictor.layout.n_changepoints == 5
    assert "weekly" not in predictor.seasonalities
    # 400 days also puts yearly below its threshold, so this nested model has
    # no seasonal columns at all -- which used to fail outright, see
    # test_a_model_with_no_seasonality_fits in test_seasonality_registry.py
    assert predictor.layout.n_regressor_columns == 0


def test_too_little_data_to_fit_a_nested_model_is_rejected(peyton_manning_df):
    """[fc] "Not enough data to fit regressor model for {name!r}".

    Driven directly rather than through fit(), because a regressor column with
    nulls never reaches this: the main fit rejects it first, in Prophet as well
    as here. The check is defensive on both sides, and pinned as such.
    """
    df = driven(peyton_manning_df.iloc[:300].reset_index(drop=True))
    df.loc[df.index[1:], "driver"] = np.nan          # one usable row

    model = CustomProphet().add_regressor("driver", regressor_predictor=True)
    with pytest.raises(ValueError, match="Not enough data to fit regressor model"):
        model._fit_regressor_models(df)


def test_a_regressor_without_a_predictor_fits_nothing_extra(peyton_manning_df,
                                                            compiled_optimizer_module):
    df = driven(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = CustomProphet().add_regressor("driver")
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert model.extra_regressors["driver"]["predictor"] is None


# -- which rows it fills ------------------------------------------------

def test_predict_needs_no_future_values(peyton_manning_df, compiled_optimizer_module):
    """The point of the feature: make_future_dataframe emits only `ds`, and
    that is now enough."""
    df = driven(peyton_manning_df.iloc[:1000].reset_index(drop=True))
    model = CustomProphet().add_regressor("driver", regressor_predictor=True)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    forecast = model.predict(model.make_future_dataframe(periods=90))

    assert len(forecast) == len(df) + 90
    assert np.all(np.isfinite(forecast["yhat"].values))


def test_predict_does_not_write_into_the_callers_frame(peyton_manning_df,
                                                       compiled_optimizer_module):
    """The filling happens on a copy. Related to #35, which is about predict
    adding `t_scaled` to the frame it is given -- this must not add to it."""
    df = driven(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = CustomProphet().add_regressor("driver", regressor_predictor=True)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=30)
    before = list(future.columns)
    model.predict(future)

    assert "driver" not in future.columns
    assert list(future.columns) == before


def test_supplied_values_win_inside_the_history(peyton_manning_df,
                                                compiled_optimizer_module):
    """The nested model fills the future only. What the caller gave for the
    history is what the history gets."""
    df = driven(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = CustomProphet().add_regressor("driver", regressor_predictor=True)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=30)
    future["driver"] = 99.0
    filled = model._ensure_regressor_values(future)["driver"].to_numpy()

    np.testing.assert_array_equal(filled[:len(df)], 99.0)
    assert not np.any(filled[len(df):] == 99.0)


def test_history_rows_left_out_come_back_from_the_fit(peyton_manning_df,
                                                      compiled_optimizer_module):
    """Not from the nested model: it forecasts, it does not re-explain what
    was observed. [fc] reads them back off self.history."""
    df = driven(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = CustomProphet().add_regressor("driver", regressor_predictor=True)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    filled = model._ensure_regressor_values(
        model.make_future_dataframe(periods=30))["driver"].to_numpy()

    np.testing.assert_allclose(filled[:len(df)], df["driver"].to_numpy(), rtol=1e-12)


def test_the_nested_forecast_tracks_the_signal(peyton_manning_df,
                                               compiled_optimizer_module):
    """A regressor built from a known trend plus a weekly cycle has to be
    forecast as such, or the nested model is decorative."""
    df = driven(peyton_manning_df.iloc[:1000].reset_index(drop=True))
    model = CustomProphet().add_regressor("driver", regressor_predictor=True)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    horizon = 90
    future = model.make_future_dataframe(periods=horizon)
    forecast = model._ensure_regressor_values(future)["driver"].to_numpy()[len(df):]

    t = np.arange(len(df), len(df) + horizon)
    truth = 3.0 + 0.002 * t + 0.8 * np.sin(2 * np.pi * t / 7)

    # the level has to be right, and the weekly shape has to be there at all
    assert np.mean(forecast) == pytest.approx(np.mean(truth), rel=0.05)
    assert np.corrcoef(forecast, truth)[0, 1] > 0.5


# -- against Prophet ----------------------------------------------------

def test_the_forecast_agrees_with_prophets(prophet_comparison,
                                           compiled_optimizer_module):
    """Two models composed, each with its own small disagreement, so this is
    looser than the single-model comparisons -- but it is the only check that
    the nested model is being used the same way."""
    Prophet, common, _ = prophet_comparison
    df = driven(common.load_data(1000))

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.add_regressor("driver", regressor_predictor=True)
    prophet_model.fit(df)
    prophet_forecast = prophet_model.predict(
        prophet_model.make_future_dataframe(periods=90))

    ours = CustomProphet().add_regressor("driver", regressor_predictor=True)
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)
    our_forecast = ours.predict(ours.make_future_dataframe(periods=90))

    n = min(len(prophet_forecast), len(our_forecast))
    scale = float(np.max(np.abs(df["y"].values)))
    difference = np.max(np.abs(prophet_forecast["yhat"].values[:n]
                               - our_forecast["yhat"].values[:n])) / scale

    assert difference < 0.02, f"forecasts differ by {difference * 100:.3f}% of scale"


def test_uncertainty_is_not_propagated(peyton_manning_df, compiled_optimizer_module):
    """Stated rather than left to be found.

    Prophet draws `predictive_samples` from the nested model and widens
    `yhat`'s interval with them. There is no sampling path here, so the
    regressor's forecast enters as a point estimate and the whole interval
    comes from the trend.

    Asserted structurally rather than by comparing two fits: `yhat_upper - yhat`
    equals `trend_upper - trend` exactly, which is only true if nothing but the
    trend contributes width. Implementing the propagation later breaks this
    test, which is the point of it.
    """
    df = driven(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = CustomProphet().add_regressor("driver", regressor_predictor=True)
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    forecast = model.predict(model.make_future_dataframe(periods=30))

    np.testing.assert_allclose(
        forecast["yhat_upper"].values - forecast["yhat"].values,
        forecast["trend_upper"].values - forecast["trend"].values, rtol=1e-12)
    np.testing.assert_allclose(
        forecast["yhat"].values - forecast["yhat_lower"].values,
        forecast["trend"].values - forecast["trend_lower"].values, rtol=1e-12)
