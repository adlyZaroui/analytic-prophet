"""
Issue #16 task 10: extra regressors. Closes #33.

`add_regressor('temperature')` names a column that the frames passed to fit()
and predict() must both carry. It joins the design matrix after the holiday
columns, with its own prior scale and its own mode.

The method it replaces was a stub:

    def add_regressor(self, regressor: pd.Series) -> None:
        pass

which accepted a series and discarded it (#33). The signature could not have
worked even had the body been written: a series carries the history's values and
no way to produce the future ones that predict() needs. Prophet reads the column
from the dataframe by name, at both ends, which is what this now does.

Two details are easy to get wrong and are pinned below:

  * `prior_scale` defaults to `holidays_prior_scale`, not to
    `seasonality_prior_scale`. That reads like a mistake in the original. It is
    not -- [fc] `prior_scale = float(self.holidays_prior_scale)`.
  * `standardize='auto'` standardizes unless the column is binary. Centring a
    0/1 indicator would turn "the flag is on" into two values neither of which
    is zero, and its coefficient would stop meaning what it did.
"""
import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet, regressor_standardization


def with_regressors(df, seed=0):
    rng = np.random.default_rng(seed)
    return df.assign(temp=rng.normal(15, 8, len(df)),
                     promo=(rng.random(len(df)) < 0.2).astype(int))


# -- the stub is gone ---------------------------------------------------

def test_add_regressor_no_longer_silently_does_nothing(peyton_manning_df,
                                                       compiled_optimizer_module):
    """#33: the stub accepted a series and discarded it, so a caller's
    regressor was simply absent from the fitted model with no error."""
    df = with_regressors(peyton_manning_df.iloc[:1000].reset_index(drop=True))

    without = AnalyticProphet()
    without.fit_cpp(df, lib_path=compiled_optimizer_module)

    with_one = AnalyticProphet().add_regressor("temp")
    with_one.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert with_one.layout.n_holiday_columns == without.layout.n_holiday_columns + 1
    assert with_one.layout.size == without.layout.size + 1


def test_passing_a_series_is_now_a_clear_error():
    """The old signature took a pd.Series. Passing one now fails on the name
    checks rather than being accepted and dropped."""
    with pytest.raises((ValueError, TypeError)):
        AnalyticProphet().add_regressor(pd.Series([1.0, 2.0, 3.0]))


# -- validation ---------------------------------------------------------

@pytest.mark.parametrize("args,kwargs", [
    (("temp",), {}),
    (("temp",), {"prior_scale": 2.0}),
    (("temp",), {"prior_scale": 0.0}),
    (("temp",), {"prior_scale": -1.0}),
    (("temp",), {"mode": "multiplicative"}),
    (("temp",), {"mode": "sideways"}),
    (("trend",), {}),
    (("yhat_lower",), {}),
    (("a_delim_b",), {}),
])
def test_accepts_and_rejects_what_prophet_does(prophet_comparison, args, kwargs):
    Prophet, _, _ = prophet_comparison

    def outcome(model):
        try:
            model.add_regressor(*args, **kwargs)
        except Exception as exc:                  # noqa: BLE001 -- the class is the assertion
            return type(exc)
        return None

    assert outcome(AnalyticProphet()) is outcome(Prophet())


def test_returns_self_so_calls_chain():
    model = AnalyticProphet()
    assert model.add_regressor("a").add_regressor("b") is model
    assert list(model.extra_regressors) == ["a", "b"]


def test_prior_scale_defaults_to_the_holidays_one_not_the_seasonality_one():
    """[fc] `prior_scale = float(self.holidays_prior_scale)`. Surprising, and
    matched deliberately -- a model with different holiday and seasonality
    scales would otherwise put regressors on the wrong one."""
    model = AnalyticProphet()
    model.seasonality_prior_scale = 3.0
    model.holidays_prior_scale = 7.0
    model.add_regressor("temp")

    assert model.extra_regressors["temp"]["prior_scale"] == 7.0


def test_a_regressor_may_not_collide_with_a_seasonality_or_holiday():
    model = AnalyticProphet().add_seasonality("monthly", 30.5, 5)
    with pytest.raises(ValueError, match="already used for a seasonality"):
        model.add_regressor("monthly")

    model = AnalyticProphet().add_country_holidays("US")
    with pytest.raises(ValueError, match="is a holiday name in US"):
        model.add_regressor("Christmas Day")


def test_the_same_regressor_may_be_re_registered():
    """[fc] check_regressors=False, so re-registering overwrites rather than
    colliding -- the way to change a prior scale after the fact."""
    model = AnalyticProphet().add_regressor("temp", prior_scale=1.0)
    model.add_regressor("temp", prior_scale=5.0)

    assert model.extra_regressors["temp"]["prior_scale"] == 5.0
    assert list(model.extra_regressors) == ["temp"]


def test_adding_a_regressor_after_a_fit_is_refused(peyton_manning_df,
                                                   compiled_optimizer_module):
    model = AnalyticProphet()
    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    with pytest.raises(RuntimeError, match="before fitting"):
        model.add_regressor("temp")


# -- standardization ----------------------------------------------------

@pytest.mark.parametrize("values,standardize,expected", [
    ([1.0, 2.0, 3.0, 4.0], "auto", "standardized"),
    ([0, 1, 1, 0], "auto", "untouched"),            # binary
    ([5.0, 5.0, 5.0], "auto", "untouched"),         # constant: nothing to divide by
    ([1.0, 2.0, 3.0, 4.0], False, "untouched"),
    ([0, 1, 1, 0], True, "standardized"),
])
def test_auto_standardizes_all_but_binary_and_constant(values, standardize, expected):
    mu, std = regressor_standardization(pd.Series(values), standardize)

    if expected == "untouched":
        assert (mu, std) == (0.0, 1.0)
    else:
        assert mu == pytest.approx(float(pd.Series(values).mean()))
        assert std == pytest.approx(float(pd.Series(values).std()))


def test_standardization_matches_prophets(prophet_comparison, peyton_manning_df,
                                          compiled_optimizer_module):
    """Prophet uses pandas' std, which is the sample standard deviation
    (ddof=1). A population std would be quietly wrong by a factor of
    sqrt((n-1)/n)."""
    Prophet, common, _ = prophet_comparison
    df = with_regressors(common.load_data(1000))

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.add_regressor("temp")
    prophet_model.add_regressor("promo")
    prophet_model.fit(df)

    ours = AnalyticProphet().add_regressor("temp").add_regressor("promo")
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    for name in ("temp", "promo"):
        theirs, mine = prophet_model.extra_regressors[name], ours.extra_regressors[name]
        assert mine["mu"] == pytest.approx(theirs["mu"])
        assert mine["std"] == pytest.approx(theirs["std"])


def test_standardization_is_fitted_on_history_and_reused_at_predict(
        peyton_manning_df, compiled_optimizer_module):
    """The future frame must not restandardize itself: the coefficient was
    fitted against the history's mean and spread, so a future frame on a
    different scale has to be mapped through the *same* transformation."""
    df = with_regressors(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = AnalyticProphet().add_regressor("temp")
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    fitted_mu = model.extra_regressors["temp"]["mu"]
    fitted_std = model.extra_regressors["temp"]["std"]

    future = model.make_future_dataframe(periods=30)
    # deliberately on a different scale from the history
    future["temp"] = np.linspace(100.0, 200.0, len(future))
    model.predict(future)

    assert model.extra_regressors["temp"]["mu"] == fitted_mu
    assert model.extra_regressors["temp"]["std"] == fitted_std


# -- the column ---------------------------------------------------------

def test_a_missing_regressor_column_is_rejected(peyton_manning_df,
                                                compiled_optimizer_module):
    model = AnalyticProphet().add_regressor("temp")

    with pytest.raises(ValueError, match="Regressor 'temp' missing"):
        model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                      lib_path=compiled_optimizer_module)


def test_a_nan_in_a_regressor_column_is_rejected(peyton_manning_df,
                                                 compiled_optimizer_module):
    df = with_regressors(peyton_manning_df.iloc[:300].reset_index(drop=True))
    df.loc[10, "temp"] = np.nan
    model = AnalyticProphet().add_regressor("temp")

    with pytest.raises(ValueError, match="Found NaN in column 'temp'"):
        model.fit_cpp(df, lib_path=compiled_optimizer_module)


def test_predict_requires_the_regressor_column(peyton_manning_df,
                                               compiled_optimizer_module):
    """The reason the stub's signature could not have worked: predict needs
    future values, which a series handed over at registration cannot supply."""
    df = with_regressors(peyton_manning_df.iloc[:400].reset_index(drop=True))
    model = AnalyticProphet().add_regressor("temp")
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    future = model.make_future_dataframe(periods=30)
    with pytest.raises(ValueError, match="Regressor 'temp' missing"):
        model.predict(future)

    future["temp"] = 15.0
    forecast = model.predict(future)
    assert np.all(np.isfinite(forecast["yhat"].values))


def test_regressor_columns_follow_the_holiday_ones(peyton_manning_df,
                                                   compiled_optimizer_module):
    """[fc] make_all_seasonality_features: seasonalities, then holidays, then
    regressors. That order is beta's layout."""
    df = with_regressors(peyton_manning_df.iloc[:1000].reset_index(drop=True))
    frame = pd.DataFrame({"holiday": "bump",
                          "ds": [pd.Timestamp(f"{y}-03-15") for y in (2008, 2009)],
                          "lower_window": 0, "upper_window": 0})

    model = (AnalyticProphet().add_holidays(frame, prior_scale=3.0)
             .add_regressor("temp", prior_scale=7.0))
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    data = model.sigmas[model.layout.holiday_block]
    assert data.shape == (2,)
    assert data[0] == 3.0       # the holiday column
    assert data[1] == 7.0       # the regressor


def test_a_multiplicative_regressor_is_marked_as_such(peyton_manning_df,
                                                      compiled_optimizer_module):
    df = with_regressors(peyton_manning_df.iloc[:1000].reset_index(drop=True))
    model = (AnalyticProphet().add_regressor("temp")
             .add_regressor("promo", mode="multiplicative"))
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert model._multiplicative
    np.testing.assert_array_equal(model.s_m[model.layout.seasonality_block], 0.0)
    np.testing.assert_array_equal(model.s_m[model.layout.holiday_block], [0.0, 1.0])


def test_a_regressor_carrying_signal_is_actually_fitted(peyton_manning_df,
                                                        compiled_optimizer_module):
    """A column built from a known effect has to be recovered, or the columns
    are being carried and ignored -- which is what #33 was about."""
    df = peyton_manning_df.iloc[:1000].reset_index(drop=True).copy()
    rng = np.random.default_rng(0)
    df["driver"] = rng.normal(size=len(df))
    df["y"] = df["y"] + 2.0 * df["driver"]

    model = AnalyticProphet().add_regressor("driver")
    model.fit_cpp(df, lib_path=compiled_optimizer_module)

    coefficient = model.params["beta"][0][model.layout.holiday_block][-1]
    std = model.extra_regressors["driver"]["std"]
    # beta is in normalized units against the standardized column
    recovered = coefficient * model.y_scale / std
    assert recovered == pytest.approx(2.0, rel=0.1)


# -- against Prophet ----------------------------------------------------

def test_design_matrix_sigmas_and_posterior_match_prophets(prophet_comparison,
                                                           compiled_optimizer_module):
    Prophet, common, bridge = prophet_comparison
    df = with_regressors(common.load_data(1000))

    def configure(model):
        model.add_regressor("temp")
        model.add_regressor("promo", prior_scale=2.0)
        return model

    prophet_model = configure(Prophet(**common.PROPHET_KWARGS))
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, prophet_params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = configure(AnalyticProphet())
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit_cpp(df, lib_path=compiled_optimizer_module)

    _, X_ours = ours._design_matrices()
    X_stan = np.asarray(stan_data["X"], dtype=float)
    assert X_ours.shape == X_stan.shape

    # exact on the regressor block: these are data carried through a fixed
    # affine map, so nothing about the Fourier basis touches them
    data = ours.layout.holiday_block
    np.testing.assert_array_equal(X_ours[:, data], X_stan[:, data])

    np.testing.assert_array_equal(ours.sigmas, np.asarray(stan_data["sigmas"], dtype=float))
    np.testing.assert_array_equal(ours.s_m, np.asarray(stan_data["s_m"], dtype=float))

    lp_ours = bridge.stan_log_prob(stan_model, stan_data, ours.params["k"][0][0],
                                   ours.params["m"][0][0], ours.params["delta"][0],
                                   ours.sigma_obs, ours.params["beta"][0])
    assert lp_ours >= lp_prophet - 1e-6
