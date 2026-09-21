"""
Issue #30: the acceptance criterion against the original Prophet.

Comparing fitted parameters was the obvious criterion and it does not work.
`beta` lives in a rotated Fourier basis -- Prophet measures days from the 1970
epoch, this implementation from the series start -- so the coefficients differ,
including in sign, while describing the same function. `delta` is indexed
against different changepoints. Neither can be compared element-wise, and a
test that tried would be measuring the parameterization, not the model.

Two criteria replace it:

  1. POSTERIOR -- our optimum is at least as good as Prophet's, scored by
     Stan's own log density. One scalar, invariant to both reparameterizations,
     and the quantity the model is actually defined by. This is the strong
     criterion: it would catch a genuine modelling error, which `beta` agreeing
     to N digits would not.

  2. PREDICTIONS -- forecasts agree within a tolerance set from measurement
     (see TOLERANCE below), on series long enough for the model to be
     identified.

The identifiability caveat is not a hedge. With under two years of history,
yearly seasonality and the trend trade off almost freely -- Prophet emits a
warning of its own at that point. On a 328-day slice the two implementations
agree to 2.6% in-sample and then diverge to 111% over a 30-day forecast, with
fitted `k` differing eightfold, while *our* posterior is the better one. That
is the model being under-determined, not either implementation being wrong,
and a forecast-agreement test over such a series would be measuring noise.
"""
import numpy as np
import pandas as pd
import pytest

from customProphet import CustomProphet, fourier_components, n_yearly, SIGMA_OBS_IDX

# Prophet's own threshold for yearly seasonality being identifiable.
MIN_IDENTIFIED_DAYS = 730

# Max |yhat difference| as a fraction of the series scale, over history plus a
# 30-day horizon. Measured across every identified slice of the Peyton Manning
# series -- T = 730, 800, 1000, 1500, 2000, 2905 -- where the observed range is
# 0.320% to 0.419%, mean 0.055% to 0.088%. 1% leaves roughly 2.4x headroom over
# the worst case while staying far tighter than a real regression: the
# under-identified slice above misses by 111%.
TOLERANCE = 0.01

HORIZON = 30


def history_span_days(df):
    return (pd.to_datetime(df["ds"].iloc[-1]) - pd.to_datetime(df["ds"].iloc[0])).days


@pytest.fixture(scope="module")
def data(request):
    import _common as benchmark_common
    return benchmark_common


def fit_ours(df, lib_path, change_points=None):
    model = CustomProphet()
    if change_points is not None:
        model._generate_change_points = lambda: setattr(model, "change_points", change_points.copy())
    model.fit_cpp(df, lib_path=lib_path)
    return model


@pytest.mark.parametrize("n_rows", [300, 1000, 2905])
def test_posterior_at_least_as_good_as_prophet(prophet_comparison, compiled_optimizer_module, n_rows):
    """Criterion 1. Scored under Stan's own log density, on the same model
    specification -- Prophet's changepoints -- so this isolates optimizer
    quality rather than folding in the placement difference of #15.

    Holds at every size including the under-identified one, which is why it is
    the criterion the project is held to.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    reported = float(np.asarray(prophet_model.params["lp__"]).ravel()[0])

    # if this scoring path does not reproduce Prophet's own number, nothing
    # built on it means anything
    lp_prophet = bridge.validate_bridge(stan_model, stan_data, prophet_params, reported)

    t_change = np.asarray(stan_data["t_change"], dtype=float)
    X_stan = np.asarray(stan_data["X"], dtype=float)

    ours = fit_ours(df, compiled_optimizer_module, change_points=t_change)
    X_ours = fourier_components(ours.t_scaled, 365.25 / ours.scale_period, n_yearly)
    beta_in_stan, residual = bridge.transfer_seasonality(
        ours.opt_params[SIGMA_OBS_IDX + 1:], X_ours, X_stan)

    # the two Fourier bases span the same space, so this transfer is exact;
    # if it were not, the comparison below would be meaningless
    assert residual < 1e-8

    lp_ours = bridge.stan_log_prob(stan_model, stan_data,
                                   ours.opt_params[0], ours.opt_params[1],
                                   ours.opt_params[2:2 + len(t_change)],
                                   ours.sigma_obs, beta_in_stan)

    assert lp_ours >= lp_prophet - 1e-6, (
        f"our posterior is worse than Prophet's at T={n_rows}: "
        f"{lp_ours:.4f} < {lp_prophet:.4f}")


@pytest.mark.parametrize("n_rows", [1000, 2905])
def test_predictions_agree_on_identified_series(prophet_comparison, compiled_optimizer_module, n_rows):
    """Criterion 2, on series long enough to identify yearly seasonality.

    Both models run in their default configuration here, because that is what
    a user gets -- unlike the posterior test, this deliberately includes the
    changepoint placement difference.
    """
    Prophet, common, _ = prophet_comparison
    df = common.load_data(n_rows)
    assert history_span_days(df) >= MIN_IDENTIFIED_DAYS

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.fit(df)
    prophet_forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=HORIZON))

    ours = fit_ours(df, compiled_optimizer_module)
    our_forecast = ours.predict(ours.make_future_dataframe(periods=HORIZON))

    n = min(len(prophet_forecast), len(our_forecast))
    assert np.array_equal(pd.to_datetime(prophet_forecast["ds"]).values[:n],
                          pd.to_datetime(our_forecast["ds"]).values[:n]), \
        "forecast frames are not date-aligned; the comparison would be meaningless"

    y_scale = float(np.max(np.abs(df["y"].values)))
    difference = np.max(np.abs(prophet_forecast["yhat"].values[:n]
                               - our_forecast["yhat"].values[:n])) / y_scale

    assert difference < TOLERANCE, (
        f"forecasts differ by {difference * 100:.3f}% of the series scale at "
        f"T={n_rows}, above the {TOLERANCE * 100:.1f}% tolerance")


def test_under_identified_series_agree_in_sample_only(prophet_comparison, compiled_optimizer_module):
    """The documented exception, pinned so it is not mistaken for a regression.

    With 328 days of history the trend and yearly seasonality are nearly
    unidentifiable. The two implementations fit the history comparably and then
    extrapolate very differently. Asserting the in-sample agreement keeps the
    behaviour honest without pretending the forecasts should match.
    """
    Prophet, common, _ = prophet_comparison
    df = common.load_data(300)
    assert history_span_days(df) < MIN_IDENTIFIED_DAYS

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.fit(df)
    prophet_fitted = prophet_model.predict(df[["ds"]])

    ours = fit_ours(df, compiled_optimizer_module)
    our_fitted = ours.predict(ours.make_future_dataframe(periods=0))

    n = len(df)
    y_scale = float(np.max(np.abs(df["y"].values)))
    in_sample = np.max(np.abs(prophet_fitted["yhat"].values[:n]
                              - our_fitted["yhat"].values[:n])) / y_scale

    # measured at 2.6%; the bound is loose because this regime is unstable by
    # nature, and the point is to document it rather than police it
    assert in_sample < 0.05
