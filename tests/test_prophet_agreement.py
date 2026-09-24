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

Both criteria used to carry an identifiability caveat, and #16 task 3 removed
it. The 328-day slice that diverged 111% over a 30-day forecast was doing so
because the benchmark *forced* yearly seasonality on both sides at a length
where Prophet's own rule disables it. With auto-selection implemented, both
sides fit weekly-only there and agree to 0.709%. The caveat was measuring a
configuration no user would get.

What remains is a milder version of the same thing, and it is now about the
trend rather than the seasonality: on short series the changepoint/rate
decomposition is loose, and Prophet stops in a flatter region than we do.
Measured, that costs up to 1.216% (at T=500) against 0.207-0.537% on series
past two years. It is not changepoint placement -- refitting on Prophet's own
changepoints moves T=500 from 1.216% to 1.207% -- and our posterior is the
better one at every size measured, T = 100 through 2905.
"""
import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet

# Prophet's own threshold for yearly seasonality being identifiable, and the
# point at which its auto rule switches yearly on. [fc] set_auto_seasonalities.
MIN_IDENTIFIED_DAYS = 730

# Max |yhat difference| as a fraction of the series scale, over history plus a
# 30-day horizon.
#
# Measured across every slice of the Peyton Manning series past two years --
# T = 730, 800, 1000, 1500, 2000, 2500, 2905 -- where the observed range is
# 0.207% to 0.537%, mean 0.044% to 0.132%. 1% leaves roughly 1.9x headroom over
# the worst case while staying far tighter than a real regression.
TOLERANCE = 0.01

# Short series, where the trend decomposition is loose. Measured 0.026% to
# 1.216% over T = 50, 100, 200, 300, 400, 500, so 2% keeps about 1.6x headroom.
# Kept separate rather than folded into TOLERANCE so that a regression on the
# well-identified series cannot hide behind the looser bound.
SHORT_SERIES_TOLERANCE = 0.02

HORIZON = 30


def history_span_days(df):
    return (pd.to_datetime(df["ds"].iloc[-1]) - pd.to_datetime(df["ds"].iloc[0])).days


@pytest.fixture(scope="module")
def data(request):
    import _common as benchmark_common
    return benchmark_common

def seasonal_block(seasonalities, df):
    """The seasonal columns for a registry, through the model's own builder.

    [fc] make_all_seasonality_features is a method because it reads the
    holiday and regressor registries too, so a test wanting only the seasonal
    part goes through a model configured with just that.
    """
    model = AnalyticProphet(yearly_seasonality=False, weekly_seasonality=False,
                          daily_seasonality=False)
    model.seasonalities = seasonalities
    return np.ascontiguousarray(model.make_all_seasonality_features(df)[0].to_numpy(dtype=float))



def fit_ours(df, lib_path, changepoints_t=None):
    model = AnalyticProphet()
    if changepoints_t is not None:
        model.set_changepoints = lambda: setattr(model, "changepoints_t", changepoints_t.copy())
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

    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    X_stan = np.asarray(stan_data["X"], dtype=float)

    ours = fit_ours(df, compiled_optimizer_module, changepoints_t=changepoints_t)
    X_ours = seasonal_block(ours.seasonalities, pd.DataFrame({"ds": ours.ds}))
    beta_in_stan, residual = bridge.transfer_seasonality(
        ours.params["beta"][0], X_ours, X_stan)

    # the two Fourier bases span the same space, so this transfer is exact;
    # if it were not, the comparison below would be meaningless
    assert residual < 1e-8

    lp_ours = bridge.stan_log_prob(stan_model, stan_data,
                                   ours.params["k"][0][0], ours.params["m"][0][0],
                                   ours.params["delta"][0],
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


@pytest.mark.parametrize("n_rows", [100, 200, 300, 500])
def test_short_series_no_longer_diverge(prophet_comparison, compiled_optimizer_module, n_rows):
    """What used to be the documented exception.

    Before #16 task 3 the benchmark forced yearly seasonality on both sides at
    every length, and a 328-day slice diverged 111% over a 30-day forecast with
    fitted `k` differing eightfold. That was a configuration Prophet's own rule
    rejects: under two years of history it disables yearly, and the model this
    project was comparing against was not the one a user gets.

    Both sides now select components by the same rule -- weekly only at these
    lengths -- and the forecasts agree to within a percent or so. The bound is
    looser than TOLERANCE because the trend decomposition is still loose here,
    not because the seasonality is.
    """
    Prophet, common, _ = prophet_comparison
    df = common.load_data(n_rows)
    assert history_span_days(df) < MIN_IDENTIFIED_DAYS

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.fit(df)
    prophet_forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=HORIZON))

    ours = fit_ours(df, compiled_optimizer_module)
    our_forecast = ours.predict(ours.make_future_dataframe(periods=HORIZON))

    # the premise of the test: neither side fits yearly at this length
    assert list(ours.seasonalities) == list(prophet_model.seasonalities) == ["weekly"]

    n = min(len(prophet_forecast), len(our_forecast))
    y_scale = float(np.max(np.abs(df["y"].values)))
    difference = np.max(np.abs(prophet_forecast["yhat"].values[:n]
                               - our_forecast["yhat"].values[:n])) / y_scale

    assert difference < SHORT_SERIES_TOLERANCE, (
        f"forecasts differ by {difference * 100:.3f}% of the series scale at "
        f"T={n_rows}, above the {SHORT_SERIES_TOLERANCE * 100:.1f}% short-series "
        f"tolerance")


@pytest.mark.parametrize("n_rows", [1000, 2905])
def test_seasonality_coefficients_agree(prophet_comparison, compiled_optimizer_module, n_rows):
    """Criterion 1b, unlocked by #16 task 1.

    Before the Fourier basis was aligned, `beta` could not be compared at all:
    a different time origin rotated it and a different column order permuted
    it. Both now match Prophet's construction exactly, so the coefficients are
    directly comparable and a modelling error in the seasonality would show up
    here rather than only as a prediction difference.

    Compared per component, each scaled by its own largest coefficient: the
    weekly coefficients are about a quarter the size of the yearly ones, so a
    single scale would let a weekly disagreement hide under the yearly block.

    Observed with both components selected (#16 task 3): yearly 0.209% at
    T=2905 and 9.156% at T=1000, weekly 0.013% and 0.273%. The bound keeps
    roughly 1.6x headroom over the worst. The T=1000 yearly figure is the
    trend/seasonality trade-off on just under three years of history, in the
    same flat region that leaves our posterior 2.76 nats ahead there -- the
    fitted curves still agree to 0.537%.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    _, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = fit_ours(df, compiled_optimizer_module, changepoints_t=changepoints_t)

    beta_ours = ours.params["beta"][0]
    beta_prophet = prophet_params["beta"]
    assert beta_ours.shape == beta_prophet.shape

    # both registries are built by the same rule, so the blocks line up; if they
    # did not, comparing them column-wise would be meaningless
    assert list(ours.seasonalities) == list(prophet_model.seasonalities)

    offset = 0
    for name, props in ours.seasonalities.items():
        block = slice(offset, offset + 2 * props["fourier_order"])
        offset = block.stop
        relative = (np.max(np.abs(beta_ours[block] - beta_prophet[block]))
                    / np.max(np.abs(beta_prophet[block])))
        assert relative < 0.15, (
            f"{name} coefficients differ by {relative * 100:.2f}% of that "
            f"component's largest coefficient at T={n_rows}")
    assert offset == len(beta_prophet)


@pytest.mark.parametrize("n_rows", [1000, 2905])
def test_fitted_noise_level_agrees(prophet_comparison, compiled_optimizer_module, n_rows):
    """`sigma_obs` is a single identifiable scalar -- unlike `k` and `delta`,
    which trade off against each other, so neither is comparable on its own.

    (That trade-off is why the trend is compared as a curve, in
    test_predictions_agree_on_identified_series, rather than parameter by
    parameter: at T=1000 Prophet reports k=-0.055 against our -0.006 while the
    fitted trends agree to well under a percent.)
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    _, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    ours = fit_ours(df, compiled_optimizer_module, changepoints_t=changepoints_t)

    relative = abs(ours.sigma_obs - prophet_params["sigma_obs"][0]) / prophet_params["sigma_obs"][0]
    # measured 1.060% at T=1000 and 0.090% at T=2905, ours the smaller of the
    # two in every case -- consistent with reaching the better optimum, since a
    # lower residual variance is what a better fit of the same data means
    assert relative < 0.02, (
        f"fitted noise level differs by {relative * 100:.2f}% at T={n_rows}")
