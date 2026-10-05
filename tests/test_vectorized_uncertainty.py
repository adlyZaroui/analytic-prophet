"""
Issue #93: Prophet's `predict` runs an approximate uncertainty sampler by
default, and now so does this one.

The two are **different computations, not two speeds of one**. That is the
whole point, and what the tests are about: that both exist, that which one ran
is knowable, that the approximation is an approximation of the right thing, and
that the exact path did not move when the approximate one was added.

[fc] the argument name, `vectorized`, and the default, True — so a script
ported from Prophet gets what Prophet would have given it.
"""
import numpy as np
import pytest

from analytic_prophet import AnalyticProphet


@pytest.fixture
def fitted(peyton_manning_df, compiled_optimizer_module):
    model = AnalyticProphet(uncertainty_samples=400)
    model.fit(peyton_manning_df.iloc[:1200].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    return model


def forecast(model, vectorized, seed=3, periods=90, **kwargs):
    future = model.make_future_dataframe(periods=periods)
    model.rng = np.random.default_rng(seed)
    return model.predict(future, vectorized=vectorized, **kwargs)


# -- both paths exist, and which one ran is knowable ----------------------

def test_the_approximation_is_the_default(fitted):
    """[fc] Prophet.predict(vectorized=True). A ported script should get what
    it got there."""
    future = fitted.make_future_dataframe(periods=30)
    fitted.rng = np.random.default_rng(1)
    fitted.predict(future)
    assert fitted.predicted_vectorized is True


def test_which_sampler_ran_is_recorded(fitted):
    """Two computations that differ by more than rounding cannot both be
    reported as 'the forecast' without saying which one it is."""
    forecast(fitted, vectorized=False)
    assert fitted.predicted_vectorized is False
    forecast(fitted, vectorized=True)
    assert fitted.predicted_vectorized is True


def test_only_the_interval_differs(fitted):
    """`yhat` is not sampled, so the choice cannot touch it — which is also
    true of Prophet's own two paths."""
    approximate = forecast(fitted, vectorized=True)
    exact = forecast(fitted, vectorized=False)

    np.testing.assert_array_equal(approximate["yhat"].values, exact["yhat"].values)
    np.testing.assert_array_equal(approximate["trend"].values, exact["trend"].values)
    assert not np.array_equal(approximate["yhat_lower"].values,
                              exact["yhat_lower"].values)


def test_the_approximation_agrees_with_the_exact_sampler_to_sampling_noise(fitted):
    """The acceptance criterion from #93: an approximation of the right thing.

    Not bit-identical — it is a different computation — but the interval widths
    must not differ systematically. The yardstick is Prophet's own two paths,
    which disagree by about 1.4%; anything of that order is the approximation
    behaving as Prophet's does.
    """
    approximate = forecast(fitted, vectorized=True)
    exact = forecast(fitted, vectorized=False)

    width = lambda f: (f["yhat_upper"] - f["yhat_lower"]).values
    ratio = np.mean(width(approximate)) / np.mean(width(exact))

    assert 0.9 < ratio < 1.1, (
        f"the approximate interval is {ratio:.2f}x the exact one on average, "
        f"which is a systematic difference rather than sampling noise")


def test_the_history_carries_no_trend_uncertainty_either_way(fitted):
    """[fc] both samplers give the history exactly zero — the band is the
    uncertainty in where the trend goes next, not where it has been (#58)."""
    for vectorized in (True, False):
        result = forecast(fitted, vectorized=vectorized, periods=60)
        history = result.iloc[:len(fitted.ds)]
        spread = (history["trend_upper"] - history["trend_lower"]).values
        assert np.max(np.abs(spread)) < 1e-9, vectorized


def test_both_paths_are_reproducible_from_a_seed(fitted):
    for vectorized in (True, False):
        first = forecast(fitted, vectorized=vectorized)
        second = forecast(fitted, vectorized=vectorized)
        np.testing.assert_array_equal(first["yhat_upper"].values,
                                      second["yhat_upper"].values)


# -- the pieces of the approximation --------------------------------------

def test_the_sampled_deviation_is_zero_wherever_the_history_is(fitted):
    """[fc] _sample_uncertainty returns zeros for `t <= 1` and only samples
    past it."""
    t = np.linspace(0.0, 1.5, 200)
    uncertainty = fitted._sample_uncertainty(t, 50)

    assert uncertainty.shape == (50, len(t))
    assert np.all(uncertainty[:, t <= 1.0] == 0.0)
    assert np.any(uncertainty[:, t > 1.0] != 0.0)


def test_a_frame_entirely_inside_the_history_has_no_deviation_at_all(fitted):
    t = np.linspace(0.0, 1.0, 50)
    assert np.all(fitted._sample_uncertainty(t, 20) == 0.0)


def test_flat_growth_has_no_slope_to_change(peyton_manning_df,
                                            compiled_optimizer_module):
    """[fc] the flat branch returns zeros: there is no rate for a changepoint
    to adjust."""
    model = AnalyticProphet(growth="flat", uncertainty_samples=200)
    model.fit(peyton_manning_df.iloc[:600].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert np.all(model._sample_uncertainty(np.linspace(0, 1.5, 100), 20) == 0.0)


def test_the_shift_matrix_lands_changes_at_about_the_historical_rate(fitted):
    """The approximation's central assumption: one coin per timestep, at the
    rate changepoints occurred in the history."""
    likelihood = 0.25
    shifts = fitted._trend_shift_matrix(1.0, likelihood, 400, 500)

    assert shifts.shape == (500, 400)
    # the trapezoidal average spreads each change over two steps, so a non-zero
    # entry means a change landed on it or just before it
    occupied = np.mean(shifts != 0)
    assert likelihood < occupied < 2.2 * likelihood


# -- logistic falls back, and says so -------------------------------------

def test_logistic_growth_uses_the_exact_sampler(peyton_manning_df,
                                                compiled_optimizer_module):
    """Prophet's approximation needs a separate derivation under logistic
    growth which this does not have yet. Falling back is slower, not wrong —
    but it must be visible rather than silent."""
    df = peyton_manning_df.iloc[:600].reset_index(drop=True)
    df = df.assign(cap=df["y"].max() * 1.5)

    model = AnalyticProphet(growth="logistic", uncertainty_samples=100)
    model.fit(df, lib_path=compiled_optimizer_module)
    future = model.make_future_dataframe(periods=60).assign(cap=df["cap"].iloc[0])

    model.rng = np.random.default_rng(1)
    result = model.predict(future, vectorized=True)

    assert model.predicted_vectorized is False, "logistic must fall back"
    assert np.all(np.isfinite(result["yhat_upper"].values))
