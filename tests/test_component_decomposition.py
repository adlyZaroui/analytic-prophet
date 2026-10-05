"""
Issue #114: the four gaps #101 found, closed.

`predict_seasonal_components` with `regressor_column_matrix` and
`add_group_component` beneath it, and `predictive_samples`. They were the only
members the parity table called `absent, gap` -- absent and arguably should
not be. Everything else missing is MCMC, plotting, or an internal folded into
something else.

The decomposition is checked against Prophet **on matched parameters**. Fitting
both and comparing would measure the two optimizers, which differ by design:
the raw comparison is 1.2e-03, which is the documented spread between the two
fits and says nothing about whether this computes the decomposition correctly.
Feeding Prophet's own `beta` through this code isolates the arithmetic, and
there it agrees to 1.1e-12 -- the design matrix's own tolerance.
"""
import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet


@pytest.fixture
def fitted(peyton_manning_df, compiled_optimizer_module):
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:600].reset_index(drop=True),
              lib_path=compiled_optimizer_module)
    return model


@pytest.fixture
def future(fitted):
    frame = fitted.make_future_dataframe(periods=30)
    return frame.assign(ds=pd.to_datetime(frame["ds"]))


# -- the decomposition -----------------------------------------------------

def test_the_components_are_prophets_on_matched_parameters(peyton_manning_df,
                                                           prophet_comparison,
                                                           compiled_optimizer_module):
    """The arithmetic, isolated from the optimizer."""
    Prophet, _common, _bridge = prophet_comparison
    import warnings

    frame = peyton_manning_df.iloc[:600].reset_index(drop=True)
    ours = AnalyticProphet()
    ours.fit(frame, lib_path=compiled_optimizer_module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = Prophet().fit(frame)

    horizon = ours.make_future_dataframe(periods=30)
    ours.params["beta"] = np.asarray(theirs.params["beta"])
    ours.y_scale = theirs.y_scale

    mine = ours.predict_seasonal_components(
        horizon.assign(ds=pd.to_datetime(horizon["ds"])))
    yours = theirs.predict_seasonal_components(
        theirs.setup_dataframe(horizon.copy()))

    assert set(mine.columns) == set(yours.columns), "different components"
    for column in yours.columns:
        assert np.max(np.abs(mine[column].to_numpy() - yours[column].to_numpy())) < 1e-9, \
            f"{column} disagrees with Prophet"


def test_the_intervals_are_degenerate_under_map(fitted, future):
    """Not a defect, and not ours: the percentile is taken over `beta`'s
    sample axis, which a MAP fit gives one row. Prophet 1.4.0 does exactly
    this, measured. Widening it would be inventing an interval the original
    does not have, so the test pins the degeneracy rather than tolerating it.
    """
    components = fitted.predict_seasonal_components(future)
    named = [c for c in components.columns if not c.endswith(("_lower", "_upper"))]
    assert named, "no components at all"
    for component in named:
        assert np.allclose(components[component], components[component + "_lower"])
        assert np.allclose(components[component], components[component + "_upper"])


def test_the_components_reconstruct_what_predict_reports(fitted, future):
    """Internal consistency: the additive total plus what the multiplicative
    part contributes at the fitted trend is `predict`'s `seasonality` column,
    which is `yhat - trend`."""
    components = fitted.predict_seasonal_components(future)
    forecast = fitted.predict(future)

    rebuilt = (components["additive_terms"].to_numpy()
               + forecast["trend"].to_numpy()
               * components["multiplicative_terms"].to_numpy())
    assert np.allclose(rebuilt, forecast["seasonality"].to_numpy(), atol=1e-8)


def test_a_component_belongs_to_exactly_one_mode(fitted, future):
    """[fc] Prophet's own guard. A column in both modes would be counted
    twice by the decomposition."""
    _f, _s, component_cols, _m = fitted.make_all_seasonality_features(future)
    both = (component_cols["additive_terms"] + component_cols["multiplicative_terms"])
    assert both.max() <= 1


def test_a_predict_frame_whose_columns_moved_is_refused(fitted, future):
    """[fc] 'A bug occurred in constructing regressors.' The fitted grouping
    is what `beta` is indexed by, so a predict-time matrix that does not match
    it is silently wrong rather than loudly."""
    # a *value* has to differ, not just the column set: the guard reindexes
    # to the fitted columns first, so dropping one selects a subset that still
    # matches. Flipping an entry is what a grouping that moved looks like.
    moved = fitted.train_component_cols.copy()
    moved.iloc[0, 0] = 1 - moved.iloc[0, 0]
    fitted.train_component_cols = moved

    with pytest.raises(Exception, match="bug occurred"):
        fitted.make_all_seasonality_features(future)


def test_a_fit_leaves_the_grouping_behind(fitted):
    """`component_modes` and `train_component_cols` were `absent, gap` rows of
    their own in the parity table."""
    assert fitted.train_component_cols is not None
    assert set(fitted.component_modes) == {"additive", "multiplicative"}
    assert "additive_terms" in fitted.component_modes["additive"]


def test_make_all_seasonality_features_returns_prophets_four(fitted, future):
    """It returned three while sharing Prophet's name -- a divergence the
    naming tests could not see, because they check that a member exists rather
    than what it gives back."""
    returned = fitted.make_all_seasonality_features(future)
    assert len(returned) == 4
    features, scales, component_cols, modes = returned
    assert features.shape[0] == len(future)
    assert len(scales) == features.shape[1]
    assert component_cols.shape[0] == features.shape[1]
    assert set(modes) == {"additive", "multiplicative"}


# -- the draws -------------------------------------------------------------

def test_predictive_samples_has_prophets_shape(fitted, future):
    """Everything internal here is `(n_samples, T)`; Prophet documents and
    returns `(T, n_samples)`. The transpose happens at the boundary because
    the point of the member is that a ported script can use it."""
    draws = fitted.predictive_samples(future)

    assert set(draws) == {"trend", "yhat"}
    for key, value in draws.items():
        assert value.shape == (len(future), fitted.uncertainty_samples), key


def test_the_draws_are_the_ones_predict_reduces(fitted, future):
    """Otherwise it would be a second sampler that happens to look similar.
    Both are seeded from the same generator state, so the quantiles of these
    draws are the bounds `predict` reports.
    """
    fitted.rng = np.random.default_rng(11)
    draws = fitted.predictive_samples(future)
    fitted.rng = np.random.default_rng(11)
    forecast = fitted.predict(future)

    lower = 100 * (1.0 - fitted.interval_width) / 2
    upper = 100 * (1.0 + fitted.interval_width) / 2
    for key in ("yhat", "trend"):
        assert np.allclose(np.percentile(draws[key], lower, axis=1),
                           forecast[key + "_lower"].to_numpy())
        assert np.allclose(np.percentile(draws[key], upper, axis=1),
                           forecast[key + "_upper"].to_numpy())


@pytest.mark.parametrize("vectorized", [True, False])
def test_both_samplers_are_reachable(fitted, future, vectorized):
    draws = fitted.predictive_samples(future, vectorized=vectorized)
    assert np.all(np.isfinite(draws["yhat"]))


def test_predictive_samples_before_a_fit_says_so():
    with pytest.raises(ValueError, match="Model has not been fit"):
        AnalyticProphet().predictive_samples(
            pd.DataFrame({"ds": pd.date_range("2013-01-01", periods=3)}))
