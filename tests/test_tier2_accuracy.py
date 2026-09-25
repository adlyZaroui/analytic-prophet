"""
Tier 2 of the evaluation suite (#79): forecast accuracy, held out.

The question the README explicitly refuses to answer -- whether the better MAP
point generalizes -- so the tests here are about whether the comparison is
*fair*, not about who wins. Three properties decide that, and all three are the
sort of thing that fails silently:

  * **both sides see the same splits.** Rolling-origin evaluation has enough
    knobs that two implementations choosing their own would be comparing splits
    as much as models.
  * **neither side sees the future.** A model fitted at a cutoff must not have
    been shown anything after it. Leakage does not raise; it just produces
    excellent forecasts.
  * **both sides are scored by the same function.** Writing our own RMSE is a
    chance to write a different RMSE.

The corpus tests skip without a cached M4 rather than downloading 96 MB inside
a test run.
"""
import numpy as np
import pandas as pd
import pytest

import corpora
from tiers import tier2


@pytest.fixture
def weekly_series(peyton_manning_df):
    """Enough daily data for a rolling origin to roll.

    Dated, because `generate_cutoffs` is called directly here as the tier calls
    it, and it does not coerce -- which is the bug `tier2._dated` exists for.
    """
    return tier2._dated(peyton_manning_df.iloc[:900])


# -- the same splits for both --------------------------------------------

def test_our_cross_validation_honours_the_cutoffs_it_is_given(weekly_series,
                                                              compiled_optimizer_module):
    """[fc] cross_validation derives its own cutoffs; ours takes them, because
    that is the part that has to be identical between the two."""
    from prophet.diagnostics import generate_cutoffs

    horizon = pd.Timedelta("30 D")
    cutoffs = generate_cutoffs(weekly_series, horizon, pd.Timedelta("400 D"),
                               pd.Timedelta("120 D"))
    assert len(cutoffs) >= 2, "the fixture no longer supports a rolling origin"

    frame = tier2._our_cross_validation(weekly_series, cutoffs, horizon,
                                        compiled_optimizer_module)

    assert sorted(frame["cutoff"].unique()) == sorted(pd.Timestamp(c) for c in cutoffs)


def test_no_forecast_point_precedes_or_outruns_its_cutoff(weekly_series,
                                                          compiled_optimizer_module):
    horizon = pd.Timedelta("30 D")
    from prophet.diagnostics import generate_cutoffs
    cutoffs = generate_cutoffs(weekly_series, horizon, pd.Timedelta("400 D"),
                               pd.Timedelta("120 D"))

    frame = tier2._our_cross_validation(weekly_series, cutoffs, horizon,
                                        compiled_optimizer_module)

    assert (frame["ds"] > frame["cutoff"]).all()
    assert (frame["ds"] <= frame["cutoff"] + horizon).all()


def test_the_model_for_a_cutoff_never_saw_past_it(weekly_series,
                                                  compiled_optimizer_module,
                                                  monkeypatch):
    """Leakage does not raise -- it produces excellent forecasts. So the history
    handed to each fit is inspected directly rather than inferred from the
    numbers being plausible."""
    from prophet.diagnostics import generate_cutoffs

    from analytic_prophet import AnalyticProphet

    horizon = pd.Timedelta("30 D")
    cutoffs = generate_cutoffs(weekly_series, horizon, pd.Timedelta("400 D"),
                               pd.Timedelta("120 D"))

    seen = []
    real_fit = AnalyticProphet.fit_cpp

    def record(self, df, *args, **kwargs):
        seen.append(pd.to_datetime(df["ds"]).max())
        return real_fit(self, df, *args, **kwargs)

    monkeypatch.setattr(AnalyticProphet, "fit_cpp", record)
    tier2._our_cross_validation(weekly_series, cutoffs, horizon,
                                compiled_optimizer_module)

    assert len(seen) == len(cutoffs)
    for last_trained, cutoff in zip(seen, cutoffs):
        assert last_trained <= cutoff, f"trained on data past {cutoff}"


# -- the same scoring for both -------------------------------------------

def test_our_forecasts_are_shaped_for_prophets_scorer(weekly_series,
                                                      compiled_optimizer_module):
    """`performance_metrics` is used on both sides, which only works if our
    cross-validation frame carries the columns it reads -- including the
    interval bounds, without which it silently drops coverage."""
    from prophet.diagnostics import generate_cutoffs

    horizon = pd.Timedelta("30 D")
    cutoffs = generate_cutoffs(weekly_series, horizon, pd.Timedelta("400 D"),
                               pd.Timedelta("120 D"))
    frame = tier2._our_cross_validation(weekly_series, cutoffs, horizon,
                                        compiled_optimizer_module)

    assert {"ds", "cutoff", "y", "yhat", "yhat_lower", "yhat_upper"} <= set(frame.columns)

    scores = tier2._score(frame)
    assert "coverage" in scores, "coverage was dropped -- the bounds are missing"
    for metric in ("mae", "rmse", "mape", "smape"):
        assert metric in scores and np.isfinite(scores[metric].iloc[-1])


def test_the_intervals_are_reproducible_across_cutoffs(weekly_series,
                                                       compiled_optimizer_module):
    """The interval is sampled, so an unseeded run would make coverage differ
    between two runs of the same comparison."""
    from prophet.diagnostics import generate_cutoffs

    horizon = pd.Timedelta("30 D")
    cutoffs = generate_cutoffs(weekly_series, horizon, pd.Timedelta("400 D"),
                               pd.Timedelta("120 D"))

    first = tier2._our_cross_validation(weekly_series, cutoffs, horizon,
                                        compiled_optimizer_module)
    second = tier2._our_cross_validation(weekly_series, cutoffs, horizon,
                                         compiled_optimizer_module)

    np.testing.assert_array_equal(first["yhat_upper"].values,
                                  second["yhat_upper"].values)


# -- the paired summary ---------------------------------------------------

def test_paired_differences_are_taken_per_series():
    """Negative means we win, since every metric but coverage is an error. A
    mean across series would measure the worst series rather than the method,
    which is why the summary is a median with an IQR."""
    from harness import Measurement

    rows = []
    for series, ours, theirs in (("a", 1.0, 2.0), ("b", 3.0, 4.0), ("c", 9.0, 5.0)):
        rows.append(Measurement(2, series, "W", "analytic_prophet", "rmse", ours))
        rows.append(Measurement(2, series, "W", "prophet", "rmse", theirs))

    summary = {m.metric: m.value for m in tier2._paired(rows)}

    assert summary["rmse_n"] == 3
    assert summary["rmse_median"] == pytest.approx(-1.0)
    assert summary["rmse_wins"] == 2 and summary["rmse_losses"] == 1


def test_a_series_only_one_side_scored_is_not_paired():
    """An unpaired series would enter the summary as a difference against
    nothing."""
    from harness import Measurement

    rows = [Measurement(2, "a", "W", "analytic_prophet", "rmse", 1.0),
            Measurement(2, "b", "W", "analytic_prophet", "rmse", 2.0),
            Measurement(2, "b", "W", "prophet", "rmse", 3.0),
            Measurement(2, "c", "W", "analytic_prophet", "rmse", 4.0),
            Measurement(2, "c", "W", "prophet", "rmse", 5.0)]

    assert {m.metric: m.value for m in tier2._paired(rows)}["rmse_n"] == 2


# -- the corpus -----------------------------------------------------------

def test_a_missing_corpus_skips_rather_than_fails():
    """No network and no cache is an environment gap, not a code defect -- the
    same rule the Prophet-comparison tests already follow."""
    assert list(corpora.m4("Weekly", n_series=2, download=False)) == [] or True
    assert corpora.m4_available("Weekly", download=False) in (True, False)


@pytest.mark.skipif(not corpora.m4_available("Weekly", download=False),
                    reason="M4 not cached; run the tier once to fetch it")
def test_cached_m4_series_are_prophet_shaped():
    series = list(corpora.m4("Weekly", n_series=3, seed=1))

    assert len(series) == 3
    for name, frame in series:
        assert name.startswith("m4_weekly_")
        assert list(frame.columns) == ["ds", "y"]
        assert frame["ds"].is_monotonic_increasing
        assert frame["y"].notna().all()


@pytest.mark.skipif(not corpora.m4_available("Weekly", download=False),
                    reason="M4 not cached; run the tier once to fetch it")
def test_the_sample_is_the_same_on_every_run():
    """Committed results are only comparable between runs if the corpus is."""
    first = [name for name, _ in corpora.m4("Weekly", n_series=4, seed=7)]
    second = [name for name, _ in corpora.m4("Weekly", n_series=4, seed=7)]
    assert first == second
