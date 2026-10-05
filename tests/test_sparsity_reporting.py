"""
Issue #95: the trend's sparsity, reported as something that means what it says.

Two claims used to sit in the documentation and in Tiers 1 and 2:

  * that this implementation scores better on the posterior **because it pays
    less in the Laplace prior**, finding a sparser trend;
  * that it fits fewer *active changepoints*, counted as `|δ| > 1e-6`.

The first is backwards and the second is a threshold artifact. The tests here
pin the facts that replace them, so the explanation cannot outlive them: if
Prophet's optimizer ever does reach exact zeros, or the L1 comparison ever does
reverse, these fail and the prose has to be rewritten rather than quietly
becoming wrong again.
"""
import numpy as np
import pytest

from analytic_prophet import AnalyticProphet

TAU = 0.05      # changepoint_prior_scale, the Laplace scale the penalty divides by


@pytest.fixture(scope="module")
def both(prophet_comparison, compiled_optimizer_module):
    """(our rates, Prophet's rates) from the same series and settings."""
    Prophet, common, _bridge = prophet_comparison
    df = common.load_data(1200)

    ours = AnalyticProphet(**common.PROPHET_KWARGS)
    ours.fit(df, lib_path=compiled_optimizer_module)
    theirs = Prophet(**common.PROPHET_KWARGS)
    theirs.fit(df)
    return (np.abs(ours.params["delta"][0]),
            np.abs(np.ravel(theirs.params["delta"])))


def test_our_rates_reach_exactly_zero_and_prophets_do_not(both):
    """The real difference, and the one that needs no threshold.

    It is the non-smooth-objective story again: the Laplace prior's kink sits at
    δ = 0, the split reformulation puts it on a bound the optimizer can land on
    exactly, and Prophet's stops nearby instead.
    """
    ours, theirs = both

    assert np.sum(ours == 0.0) > 0, "the split reformulation should reach the kink"
    assert np.sum(theirs == 0.0) == 0, (
        "Prophet now reaches exact zeros, which is the premise of the whole "
        "sparsity discussion in #95 -- the prose needs revisiting")


def test_counting_active_changepoints_measures_the_threshold(both):
    """Why the count was replaced. Ours is flat across five orders of
    magnitude; Prophet's roughly halves over the same range."""
    ours, theirs = both
    thresholds = (1e-8, 1e-6, 1e-4, 1e-3)

    our_counts = [int((ours > t).sum()) for t in thresholds]
    their_counts = [int((theirs > t).sum()) for t in thresholds]

    assert len(set(our_counts)) == 1, our_counts
    assert their_counts[0] > their_counts[-1] * 1.4, (
        f"Prophet's count no longer moves with the threshold: {their_counts}")


def test_we_pay_more_prior_not_less(both):
    """The documented mechanism, backwards. We reach the better posterior with
    a *higher* L1 cost, so the likelihood gain more than covers it -- no appeal
    to sparsity is needed, and the one that was made is not supported."""
    ours, theirs = both

    assert ours.sum() > theirs.sum(), (
        f"Σ|δ| ours {ours.sum():.5f} vs prophet {theirs.sum():.5f} -- if this "
        f"reverses, the correction in docs/non-smooth-objective.md is wrong")
    assert ours.sum() / TAU > theirs.sum() / TAU


def test_we_also_fit_the_training_data_better(prophet_comparison,
                                              compiled_optimizer_module):
    """The other half of the old explanation. It said Prophet fits the training
    data marginally better; on the default configuration it does not."""
    Prophet, common, _bridge = prophet_comparison
    df = common.load_data(1200)

    def sse(model, fitted):
        residual = df["y"].to_numpy() - fitted["yhat"].to_numpy()[:len(df)]
        return float((residual ** 2).sum())

    ours = AnalyticProphet(**common.PROPHET_KWARGS)
    ours.fit(df, lib_path=compiled_optimizer_module)
    theirs = Prophet(**common.PROPHET_KWARGS)
    theirs.fit(df)

    assert sse(ours, ours.predict(df.copy())) < sse(theirs, theirs.predict(df.copy()))
