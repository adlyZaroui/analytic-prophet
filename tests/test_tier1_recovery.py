"""
Tier 1 of the evaluation suite (#79): parameter recovery on synthetic data.

On real data "who is right" has no answer -- there is no true parameter vector,
so agreement between two fits is all anyone can measure. Generated from the
model there is one, which is the whole reason this tier exists.

Two properties carry the tier, and both are checked here rather than assumed:

  * **the generator generates the model**, so a fit recovers what made the
    series, with the error shrinking as the series lengthens. A generator that
    quietly produced something else would make every number downstream a
    measurement of that mistake.
  * **the recovery metric is a distance, not a loss gap.** The distinction is
    easy to lose and produced a tautology on the first attempt; the test below
    pins the reasoning so it cannot come back.
"""
import numpy as np
import pytest

import metrics
import synthetic
from tiers import tier1

from analytic_prophet import AnalyticProphet


# -- the generator --------------------------------------------------------

@pytest.mark.parametrize("n_rows", [100, 300])
def test_the_generated_series_has_the_shape_the_model_expects(n_rows):
    truth = synthetic.generate(n_rows, seed=1)

    assert list(truth.frame.columns) == ["ds", "y"]
    assert len(truth.frame) == n_rows
    assert truth.theta.size == truth.layout.size
    assert np.all(np.isfinite(truth.theta))
    assert 0 < truth.active <= truth.layout.n_changepoints


def test_the_generator_places_changepoints_where_a_fit_will(compiled_optimizer_module):
    """The property that makes a recovered `delta` mean anything.

    `delta` indexes breakpoints. If the generator put them anywhere else, every
    entry would be compared against a different breakpoint and the error would
    be meaningless however the fit went.
    """
    truth = synthetic.generate(300, seed=1)

    model = AnalyticProphet()
    model.fit_cpp(truth.frame, lib_path=compiled_optimizer_module)

    np.testing.assert_array_equal(model.changepoints_t, truth.changepoints_t)


def test_the_truth_is_recorded_in_the_space_the_fit_works_in():
    """Both implementations fit y/max|y|, and every parameter scales with that
    divisor. Comparing a raw generating vector against a fitted one would report
    an error that is entirely the normalization."""
    truth = synthetic.generate(300, noise=0.1, seed=1)
    y_scale = float(np.max(np.abs(truth.frame["y"])))

    assert truth.sigma_obs == pytest.approx(0.1 / y_scale)
    assert truth.theta[truth.layout.sigma_obs_idx] == pytest.approx(truth.sigma_obs)


def test_recovery_improves_with_the_length_of_the_series(compiled_optimizer_module):
    """The generator generates the model, checked the only way that means
    anything: fit it back and watch the error fall.

    Raw distance, deliberately. The identified error is in **nats**, and the
    curvature it weights by grows with the sample -- more data means the same
    parameter error costs more -- so it is comparable between implementations on
    one series and not across series of different lengths. That is why the tier
    pairs per series and never pools.
    """
    errors = {}
    for n_rows in (100, 300, 1000):
        truth = synthetic.generate(n_rows, noise=0.05, seed=1)
        model = AnalyticProphet()
        model.fit_cpp(truth.frame, lib_path=compiled_optimizer_module)
        errors[n_rows] = float(np.linalg.norm(model.get_parameters() - truth.theta))

    assert errors[1000] < errors[300] < errors[100]


def test_the_identified_error_grows_with_the_sample(compiled_optimizer_module):
    """The caveat above, asserted rather than left in a docstring: a reader who
    pools this metric across series lengths is measuring how much data each one
    had."""
    sizes = {}
    for n_rows in (100, 1000):
        truth = synthetic.generate(n_rows, noise=0.05, seed=1)
        model = AnalyticProphet()
        model.fit_cpp(truth.frame, lib_path=compiled_optimizer_module)
        sizes[n_rows] = metrics.quadratic_distance(
            model, truth.theta, model.get_parameters())

    assert sizes[1000] > sizes[100], (
        "if this ever reverses, the metric's scale has changed and the "
        "no-pooling rule in tier1.py should be revisited")


# -- the metric, and the tautology it replaced ---------------------------

def test_the_recovery_metric_is_a_distance_not_a_loss_gap(compiled_optimizer_module):
    """The reasoning behind `quadratic_distance`, pinned so it cannot come back.

    `quadratic_change` expanded around a *fitted* point carries the gradient
    term, so it reports the loss gap -- and whichever implementation reached the
    lower objective is then further from the truth **by construction**, whatever
    it recovered. The first version of this tier used it and produced a clean
    0/18 that turned out to be a restatement of the posterior comparison.

    A distance with the ruler fixed at the truth has no such bias: it is
    symmetric in the two candidates and depends on neither one's objective.
    """
    truth = synthetic.generate(300, noise=0.1, seed=1)
    model = AnalyticProphet()
    model.fit_cpp(truth.frame, lib_path=compiled_optimizer_module)
    design = model._design_matrices()
    fitted = model.get_parameters().copy()

    # a deliberately worse point, in an identified direction
    worse = fitted.copy()
    worse[model.layout.beta] += 0.05

    objective = lambda t: model._minus_log_posterior(t, design=design)
    assert objective(worse) > objective(fitted), "the worse point should score worse"

    # the loss-gap reading: the better fit is "further from the truth"
    gap_fitted = metrics.quadratic_change(model, fitted, truth.theta, design=design)
    gap_worse = metrics.quadratic_change(model, worse, truth.theta, design=design)
    assert gap_fitted > gap_worse, (
        "this is the tautology: the better-scoring point reports the larger gap "
        "regardless of what it recovered")

    # the distance reading: one ruler, and it prefers whichever is actually nearer
    near = metrics.quadratic_distance(model, truth.theta, fitted, design=design)
    far = metrics.quadratic_distance(model, truth.theta, worse, design=design)
    assert near < far


def test_the_distance_is_zero_at_the_reference(compiled_optimizer_module):
    truth = synthetic.generate(300, seed=1)
    model = AnalyticProphet()
    model.fit_cpp(truth.frame, lib_path=compiled_optimizer_module)
    assert metrics.quadratic_distance(model, truth.theta, truth.theta) == \
        pytest.approx(0.0, abs=1e-9)


def test_the_distance_uses_one_ruler_for_both_candidates(compiled_optimizer_module):
    """Curvature at the reference, so swapping which candidate is measured first
    cannot change either number."""
    truth = synthetic.generate(300, seed=1)
    model = AnalyticProphet()
    model.fit_cpp(truth.frame, lib_path=compiled_optimizer_module)
    design = model._design_matrices()
    a = model.get_parameters().copy()
    b = a + 0.01

    first = [metrics.quadratic_distance(model, truth.theta, x, design=design) for x in (a, b)]
    second = [metrics.quadratic_distance(model, truth.theta, x, design=design) for x in (b, a)]

    assert first == [second[1], second[0]]


# -- the tier itself ------------------------------------------------------

@pytest.fixture(scope="module")
def one_case(compiled_optimizer_module, prophet_comparison):
    # prophet_comparison is for the skip, not its value: tier1 imports
    # prophet inside collect() (#103).
    return tier1.collect(sizes=(300,), noises=(0.1,), seeds=(1,),
                         lib_path=compiled_optimizer_module)


def test_both_implementations_are_measured_against_the_truth(one_case):
    reported = {(m.implementation, m.metric) for m in one_case}
    for implementation in ("analytic_prophet", "prophet"):
        for metric in ("theta_error_l2", "identified_error", "excess_over_truth",
                       "delta_error_l2", "beta_error_l2", "sum_abs_delta",
                       "exact_zeros"):
            assert (implementation, metric) in reported, (implementation, metric)


def test_the_two_fits_are_comparable_before_they_are_compared(one_case):
    """The layouts and the normalizing scale have to match, or the per-block
    errors are partly a comparison of divisors and column orders."""
    by_metric = {m.metric: m.value for m in one_case if m.implementation == "both"}
    assert by_metric["y_scale_abs_diff"] == 0.0
    assert by_metric["layout_mismatch"] == 0.0


def test_a_map_fit_scores_better_than_the_truth_on_its_own_sample(one_case):
    """Not a defect -- it is what maximizing a posterior on a finite sample
    does, and the reason `excess_over_truth` is reported as a diagnostic rather
    than as a score."""
    for m in one_case:
        if m.metric == "excess_over_truth":
            assert m.value < 0, m.implementation


# -- sparsity, reported without a threshold (#95) -------------------------

def test_the_sparsity_readouts_need_no_threshold(one_case):
    """`active changepoints` used to stand here, as `|delta| > 1e-6`. It is
    almost entirely a function of that cutoff on Prophet's side, so it is
    replaced by what the prior actually charges for and by the count of rates
    that are exactly zero -- neither of which has a cutoff to argue about."""
    reported = {(m.implementation, m.metric) for m in one_case}
    for implementation in ("analytic_prophet", "prophet"):
        assert (implementation, "sum_abs_delta") in reported
        assert (implementation, "exact_zeros") in reported
    assert not any(m.metric == "active_changepoints" for m in one_case)


def test_the_generating_sparsity_is_reported_for_reference(one_case):
    """Reference, not target. MAP with an L1 prior over nested, collinear step
    functions is not a support-recovery procedure, so neither fit is expected
    to match the vector that made the series (#88)."""
    by_metric = {m.metric: m.value for m in one_case if m.implementation == "both"}
    assert "sum_abs_delta_true" in by_metric
    assert "exact_zeros_true" in by_metric
