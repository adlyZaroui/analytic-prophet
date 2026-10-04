"""
Tier 0 of the evaluation suite (#79): the correctness gate.

Its job is to refuse to let the other tiers assume what they are built on. They
all measure the difference between two *optimizers*; if the two implementations
are optimizing different specifications, every number they produce is about
something nobody asked.

So the tests here are about the gate's judgement, not about the model: that it
passes on agreement, and -- the one that matters -- that it *trips* when the two
sides really do differ. A gate that cannot fail is a comment.
"""
import numpy as np
import pytest

import harness
from tiers import tier0


@pytest.fixture(scope="module")
def gate_at_300(compiled_optimizer_module, prophet_comparison):
    """One size, because these tests are about the gate rather than the model
    and a full sweep is the benchmark's job."""
    # prophet_comparison is for the skip, not its value: tier0 imports
    # prophet inside collect() (#103).
    return tier0.collect(sizes=(300,), lib_path=compiled_optimizer_module)


def test_the_gate_passes_on_the_real_model(gate_at_300):
    failed = [m for m in gate_at_300 if m.metric == "gate_failed"]
    assert len(failed) == 1
    assert failed[0].value == 0.0, (
        "the two implementations are no longer fitting the same model -- "
        "every other tier is measuring something else until this is resolved")


def test_it_checks_the_model_before_it_checks_the_optimum(gate_at_300):
    """Four checks, and the posterior is the last of them. A difference in the
    design matrix is a modelling difference; reporting only the posterior would
    let one hide inside the other."""
    metrics = {m.metric for m in gate_at_300}
    assert {"design_matrix_max_abs_diff", "prior_scales_max_abs_diff",
            "changepoints_max_abs_diff", "lp__", "lp___difference"} <= metrics


def test_the_design_matrices_agree_to_arithmetic(gate_at_300):
    """The Fourier basis is evaluated in a different order (see the README),
    which costs about 1e-12. Anything larger is a real difference."""
    by_metric = {m.metric: m.value for m in gate_at_300}
    assert by_metric["design_matrix_max_abs_diff"] < tier0.DESIGN_TOLERANCE
    assert by_metric["prior_scales_max_abs_diff"] == 0.0
    assert by_metric["changepoints_max_abs_diff"] == 0.0


def test_our_posterior_is_the_better_one(gate_at_300):
    by_metric = {}
    for m in gate_at_300:
        by_metric[(m.metric, m.implementation)] = m.value
    assert by_metric[("lp__", "analytic_prophet")] >= by_metric[("lp__", "prophet")]
    assert by_metric[("lp___difference", "both")] > 0


# -- the gate has to be able to fail --------------------------------------

def test_a_different_model_trips_the_gate(monkeypatch, compiled_optimizer_module,
                                          capsys, prophet_comparison):
    """The test that gives the others their meaning.

    A seasonality Prophet does not have makes the two design matrices genuinely
    different -- the same divergence a modelling bug would produce -- and the
    gate has to notice. Without this, every assertion above is consistent with a
    gate that returns "fine" unconditionally.
    """
    from analytic_prophet import AnalyticProphet

    original = AnalyticProphet.set_auto_seasonalities

    def extra_component(self):
        original(self)
        self.add_seasonality("fortnightly", 14.0, 3)

    monkeypatch.setattr(AnalyticProphet, "set_auto_seasonalities", extra_component)
    measurements = tier0.collect(sizes=(300,), lib_path=compiled_optimizer_module)

    failed = [m for m in measurements if m.metric == "gate_failed"]
    assert failed[0].value == 1.0, "the gate passed two different models"
    assert "gate failed" in capsys.readouterr().out


def test_a_worse_posterior_trips_the_gate(monkeypatch, compiled_optimizer_module,
                                          prophet_comparison):
    """The other half: identical models, ours stopping short. This is the claim
    the project makes, and the gate is what stops the rest of the suite
    assuming it."""
    real = harness.stan_log_prob

    def understate(*args, **kwargs):
        return real(*args, **kwargs) - 1000.0

    monkeypatch.setattr(tier0.harness, "stan_log_prob", understate)
    measurements = tier0.collect(sizes=(300,), lib_path=compiled_optimizer_module)

    assert [m for m in measurements if m.metric == "gate_failed"][0].value == 1.0


def test_the_runner_stops_when_the_gate_trips(tmp_path, monkeypatch):
    """End to end through run.py, since the gate is only worth having if the
    runner acts on it."""
    import run as runner

    after = []
    monkeypatch.setattr(harness, "TIERS", {
        0: ("agreement", lambda: [harness.Measurement(
            0, "s", "default", "both", "gate_failed", 1.0)], True),
        1: ("recovery", lambda: after.append(1) or [], False),
    })
    monkeypatch.setattr(runner, "TIERS", harness.TIERS)
    monkeypatch.setattr(runner, "_load_tiers", lambda: None)

    assert runner.main(["--results", str(tmp_path)]) == 1
    assert after == []
