"""
The evaluation suite's scaffolding (#79), which four tiers will land on.

Nothing here measures the model. What it checks is the contract the tiers share,
because four branches off main would otherwise each invent their own: the
results schema, the run metadata, the gate that stops a run, and the two metrics
that appear in more than one tier.

`metrics.quadratic_change` is the piece worth the most attention. It is what
makes a parameter-space distance mean anything: raw distance between two fits is
dominated by the directions the data does not identify -- `k` against `delta`,
which trade off almost freely -- so two fits can be far apart and be the same
function. Weighted by the curvature, the distance *is* the loss gap.
"""
import numpy as np
import pytest

# evaluation/ reaches sys.path through pyproject.toml's `pythonpath`, the same
# way benchmark/ does -- neither is a package, both are directories of scripts.
import corpora
import harness
import metrics
import run as runner

from analytic_prophet import AnalyticProphet


# -- the results contract -------------------------------------------------

def test_measurements_round_trip_through_the_results_file(tmp_path):
    written = [
        harness.Measurement(2, "m4_daily_17", "plain", "prophet", "rmse", 1.5, "y"),
        harness.Measurement(2, "m4_daily_17", "plain", "fit_cpp", "rmse", 1.25, "y", 7),
    ]
    csv_path, json_path = harness.write("tier2_x", written, results_dir=tmp_path)

    rows = harness.read("tier2_x", results_dir=tmp_path)

    assert csv_path.exists() and json_path.exists()
    assert [r["metric"] for r in rows] == ["rmse", "rmse"]
    # sorted, so fit_cpp precedes prophet regardless of the order written
    assert rows[0]["implementation"] == "fit_cpp"
    assert rows[0]["value"] == 1.25 and rows[0]["horizon"] == 7
    assert rows[1]["value"] == 1.5 and rows[1]["horizon"] is None


def test_results_are_sorted_so_a_rerun_diffs_cleanly(tmp_path):
    """Committed results are only useful if an unchanged rerun produces an
    unchanged file -- otherwise every run is a diff nobody can read."""
    rows = [harness.Measurement(1, s, c, i, "lp", 1.0)
            for s in ("b", "a") for c in ("z", "y") for i in ("q", "p")]

    first, _ = harness.write("tier1_x", rows, results_dir=tmp_path)
    original = first.read_text()
    harness.write("tier1_x", list(reversed(rows)), results_dir=tmp_path)

    assert first.read_text() == original


def test_metadata_records_what_makes_a_number_reproducible():
    """The differences between two runs of the same code are usually in here."""
    metadata = harness.run_metadata({"tier": 3})

    assert metadata["seed"] == harness.SEED
    assert metadata["tier"] == 3
    assert metadata["python"] and metadata["platform"]
    assert set(metadata["versions"]) >= {"numpy", "pandas", "scipy", "prophet"}
    assert metadata["versions"]["numpy"]


def test_the_benchmark_helpers_are_reached_through_one_place():
    """Tiers run as scripts, so they miss pyproject's pythonpath. Re-exporting
    here is what keeps the day those helpers move to one import to change."""
    for name in ("build_extension", "prophet_available", "peak_rss_bytes",
                 "capture_stan_model", "stan_log_prob", "validate_bridge"):
        assert callable(getattr(harness, name)), name


# -- the runner -----------------------------------------------------------

def test_an_empty_registry_is_not_an_error(capsys):
    """The normal state until the first tier lands."""
    assert runner.main(["--list"]) == 0


def test_a_failing_gate_stops_the_run(tmp_path, monkeypatch):
    """Tier 0 is a gate, not a measurement: if the two implementations are not
    fitting the same model, the other tiers measure something nobody asked
    about. The run has to stop rather than produce numbers."""
    ran = []
    monkeypatch.setattr(runner, "TIERS", {
        0: ("gate", lambda: [harness.Measurement(0, "s", "c", "both",
                                                 "gate_failed", 1.0)], True),
        3: ("cost", lambda: ran.append(3) or [], False),
    })
    monkeypatch.setattr(runner, "_load_tiers", lambda: None)

    assert runner.main(["--results", str(tmp_path)]) == 1
    assert ran == [], "tier 3 ran after the gate failed"


def test_a_passing_gate_lets_the_run_continue(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(runner, "TIERS", {
        0: ("gate", lambda: [harness.Measurement(0, "s", "c", "both",
                                                 "gate_failed", 0.0)], True),
        3: ("cost", lambda: ran.append(3) or [], False),
    })
    monkeypatch.setattr(runner, "_load_tiers", lambda: None)

    assert runner.main(["--results", str(tmp_path)]) == 0
    assert ran == [3]


# -- corpora --------------------------------------------------------------

def test_the_local_corpus_yields_prophet_shaped_frames():
    for name, frame in corpora.local():
        assert isinstance(name, str) and name
        assert list(frame.columns)[:2] == ["ds", "y"]
        assert len(frame) > 0


# -- the metrics that more than one tier needs ----------------------------

@pytest.fixture(scope="module")
def fitted(request):
    lib = request.getfixturevalue("compiled_optimizer_module")
    model = AnalyticProphet()
    model.fit_cpp(corpora.peyton_manning(300), lib_path=lib)
    return model


@pytest.mark.parametrize("scale", [1e-5, 1e-3, 1e-1])
def test_the_quadratic_change_is_the_loss_gap(fitted, scale):
    """The property the metric exists for, checked against the real thing.

    It turns out to be near-exact rather than merely second-order, and for a
    reason worth knowing: with a Gaussian likelihood, Gaussian priors and a
    model linear in its parameters, the objective *is* quadratic in these
    coordinates. The only departures are `sigma_obs`, held fixed below, and the
    Laplace term, which `quadratic_change` handles exactly rather than
    expanding. Agreement is to ~1e-10 relative even where the gap is thousands
    of nats.
    """
    theta = fitted.get_parameters().copy()
    design = fitted._design_matrices()
    step = np.random.default_rng(0).normal(0, scale, theta.size)
    step[fitted.layout.sigma_obs_idx] = 0.0
    other = theta + step

    predicted = metrics.quadratic_change(fitted, theta, other, design=design)
    actual = (fitted._minus_log_posterior(other, design=design)
              - fitted._minus_log_posterior(theta, design=design))

    assert predicted == pytest.approx(actual, rel=1e-6)
    assert actual > 0, "a perturbation away from the optimum should cost"


def test_the_quadratic_change_is_zero_at_the_point_itself(fitted):
    theta = fitted.get_parameters()
    assert metrics.quadratic_change(fitted, theta, theta) == pytest.approx(0.0, abs=1e-9)


def test_the_curvature_spectrum_finds_the_flat_directions(peyton_manning_df,
                                                          compiled_optimizer_module):
    """The project's central qualitative claim, as a number.

    Under linear growth `k` and `delta` trade off almost freely but the
    likelihood does constrain them, so the curvature is small and positive.
    Under flat growth the likelihood never sees them at all and the Laplace
    prior holding `delta` contributes no curvature, so the trend block is
    **exactly singular** -- which is why flat growth ties with Prophet exactly
    and linear growth does not.
    """
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    blocks = {}
    for growth in ("linear", "flat"):
        model = AnalyticProphet(growth=growth)
        model.fit_cpp(df, lib_path=compiled_optimizer_module)
        hessian = metrics.smooth_hessian(model, model.get_parameters())
        layout = model.layout
        trend = np.r_[[layout.k_idx, layout.m_idx],
                      np.arange(layout.delta.start, layout.delta.stop)]
        blocks[growth] = np.linalg.eigvalsh(hessian[np.ix_(trend, trend)])

    assert blocks["linear"].min() > 1e-3, "linear growth should be identified"
    assert abs(blocks["flat"].min()) < 1e-9, "flat growth's trend block is singular"
    assert blocks["linear"].min() > abs(blocks["flat"].min()) * 1e6


def test_the_spectrum_covers_every_parameter(fitted):
    spectrum = metrics.curvature_spectrum(fitted, fitted.get_parameters())
    assert spectrum.size == fitted.layout.size
    assert np.all(np.diff(spectrum) <= 0), "returned largest first"


def test_paired_summary_reports_median_iqr_and_a_signed_rank_test():
    """A mean is the wrong summary for forecast errors -- heavy-tailed, and one
    series dominates it."""
    summary = metrics.paired_summary([-1.0, -2.0, -0.5, -3.0, -1.5, 0.2, -0.8])

    assert summary["n"] == 7
    assert summary["median"] == pytest.approx(-1.0)
    assert summary["wins"] == 6 and summary["losses"] == 1
    assert 0.0 < summary["p_value"] < 0.05


def test_paired_summary_withholds_a_p_value_it_cannot_support():
    """Six pairs is the fewest a signed-rank test can distinguish from chance;
    below that it reports nan rather than a number that reads like evidence."""
    assert np.isnan(metrics.paired_summary([-1.0, -2.0, -0.5])["p_value"])
    assert np.isnan(metrics.paired_summary([])["p_value"])
    assert metrics.paired_summary([])["n"] == 0
