"""
Issue #171: the solution path in tau.

`paper/experiments/tau_path.py` sweeps lambda = 1/tau for Prophet and for the
exact path. The paper reads the result as a difference in kind -- the exact
path is continuous and piecewise linear, Prophet's jumps -- and that reading
rests on a few checkable facts, each tested here against the committed data:

  * **the exact path is the optimum** -- certified at every point;
  * **its steps behave as a piecewise-linear path's must** -- constant on a
    uniform grid, and halving each time the interval is halved;
  * **Prophet's do not shrink under refinement** -- a jump, not steep motion;
  * **prophet#1280's change** moves Prophet's solution and not the optimum;
  * **the code reproduces a committed point** through Prophet's own binary.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import recorded_here

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"
RESULTS = PAPER / "results"
S = 25


def _module():
    sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("tau_path", EXPERIMENTS / "tau_path.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tau_path = _module()


@pytest.fixture(scope="module")
def rows():
    return tau_path.read(tau_path.PATH)


@pytest.fixture(scope="module")
def refine():
    with open(tau_path.REFINE, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def pairs():
    with open(tau_path.PAIRS, newline="") as handle:
        return list(csv.DictReader(handle))


def _path(rows, grid, setting, name):
    cells = sorted((r for r in rows if r["grid"] == grid and r["setting"] == setting
                    and r["path"] == name), key=lambda r: r["lam"])
    return cells, np.array([[r[f"delta_{j + 1}"] for j in range(S)] for r in cells])


def _steps(rows, grid, setting, name):
    return np.max(np.abs(np.diff(_path(rows, grid, setting, name)[1], axis=0)), axis=1)


def test_every_path_covers_its_grid(rows):
    for setting in ("fixed", "free"):
        for name in ("prophet", "ours_exact"):
            assert len(_path(rows, "global", setting, name)[0]) == len(tau_path.GLOBAL)
            assert len(_path(rows, "zoom", setting, name)[0]) == len(tau_path.ZOOM)
    assert len(_path(rows, "global", "free", "ours_default")[0]) == len(tau_path.GLOBAL)


def test_the_exact_path_is_certified_at_every_point(rows):
    residuals = [r["residual"] for r in rows if r["path"] == "ours_exact"]
    assert max(residuals) < 1e-3


def test_prophet_never_reaches_the_exact_path(rows):
    for setting in ("fixed", "free"):
        exact, _ = _path(rows, "global", setting, "ours_exact")
        prophet, _ = _path(rows, "global", setting, "prophet")
        assert all(e["lp"] > p["lp"] for e, p in zip(exact, prophet))


def test_the_exact_paths_steps_are_constant_on_the_zoom_and_prophets_are_not(rows):
    """'move by a constant ... per step, as a piecewise-linear path must'."""
    for setting in ("fixed", "free"):
        exact = _steps(rows, "zoom", setting, "ours_exact")
        assert exact.max() / np.median(exact) < 1.2
        prophet = _steps(rows, "zoom", setting, "prophet")
        assert prophet.max() / np.median(prophet) > 3
        assert prophet.max() > 10 * exact.max()


def test_bisection_halves_the_exact_step_and_leaves_prophets(refine):
    """The distinction between steep motion and a jump."""
    for setting in ("fixed", "free"):
        exact = sorted((r for r in refine if r["setting"] == setting and r["path"] == "ours_exact"),
                       key=lambda r: int(r["level"]))
        ratios = [float(b["step"]) / float(a["step"]) for a, b in zip(exact, exact[1:])]
        assert all(abs(ratio - 0.5) < 0.05 for ratio in ratios), ratios
        prophet = sorted((r for r in refine if r["setting"] == setting and r["path"] == "prophet"),
                         key=lambda r: int(r["level"]))
        assert float(prophet[-1]["step"]) > 100 * float(exact[-1]["step"])
        assert float(prophet[-1]["step"]) > 0.25 * float(prophet[0]["step"])


def test_prophet_1280s_change_moves_prophet_and_not_the_optimum(pairs):
    for row in pairs:
        if row["path"] == "ours_exact":
            assert float(row["delta_change"]) < 1e-6
        elif float(row["tau"]) == 0.01:
            assert float(row["delta_change"]) > 1e-3


def test_the_paper_does_not_claim_the_threads_two_percent():
    text = (PAPER / "sections" / "consequences.tex").read_text()
    assert "It does not reproduce the thread's 2" in text
    assert r"\cref{sec:evidence:path}" in text


def test_prophets_bundled_binary_reproduces_a_committed_point(rows, prophet_comparison):
    """lambda = 20, sigma free: Prophet().fit with changepoint_prior_scale = 0.05,
    no CmdStan install needed.

    By conftest.recorded_here's rule. **Anywhere**: the point is below the
    exact path at the same lambda, as the paper says of every one of Prophet's.
    **Here**: the committed lp__ exactly."""
    import logging

    import _common
    import _prophet_bridge as bridge
    import kkt
    import corpora

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    committed = next(r for r in rows if r["grid"] == "zoom" and r["setting"] == "free"
                     and r["path"] == "prophet" and abs(r["lam"] - 20.0) < 1e-9)
    Prophet = prophet_comparison[0]
    stan_model, stan_data, params = bridge.capture_stan_model(
        Prophet(**{**_common.PROPHET_KWARGS, "changepoint_prior_scale": 1 / 20.0}),
        corpora.peyton_manning(), sig_figs=18)
    # at full precision, as the committed point was scored: the bridge's
    # stan_log_prob returns lp__ as CmdStan prints it, to eight digits
    lp, _ = kkt.stan_gradient(stan_model, stan_data, params["k"][0], params["m"][0],
                              params["delta"], params["sigma_obs"][0], params["beta"])
    exact = next(r for r in rows if r["grid"] == "zoom" and r["setting"] == "free"
                 and r["path"] == "ours_exact" and abs(r["lam"] - 20.0) < 1e-9)
    assert exact["lp"] - lp > 1e-3
    if recorded_here(tau_path.PATH.with_suffix(".meta.json")):
        assert lp == pytest.approx(committed["lp"], abs=1e-6)


def test_the_figure_redraws_identically(rows, tmp_path):
    pytest.importorskip("matplotlib")
    assert tau_path.FIGURE.exists() and tau_path.FIGURE.stat().st_size > 10_000
    assert tau_path.draw(rows, tmp_path / "a.pdf").read_bytes() == \
        tau_path.draw(rows, tmp_path / "b.pdf").read_bytes()
