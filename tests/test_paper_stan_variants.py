"""
Issue #183: modified copies of Prophet's own Stan program, and one runner.

#168, #169 and #171 change Prophet's program rather than call it, and each is
only evidence if the change is the *only* difference. Three things make that
true rather than assumed, and each is checked here:

  * **each variant is the diff it claims to be** -- read off the files, with
    no toolchain needed, so this part runs everywhere;
  * **the unmodified program compiled here is Prophet's binary**: the same
    optimum, `lp__` and gradient. If a self-compiled original disagreed with
    the one Prophet ships, every variant would be measuring the build;
  * **each variant changes the density by exactly what its diff says**, and
    nothing else.

The compiled checks skip without CmdStan, which CI does not install: building
four Stan programs is a minute and a half that no per-push question needs.
They fail, rather than skip, on a CmdStan that is not 2.37.0.
"""
import difflib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
STAN = REPO / "paper" / "stan"
EXPERIMENTS = REPO / "paper" / "experiments"


def _runner():
    spec = importlib.util.spec_from_file_location("stan_runner", EXPERIMENTS / "stan_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _runner()


def _changes(variant):
    """(removed, added) lines of a variant against the original."""
    original = (STAN / "prophet.stan").read_text().splitlines()
    modified = (STAN / runner.VARIANTS[variant]).read_text().splitlines()
    diff = list(difflib.unified_diff(original, modified, lineterm="", n=0))
    removed = [line[1:].strip() for line in diff
               if line.startswith("-") and not line.startswith("---")]
    added = [line[1:].strip() for line in diff
             if line.startswith("+") and not line.startswith("+++")]
    return removed, added


SIGMA_DECLARATION = "real<lower=0> sigma_obs;  // Observation noise"
EPSILON_DECLARATION = ("real<lower=0> epsilon;  // Smoothing of |delta|: "
                       "sqrt(delta^2 + epsilon^2) - epsilon")
L1_PRIOR = "delta ~ double_exponential(0, tau);"
SMOOTH_PRIOR = "target += -(sum(sqrt(square(delta) + square(epsilon))) - S * epsilon) / tau;"


# ---------------------------------------------------------------------------
# The diffs. No toolchain.

def test_the_original_is_prophets_program_verbatim(prophet_comparison):
    """The baseline every diff is read against has to be the program Prophet
    actually compiled, byte for byte, not a copy that drifted."""
    import prophet
    bundled = Path(prophet.__file__).parent / "stan_model" / "prophet.stan"
    assert (STAN / "prophet.stan").read_bytes() == bundled.read_bytes()


def test_fixed_sigma_moves_one_declaration_and_nothing_else():
    """sigma_obs leaves `parameters` for `data`. The prior line stays: on data
    it is a constant, which `~` drops, so removing it would be a second change
    with no effect."""
    removed, added = _changes("fixed_sigma")
    assert removed == [SIGMA_DECLARATION]
    assert added == [SIGMA_DECLARATION]
    text = (STAN / runner.VARIANTS["fixed_sigma"]).read_text()
    data_block = text.split("data {", 1)[1].split("}", 1)[0]
    parameters_block = text.split("\nparameters {", 1)[1].split("}", 1)[0]
    assert "sigma_obs" in data_block and "sigma_obs" not in parameters_block


def test_smooth_replaces_the_l1_prior_and_adds_epsilon():
    removed, added = _changes("smooth")
    assert removed == [L1_PRIOR]
    assert sorted(added) == sorted([EPSILON_DECLARATION, SMOOTH_PRIOR])


def test_fixed_sigma_smooth_is_exactly_both():
    removed, added = _changes("fixed_sigma_smooth")
    assert sorted(removed) == sorted([SIGMA_DECLARATION, L1_PRIOR])
    assert sorted(added) == sorted([SIGMA_DECLARATION, EPSILON_DECLARATION, SMOOTH_PRIOR])


@pytest.mark.parametrize("variant, kwargs, message", [
    ("fixed_sigma", {}, "positive sigma_obs"),
    ("fixed_sigma", {"sigma_obs": 0.0}, "positive sigma_obs"),
    ("original", {"sigma_obs": 0.1}, "cannot be given one"),
    ("smooth", {}, "positive epsilon"),
    ("smooth", {"epsilon": 0.0}, "positive epsilon"),
    ("original", {"epsilon": 1e-3}, "has no epsilon"),
])
def test_the_runner_refuses_inputs_a_variant_cannot_use(variant, kwargs, message):
    """Reject rather than ignore: a sigma handed to a program that estimates it
    would otherwise be dropped without a word, and epsilon = 0 makes the
    surrogate's gradient 0/0 at Prophet's starting point delta = 0."""
    with pytest.raises(ValueError, match=message):
        runner.inputs_for(variant, {"T": 10}, {"sigma_obs": 1.0}, **kwargs)


def test_fixing_sigma_moves_it_from_inits_to_data():
    data, inits = runner.inputs_for("fixed_sigma", {"T": 10}, {"k": 0.1, "sigma_obs": 1.0},
                                    sigma_obs=0.3)
    assert data["sigma_obs"] == 0.3 and "sigma_obs" not in inits and inits["k"] == 0.1


def test_the_runner_reads_cmdstans_version_from_its_makefile(tmp_path):
    (tmp_path / "makefile").write_text("X := 1\nCMDSTAN_VERSION := 2.36.0\n")
    assert runner._makefile_version(tmp_path) == "2.36.0"


# ---------------------------------------------------------------------------
# Compiled. Skips without CmdStan; fails on the wrong one.

@pytest.fixture(scope="module")
def cmdstan(prophet_comparison):
    import cmdstanpy
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip(f"CmdStan is not installed; the paper's Stan variants need "
                    f"{runner.CMDSTAN_VERSION} (cmdstanpy.install_cmdstan(version="
                    f"'{runner.CMDSTAN_VERSION}'))")
    return runner.cmdstan_path()


def test_cmdstan_is_the_version_prophet_bundles(cmdstan):
    """Asserted, not assumed: another version changes the math library."""
    assert runner._makefile_version(cmdstan) == runner.CMDSTAN_VERSION


@pytest.fixture(scope="module", params=[60, 300])
def reference(request, cmdstan):
    """Prophet fitted on a reference series, with what it handed Stan.

    60 observations takes Prophet's Newton branch and 300 its L-BFGS one, so
    the runner's choice between them is exercised both ways.
    """
    import logging

    import pandas as pd
    from prophet import Prophet

    import _common
    from conftest import DATA_PATH

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = pd.read_csv(DATA_PATH).iloc[:request.param].reset_index(drop=True)
    model = Prophet(**_common.PROPHET_KWARGS)
    data, inits = runner.prophet_inputs(model, df)
    point = {name: np.ravel(value) for name, value in model.params.items()}
    point = {"k": point["k"][0], "m": point["m"][0], "delta": point["delta"],
             "beta": point["beta"], "sigma_obs": point["sigma_obs"][0],
             "lp__": point["lp__"][0]}
    return data, inits, point


def test_the_original_compiled_here_reproduces_prophets_optimum(reference):
    """The parity check the issue puts before anything else. Exact on the
    machine it was written on; the tolerance only allows a different compiler
    its last bits."""
    data, inits, theirs = reference
    ours, _ = runner.fit("original", data, inits)
    for name in ("k", "m", "sigma_obs", "lp__"):
        assert ours[name] == pytest.approx(theirs[name], rel=1e-10, abs=1e-12), name
    for name in ("delta", "beta"):
        np.testing.assert_allclose(ours[name], theirs[name], rtol=1e-10, atol=1e-12)


def test_the_original_compiled_here_scores_as_prophets_binary(reference):
    """Same point, two executables: the density and its gradient agree."""
    import importlib.resources

    from cmdstanpy import CmdStanModel

    data, _, point = reference
    _, here = runner.log_prob("original", data, point)
    bundled = CmdStanModel(exe_file=str(
        importlib.resources.files("prophet") / "stan_model" / "prophet_model.bin"))
    there = bundled.log_prob(
        params={"k": point["k"], "m": point["m"], "delta": point["delta"].tolist(),
                "beta": point["beta"].tolist(), "sigma_obs": point["sigma_obs"]},
        data=data, jacobian=False, sig_figs=18)
    assert list(here.columns) == list(there.columns)
    np.testing.assert_allclose(here.to_numpy(), there.to_numpy(), rtol=1e-12, atol=1e-12)


def test_fixing_sigma_changes_the_density_by_its_constant_and_nothing_else(reference):
    """With sigma as data, `~` drops the terms that only involve it: T log sigma
    from the likelihood and 2 sigma^2 from its half-normal prior. That is the
    whole difference -- the gradient over (k, m, delta, beta) is unchanged."""
    data, _, point = reference
    sigma = point["sigma_obs"]
    lp_original, original = runner.log_prob("original", data, point)
    lp_fixed, fixed = runner.log_prob("fixed_sigma", data, point, sigma_obs=sigma)
    assert lp_fixed - lp_original == pytest.approx(
        data["T"] * np.log(sigma) + 2 * sigma ** 2, rel=1e-12)
    shared = [column for column in fixed.columns if column != "lp__"]
    assert "g_sigma_obs" not in shared and len(shared) == len(original.columns) - 2
    # not bit-identical: with sigma a constant the compiler is free to order the
    # arithmetic differently, which costs the last bits and nothing more
    np.testing.assert_allclose(fixed[shared].to_numpy(), original[shared].to_numpy(),
                               rtol=1e-12, atol=1e-12)


def test_the_smoothed_prior_tends_to_the_original_from_above(reference):
    """sqrt(delta^2 + eps^2) - eps never exceeds |delta|, so the smoothed lp__ is
    at least the original's at every point, and the gap closes as eps -> 0 --
    which is what lets #168 score one problem by the other's density."""
    data, _, point = reference
    lp_original, _ = runner.log_prob("original", data, point)
    gaps = [runner.log_prob("smooth", data, point, epsilon=eps)[0] - lp_original
            for eps in (1e-1, 1e-3, 1e-6, 1e-12)]
    assert all(gap >= 0 for gap in gaps)
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < len(point["delta"]) * 1e-12 / data["tau"] * 10


@pytest.mark.parametrize("variant, kwargs", [
    ("fixed_sigma", {"sigma_obs": "theirs"}),
    ("smooth", {"epsilon": 1e-3}),
    ("fixed_sigma_smooth", {"sigma_obs": "theirs", "epsilon": 1e-3}),
])
def test_every_variant_fits_from_prophets_own_start(reference, variant, kwargs):
    data, inits, theirs = reference
    kwargs = {key: theirs["sigma_obs"] if value == "theirs" else value
              for key, value in kwargs.items()}
    params, result = runner.fit(variant, data, inits, **kwargs)
    assert params["delta"].shape == theirs["delta"].shape
    assert params["beta"].shape == theirs["beta"].shape
    assert np.isfinite(params["lp__"])
    if "sigma_obs" in kwargs:
        assert params["sigma_obs"] == kwargs["sigma_obs"]
