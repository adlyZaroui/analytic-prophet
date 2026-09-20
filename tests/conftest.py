"""
Shared fixtures for the analytic-prophet test suite.

These currently exercise the pure-Python reference implementation in
CustomProphet (legacy/customProphet.py). compiled_optimizer_module below
builds the C++ core (legacy/optimize.cpp) as a pybind11 extension on the
fly, so the parity and convergence tests can import and call it directly
without a built extension checked into the repo.
"""
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "legacy"))
from customProphet import (CustomProphet, N_CHANGE_POINTS, n_yearly,  # noqa: E402
                           SIGMA_OBS_IDX, SIGMA_OBS_PRIOR_SCALE, CPP_MODULE_NAME,
                           load_cpp_module)

DATA_PATH = Path(__file__).parent / "data" / "peyton_manning.csv"
CPP_SOURCE = Path(__file__).parent.parent / "legacy" / "optimize.cpp"
PARAM_SIZE = 2 + N_CHANGE_POINTS + 1 + 2 * n_yearly  # k, m, delta, sigma_obs, beta -> 48


@pytest.fixture
def peyton_manning_df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def prepared_model(peyton_manning_df):
    """A CustomProphet with data loaded and preprocessed but not yet fit --
    gives direct access to t_scaled / change_points / normalized_y without
    paying for a full optimize() run in every test."""
    model = CustomProphet()
    model.y = peyton_manning_df["y"].values
    model.ds = pd.to_datetime(peyton_manning_df["ds"])
    model.t_scaled = np.array(
        (model.ds - model.ds.min()) / (model.ds.max() - model.ds.min())
    )
    model.T = peyton_manning_df.shape[0]
    model.scale_period = (model.ds.max() - model.ds.min()).days
    model._normalize_y()
    model._generate_change_points()
    return model


@pytest.fixture
def param_size():
    return PARAM_SIZE


def _find_eigen_include():
    candidates = [os.environ.get("EIGEN_INCLUDE_DIR")]
    try:
        prefix = subprocess.run(
            ["brew", "--prefix", "eigen"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if prefix:
            candidates.append(str(Path(prefix, "include", "eigen3")))
    except (OSError, subprocess.SubprocessError):
        pass
    candidates += [
        "/opt/homebrew/opt/eigen/include/eigen3",
        "/usr/local/opt/eigen/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "Eigen" / "Dense").exists():
            return candidate
    return None


def _build_cpp_extension(tmp_path_factory):
    """Builds legacy/optimize.cpp into an importable pybind11 extension, using
    the compile command documented in that file's trailing comment. Skips
    (rather than fails) the tests that depend on it when the C++ toolchain,
    Eigen, pybind11 or liblbfgs aren't available -- that's an environment gap,
    not a code defect."""
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler found to build the C++ core")

    eigen_include = _find_eigen_include()
    if eigen_include is None:
        pytest.skip("Eigen headers not found (set EIGEN_INCLUDE_DIR) -- can't build the C++ core")

    try:
        import pybind11
    except ImportError:
        pytest.skip("pybind11 is not installed -- can't build the C++ core")

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_path = tmp_path_factory.mktemp("cpp_core") / f"{CPP_MODULE_NAME}{suffix}"

    cmd = [
        compiler, "-std=c++17", "-shared", "-fPIC", "-O2",
        "-o", str(out_path), str(CPP_SOURCE),
        f"-I{eigen_include}",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",
        "-L/usr/local/lib", "-L/opt/homebrew/lib", "-llbfgs",
    ]
    if sys.platform == "darwin":
        # Extension modules resolve CPython's symbols from the host interpreter
        # at load time rather than linking libpython.
        cmd += ["-undefined", "dynamic_lookup"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        pytest.skip(f"could not build the C++ core (liblbfgs missing?): {result.stderr[-500:]}")

    return str(out_path)


@pytest.fixture(scope="session")
def compiled_optimizer_module(tmp_path_factory):
    """Path to the freshly built pybind11 extension, as fit_cpp(lib_path=...)
    wants it."""
    return _build_cpp_extension(tmp_path_factory)


@pytest.fixture(scope="session")
def cpp_module(compiled_optimizer_module):
    """The built extension, imported."""
    return load_cpp_module(compiled_optimizer_module)


@pytest.fixture(scope="session")
def cpp_loss_offset():
    """The C++ objective omits the two sigma_obs terms the Python one carries
    (T*log(sigma_obs) and the prior), because the C++ core never estimates
    sigma_obs. With sigma_obs pinned they are an additive constant, so adding
    this to a C++ loss makes it directly comparable with a Python one."""
    def offset(model, sigma_obs):
        return model.T * np.log(sigma_obs) + sigma_obs**2 / (2 * SIGMA_OBS_PRIOR_SCALE**2)

    return offset


@pytest.fixture(scope="session")
def cpp_mlp_and_gradient(cpp_module):
    """Calls the C++ objective+gradient the optimizer itself drives.

    Takes a model and the 47-length (k, m, delta, beta) vector the C++ side
    uses -- it has no sigma_obs slot, since the C++ core never estimates it --
    and returns (minus_log_posterior, gradient)."""
    def call(model, cpp_params, sigma_obs, include_l1_prior=True):
        return cpp_module.minus_log_posterior_and_gradient(
            params=cpp_params,
            t_scaled=model.t_scaled,
            change_points=model.change_points,
            scale_period=model.scale_period,
            normalized_y=model.normalized_y,
            sigma_obs=sigma_obs,
            sigma_k=model.sigma_k,
            sigma_m=model.sigma_m,
            sigma=model.sigma,
            tau=model.tau,
            include_l1_prior=include_l1_prior,
        )

    return call


@pytest.fixture
def random_params(param_size):
    """A point in parameter space away from delta=0, so the plain
    numerical-gradient check lands in a smooth region. The kink itself
    (delta=0 exactly) gets its own dedicated test.

    sigma_obs is forced positive after the draw: it's a standard deviation
    (appears as log(sigma_obs) and 1/sigma_obs**3 in the posterior/gradient),
    so a random draw landing at/below zero would make the objective undefined
    rather than exercising a legitimate point in parameter space.
    """
    rng = np.random.default_rng(seed=0)
    params = rng.normal(scale=0.5, size=param_size)
    params[SIGMA_OBS_IDX] = abs(params[SIGMA_OBS_IDX]) + 0.1
    return params
