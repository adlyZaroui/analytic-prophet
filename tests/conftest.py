"""
Shared fixtures for the analytic-prophet test suite.

These currently exercise the pure-Python reference implementation in
CustomProphet (legacy/customProphet.py). compiled_optimizer_lib below
builds the C++ core (legacy/optimize.cpp, the analytic gradient fit_cpp()
calls into) on the fly so the fit() vs fit_cpp() parity test can load and
call it directly, without a compiled liboptimization.so checked into the
repo.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "legacy"))
from customProphet import CustomProphet, N_CHANGE_POINTS, n_yearly, SIGMA_OBS_IDX  # noqa: E402

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


@pytest.fixture(scope="session")
def compiled_optimizer_lib(tmp_path_factory):
    """Builds legacy/optimize.cpp into a shared library once per test
    session, using the compile command documented in the source file's own
    trailing comment. Skips (rather than fails) the tests that depend on it
    when the C++ toolchain, Eigen, or liblbfgs aren't available -- that's an
    environment gap, not a code defect."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler (g++/clang++) found to build legacy/optimize.cpp")

    eigen_include = _find_eigen_include()
    if eigen_include is None:
        pytest.skip("Eigen headers not found (set EIGEN_INCLUDE_DIR) -- can't build legacy/optimize.cpp")

    lib_path = tmp_path_factory.mktemp("cpp_core") / "liboptimization.so"
    cmd = [
        compiler, "-std=c++17", "-shared", "-fPIC", "-O2",
        "-o", str(lib_path), str(CPP_SOURCE),
        f"-I{eigen_include}",
        "-L/usr/local/lib", "-L/opt/homebrew/lib",
        "-llbfgs",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        pytest.skip(f"could not build legacy/optimize.cpp (liblbfgs missing?): {result.stderr[-500:]}")

    return str(lib_path)


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
