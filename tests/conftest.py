"""
Shared fixtures for the analytic-prophet test suite.

These currently exercise the pure-Python reference implementation in
CustomProphet (legacy/customProphet.py). compiled_optimizer_lib below
builds the C++ core (legacy/optimize.cpp, the analytic gradient fit_cpp()
calls into) on the fly so the fit() vs fit_cpp() parity test can load and
call it directly, without a compiled liboptimization.so checked into the
repo.
"""
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "legacy"))
from customProphet import (CustomProphet, N_CHANGE_POINTS, n_yearly,  # noqa: E402
                           SIGMA_OBS_IDX, SIGMA_OBS_PRIOR_SCALE)

DATA_PATH = Path(__file__).parent / "data" / "peyton_manning.csv"
CPP_SOURCE = Path(__file__).parent.parent / "legacy" / "optimize.cpp"
CPP_GRADIENT_SOURCE = Path(__file__).parent.parent / "legacy" / "minus_log_posterior_and_gradient.cpp"
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


def _build_cpp(source, out_name, tmp_path_factory, link_lbfgs):
    """Builds one of the legacy/*.cpp sources into a shared library, using the
    compile command documented in that source file's own trailing comment.
    Skips (rather than fails) the tests that depend on it when the C++
    toolchain, Eigen, or liblbfgs aren't available -- that's an environment
    gap, not a code defect."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip(f"no C++ compiler (g++/clang++) found to build {source.name}")

    eigen_include = _find_eigen_include()
    if eigen_include is None:
        pytest.skip(f"Eigen headers not found (set EIGEN_INCLUDE_DIR) -- can't build {source.name}")

    lib_path = tmp_path_factory.mktemp("cpp_core") / out_name
    cmd = [
        compiler, "-std=c++17", "-shared", "-fPIC", "-O2",
        "-o", str(lib_path), str(source),
        f"-I{eigen_include}",
    ]
    if link_lbfgs:
        cmd += ["-L/usr/local/lib", "-L/opt/homebrew/lib", "-llbfgs"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        pytest.skip(f"could not build {source.name} (liblbfgs missing?): {result.stderr[-500:]}")

    return str(lib_path)


@pytest.fixture(scope="session")
def compiled_optimizer_lib(tmp_path_factory):
    """legacy/optimize.cpp -- the compiled L-BFGS optimizer fit_cpp() drives."""
    return _build_cpp(CPP_SOURCE, "liboptimization.so", tmp_path_factory, link_lbfgs=True)


@pytest.fixture(scope="session")
def compiled_gradient_lib(tmp_path_factory):
    """legacy/minus_log_posterior_and_gradient.cpp -- the C++ objective and
    analytic gradient, exposed on its own so they can be cross-checked against
    the Python reference without going through the optimizer."""
    return _build_cpp(CPP_GRADIENT_SOURCE, "libmlpg.so", tmp_path_factory, link_lbfgs=False)


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
def cpp_mlp_and_gradient(compiled_gradient_lib):
    """Callable wrapping the C++ objective+gradient entry point.

    Takes a model and the 47-length (k, m, delta, beta) vector the C++ side
    uses -- it has no sigma_obs slot, since the C++ core never estimates it --
    and returns (minus_log_posterior, gradient)."""
    lib = ctypes.CDLL(compiled_gradient_lib)
    nd = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.minus_log_posterior_and_gradient.argtypes = [
        nd, ctypes.c_int, nd, ctypes.c_int, nd, ctypes.c_int, ctypes.c_double,
        nd, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, nd, nd,
    ]
    lib.minus_log_posterior_and_gradient.restype = None

    def call(model, cpp_params, sigma_obs):
        cpp_params = np.ascontiguousarray(cpp_params, dtype=np.float64)
        mlp_out = np.zeros(1)
        grad_out = np.zeros(len(cpp_params))
        lib.minus_log_posterior_and_gradient(
            cpp_params, len(cpp_params),
            np.ascontiguousarray(model.t_scaled, dtype=np.float64), len(model.t_scaled),
            np.ascontiguousarray(model.change_points, dtype=np.float64), len(model.change_points),
            float(model.scale_period),
            np.ascontiguousarray(model.normalized_y, dtype=np.float64), len(model.normalized_y),
            sigma_obs, model.sigma_k, model.sigma_m, model.sigma, model.tau,
            mlp_out, grad_out,
        )
        return mlp_out[0], grad_out

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
