"""
Shared fixtures for the analytic-prophet test suite.

These currently exercise the pure-Python reference implementation in
AnalyticProphet (analytic_prophet/forecaster.py). compiled_optimizer_module
below builds the C++ core (analytic_prophet/optimize.cpp) as an extension on the
fly, so the parity and convergence tests can import and call it directly
without a built extension checked into the repo.
"""
import collections
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Both the repo root and benchmark/ reach sys.path through `pythonpath` in
# pyproject.toml, so a fresh clone runs `pytest` with no install and no
# PYTHONPATH. benchmark/ is there because the Prophet-comparison plumbing lives
# with the benchmarks -- that is also where it is exercised interactively, and
# the agreement tests reuse it rather than keeping a second copy in step.
from analytic_prophet import (AnalyticProphet, N_CHANGE_POINTS, n_yearly,
                           SIGMA_OBS_IDX, SIGMA_OBS_PRIOR_SCALE, CPP_MODULE_NAME,
                           load_cpp_module, seasonal_time, seasonality,
                           condition_matrix, TREND_INDICATORS, YEARLY_PERIOD)

DATA_PATH = Path(__file__).parent / "data" / "peyton_manning.csv"
CPP_SOURCE = Path(__file__).parent.parent / "analytic_prophet" / "optimize.cpp"
PARAM_SIZE = 2 + N_CHANGE_POINTS + 1 + 2 * n_yearly  # k, m, delta, sigma_obs, beta -> 48

# ---------------------------------------------------------------------------
# Environment gaps: a skip on a laptop, a failure where the environment was
# built on purpose.
#
# #103. A runner without Eigen and LBFGSpp reports a green suite that silently
# skipped the compiled path -- which is worse than a red one, because it looks
# like evidence. The skip is right on a contributor's machine and wrong in CI,
# so which one it is belongs to the caller rather than to the test.
ENVIRONMENT_PREFIX = "environment: "

REQUIREMENTS = {
    "cpp": ("--require-cpp",
            "the C++ toolchain (a compiler, Eigen, LBFGSpp, pybind11)"),
    "prophet": ("--require-prophet", "prophet"),
}


def pytest_addoption(parser):
    group = parser.getgroup("analytic-prophet")
    for option, what in sorted(REQUIREMENTS.values()):
        group.addoption(option, action="store_true", default=False,
                        help=f"fail rather than skip when {what} is missing")


def environment_gap(config, requirement, reason):
    """Skip because this environment cannot run it -- or fail, because the
    caller said this environment was supposed to be able to."""
    option, what = REQUIREMENTS[requirement]
    if config.getoption(option):
        pytest.fail(f"{reason}\n\nThis run was given {option}, so {what} was "
                    f"expected to be installed and its absence is a failure "
                    f"rather than a skip.", pytrace=False)
    pytest.skip(ENVIRONMENT_PREFIX + reason)


def _skip_reason(report):
    """The reason out of a skip report, whatever shape it arrived in."""
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say what was skipped and why, grouped.

    "Green" has to be readable: without this, a run that skipped every test
    touching the compiled core looks exactly like a run that passed them. The
    count is the thing CI publishes (#103).
    """
    reports = terminalreporter.stats.get("skipped", [])
    if not reports:
        return
    counts = collections.Counter(_skip_reason(report) for report in reports)
    terminalreporter.write_sep("=", "skipped, by reason", yellow=True)
    for reason, count in counts.most_common():
        terminalreporter.write_line(f"  {count:4d}  {reason}")

    gaps = sum(count for reason, count in counts.items()
               if reason.startswith(ENVIRONMENT_PREFIX))
    if gaps:
        options = " ".join(option for option, _ in sorted(REQUIREMENTS.values()))
        terminalreporter.write_line(
            f"  {gaps} of these are gaps in this environment rather than "
            f"choices about what to run.")
        terminalreporter.write_line(
            f"  Run with {options} to make them failures instead.")


def pin_yearly_only(model):
    """Force the pre-#16-task-3 component set: yearly at order 10, nothing else.

    Auto-selection would give weekly-only on the short slices these tests use,
    changing the model they certify and invalidating the residuals and loss
    values recorded in their comments. They are about the optimizer, not about
    which components a model picks, so the component set is pinned and the
    recorded numbers keep meaning what they say.
    """
    model.yearly_seasonality = n_yearly
    model.weekly_seasonality = False
    model.daily_seasonality = False
    return model


@pytest.fixture
def peyton_manning_df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def prepared_model(peyton_manning_df):
    """A AnalyticProphet with data loaded and preprocessed but not yet fit --
    gives direct access to t / changepoints_t / y_scaled without
    paying for a full optimize() run in every test.

    Yearly seasonality is registered by hand rather than by the auto rule of
    #16 task 3, which would add weekly on this series. Everything built on this
    fixture is about the objective and its gradient at a fixed shape, so pinning
    the 48-parameter layout (PARAM_SIZE) keeps those tests saying what they say.
    """
    model = AnalyticProphet()
    model.seasonalities = {"yearly": seasonality(YEARLY_PERIOD, n_yearly)}
    model.y = peyton_manning_df["y"].values
    model.ds = pd.to_datetime(peyton_manning_df["ds"])
    model.t = np.array(
        (model.ds - model.ds.min()) / (model.ds.max() - model.ds.min())
    )
    model.T = peyton_manning_df.shape[0]
    model.t_seasonality = seasonal_time(model.ds)
    model._normalize_y()
    model._build_layout()
    model.set_changepoints()
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


def _build_cpp_extension(tmp_path_factory, config):
    """Builds analytic_prophet/optimize.cpp into an importable extension, using
    the compile command documented in that file's trailing comment. Skips
    (rather than fails) the tests that depend on it when the C++ toolchain,
    Eigen, pybind11 or LBFGSpp aren't available -- that's an environment gap,
    not a code defect. `--require-cpp` says the environment was built to have
    them, and turns each of these into a failure (#103)."""
    gap = lambda reason: environment_gap(config, "cpp", reason)

    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        gap("no C++ compiler found to build the C++ core")

    eigen_include = _find_eigen_include()
    if eigen_include is None:
        gap("Eigen headers not found (set EIGEN_INCLUDE_DIR) -- can't build the C++ core")

    try:
        import pybind11
    except ImportError:
        gap("pybind11 is not installed -- can't build the C++ core")

    lbfgspp_include = _find_lbfgspp_include()
    if lbfgspp_include is None:
        gap("LBFGSpp headers not found (set LBFGSPP_INCLUDE_DIR, or "
            "`brew install lbfgspp`) -- can't build the C++ core")

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out_path = tmp_path_factory.mktemp("cpp_core") / f"{CPP_MODULE_NAME}{suffix}"

    cmd = [
        compiler, "-std=c++17", "-shared", "-fPIC", "-O2",
        "-o", str(out_path), str(CPP_SOURCE),
        f"-I{eigen_include}",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{lbfgspp_include}",
    ]
    if sys.platform == "darwin":
        # Extension modules resolve CPython's symbols from the host interpreter
        # at load time rather than linking libpython.
        cmd += ["-undefined", "dynamic_lookup"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        gap(f"could not build the C++ core: {result.stderr[-500:]}")

    return str(out_path)


def _find_lbfgspp_include():
    """LBFGSpp headers (L-BFGS-B). Header-only, so nothing is linked.

    NOTE: LBFGSpp ships LBFGS.h, which shadows liblbfgs's lbfgs.h on a
    case-insensitive filesystem. That no longer bites since #23 removed the
    liblbfgs dependency, but it is why the two must not both be on the
    include path.
    """
    candidates = [os.environ.get("LBFGSPP_INCLUDE_DIR")]
    try:
        prefix = subprocess.run(["brew", "--prefix", "lbfgspp"], capture_output=True,
                                text=True, timeout=5).stdout.strip()
        if prefix:
            candidates.append(str(Path(prefix, "include")))
    except (OSError, subprocess.SubprocessError):
        pass
    candidates += ["/opt/homebrew/include", "/usr/local/include", "/usr/include"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "LBFGSB.h").exists():
            return candidate
    return None


@pytest.fixture(scope="session")
def compiled_optimizer_module(tmp_path_factory, pytestconfig):
    """Path to the freshly built pybind11 extension, as fit_cpp(lib_path=...)
    wants it."""
    return _build_cpp_extension(tmp_path_factory, pytestconfig)


@pytest.fixture(scope="session")
def cpp_module(compiled_optimizer_module):
    """The built extension, imported."""
    return load_cpp_module(compiled_optimizer_module)


@pytest.fixture(scope="session")
def cpp_mlp_and_gradient(cpp_module):
    """Calls the C++ objective+gradient the optimizer itself drives.

    `cpp_params` is in the C++ layout -- (k, m, delta, beta, zeta) with
    zeta = log(sigma_obs) last, length 2 + S + K + 1. Returns
    (minus_log_posterior, gradient). Since sigma_obs is now estimated on both
    sides, this value is directly comparable with the Python objective: no
    constant offset separates them any more."""
    def call(model, cpp_params, include_l1_prior=True):
        return cpp_module.minus_log_posterior_and_gradient(
            params=cpp_params,
            t=model.t,
            changepoints_t=model.changepoints_t,
            t_seasonality=model.t_seasonality,
            y_scaled=model.y_scaled,
            sigma_obs_prior_scale=SIGMA_OBS_PRIOR_SCALE,
            sigma_k=model.sigma_k,
            sigma_m=model.sigma_m,
            sigmas=model.sigmas,
            s_m=model.s_m,
            cap_scaled=(model.cap_scaled if model.growth == 'logistic'
                        else np.empty(0)),
            trend_indicator=TREND_INDICATORS[model.growth],
            changepoint_prior_scale=model.changepoint_prior_scale,
            # from the model's registry, so a test that registers a second
            # seasonality gets the design matrix it asked for
            fourier_orders=[p["fourier_order"] for p in model.seasonalities.values()],
            seasonality_periods=[p["period"] for p in model.seasonalities.values()],
            seasonality_conditions=condition_matrix(model.seasonalities,
                                                    model.condition_masks,
                                                    len(model.t)),
            data_columns=(model._data_columns
                              if model._holiday_columns
                              else np.empty((len(model.t), 0))),
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


@pytest.fixture(scope="session")
def prophet_comparison(pytestconfig):
    """Handles for comparing against the original Prophet.

    Skips when prophet is not installed: it is a heavy optional dependency
    (it pulls cmdstanpy and a compiled Stan model), so the rest of the suite
    must not require it. `--require-prophet` is for the one CI job that
    installs it on purpose, where a skip would mean the agreement checks
    quietly did not run (#103).

    Some tests request this fixture **by name alone**, without using what it
    returns: the evaluation tiers import prophet inside their own `collect`
    functions, so those tests need the skip without needing the handles. They
    used to raise `ImportError` instead, which made 21 tests fail rather than
    skip on a machine without prophet -- found by rehearsing a CI matrix job
    before there was a CI matrix (#103).
    """
    if not importlib.util.find_spec("prophet"):
        environment_gap(pytestconfig, "prophet",
                        "prophet is not installed -- `pip install prophet` to run "
                        "the agreement checks against the original")

    import _common as benchmark_common
    import _prophet_bridge as bridge
    from prophet import Prophet

    return Prophet, benchmark_common, bridge
