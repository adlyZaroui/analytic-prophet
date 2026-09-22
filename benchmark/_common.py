"""Shared plumbing for the benchmarks.

Kept separate from tests/conftest.py on purpose: the benchmarks are standalone
scripts, not part of `pytest tests/`, and should stay runnable without pytest.
"""
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LEGACY = REPO / "legacy"
DATA_PATH = REPO / "tests" / "data" / "peyton_manning.csv"

sys.path.insert(0, str(LEGACY))

# Sizes worth reporting. 50 is below Prophet's T < 100 cutoff, where it uses
# Newton rather than L-BFGS (see issue #25) -- kept in deliberately, since a
# mismatch there is a finding rather than noise.
DEFAULT_SIZES = (50, 100, 300, 1000, 2905)

# These used to pin Prophet's seasonality to yearly-only, because that was the
# only component this project could fit: comparing its 26-column design matrix
# against our 20-column one would have called a modelling gap "performance".
# #16 task 3 closed that -- both sides now choose their components from the
# history by the same rule -- so the seasonality arguments are back on 'auto'
# and the comparison is against Prophet as a user actually gets it.
#
# Everything left here is Prophet's own default, restated rather than relied
# on, so that a change in their defaults shows up as a benchmark change rather
# than silently moving the baseline.
PROPHET_KWARGS = dict(
    growth="linear",
    n_changepoints=25,
    changepoint_range=0.8,
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_mode="additive",
    seasonality_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    mcmc_samples=0,
)

PROPHET_INSTALL_HINT = (
    "prophet is not installed, so the comparison this benchmark exists for is "
    "unavailable. Install it with `pip install prophet` (it pulls cmdstanpy and "
    "needs a cmdstan toolchain). Our two paths are still measured below."
)


def load_data(n_rows=None):
    df = pd.read_csv(DATA_PATH)
    return df if n_rows is None else df.iloc[:n_rows].reset_index(drop=True)


def build_cpp_extension(out_dir):
    """Compile legacy/optimize.cpp into an importable extension.

    Built once and reused: compilation is not part of what is being measured.
    Returns the path, or None with a reason printed if the toolchain is absent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("analytic_prophet_cpp" + sysconfig.get_config_var("EXT_SUFFIX"))
    if out_path.exists():
        return str(out_path)

    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        print("no C++ compiler found; skipping fit_cpp", file=sys.stderr)
        return None
    try:
        import pybind11
    except ImportError:
        print("pybind11 not installed; skipping fit_cpp", file=sys.stderr)
        return None

    eigen = _find_eigen()
    if eigen is None:
        print("Eigen headers not found (set EIGEN_INCLUDE_DIR); skipping fit_cpp", file=sys.stderr)
        return None

    lbfgspp = _find_lbfgspp()
    if lbfgspp is None:
        print("LBFGSpp headers not found (`brew install lbfgspp`); skipping fit_cpp", file=sys.stderr)
        return None

    cmd = [compiler, "-std=c++17", "-shared", "-fPIC", "-O3",
           "-o", str(out_path), str(LEGACY / "optimize.cpp"),
           f"-I{eigen}", f"-I{pybind11.get_include()}",
           f"-I{sysconfig.get_paths()['include']}", f"-I{lbfgspp}"]
    if sys.platform == "darwin":
        cmd += ["-undefined", "dynamic_lookup"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"could not build the C++ core: {result.stderr[-400:]}", file=sys.stderr)
        return None
    return str(out_path)


def _find_lbfgspp():
    """LBFGSpp headers (L-BFGS-B), header-only -- nothing is linked."""
    candidates = [os.environ.get("LBFGSPP_INCLUDE_DIR")]
    try:
        prefix = subprocess.run(["brew", "--prefix", "lbfgspp"], capture_output=True,
                                text=True, timeout=5).stdout.strip()
        if prefix:
            candidates.append(str(Path(prefix, "include")))
    except (OSError, subprocess.SubprocessError):
        pass
    candidates += ["/opt/homebrew/include", "/usr/local/include", "/usr/include"]
    for c in candidates:
        if c and (Path(c) / "LBFGSB.h").exists():
            return c
    return None


def _find_eigen():
    candidates = [os.environ.get("EIGEN_INCLUDE_DIR")]
    try:
        prefix = subprocess.run(["brew", "--prefix", "eigen"], capture_output=True,
                                text=True, timeout=5).stdout.strip()
        if prefix:
            candidates.append(str(Path(prefix, "include", "eigen3")))
    except (OSError, subprocess.SubprocessError):
        pass
    candidates += ["/opt/homebrew/opt/eigen/include/eigen3",
                   "/usr/local/opt/eigen/include/eigen3",
                   "/usr/local/include/eigen3", "/usr/include/eigen3"]
    for c in candidates:
        if c and (Path(c) / "Eigen" / "Dense").exists():
            return c
    return None


def prophet_available():
    try:
        import prophet  # noqa: F401
        return True
    except ImportError:
        return False


# --- the three things being compared -------------------------------------
# Each takes a dataframe and performs one full fit, exactly as a user would
# call it, so setup (scaling, changepoint placement) is inside the measurement
# on both sides.

def fit_prophet(df):
    from prophet import Prophet
    model = Prophet(**PROPHET_KWARGS)
    model.fit(df)
    return model


def fit_python(df, analytic=True):
    from customProphet import CustomProphet
    model = CustomProphet()
    model.fit(df, analytic=analytic)
    return model


def fit_cpp(df, lib_path):
    from customProphet import CustomProphet
    model = CustomProphet()
    model.fit_cpp(df, lib_path=lib_path)
    return model


IMPLEMENTATIONS = ("prophet", "fit(analytic=True)", "fit(numeric grad)", "fit_cpp")


def run_one(name, df, lib_path=None):
    """Dispatch by implementation name. Raises if the implementation is
    unavailable, so callers decide how to report that."""
    if name == "prophet":
        return fit_prophet(df)
    if name == "fit(analytic=True)":
        return fit_python(df, analytic=True)
    if name == "fit(numeric grad)":
        return fit_python(df, analytic=False)
    if name == "fit_cpp":
        if lib_path is None:
            raise RuntimeError("the C++ extension was not built")
        return fit_cpp(df, lib_path)
    raise ValueError(f"unknown implementation {name!r}")


def peak_rss_bytes():
    """Peak resident set size of this process and any children it waited on.

    Children matter: Prophet runs the actual optimization in a cmdstan
    subprocess, so measuring only RUSAGE_SELF would report almost none of its
    memory. ru_maxrss is bytes on macOS and kilobytes on Linux.
    """
    import resource
    scale = 1 if sys.platform == "darwin" else 1024
    return scale * max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:,.1f} {unit}"
        n /= 1024
