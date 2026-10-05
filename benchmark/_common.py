"""Shared plumbing for the benchmarks.

Kept separate from tests/conftest.py on purpose: the benchmarks are standalone
scripts, not part of `pytest tests/`, and should stay runnable without pytest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "analytic_prophet"
DATA_PATH = REPO / "tests" / "data" / "peyton_manning.csv"

# Before importing the package: these are standalone scripts, so sys.path[0]
# is benchmark/ and the repo root is not on it. Every other import of
# analytic_prophet in this file is inside a function for the same reason.
sys.path.insert(0, str(REPO))

from analytic_prophet.build import ToolchainMissing  # noqa: E402
from analytic_prophet.build import (  # noqa: E402
    build_cpp_extension as build_cpp_extension_shared)

# Sizes worth reporting. 50 is below the T < 100 cutoff, where both sides use
# Newton rather than L-BFGS (see issue #25) -- kept in deliberately, since that
# regime has its own cost profile and a mismatch there is a finding, not noise.
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
    """Compile the C++ core into `out_dir`, or return None with a reason.

    The compile itself lives in `analytic_prophet.build` since #108, because
    the package needs to be able to build -- `fit(df)` runs the compiled core
    and nothing used to build it. This was a second copy of those ~45 lines,
    and a drifted one: it compiled at -O3 where tests/conftest.py compiled at
    -O2, so every published timing came from a different binary than the
    parity tests verified. The shared builder compiles at -O3, which is the
    one the timings were measured with.

    None-with-a-message rather than raising, which is what the benchmarks
    want: they report the paths they can run and say what they skipped.
    """
    try:
        return build_cpp_extension_shared(dest=out_dir)
    except ToolchainMissing as missing:
        print(f"{missing}; skipping the compiled path", file=sys.stderr)
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
    from analytic_prophet import AnalyticProphet
    model = AnalyticProphet()
    model.fit(df, backend="python", analytic=analytic)
    return model


def fit_compiled(df, lib_path):
    from analytic_prophet import AnalyticProphet
    model = AnalyticProphet()
    model.fit(df, lib_path=lib_path)
    return model


IMPLEMENTATIONS = ("prophet", "python", "python(numeric grad)", "compiled")


def run_one(name, df, lib_path=None):
    """Dispatch by implementation name. Raises if the implementation is
    unavailable, so callers decide how to report that."""
    if name == "prophet":
        return fit_prophet(df)
    if name == "python":
        return fit_python(df, analytic=True)
    if name == "python(numeric grad)":
        return fit_python(df, analytic=False)
    if name == "compiled":
        if lib_path is None:
            raise RuntimeError("the C++ extension was not built")
        return fit_compiled(df, lib_path)
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
