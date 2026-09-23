"""
Issue #43: every dimension and scale the C++ core relies on, checked.

Eigen answers a size mismatch with an assertion, and an assertion calls
`abort()`. The interpreter dies with no traceback, no line number and -- in a
test run -- no failing test name. Three separate mismatches reached that state
during #16, each found by tripping over it: a stale `params` length (#38), a
`beta` narrower than the design matrix (#39), and a short `sigmas` (#42). Each
was guarded individually afterwards, which left `optimize()` well defended and
the core objective almost undefended, even though the core is a public entry
point that the fixtures in conftest.py call directly.

A second, quieter family does not abort at all: a non-positive `tau` or prior
scale divides into the objective and returns a plausible wrong number, and a
zero or non-finite period fills the design matrix with NaN.

**Every case here runs in one subprocess**, not in the test process. That is the
point of the module rather than an implementation detail: if a guard is removed,
these tests have to *fail*, and a test that aborts the interpreter reports
nothing at all -- it takes the whole run with it and names no culprit.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

LEGACY = Path(__file__).parent.parent / "legacy"
BENCHMARK = Path(__file__).parent.parent / "benchmark"

# case -> a statement mutating `kw`, the otherwise-valid argument dict
CASES = {
    "t_seasonality_short": "kw['t_seasonality'] = kw['t_seasonality'][:-5]",
    "t_seasonality_long": "kw['t_seasonality'] = np.r_[kw['t_seasonality'], kw['t_seasonality'][:5]]",
    "normalized_y_short": "kw['normalized_y'] = kw['normalized_y'][:-5]",
    "fourier_order_negative": "kw['fourier_orders'] = [-3]",
    "fourier_order_zero": "kw['fourier_orders'] = [0]",
    "period_negative": "kw['seasonality_periods'] = [-365.25]",
    "period_zero": "kw['seasonality_periods'] = [0.0]",
    "period_nan": "kw['seasonality_periods'] = [float('nan')]",
    "period_inf": "kw['seasonality_periods'] = [float('inf')]",
    "empty_series": "kw['t_scaled'] = np.zeros(0); kw['normalized_y'] = np.zeros(0); kw['t_seasonality'] = np.zeros(0)",
    "orders_periods_mismatched": "kw['fourier_orders'] = [10, 3]",
    "sigmas_short": "kw['sigmas'] = np.full(19, 10.0)",
    "sigmas_non_positive": "kw['sigmas'] = np.r_[np.full(19, 10.0), 0.0]",
    "params_too_long": "kw['params'] = np.zeros(60)",
    "params_too_short": "kw['params'] = np.zeros(20)",
    "tau_negative": "kw['tau'] = -0.05",
    "tau_zero": "kw['tau'] = 0.0",
    "sigma_obs_prior_scale_zero": "kw['sigma_obs_prior_scale'] = 0.0",
    "sigma_k_zero": "kw['sigma_k'] = 0.0",
}

ENTRY_POINTS = ("minus_log_posterior_and_gradient", "optimize")

RUNNER = '''
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, {legacy!r})
sys.path.insert(0, {benchmark!r})
import _common as common
from customProphet import CustomProphet, load_cpp_module, seasonal_time, seasonality

CASES = json.loads({cases!r})
cpp = load_cpp_module({lib!r})

df = common.load_data(300)
model = CustomProphet()
model.yearly_seasonality = model.weekly_seasonality = model.daily_seasonality = False
model.seasonalities = {{"yearly": seasonality(365.25, 10)}}
model.y = df["y"].values
model.ds = pd.to_datetime(df["ds"])
model.t_scaled = np.array((model.ds - model.ds.min()) / (model.ds.max() - model.ds.min()))
model.T = len(df)
model.t_seasonality = seasonal_time(model.ds)
model._normalize_y(); model._build_layout(); model._generate_change_points()

def fresh():
    return dict(params=np.zeros(model.layout.size), t_scaled=model.t_scaled,
                change_points=model.change_points, t_seasonality=model.t_seasonality,
                normalized_y=model.normalized_y, sigma_obs_prior_scale=0.5,
                sigma_k=5.0, sigma_m=5.0, sigmas=np.full(20, 10.0), tau=0.05,
                fourier_orders=[10], seasonality_periods=[365.25])

for entry in {entries!r}:
    for case, mutation in CASES.items():
        kw = fresh()
        exec(mutation)
        # flushed before the call, so a case that aborts is identifiable from
        # the output rather than merely absent
        print(json.dumps({{"event": "start", "entry": entry, "case": case}}), flush=True)
        try:
            result = getattr(cpp, entry)(**kw)
            value = result[0] if isinstance(result, tuple) else None
            outcome = "silent-nan" if value is not None and not np.isfinite(value) else "silent"
            message = ""
        except ValueError as exc:
            outcome, message = "raises", str(exc)
        except Exception as exc:
            outcome, message = type(exc).__name__, str(exc)
        print(json.dumps({{"event": "done", "entry": entry, "case": case,
                           "outcome": outcome, "message": message}}), flush=True)
'''


@pytest.fixture(scope="module")
def outcomes(compiled_optimizer_module):
    """Run every case in one subprocess and collect what each one did.

    One subprocess rather than one per case: 38 interpreter startups would
    dominate the suite's runtime, and the abort-safety that matters comes from
    being out of process at all, not from being alone in it.
    """
    script = RUNNER.format(legacy=str(LEGACY), benchmark=str(BENCHMARK),
                           lib=compiled_optimizer_module,
                           cases=json.dumps(CASES), entries=list(ENTRY_POINTS))
    completed = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=900)

    started, finished = [], {}
    for line in completed.stdout.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        key = (record["entry"], record["case"])
        if record["event"] == "start":
            started.append(key)
        else:
            finished[key] = record

    if started and started[-1] not in finished:
        entry, case = started[-1]
        pytest.fail(
            f"the runner died during {entry}({case}) -- almost certainly an Eigen "
            f"assertion calling abort(), which is the failure this module exists "
            f"to prevent. Last stderr:\n{completed.stderr[-2000:]}")
    return finished


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_bad_input_raises(outcomes, entry, case):
    """A `ValueError`, through both entry points, for every case.

    Both matter: `optimize()` is what `fit_cpp` drives, and
    `minus_log_posterior_and_gradient` is what the test fixtures and anyone
    reading the pybind11 signatures will call.
    """
    record = outcomes.get((entry, case))
    assert record is not None, f"{entry}({case}) produced no result at all"
    assert record["outcome"] == "raises", (
        f"{entry}({case}) did not raise: it returned {record['outcome']!r}. "
        f"A wrong answer that looks plausible is worse than an abort.")


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("case,fragment", [
    ("t_seasonality_short", "same observations as t_scaled"),
    ("t_seasonality_long", "same observations as t_scaled"),
    ("normalized_y_short", "same length"),
    ("fourier_order_negative", "Fourier order must be positive"),
    ("fourier_order_zero", "Fourier order must be positive"),
    ("period_negative", "positive and finite"),
    ("period_zero", "positive and finite"),
    ("period_nan", "positive and finite"),
    ("period_inf", "positive and finite"),
    ("empty_series", "nothing to fit"),
    ("orders_periods_mismatched", "one period per seasonality"),
    ("sigmas_short", "one prior scale per column"),
    ("sigmas_non_positive", "sigmas must be positive"),
    ("tau_negative", "tau must be positive"),
    ("tau_zero", "tau must be positive"),
    ("sigma_obs_prior_scale_zero", "sigma_obs_prior_scale must be positive"),
    ("sigma_k_zero", "sigma_k and sigma_m must be positive"),
])
def test_the_message_says_which_input_was_wrong(outcomes, entry, case, fragment):
    """The message is the whole value of the change: the caller has to learn
    *which* argument was wrong, not merely that the call failed."""
    record = outcomes[(entry, case)]
    assert fragment in record["message"], (
        f"{entry}({case}) raised {record['message']!r}, which does not say "
        f"{fragment!r}")


def test_valid_inputs_are_still_accepted(prepared_model, cpp_module):
    """The guards must not have narrowed what the model itself passes -- every
    other test in the suite would catch a false rejection, but stating it here
    keeps this module self-contained."""
    import numpy as np

    value, gradient = cpp_module.minus_log_posterior_and_gradient(
        params=np.zeros(prepared_model.layout.size), t_scaled=prepared_model.t_scaled,
        change_points=prepared_model.change_points,
        t_seasonality=prepared_model.t_seasonality,
        normalized_y=prepared_model.normalized_y, sigma_obs_prior_scale=0.5,
        sigma_k=prepared_model.sigma_k, sigma_m=prepared_model.sigma_m,
        sigmas=prepared_model.sigmas, tau=prepared_model.tau,
        fourier_orders=[10], seasonality_periods=[365.25])

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
