"""Tier 0 -- the correctness gate (#79).

**Are the two implementations fitting the same model at all?** Everything the
other tiers measure is about the difference between two optimizers; if they are
optimizing different specifications, those numbers are about something nobody
asked. So this runs first, and a failure stops the run.

Four checks, in order of how fundamental they are:

  1. **the design matrix** -- every column of X, element for element. A
     difference here is a modelling difference, not an optimizer one.
  2. **the prior scales** -- `sigmas`, which weight beta's prior per column.
  3. **the changepoints** -- `t_change`, since #15 made them identical rather
     than merely similar.
  4. **the posterior** -- our `lp__` at least Prophet's, under Stan's own
     density. This is the one claim the project makes, so it is the one the
     gate refuses to let the rest of the suite assume.

The first three are equalities and the tolerance on them is the arithmetic, not
the model: the Fourier basis is evaluated in a different order (see the README),
which costs about 1e-10 and nothing more.
"""
import tempfile

import numpy as np

import corpora
import harness
from harness import Measurement

# Element-wise agreement the different evaluation order allows. Deliberately
# tight: this is floating-point noise, and anything larger is a real difference.
DESIGN_TOLERANCE = 1e-9

# How far below Prophet our posterior may fall before the gate trips. Not zero,
# because the basis transfer that makes beta comparable carries its own
# rounding, but far tighter than any margin the project reports.
POSTERIOR_TOLERANCE = 1e-6

SIZES = (300, 1000, 2905)


def _compare(size, lib_path):
    """One size, as (measurements, ok)."""
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    df = corpora.peyton_manning(size)
    series = f"peyton_manning[:{size}]"

    prophet_model = Prophet(**harness.PROPHET_KWARGS)
    stan_model, stan_data, prophet_params = harness.capture_stan_model(prophet_model, df)
    reported = float(np.asarray(prophet_model.params["lp__"]).ravel()[0])
    lp_prophet = harness.validate_bridge(stan_model, stan_data, prophet_params, reported)

    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    X_stan = np.asarray(stan_data["X"], dtype=float)
    sigmas_stan = np.asarray(stan_data["sigmas"], dtype=float)

    # fitted on Prophet's changepoints, so `delta` indexes the same breakpoints
    # and the comparison is of optimizers rather than of specifications
    ours = AnalyticProphet()
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=lib_path)

    X_ours = np.ascontiguousarray(
        ours.make_all_seasonality_features(df)[0].to_numpy(dtype=float))

    same_shape = X_ours.shape == X_stan.shape
    design_gap = float(np.max(np.abs(X_ours - X_stan))) if same_shape else float("inf")
    sigmas_gap = (float(np.max(np.abs(ours.sigmas - sigmas_stan)))
                  if ours.sigmas.shape == sigmas_stan.shape else float("inf"))
    changepoint_gap = (float(np.max(np.abs(
        np.asarray(ours.changepoints_t) - changepoints_t)))
        if ours.changepoints_t.shape == changepoints_t.shape else float("inf"))

    def row(metric, value, unit="", implementation="both"):
        return Measurement(0, series, "default", implementation, metric, value, unit)

    shape_rows = [
        row("design_matrix_max_abs_diff", design_gap),
        row("prior_scales_max_abs_diff", sigmas_gap),
        row("changepoints_max_abs_diff", changepoint_gap),
        row("design_columns", float(X_stan.shape[1])),
        row("design_columns", float(X_ours.shape[1]), "", "analytic_prophet"),
        row("observations", float(len(df))),
    ]
    if not same_shape:
        # The posterior cannot be scored at all: Stan would be handed a `beta`
        # of the wrong length and raise from inside the model. A shape mismatch
        # is the most basic way the gate can fail, so it reports rather than
        # propagating -- the point of a gate is to say what is wrong, not to
        # crash on the way to finding out.
        print(f"  T={size}: design matrices differ in shape, "
              f"{X_ours.shape[1]} columns against Stan's {X_stan.shape[1]}")
        return shape_rows + [row("lp__", lp_prophet, "nats", "prophet")], False

    # beta is passed straight across rather than projected into Stan's basis.
    # That is only legitimate because the design matrices are identical, which
    # is what the check above establishes -- since #36 aligned the Fourier basis
    # there is no rotation left to undo, and a projection step here would hide a
    # real difference by absorbing it.
    lp_ours = harness.stan_log_prob(
        stan_model, stan_data, ours.params["k"][0][0], ours.params["m"][0][0],
        ours.params["delta"][0], ours.sigma_obs, ours.params["beta"][0])

    measurements = shape_rows + [
        row("lp__", lp_prophet, "nats", "prophet"),
        row("lp__", lp_ours, "nats", "analytic_prophet"),
        row("lp___difference", lp_ours - lp_prophet, "nats"),
    ]
    ok = (design_gap < DESIGN_TOLERANCE
          and sigmas_gap < DESIGN_TOLERANCE
          and changepoint_gap < DESIGN_TOLERANCE
          and lp_ours >= lp_prophet - POSTERIOR_TOLERANCE)
    return measurements, ok


def collect(sizes=SIZES, lib_path=None):
    """Every check at every size, plus the single gate_failed the runner reads."""
    lib_path = lib_path or harness.build_extension(tempfile.mkdtemp())
    measurements, failures = [], []
    for size in sizes:
        rows, ok = _compare(size, lib_path)
        measurements += rows
        if not ok:
            failures.append(size)
    measurements.append(Measurement(
        0, "all", "default", "both", "gate_failed", float(bool(failures))))
    if failures:
        print(f"  the gate failed at T = {failures}: the two implementations are "
              f"not fitting the same model")
    return measurements


harness.register(0, "agreement", collect, gate=True)
