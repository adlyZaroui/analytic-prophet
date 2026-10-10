"""One Newton method, two parameterizations (#174).

The paper's in-house control for the kink: **the same second-order code, the
same start, the same damping and stopping tests, varying only how the
changepoint rates are represented.**

    natural   (k, m, delta, sigma_obs, beta). The Laplace prior enters as
              |delta| / tau and its gradient as the subgradient sign(delta) / tau,
              with sign(0) = 0 -- Stan's convention (#170).
    split     delta = delta_pos - delta_neg, both >= 0, so the prior is linear.
              This is what `fit(backend="python", algorithm="Newton")` runs, and
              the script checks it lands exactly where that does.

Both go through `analytic_prophet.optimizer.projected_newton`, unmodified: the
natural arm with no bounds but the one on sigma_obs, which both arms share. The
start is Prophet's own initialization (`calculate_initial_params`): every rate
at zero, which is on the kink.

**Stan's own Newton is reported beside them**, as a reference rather than a
third arm: Prophet fitted with `algorithm="Newton"`. It is not the same code.
Its Hessian is a four-point finite difference of the gradient with a step of
1e-3 (`stan/model/grad_hess_log_prob.hpp`), against about 1.5e-8 here, its
damping flips the sign of negative eigenvalues instead of adding to the
diagonal, and its stopping rule is CmdStan's. It is there so the paper does not
describe Stan's Newton by a measurement of a different one.

Every point is scored by Stan's density of Prophet's compiled program, fitted on
Prophet's own changepoints -- Tier 0's protocol, and #173's.

    python paper/experiments/newton.py

Writes `paper/results/newton.csv` (one row per length and arm),
`paper/results/newton_traces.csv` (the objective at every iteration of both
in-house arms) and `newton.meta.json`.
"""
import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kkt                                          # noqa: E402  (sets up paths)
import corpora                                      # noqa: E402
import harness                                      # noqa: E402

RESULTS = PAPER / "results" / "newton.csv"
TRACES = PAPER / "results" / "newton_traces.csv"

# Short lengths, where Prophet itself runs Newton, up to the whole series.
LENGTHS = (50, 99, 150, 300, 1000, 2905)
ARMS = ("split", "natural", "stan_newton")
# The tail over which per-step progress is summarized: long enough to be a
# rate, short enough to describe how a run ends rather than how it began.
TAIL = 100

FIELDS = ["observations", "arm", "iterations", "stopped_by", "lp", "shortfall",
          "median_progress_tail", "sign_flips", "coordinates_flipped",
          "exact_zeros", "min_abs_delta", "matches_shipped"]
TRACE_FIELDS = ["observations", "arm", "iteration", "objective"]


class _Iterates:
    """The accepted iterates of a `projected_newton` run, read off its calls.

    projected_newton evaluates the gradient once at the start, then per
    iteration 2n times for the Hessian and once at the accepted point -- so
    every (2n + 1)-th call after the first is an iterate. That pattern is the
    shipped code's, not a guess: `run_arm` checks every recovered iterate
    against the run's own loss trace and refuses to report otherwise.
    """

    def __init__(self, gradient_fn, n):
        self.gradient_fn, self.n, self.calls, self.points = gradient_fn, n, 0, []

    def __call__(self, x):
        index = self.calls
        self.calls += 1
        if index == 0 or (index - 1) % (2 * self.n + 1) == 2 * self.n:
            self.points.append(np.array(x, dtype=float))
        return self.gradient_fn(x)


def _stopped_by(result, iterates, objective):
    """Which of Stan's tests ended the run, read off its last step."""
    from analytic_prophet.optimizer import (STAN_EPS, STAN_TOL_OBJ, STAN_TOL_PARAM,
                                            STAN_TOL_REL_OBJ)
    if result.status != 0:
        return "iteration cap"
    if result.message != "converged":
        return result.message
    if len(iterates) < 2:
        return "gradient"
    before, after = iterates[-2], iterates[-1]
    old, new = objective(before), objective(after)
    tests = []
    if old - new < STAN_TOL_OBJ:
        tests.append("tol_obj")
    if (old - new) / max(abs(old), abs(new), 1.0) < STAN_TOL_REL_OBJ * STAN_EPS:
        tests.append("tol_rel_obj")
    if np.max(np.abs(after - before)) < STAN_TOL_PARAM:
        tests.append("tol_param")
    return "+".join(tests) or "gradient"


def run_arm(objective, gradient_fn, x0, lower, upper, to_canonical, delta_slice):
    """One projected_newton run, with its iterates recovered and checked."""
    from analytic_prophet.optimizer import projected_newton

    recorder = _Iterates(gradient_fn, np.asarray(x0).size)
    result = projected_newton(objective, recorder, x0, lower, upper)
    trace = np.asarray(result.loss_trace)
    iterates = recorder.points
    if len(iterates) != len(trace) or not all(
            objective(point) == value for point, value in zip(iterates, trace)):
        raise RuntimeError("recovered iterates do not reproduce the loss trace; "
                           "projected_newton's call pattern has changed")
    deltas = np.array([to_canonical(point)[delta_slice] for point in iterates])
    flips = (np.sign(deltas[1:]) * np.sign(deltas[:-1])) < 0
    progress = -np.diff(trace)[-TAIL:]
    final = to_canonical(result.x)
    return {
        "result": result,
        "canonical": final,
        "trace": trace,
        "iterations": int(result.nit),
        "stopped_by": _stopped_by(result, iterates, objective),
        "median_progress_tail": float(np.median(progress)) if progress.size else 0.0,
        "sign_flips": int(flips.sum()),
        "coordinates_flipped": int(flips.any(axis=0).sum()),
        "exact_zeros": int(np.sum(final[delta_slice] == 0.0)),
        "min_abs_delta": float(np.min(np.abs(final[delta_slice]))),
    }


def measure(size, lib_path=None):
    """Every arm at one length. Returns (rows, trace rows)."""
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet
    from analytic_prophet.layout import (canonical_to_split, from_dict_to_array,
                                         split_to_canonical)

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = corpora.peyton_manning(size)
    df = df.assign(ds=pd.to_datetime(df["ds"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stan_model, stan_data, _ = harness.capture_stan_model(
            Prophet(**harness.PROPHET_KWARGS), df, sig_figs=18)
        _, _, stan_newton = harness.capture_stan_model(
            Prophet(**harness.PROPHET_KWARGS), df, algorithm="Newton", sig_figs=18)

    # The shipped split Newton, which also leaves the model prepared: its
    # history, scaling and design matrices are what both arms evaluate.
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    model = AnalyticProphet(**harness.PROPHET_KWARGS)
    model.set_changepoints = lambda: setattr(model, "changepoints_t", changepoints_t.copy())
    model.fit(df, backend="python", algorithm="Newton")
    shipped = float(model.opt.fun)

    layout = model.layout
    design = model._design_matrices()
    start = from_dict_to_array(model.calculate_initial_params(), layout)
    n_delta = layout.n_changepoints
    sigma_floor = 1e-6                    # the split path's own bound on sigma_obs

    natural_lower = np.full(start.size, -np.inf)
    natural_lower[layout.sigma_obs_idx] = sigma_floor
    split_start = canonical_to_split(start, layout)
    split_lower = np.full(split_start.size, -np.inf)
    split_lower[2:2 + 2 * n_delta] = 0.0
    split_lower[2 + 2 * n_delta] = sigma_floor

    arms = {
        "natural": run_arm(
            lambda x: model._minus_log_posterior(x, design=design),
            lambda x: model._gradient(x, design=design),
            start, natural_lower, np.full(start.size, np.inf),
            lambda x: x, layout.delta),
        "split": run_arm(
            lambda z: model._split_minus_log_posterior(z, design=design),
            lambda z: model._split_gradient(z, design=design),
            split_start, split_lower, np.full(split_start.size, np.inf),
            lambda z: split_to_canonical(z, n_delta), layout.delta),
    }

    def score(k, m, delta, sigma_obs, beta):
        return kkt.stan_gradient(stan_model, stan_data, k, m, delta, sigma_obs, beta)[0]

    lp = {}
    for name, arm in arms.items():
        x = arm["canonical"]
        lp[name] = score(x[0], x[1], x[layout.delta], x[layout.sigma_obs_idx], x[layout.beta])
    lp["stan_newton"] = score(stan_newton["k"][0], stan_newton["m"][0], stan_newton["delta"],
                              stan_newton["sigma_obs"][0], stan_newton["beta"])

    rows, traces = [], []
    for name in ARMS:
        row = {"observations": size, "arm": name, "lp": lp[name],
               "shortfall": lp["split"] - lp[name]}
        if name in arms:
            arm = arms[name]
            row.update({key: arm[key] for key in (
                "iterations", "stopped_by", "median_progress_tail", "sign_flips",
                "coordinates_flipped", "exact_zeros", "min_abs_delta")})
            row["matches_shipped"] = (name == "split" and float(arm["result"].fun) == shipped)
            traces += [{"observations": size, "arm": name, "iteration": i,
                        "objective": float(value)} for i, value in enumerate(arm["trace"])]
        else:
            delta = np.asarray(stan_newton["delta"], dtype=float)
            row.update({"iterations": "", "stopped_by": "CmdStan", "median_progress_tail": "",
                        "sign_flips": "", "coordinates_flipped": "",
                        "exact_zeros": int(np.sum(delta == 0.0)),
                        "min_abs_delta": float(np.min(np.abs(delta))),
                        "matches_shipped": ""})
        rows.append(row)
    return rows, traces


def collect(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(measure, LENGTHS))
    rows = [row for chunk, _ in results for row in chunk]
    traces = [row for _, chunk in results for row in chunk]
    return rows, traces


def _write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(value) if isinstance(value, float) else value)
                             for key, value in row.items()})


def read(path=RESULTS):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int,
                        default=min(len(LENGTHS), max(1, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args(argv)
    rows, traces = collect(args.workers)
    _write(RESULTS, FIELDS, rows)
    _write(TRACES, TRACE_FIELDS, traces)
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"lengths": list(LENGTHS), "tail": TAIL}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} rows) and "
          f"{TRACES.relative_to(REPO)} ({len(traces)} iterations)")


if __name__ == "__main__":
    main()
