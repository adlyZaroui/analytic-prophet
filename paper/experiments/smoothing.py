"""Smoothing the prior inside Prophet's own Stan program (#168).

The strongest attribution available: hold Prophet's optimizer, data,
initialization and software stack fixed and vary only the smoothness of the
objective. Prophet's program with |delta| replaced by sqrt(delta^2 + eps^2) - eps
(`paper/stan/prophet_smooth.stan`, #183) is fitted by Stan's own L-BFGS across a
grid of eps, on the series and configuration of the lp__ tables.

**Two quantities, not one.** The naive reading -- "the gap to the l1 optimum
closes as eps grows" -- mixes how well Stan solves the problem it was given with
how far that problem has moved. So at every (T, eps) there are three points:

    A    Stan's L-BFGS on the smoothed program, from Prophet's own inits, with a
         fixed seed, read at full precision
    B    the smoothed problem's own optimum, solved independently and tightly
    L1   the l1 optimum: the split reformulation's Newton solution (#174)

and two curves:

    convergence gap   B - A, how far Stan stops from the problem it was given
    smoothing bias    L1 - B, how far that problem's optimum sits from Prophet's

each scored under both densities -- the smoothed program's and the original's --
so that neither is read with its own ruler alone. Higher lp__ is better, so
every difference is written as (better point) - (worse point).

**How B is found, and how it is checked.** A damped Newton method on
F + sum(sqrt(delta^2 + eps^2) - eps) / tau, where F is this implementation's
smooth part (which equals minus Prophet's lp__ without the prior, #170). The
smoothing term's Hessian is diagonal and exact; F's is central differences of
the analytic gradient. It is started from A and from L1, and the two answers
are recorded. B is certified three ways: Stan's own gradient of the smoothed
program at B, the Newton decrement (the improvement a further full Newton step
predicts), and the agreement of the two starts.

    python paper/experiments/smoothing.py

Writes `paper/results/smoothing.csv` and its `.meta.json`.
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
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
import stan_runner as runner                        # noqa: E402

RESULTS = PAPER / "results" / "smoothing.csv"
LENGTHS = (300, 1000, 2905)
# From effectively l1 to effectively quadratic: at 1e-8 the surrogate differs
# from |delta| by at most 1e-8, at 1 it is quadratic over every rate's range.
EPSILONS = tuple(10.0 ** (half_decade / 2) for half_decade in range(-16, 1))
NEWTON_ITERATIONS = 500

FIELDS = ["observations", "epsilon",
          "lp_smooth_A", "lp_smooth_B", "lp_smooth_L1",
          "lp_l1_A", "lp_l1_B", "lp_l1_L1", "lp_l1_prophet",
          "gap_smooth", "gap_l1", "bias_l1", "bias_smooth", "total_l1",
          "stan_algorithm", "stan_iterations", "stan_grad_norm", "stan_stopped_by",
          "newton_iterations", "newton_decrement", "starts_disagree", "stan_grad_at_B",
          "l1_certificate"]


def _parse_stdout(path):
    """(algorithm, iterations, ||grad||, termination reason) from CmdStan's log.

    The algorithm is read from CmdStan's own echo of its configuration rather
    than inferred, because the runner falls back to Newton when L-BFGS exits
    abnormally and the run should say which one it reports.
    """
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    algorithm = next((line.split("=")[1].split()[0] for line in lines
                      if line.startswith("algorithm =")), "")
    rows = [line.split() for line in lines if re.match(r"^\d+\s+-?[\d.]", line)]
    iterations = int(rows[-1][0]) if rows else 0
    grad = float(rows[-1][3]) if rows else float("nan")
    reason = ""
    for index, line in enumerate(lines):
        if line.startswith("Optimization terminated"):
            reason = " ".join(lines[index:index + 2]).replace("Optimization terminated normally: ", "")
    return algorithm, iterations, grad, reason


def tight_solve(model, design, epsilon, start, iterations=NEWTON_ITERATIONS):
    """The smoothed problem's optimum, by damped Newton. Returns (x, f, its, decrement)."""
    from analytic_prophet.optimizer import finite_difference_hessian

    layout, tau = model.layout, model.changepoint_prior_scale
    delta_index = np.arange(start.size)[layout.delta]

    def smooth_part(x):
        if not x[layout.sigma_obs_idx] > 0:
            return np.inf
        return model._minus_log_posterior(x, include_l1_prior=False, design=design)

    def smooth_gradient(x):
        return model._gradient(x, include_l1_prior=False, design=design)

    def value(x):
        delta = x[layout.delta]
        return smooth_part(x) + np.sum(np.sqrt(delta ** 2 + epsilon ** 2) - epsilon) / tau

    def gradient(x):
        out = smooth_gradient(x).copy()
        delta = x[layout.delta]
        out[layout.delta] += delta / np.sqrt(delta ** 2 + epsilon ** 2) / tau
        return out

    def hessian(x):
        H = finite_difference_hessian(smooth_gradient, x)
        delta = x[layout.delta]
        H[delta_index, delta_index] += epsilon ** 2 / (delta ** 2 + epsilon ** 2) ** 1.5 / tau
        return H

    x = np.array(start, dtype=float)
    fx, damping, iteration = value(x), 1e-8, 0
    for iteration in range(iterations):
        gx = gradient(x)
        if np.max(np.abs(gx)) < 1e-10:
            break
        H = hessian(x)
        improved, alpha = False, 1.0
        for _ in range(60):
            try:
                step = np.linalg.solve(H + damping * np.eye(x.size), -gx)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            alpha = 1.0
            for _ in range(40):
                candidate = x + alpha * step
                fc = value(candidate)
                if fc < fx:
                    improved = True
                    break
                alpha *= 0.5
            if improved:
                break
            damping *= 10
        if not improved:
            break
        damping = max(damping * 0.3, 1e-14) if alpha == 1.0 else damping * 3
        x, fx = candidate, fc
    gx = gradient(x)
    try:
        decrement = float(0.5 * gx @ np.linalg.solve(hessian(x), gx))
    except np.linalg.LinAlgError:
        decrement = float("nan")
    return x, fx, iteration, decrement


def _point(x, layout):
    return {"k": x[0], "m": x[1], "delta": x[layout.delta],
            "sigma_obs": x[layout.sigma_obs_idx], "beta": x[layout.beta]}


def measure(size):
    """Every eps at one length, as rows."""
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet
    from analytic_prophet.layout import split_to_canonical

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    warnings.simplefilter("ignore")
    df = corpora.peyton_manning(size)
    df = df.assign(ds=pd.to_datetime(df["ds"]))
    prophet_model = Prophet(**harness.PROPHET_KWARGS)
    data, inits = runner.prophet_inputs(prophet_model, df, sig_figs=18)
    theirs = {name: np.ravel(value) for name, value in prophet_model.params.items()}

    changepoints_t = np.asarray(data["t_change"], dtype=float)
    model = AnalyticProphet(**harness.PROPHET_KWARGS)
    model.set_changepoints = lambda: setattr(model, "changepoints_t", changepoints_t.copy())
    model.fit(df, backend="python", algorithm="Newton")
    layout, design = model.layout, model._design_matrices()
    l1 = split_to_canonical(model.opt.x, layout.n_changepoints)

    def lp_l1(x):
        return runner.log_prob("original", data, _point(x, layout))

    def lp_smooth(x, epsilon):
        return runner.log_prob("smooth", data, _point(x, layout), epsilon=epsilon)

    lp_star, frame = lp_l1(l1)
    gradient = {"k": frame["g_k"].iloc[0], "m": frame["g_m"].iloc[0],
                "delta": np.array([frame[f"g_delta.{j + 1}"].iloc[0] for j in range(layout.n_changepoints)]),
                "beta": np.array([frame[f"g_beta.{j + 1}"].iloc[0] for j in range(len(l1[layout.beta]))])}
    certificate = kkt.summarise(kkt.certificate(gradient, l1[layout.delta],
                                                float(data["tau"])), l1[layout.delta])["norm_strict"]
    prophet_point = np.r_[theirs["k"], theirs["m"], theirs["delta"], theirs["sigma_obs"], theirs["beta"]]
    lp_prophet = lp_l1(prophet_point)[0]

    rows = []
    for epsilon in EPSILONS:
        A, result = runner.fit("smooth", data, inits, epsilon=epsilon, sig_figs=18,
                               seed=harness.SEED)
        algorithm, iterations, grad_norm, reason = _parse_stdout(result.runset.stdout_files[0])
        xa = np.r_[A["k"], A["m"], A["delta"], A["sigma_obs"], A["beta"]]

        solutions = [tight_solve(model, design, epsilon, start) for start in (xa, l1)]
        xb, fb, newton_iterations, decrement = min(solutions, key=lambda s: s[1])
        lp_smooth_B, frame_B = lp_smooth(xb, epsilon)
        lp_smooth_A, lp_smooth_L1 = lp_smooth(xa, epsilon)[0], lp_smooth(l1, epsilon)[0]
        lp_l1_A, lp_l1_B = lp_l1(xa)[0], lp_l1(xb)[0]
        rows.append({
            "observations": size, "epsilon": epsilon,
            "lp_smooth_A": lp_smooth_A, "lp_smooth_B": lp_smooth_B, "lp_smooth_L1": lp_smooth_L1,
            "lp_l1_A": lp_l1_A, "lp_l1_B": lp_l1_B, "lp_l1_L1": lp_star, "lp_l1_prophet": lp_prophet,
            "gap_smooth": lp_smooth_B - lp_smooth_A, "gap_l1": lp_l1_B - lp_l1_A,
            "bias_l1": lp_star - lp_l1_B, "bias_smooth": lp_smooth_B - lp_smooth_L1,
            "total_l1": lp_star - lp_l1_A,
            "stan_algorithm": algorithm, "stan_iterations": iterations,
            "stan_grad_norm": grad_norm, "stan_stopped_by": reason,
            "newton_iterations": newton_iterations, "newton_decrement": decrement,
            "starts_disagree": abs(solutions[0][1] - solutions[1][1]),
            "stan_grad_at_B": float(np.max(np.abs(frame_B.drop(columns=["lp__"]).to_numpy()))),
            "l1_certificate": certificate,
        })
    return rows


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in FIELDS:
            if key in ("stan_algorithm", "stan_stopped_by"):
                continue
            row[key] = float(row[key])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=min(len(LENGTHS), os.cpu_count() or 1))
    args = parser.parse_args(argv)
    runner.cmdstan_path()
    for variant in ("original", "smooth"):
        runner.model(variant)                 # compile once, before the pool
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for chunk in pool.map(measure, LENGTHS) for row in chunk]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(value) if isinstance(value, float) else value)
                             for key, value in row.items()})
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"lengths": list(LENGTHS), "epsilons": list(EPSILONS),
                                        "cmdstan": runner.CMDSTAN_VERSION}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
