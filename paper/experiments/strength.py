"""An unconstrained L-BFGS against the prior's strength (#175).

**This replaces a measurement that can no longer be rerun.** This project's
first backend was liblbfgs, and a sweep of the prior's strength on it found the
iterations before its line search failed tracking the l1 term: 2 at
1/tau = 20, 3528 at 1/tau = 1e-6. liblbfgs was never vendored here -- it was a
system library -- and the backend was replaced by OWL-QN and then by the split
reformulation, so nothing reruns that sweep. Its numbers survive only in prose,
and the paper does not quote them.

The same question is asked here of an L-BFGS that is installed: SciPy's
L-BFGS-B (Byrd et al. 1995), given no bounds, on the natural parameterization
(k, m, delta, beta) -- the prior entering as |delta| / tau and its gradient as
sign(delta) / tau with sign(0) = 0, Stan's convention -- through this
implementation's own objective and gradient, from Prophet's initialization.

**sigma_obs is held at Prophet's own estimate**, so that at every strength the
problem is convex with a unique minimiser (#169's argument) and a shortfall has
one meaning. Then the prediction is linear in x = (k, m, delta, beta),

    y_hat = Z x,    Z = [t, 1, a_j(t) (t - s_j), X],

and the objective is exactly a quadratic plus the l1 term,

    F(x) = x'Qx / 2 - c'x + lambda * sum_j |delta_j| + const,
    Q = Z'Z / sigma^2 + diag(1/sigma_k^2, 1/sigma_m^2, 0, 1/sigma_beta^2),
    c = Z'y / sigma^2.

**The reference is that problem's exact minimiser**, by an active-set method
(feature-sign search: Lee, Battle, Raina & Ng 2007), which solves a linear
system per sign pattern and so is exact to the linear algebra at any strength.
This implementation's split Newton is not used for it: as lambda falls the
direction (delta+ + t, delta- + t) costs only 2 lambda t with no curvature, and
the method crawls. Every reference is certified by Stan's gradient of
Prophet's compiled density, so the active-set code does not grade itself.

    lambda    1/tau over 1e-6 ... 1e2, half a decade apart, plus Prophet's
              default 20
    runs      default: SciPy's tolerances and Prophet's budget of 10^4
              iterations; none: ftol = gtol = 0 and a budget of 10^5, so that
              the run ends only when an iteration makes no progress at all

Peyton Manning at the lengths of the lp tables, Prophet's changepoints.

    python paper/experiments/strength.py

Writes `paper/results/strength.csv` and its `.meta.json`.
"""
import argparse
import csv
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kkt                                          # noqa: E402  (sets up paths)
import harness                                      # noqa: E402
import multistart                                   # noqa: E402
import stan_runner as runner                        # noqa: E402

RESULTS = PAPER / "results" / "strength.csv"
LENGTHS = (300, 1000, 2905)
LAMBDAS = tuple(sorted({*(float(v) for v in np.round(np.logspace(-6, 2, 17), 12)), 20.0}))
DEFAULT_LAMBDA = 20.0
RUNS = {"default": {"maxiter": int(1e4)},
        "none": {"maxiter": int(1e5), "ftol": 0.0, "gtol": 0.0}}
FIELDS = ["observations", "lam", "run", "iterations", "evaluations", "message", "lp",
          "shortfall", "residual", "exact_zeros", "reference_lp", "reference_residual",
          "reference_zeros", "reference_patterns"]


def quadratic(model, sigma):
    """(Q, c, penalized): F at fixed sigma as x'Qx/2 - c'x + lambda |x[penalized]|."""
    layout = model.layout
    A, X = model._design_matrices()
    t, s = np.asarray(model.t, dtype=float), np.asarray(model.changepoints_t, dtype=float)
    Z = np.column_stack([t, np.ones_like(t), A * (t[:, None] - s[None, :]), X])
    prior = np.concatenate([[1 / model.sigma_k ** 2, 1 / model.sigma_m ** 2],
                            np.zeros(layout.n_changepoints),
                            1 / np.asarray(model.sigmas, dtype=float) ** 2])
    Q = Z.T @ Z / sigma ** 2 + np.diag(prior)
    c = Z.T @ np.asarray(model.y_scaled, dtype=float) / sigma ** 2
    penalized = np.zeros(c.size, dtype=bool)
    penalized[2:2 + layout.n_changepoints] = True
    return Q, c, penalized


def exact(Q, c, penalized, lam, patterns=10000):
    """min x'Qx/2 - c'x + lam |x[penalized]|, Q positive definite. Returns (x, patterns).

    Feature-sign search: hold a sign for each nonzero penalized coordinate,
    solve the quadratic those signs make smooth, and walk towards its solution
    only as far as the lowest objective along the way, which is either the
    solution or a point where a coordinate changes sign (and leaves the active
    set). When the active coordinates are optimal, the zero coordinate that
    violates |dF/dx_j| <= lam most enters, with the sign that descends.
    """
    def objective(x):
        return 0.5 * x @ Q @ x - c @ x + lam * np.abs(x[penalized]).sum()

    x = np.zeros(c.size)
    active = ~penalized
    x[active] = np.linalg.solve(Q[np.ix_(active, active)], c[active])
    theta = np.zeros(c.size)
    for count in range(1, patterns + 1):
        gradient = Q @ x - c
        idle = penalized & (x == 0.0)
        violation = np.where(idle & ~active, np.abs(gradient), 0.0)
        smooth_ok = np.all(np.abs(gradient[active & ~penalized]) <= 1e-9 * (1 + np.abs(c).max()))
        signed = active & penalized & (x != 0.0)
        signs_ok = np.all(np.abs(gradient[signed] + lam * np.sign(x[signed]))
                          <= 1e-9 * (1 + np.abs(c).max()))
        if smooth_ok and signs_ok:
            j = int(np.argmax(violation))
            if violation[j] <= lam:
                return x, count
            active[j] = True
            theta[j] = -np.sign(gradient[j])
        index = np.flatnonzero(active)
        target = np.linalg.solve(Q[np.ix_(index, index)], c[index] - lam * theta[index])
        current = x[index]
        crossings = [0.0 if a == 0.0 else a / (a - b) for a, b, p in
                     zip(current, target, penalized[index]) if p and a * b < 0]
        candidates = [1.0] + [t for t in crossings if 0.0 < t < 1.0]
        best, best_value = None, np.inf
        for step in candidates:
            trial = x.copy()
            trial[index] = current + step * (target - current)
            for i, a, b in zip(index, current, target):
                if penalized[i] and a * b < 0 and np.isclose(step, a / (a - b), rtol=0, atol=1e-15):
                    trial[i] = 0.0
            value = objective(trial)
            if value < best_value:
                best, best_value = trial, value
        x = best
        leaving = active & penalized & (x == 0.0)
        active &= ~leaving
        theta = np.where(penalized, np.sign(x), 0.0)
        theta[active & penalized & (x == 0.0)] = 0.0
    raise RuntimeError(f"no optimal sign pattern after {patterns} patterns")


class Problem(multistart.Problem):
    """multistart's: Prophet's data, inits, sigma and changepoints at one length."""

    def model(self, lam):
        from analytic_prophet import AnalyticProphet

        model = AnalyticProphet(**{**harness.PROPHET_KWARGS, "changepoint_prior_scale": 1 / lam})
        model.set_changepoints = lambda: setattr(model, "changepoints_t", self.changepoints_t.copy())
        model.preprocess(self.df)          # history, scaling and designs; no fit
        return model

    def score(self, point, lam):
        tau = 1 / lam
        lp, frame = runner.log_prob("original", dict(self.data, tau=tau), point)
        row = frame.iloc[0]
        delta = np.asarray(point["delta"], dtype=float)
        gradient = {"k": row["g_k"], "m": row["g_m"],
                    "delta": np.array([row[f"g_delta.{j + 1}"] for j in range(len(delta))]),
                    "beta": np.array([row[c] for c in frame.columns if c.startswith("g_beta")])}
        return lp, kkt.summarise(kkt.certificate(gradient, delta, tau), delta)["norm_relaxed"]


def _point(x, layout, sigma):
    return {"k": float(x[0]), "m": float(x[1]), "delta": x[2:2 + layout.n_changepoints],
            "sigma_obs": sigma, "beta": x[2 + layout.n_changepoints:]}


def measure(size):
    """Every strength and both runs at one length, as rows."""
    from scipy.optimize import minimize

    from analytic_prophet.layout import from_dict_to_array

    warnings.simplefilter("ignore")
    problem = Problem(size)
    sigma = problem.sigma
    rows = []
    for lam in LAMBDAS:
        model = problem.model(lam)
        layout = model.layout
        design = model._design_matrices()
        Q, c, penalized = quadratic(model, sigma)
        reference, patterns = exact(Q, c, penalized, lam)
        reference_point = _point(reference, layout, sigma)
        reference_lp, reference_residual = problem.score(reference_point, lam)

        start = from_dict_to_array(model.calculate_initial_params(), layout)
        keep = np.ones(start.size, dtype=bool)
        keep[layout.sigma_obs_idx] = False

        def full(x):
            out = np.empty(start.size)
            out[keep] = x
            out[layout.sigma_obs_idx] = sigma
            return out

        def objective(x):
            return model._minus_log_posterior(full(x), design=design)

        def gradient(x):
            return model._gradient(full(x), design=design)[keep]

        for run, options in RUNS.items():
            result = minimize(objective, start[keep], jac=gradient, method="L-BFGS-B",
                              options={"maxfun": 10 * options["maxiter"], **options})
            point = _point(result.x, layout, sigma)
            lp, residual = problem.score(point, lam)
            rows.append({"observations": size, "lam": lam, "run": run,
                         "iterations": int(result.nit), "evaluations": int(result.nfev),
                         "message": str(result.message), "lp": lp,
                         "shortfall": reference_lp - lp, "residual": residual,
                         "exact_zeros": int(np.sum(point["delta"] == 0.0)),
                         "reference_lp": reference_lp,
                         "reference_residual": reference_residual,
                         "reference_zeros": int(np.sum(reference_point["delta"] == 0.0)),
                         "reference_patterns": patterns})
    return rows


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("lam", "lp", "shortfall", "residual", "reference_lp", "reference_residual"):
            row[key] = float(row[key])
        for key in ("observations", "iterations", "evaluations", "exact_zeros",
                    "reference_zeros", "reference_patterns"):
            row[key] = int(row[key])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=min(len(LENGTHS), os.cpu_count() or 1))
    args = parser.parse_args(argv)
    runner.cmdstan_path()
    runner.model("original")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for chunk in pool.map(measure, LENGTHS) for row in chunk]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(float(value)) if isinstance(value, (float, np.floating))
                                   else value) for key, value in row.items()})
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"lengths": list(LENGTHS), "lambdas": list(LAMBDAS),
                                        "runs": RUNS}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
