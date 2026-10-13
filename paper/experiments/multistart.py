"""Prophet from many random starts, where the optimum is unique (#169).

**A proof of non-convergence that needs no reference implementation.** With
linear growth and additive seasonality the prediction is linear in
(k, m, delta, beta), so with sigma_obs held fixed the negative log posterior is
a convex quadratic plus the convex |delta| / tau: its minimiser is unique, and a
method that converges reaches it from any start. If Prophet's fits from
different starts scatter, Prophet's optimizer is not converging -- whatever
this implementation does.

Peyton Manning at the lengths of the lp__ tables, Prophet's default
configuration (linear growth, additive seasonality). Two settings:

    fixed   sigma_obs held at Prophet's own estimate from its default fit, via
            `paper/stan/prophet_fixed_sigma.stan` (#183): convex, so scatter
            is non-convergence
    free    sigma_obs estimated, Prophet's own program: joint convexity fails,
            so scatter there proves nothing on its own -- reported beside the
            fixed setting for that reason

**The starts**, drawn with `harness.SEED` plus the length, the same for both
settings: k ~ N(0, 1), m ~ N(0, 1) (the data are scaled to [0, 1]),
delta_j ~ Laplace(0, tau) -- Prophet's own prior -- beta_j ~ N(0, 1), and in the
free setting sigma_obs log-uniform on [0.01, 1]. Prophet's own default start is
run as well.

**The reference** is this implementation's optimum, solved by Newton on the
split reformulation from a cold start and certified by Stan's gradient (#170).
A further `OURS` starts of this implementation, from the same draws, show the
same optimum reached from each.

    python paper/experiments/multistart.py

Writes `paper/results/multistart.csv` and its `.meta.json`.
"""
import argparse
import csv
import json
import logging
import os
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
import smoothing                                    # noqa: E402
import stan_runner as runner                        # noqa: E402

RESULTS = PAPER / "results" / "multistart.csv"
LENGTHS = (300, 1000, 2905)
STARTS = 50
OURS = 10
TAU = 0.05
FIELDS = ["observations", "setting", "implementation", "start", "lp", "shortfall",
          "residual", "delta_distance", "theta_distance", "sigma_obs", "exact_zeros",
          "iterations", "stopped_by"]


def starts(size, n_changepoints, n_beta, count=STARTS):
    """The random starts for one length, the same for both settings."""
    rng = np.random.default_rng(harness.SEED + size)
    out = []
    for _ in range(count):
        out.append({"k": float(rng.normal(0, 1)), "m": float(rng.normal(0, 1)),
                    "delta": rng.laplace(0, TAU, n_changepoints),
                    "beta": rng.normal(0, 1, n_beta),
                    "sigma_obs": float(np.exp(rng.uniform(np.log(0.01), np.log(1.0))))})
    return out


class Problem:
    def __init__(self, size):
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        warnings.simplefilter("ignore")
        self.size = size
        self.df = corpora.peyton_manning(size)
        self.df = self.df.assign(ds=pd.to_datetime(self.df["ds"]))
        model = Prophet(**harness.PROPHET_KWARGS)
        self.data, self.inits = runner.prophet_inputs(model, self.df, sig_figs=18)
        self.default = {key: np.ravel(value) for key, value in model.params.items()}
        self.sigma = float(self.default["sigma_obs"][0])
        self.changepoints_t = np.asarray(self.data["t_change"], dtype=float)

    def ours(self, sigma=None, start=None, exact=False):
        from analytic_prophet import AnalyticProphet

        fitted = AnalyticProphet(**harness.PROPHET_KWARGS)
        fitted.set_changepoints = lambda: setattr(fitted, "changepoints_t", self.changepoints_t.copy())
        kwargs = {"backend": "python"}
        if exact:
            kwargs["algorithm"] = "Newton"
        if sigma is not None:
            kwargs["fixed_sigma_obs"] = sigma
        if start is not None:
            kwargs["initial_params"] = dict(start, sigma_obs=sigma if sigma is not None
                                            else start["sigma_obs"])
        fitted.fit(self.df, **kwargs)
        return {"k": float(fitted.params["k"][0][0]), "m": float(fitted.params["m"][0][0]),
                "delta": np.asarray(fitted.params["delta"][0], dtype=float),
                "sigma_obs": float(fitted.sigma_obs),
                "beta": np.asarray(fitted.params["beta"][0], dtype=float)}

    def prophet(self, start, sigma=None):
        """Stan's L-BFGS from `start`; (point, iterations, reason)."""
        variant = "original" if sigma is None else "fixed_sigma"
        inits = {"k": start["k"], "m": start["m"], "delta": list(start["delta"]),
                 "beta": list(start["beta"])}
        if sigma is None:
            inits["sigma_obs"] = start["sigma_obs"]
        params, result = runner.fit(variant, self.data, inits, sigma_obs=sigma,
                                    sig_figs=18, seed=harness.SEED)
        _, iterations, _, reason = smoothing._parse_stdout(result.runset.stdout_files[0])
        return params, iterations, reason

    def score(self, point):
        lp, frame = runner.log_prob("original", self.data, point)
        row = frame.iloc[0]
        delta = np.asarray(point["delta"], dtype=float)
        gradient = {"k": row["g_k"], "m": row["g_m"],
                    "delta": np.array([row[f"g_delta.{j + 1}"] for j in range(len(delta))]),
                    "beta": np.array([row[c] for c in frame.columns if c.startswith("g_beta")])}
        return lp, kkt.summarise(kkt.certificate(gradient, delta, TAU), delta)["norm_relaxed"]


def _row(problem, setting, implementation, index, point, reference, iterations="", reason=""):
    lp, residual = problem.score(point)
    theta = lambda p: np.concatenate([[p["k"], p["m"]], np.ravel(p["beta"])])
    return {"observations": problem.size, "setting": setting, "implementation": implementation,
            "start": index, "lp": lp, "shortfall": reference["lp"] - lp, "residual": residual,
            "delta_distance": float(np.max(np.abs(np.asarray(point["delta"]) - reference["delta"]))),
            "theta_distance": float(np.max(np.abs(theta(point) - theta(reference)))),
            "sigma_obs": float(point["sigma_obs"]),
            "exact_zeros": int(np.sum(np.asarray(point["delta"]) == 0.0)),
            "iterations": iterations, "stopped_by": reason}


def measure(size):
    """Every start at one length, both settings, as rows."""
    problem = Problem(size)
    n_beta = int(problem.data["K"])
    draws = starts(size, int(problem.data["S"]), n_beta)
    rows = []
    for setting, sigma in (("fixed", problem.sigma), ("free", None)):
        reference = problem.ours(sigma=sigma, exact=True)
        reference["lp"] = problem.score(reference)[0]
        rows.append(_row(problem, setting, "reference", "", reference, reference))
        default = {"k": problem.inits["k"], "m": problem.inits["m"],
                   "delta": np.asarray(problem.inits["delta"], dtype=float),
                   "beta": np.asarray(problem.inits["beta"], dtype=float),
                   "sigma_obs": problem.inits["sigma_obs"]}
        for index, start in [("default", default)] + list(enumerate(draws)):
            point, iterations, reason = problem.prophet(start, sigma)
            rows.append(_row(problem, setting, "prophet", index, point, reference,
                             iterations, reason))
        for index, start in enumerate(draws[:OURS]):
            point = problem.ours(sigma=sigma, start=start)
            rows.append(_row(problem, setting, "analytic_prophet", index, point, reference))
    return rows


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("lp", "shortfall", "residual", "delta_distance", "theta_distance",
                    "sigma_obs"):
            row[key] = float(row[key])
        row["observations"] = int(row["observations"])
        row["exact_zeros"] = int(row["exact_zeros"])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=min(len(LENGTHS), os.cpu_count() or 1))
    args = parser.parse_args(argv)
    runner.cmdstan_path()
    for variant in ("original", "fixed_sigma"):
        runner.model(variant)
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
        json.dump(harness.run_metadata({"lengths": list(LENGTHS), "starts": STARTS,
                                        "ours": OURS, "tau": TAU}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
