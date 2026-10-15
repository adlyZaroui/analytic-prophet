"""The fix with Prophet's Stan model frozen: polishing from outside Stan (#208).

Prophet's maintainers do not plan to change its Stan model (#207's quotation),
and the fix does not need them to. Stan's own gradient of Prophet's compiled
program is reachable from outside Stan, and the split reformulation needs
nothing else. Stan's gradient of lp__ in delta_j is minus the smooth part's,
minus sign(delta_j)/tau, with sign(0) = 0 (#170). So the smooth part's gradient
is recovered exactly by adding the prior's known term back, and the split
problem's objective and gradient follow from one `log_prob` call.

This runs that step: scipy's L-BFGS-B on the split problem, started from
Prophet's own solution (read at full precision), with every objective and
gradient taken from cmdstanpy's `CmdStanModel.log_prob` on Prophet's
unmodified program. sigma_obs is optimized on Stan's own unconstrained scale,
log(sigma_obs), on which `log_prob` reports its gradient. The settings are the
ones this implementation's Python path uses (#24). On the Peyton Manning
series at the lengths of the lp tables, it records:

    where it ends     lp__, the shortfall from the corrected optimum (this
                      implementation's fit on Prophet's changepoints), the
                      relaxed residual under Stan's gradient (#170), and how
                      many rates are exactly zero;
    what it cost      the number of `log_prob` calls -- each a CmdStan
                      subprocess through cmdstanpy -- and the wall time.

    python paper/experiments/polish.py

Writes `paper/results/polish.csv` and its `.meta.json`.
"""
import argparse
import csv
import json
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kkt                                          # noqa: E402  (sets up paths)
import corpora                                      # noqa: E402
import harness                                      # noqa: E402

RESULTS = PAPER / "results" / "polish.csv"
LENGTHS = (300, 1000, 2905)
FIELDS = ["observations", "calls", "seconds", "iterations", "status",
          "lp_prophet", "lp_polished", "lp_optimum",
          "shortfall_prophet", "shortfall_polished",
          "relaxed_prophet", "relaxed_polished", "relaxed_optimum",
          "exact_zeros_polished", "exact_zeros_optimum"]


class FrozenModel:
    """The split problem's objective and gradient from `log_prob` alone.

    z = (k, m, delta_pos(S), delta_neg(S), zeta, beta(K)), zeta = log(sigma_obs).
    """

    def __init__(self, stan_model, stan_data, S, K):
        self.stan_model, self.stan_data, self.S, self.K = stan_model, stan_data, S, K
        self.tau = float(stan_data["tau"])
        self.calls = 0

    def unpack(self, z):
        S, K = self.S, self.K
        delta = z[2:2 + S] - z[2 + S:2 + 2 * S]
        return z[0], z[1], delta, np.exp(z[2 + 2 * S]), z[3 + 2 * S:3 + 2 * S + K]

    def pack(self, k, m, delta, sigma_obs, beta):
        delta = np.asarray(delta, dtype=float)
        return np.concatenate([[k, m], np.maximum(delta, 0.0), np.maximum(-delta, 0.0),
                               [np.log(sigma_obs)], np.asarray(beta, dtype=float)])

    def __call__(self, z):
        """(Q, grad Q): minus lp__, with the prior's |delta| traded for the split's."""
        S = self.S
        k, m, delta, sigma_obs, beta = self.unpack(z)
        frame = self.stan_model.log_prob(
            params={"k": float(k), "m": float(m), "delta": [float(v) for v in delta],
                    "sigma_obs": float(sigma_obs), "beta": [float(v) for v in beta]},
            data=self.stan_data, jacobian=False, sig_figs=18)
        self.calls += 1
        row = frame.iloc[0]
        lp = float(row["lp__"])
        g_delta = np.array([row[f"g_delta.{j + 1}"] for j in range(S)], dtype=float)
        value = -lp - np.sum(np.abs(delta)) / self.tau + np.sum(z[2:2 + 2 * S]) / self.tau
        # Stan: d lp/d delta_j = -dF/d delta_j - sign(delta_j)/tau, with sign(0) = 0
        smooth = -g_delta - np.sign(delta) / self.tau
        gradient = np.concatenate([
            [-float(row["g_k"]), -float(row["g_m"])],
            smooth + 1.0 / self.tau, -smooth + 1.0 / self.tau,
            [-float(row["g_sigma_obs"])],          # d lp / d log(sigma_obs), jacobian=False
            -np.array([row[f"g_beta.{j + 1}"] for j in range(self.K)], dtype=float)])
        return value, gradient


def polish(stan_model, stan_data, start):
    """L-BFGS-B on the split problem from `start`, through `log_prob` only."""
    from scipy.optimize import minimize

    from analytic_prophet.optimizer import (SCIPY_TOL_REL_OBJ, STAN_MAX_ITERATIONS,
                                            STAN_TOL_GRAD)

    S, K = len(start["delta"]), len(start["beta"])
    model = FrozenModel(stan_model, stan_data, S, K)
    z0 = model.pack(start["k"], start["m"], start["delta"], start["sigma_obs"], start["beta"])
    bounds = [(None, None)] * 2 + [(0.0, None)] * (2 * S) + [(None, None)] * (1 + K)
    begun = time.perf_counter()
    result = minimize(model, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": STAN_MAX_ITERATIONS, "maxfun": 10 * STAN_MAX_ITERATIONS,
                               "ftol": SCIPY_TOL_REL_OBJ, "gtol": STAN_TOL_GRAD})
    seconds = time.perf_counter() - begun
    k, m, delta, sigma_obs, beta = model.unpack(result.x)
    point = {"k": float(k), "m": float(m), "delta": delta, "sigma_obs": float(sigma_obs),
             "beta": beta}
    return point, model.calls, seconds, int(result.nit), str(result.message)


def _graded(stan_model, stan_data, point):
    delta = np.asarray(point["delta"], dtype=float)
    lp, gradient = kkt.stan_gradient(stan_model, stan_data, point["k"], point["m"], delta,
                                     point["sigma_obs"], point["beta"])
    relaxed = kkt.summarise(kkt.certificate(gradient, delta, float(stan_data["tau"])),
                            delta)["norm_relaxed"]
    return lp, relaxed, int(np.sum(delta == 0.0))


def measure(size, lib_path):
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    df = corpora.peyton_manning(size)
    df = df.assign(ds=pd.to_datetime(df["ds"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stan_model, stan_data, theirs = harness.capture_stan_model(
            Prophet(**harness.PROPHET_KWARGS), df, sig_figs=18)
    prophet = {"k": float(theirs["k"][0]), "m": float(theirs["m"][0]),
               "delta": np.asarray(theirs["delta"], dtype=float),
               "sigma_obs": float(theirs["sigma_obs"][0]),
               "beta": np.asarray(theirs["beta"], dtype=float)}
    polished, calls, seconds, iterations, status = polish(stan_model, stan_data, prophet)

    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    ours = AnalyticProphet(**harness.PROPHET_KWARGS)
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=lib_path)
    optimum = {"k": ours.params["k"][0][0], "m": ours.params["m"][0][0],
               "delta": ours.params["delta"][0], "sigma_obs": ours.sigma_obs,
               "beta": ours.params["beta"][0]}

    lp_prophet, relaxed_prophet, _ = _graded(stan_model, stan_data, prophet)
    lp_polished, relaxed_polished, zeros_polished = _graded(stan_model, stan_data, polished)
    lp_optimum, relaxed_optimum, zeros_optimum = _graded(stan_model, stan_data, optimum)
    return {"observations": size, "calls": calls, "seconds": seconds,
            "iterations": iterations, "status": status,
            "lp_prophet": lp_prophet, "lp_polished": lp_polished, "lp_optimum": lp_optimum,
            "shortfall_prophet": lp_optimum - lp_prophet,
            "shortfall_polished": lp_optimum - lp_polished,
            "relaxed_prophet": relaxed_prophet, "relaxed_polished": relaxed_polished,
            "relaxed_optimum": relaxed_optimum,
            "exact_zeros_polished": zeros_polished, "exact_zeros_optimum": zeros_optimum}


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("observations", "calls", "iterations", "exact_zeros_polished",
                    "exact_zeros_optimum"):
            row[key] = int(row[key])
        for key in FIELDS:
            if key not in ("observations", "calls", "iterations", "status",
                           "exact_zeros_polished", "exact_zeros_optimum"):
                row[key] = float(row[key])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS),
                        help="anything but the default is a trial run")
    args = parser.parse_args(argv)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    lib_path = harness.build_extension(tempfile.mkdtemp())
    rows = [measure(size, lib_path) for size in args.lengths]
    if tuple(args.lengths) != LENGTHS:
        for row in rows:
            print(row)
        return
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(float(value)) if isinstance(value, (float, np.floating))
                                   else value) for key, value in row.items()})
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"lengths": list(LENGTHS)}), handle, indent=1,
                  sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
