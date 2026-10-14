"""The fix inside Stan: two smooth reformulations on Stan's own L-BFGS (#202).

A reviewer will ask why the paper's fix needs a solver outside Stan. Two
reformulations of Prophet's program are smooth and keep the minimisers in
delta, so both run on Stan's own L-BFGS with Prophet's settings and nothing
else changed (`paper/stan/`, registered in `stan_runner`):

    hadamard   delta = delta_u .* delta_v, and the Laplace prior's |delta| / tau
               replaced by (|delta_u|^2 + |delta_v|^2) / (2 tau) (Hoff 2017). The
               minimum of (u^2 + v^2) / 2 over u v = delta is |delta|, so the
               minimisers in delta are unchanged; the objective is smooth and
               unconstrained, though no longer convex in (u, v).
    split      delta = delta_pos - delta_neg, both `<lower=0>`, the prior linear
               in them: the paper's reformulation, as Stan has to state it. Stan
               optimizes a bounded parameter through its logarithm, so a rate at
               zero is approached as that logarithm goes to minus infinity and is
               never reached.

**The question is what each reaches.** Peyton Manning at the lengths of the lp
tables, with sigma_obs held at Prophet's estimate -- convex in delta, so the
optimum is unique (#169) -- and free, Prophet's actual problem. Each variant
starts from Prophet's own initialization and from the first ten of #169's
seeded starts.

**Where a rate starts at zero, each variant needs a choice Prophet does not
make**, and the result depends on it, so it is measured rather than fixed:

    hadamard   a rate at zero starts at (delta_u, delta_v) = (c, 0), since (0, 0)
               is a stationary point; c in sqrt(tau) / 10, sqrt(tau), 1
    split      both parts start at a floor, since Stan cannot start a bounded
               parameter on its bound; floor in tau / 50, tau, 1

Prophet's own start, where every rate is zero, is run at each value; the
random starts, whose rates are never zero, use the middle one (the defaults of
`stan_runner.hadamard_factors` and `split_parts`).

Every fit is scored under Prophet's own density, the original program, and
measured against #169's reference: this implementation's Newton on the split,
certified by Stan's gradient (#170). Per fit: the shortfall, the relaxed
residual, how many rates are exactly zero and how many fall below 1e-8 in
magnitude, the largest |delta_j| among the rates the optimum puts at zero, and
how the run ended -- including whether Prophet's rule fell back to Newton.

    python paper/experiments/inside_stan.py

Writes `paper/results/inside_stan.csv` and its `.meta.json`.
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
import issue842                                     # noqa: E402
import multistart                                   # noqa: E402
import smoothing                                    # noqa: E402
import stan_runner as runner                        # noqa: E402

RESULTS = PAPER / "results" / "inside_stan.csv"
LENGTHS = (300, 1000, 2905)
STARTS = 10
FAMILIES = {"hadamard": {"fixed": "fixed_sigma_hadamard", "free": "hadamard"},
            "split": {"fixed": "fixed_sigma_split", "free": "split"}}
SMALL = 1e-8
# The value a rate at zero starts from, as a multiple of what it is scaled by:
# sqrt(tau) for hadamard's c, tau for split's floor. The middle one is the
# default the random starts use.
SCALES = {"hadamard": (lambda tau: np.sqrt(tau) / 10, lambda tau: np.sqrt(tau),
                       lambda tau: 1.0),
          "split": (lambda tau: tau / 50, lambda tau: tau, lambda tau: 1.0)}
FIELDS = ["observations", "setting", "variant", "start", "scale", "lp", "shortfall", "residual",
          "exact_zeros", "small_rates", "reference_zeros", "largest_on_zero_set",
          "delta_distance", "algorithm", "iterations", "stopped_by"]


def _row(size, setting, variant, start, point, problem, reference, algorithm="",
         iterations="", reason="", scale=""):
    lp, residual = problem.score(point)
    delta = np.asarray(point["delta"], dtype=float)
    zero = np.asarray(reference["delta"], dtype=float) == 0.0
    return {"observations": size, "setting": setting, "variant": variant, "start": start,
            "scale": scale, "lp": lp, "shortfall": reference["lp"] - lp, "residual": residual,
            "exact_zeros": int(np.sum(delta == 0.0)),
            "small_rates": int(np.sum(np.abs(delta) < SMALL)),
            "reference_zeros": int(np.sum(zero)),
            "largest_on_zero_set": float(np.max(np.abs(delta[zero]), initial=0.0)),
            "delta_distance": float(np.max(np.abs(delta - reference["delta"]))),
            "algorithm": algorithm, "iterations": iterations, "stopped_by": reason}


def measure(size):
    """Every variant, setting and start at one length, as rows."""
    warnings.simplefilter("ignore")
    problem = multistart.Problem(size)
    draws = multistart.starts(size, int(problem.data["S"]), int(problem.data["K"]))[:STARTS]
    default = {"k": problem.inits["k"], "m": problem.inits["m"],
               "delta": np.asarray(problem.inits["delta"], dtype=float),
               "beta": np.asarray(problem.inits["beta"], dtype=float),
               "sigma_obs": problem.inits["sigma_obs"]}
    rows = []
    for setting, sigma in (("fixed", problem.sigma), ("free", None)):
        reference = problem.ours(sigma=sigma, exact=True)
        reference["lp"] = problem.score(reference)[0]
        rows.append(_row(size, setting, "reference", "", reference, problem, reference))
        tau = float(problem.data["tau"])
        for family, variants in FAMILIES.items():
            runs = [("default", default, scale(tau)) for scale in SCALES[family]]
            runs += [(index, start, SCALES[family][1](tau)) for index, start in enumerate(draws)]
            for index, start, scale in runs:
                inits = {"k": start["k"], "m": start["m"], "beta": list(start["beta"])}
                if sigma is None:
                    inits["sigma_obs"] = start["sigma_obs"]
                delta = np.asarray(start["delta"], dtype=float)
                if family == "hadamard":
                    u, v = runner.hadamard_factors(delta)
                    inits["delta_u"] = np.where(delta == 0, scale, u).tolist()
                    inits["delta_v"] = np.where(delta == 0, 0.0, v).tolist()
                else:
                    positive, negative = runner.split_parts(delta, scale)
                    inits["delta_pos"], inits["delta_neg"] = positive.tolist(), negative.tolist()
                params, result = runner.fit(variants[setting], problem.data, inits,
                                            sigma_obs=sigma, sig_figs=18, seed=harness.SEED)
                path = result.runset.stdout_files[0]
                algorithm = smoothing._parse_stdout(path)[0]
                iterations, reason = issue842._parse_log(path)
                rows.append(_row(size, setting, family, index, params, problem, reference,
                                 algorithm, iterations, reason, scale))
    return rows


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("lp", "shortfall", "residual", "largest_on_zero_set", "delta_distance"):
            row[key] = float(row[key])
        row["scale"] = float(row["scale"]) if row["scale"] else None
        for key in ("observations", "exact_zeros", "small_rates", "reference_zeros"):
            row[key] = int(row[key])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=min(len(LENGTHS), os.cpu_count() or 1))
    args = parser.parse_args(argv)
    runner.cmdstan_path()
    for variants in FAMILIES.values():
        for variant in variants.values():
            runner.model(variant)
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
        json.dump(harness.run_metadata({"lengths": list(LENGTHS), "starts": STARTS,
                                        "families": FAMILIES, "small": SMALL,
                                        "scales": {"hadamard": "c = sqrt(tau)/10, sqrt(tau), 1",
                                                   "split": "floor = tau/50, tau, 1"}}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
