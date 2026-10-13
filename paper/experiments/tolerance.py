"""Prophet's L-BFGS with its stopping tolerances removed (#175).

A shortfall from the optimum invites one reply before any other: Prophet stops
early, and tighter tolerances would close the gap. Stan's L-BFGS has five
stopping tests besides its iteration budget, and this sets them to machine
precision or below and lets it run.

    default   CmdStan's defaults, which is what Prophet runs
    four      tol_rel_obj = 1 (in units of machine epsilon) and tol_obj =
              tol_grad = tol_param = 1e-20, tol_rel_grad left at its default --
              the run docs/non-smooth-objective.md once reported
    five      the same and tol_rel_grad = 1: every test at machine precision,
              so the run ends only on the budget or a failed line search

**Each setting is run twice.** First Stan's L-BFGS alone, through the runner
(#183) with no fall-back and `require_converged=False`, so its endpoint is read
even when it exits abnormally. Then through `Prophet.fit(**kwargs)`, to record
what a user who asked for the same tolerances gets: Prophet falls back to
Newton when L-BFGS exits abnormally, passing the same arguments, and Newton
takes no tolerances.

Peyton Manning at the lengths of the lp tables in Prophet's default
configuration, and at T = 2905 with yearly seasonality alone, the cell the
note's run was made in. Prophet's own program, sigma_obs free -- the problem
Prophet's users fit. The reference is this implementation's optimum, Newton on
the split reformulation, certified by Stan's gradient, as for #169.

    python paper/experiments/tolerance.py

Writes `paper/results/tolerance.csv` and its `.meta.json`.
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

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kkt                                          # noqa: E402  (sets up paths)
import harness                                      # noqa: E402
import margin                                       # noqa: E402
import multistart                                   # noqa: E402
import smoothing                                    # noqa: E402
import stan_runner as runner                        # noqa: E402

RESULTS = PAPER / "results" / "tolerance.csv"
CELLS = (("default", 300), ("default", 1000), ("default", 2905), ("yearly_only", 2905))
_FOUR = {"tol_rel_obj": 1, "tol_obj": 1e-20, "tol_grad": 1e-20, "tol_param": 1e-20}
SETTINGS = {"default": {}, "four": _FOUR, "five": {**_FOUR, "tol_rel_grad": 1}}
FIELDS = ["configuration", "observations", "tolerances", "iterations", "stopped_by", "grad_norm", "lp",
          "shortfall", "residual", "prophet_outcome", "prophet_lp", "prophet_shortfall",
          "reference_lp", "reference_residual"]


def through_prophet(problem, kwargs):
    """What `Prophet.fit(df, **kwargs)` gives: (outcome, params or None)."""
    from prophet import Prophet

    fallback = _Seen("Falling back to Newton")
    logger = logging.getLogger("prophet")
    logger.addHandler(fallback)
    try:
        model = Prophet(**problem.kwargs)
        model.fit(problem.df, sig_figs=18, **kwargs)
    except Exception as error:                       # recorded, not hidden
        return f"raised {type(error).__name__}: {error}", None
    finally:
        logger.removeHandler(fallback)
    params = {key: np.ravel(value) for key, value in model.params.items()}
    point = {"k": float(params["k"][0]), "m": float(params["m"][0]), "delta": params["delta"],
             "sigma_obs": float(params["sigma_obs"][0]), "beta": params["beta"]}
    return ("LBFGS, then Newton" if fallback.seen else "LBFGS"), point


class _Seen(logging.Handler):
    def __init__(self, text):
        super().__init__()
        self.text, self.seen = text, False

    def emit(self, record):
        self.seen |= self.text in record.getMessage()


def measure(cell):
    """Every setting in one (configuration, length) cell, as rows."""
    configuration, size = cell
    logging.getLogger("prophet").setLevel(logging.WARNING)
    warnings.simplefilter("ignore")
    problem = multistart.Problem(size, **margin.CONFIGURATIONS[configuration])
    reference = problem.ours(exact=True)
    reference_lp, reference_residual = problem.score(reference)
    rows = []
    for name, kwargs in SETTINGS.items():
        params, result = runner.fit("original", problem.data, problem.inits, sig_figs=18,
                                    seed=harness.SEED, newton_fallback=False,
                                    require_converged=False, **kwargs)
        _, iterations, grad_norm, reason = smoothing._parse_stdout(result.runset.stdout_files[0])
        lp, residual = problem.score(params)
        outcome, point = through_prophet(problem, kwargs)
        prophet_lp = problem.score(point)[0] if point is not None else float("nan")
        rows.append({"configuration": configuration, "observations": size, "tolerances": name, "iterations": iterations,
                     "stopped_by": reason, "grad_norm": grad_norm, "lp": lp,
                     "shortfall": reference_lp - lp, "residual": residual,
                     "prophet_outcome": outcome, "prophet_lp": prophet_lp,
                     "prophet_shortfall": reference_lp - prophet_lp,
                     "reference_lp": reference_lp, "reference_residual": reference_residual})
    return rows


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("grad_norm", "lp", "shortfall", "residual", "prophet_lp",
                    "prophet_shortfall", "reference_lp", "reference_residual"):
            row[key] = float(row[key])
        row["observations"] = int(row["observations"])
        row["iterations"] = int(row["iterations"])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=min(len(CELLS), os.cpu_count() or 1))
    args = parser.parse_args(argv)
    runner.cmdstan_path()
    runner.model("original")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for chunk in pool.map(measure, CELLS) for row in chunk]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(float(value)) if isinstance(value, (float, np.floating))
                                   else value) for key, value in row.items()})
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"cells": [list(cell) for cell in CELLS],
                                        "settings": SETTINGS}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
