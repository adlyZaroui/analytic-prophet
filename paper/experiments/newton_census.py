"""Prophet's own Newton method on a sample of the census (#208).

The discussion's first question is what a Prophet user can do today. Prophet
already ships a second optimizer, `algorithm="Newton"`, and on the Peyton
Manning series Stan's Newton stops far closer to the optimum than its L-BFGS
(#174). One series is not a recommendation, so this measures it on the random
sample of the census that `rounding.py` draws (`kkt.py --sample`'s rule), with
what a user would weigh: how far each of Prophet's optimizers stops from the
optimum, and how long each takes.

Per series, on the full history, in Prophet's default configuration:

    lbfgs    Prophet as users fit it -- L-BFGS, since every census series has
             at least 100 observations;
    newton   the same, with algorithm="Newton";
    optimum  this implementation on Prophet's changepoints: the corrected
             optimum the shortfalls are measured from.

Every point is scored by Stan's log density of Prophet's compiled program, read
at full precision (`sig_figs=18`), and graded by Stan's gradient with #170's
relaxed residual, so neither implementation grades itself. The wall time of
each Prophet fit is recorded as a user would see it, CmdStan call and all;
fits run in parallel workers, so it is the ratio between the two on one series
that carries the comparison.

    python paper/experiments/newton_census.py

Writes `paper/results/newton_census.csv` and its `.meta.json`.
"""
import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
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

RESULTS = PAPER / "results" / "newton_census.csv"
SAMPLE = 200
FIELDS = ["series", "frequency", "observations", "lp_optimum", "lp_lbfgs", "lp_newton",
          "shortfall_lbfgs", "shortfall_newton", "relaxed_lbfgs", "relaxed_newton",
          "exact_zeros_newton", "seconds_lbfgs", "seconds_newton", "newton_error"]


def _timed_fit(df, **fit_kwargs):
    from prophet import Prophet

    start = time.perf_counter()
    stan_model, stan_data, point = harness.capture_stan_model(
        Prophet(**harness.PROPHET_KWARGS), df, sig_figs=18, **fit_kwargs)
    return stan_model, stan_data, point, time.perf_counter() - start


def _graded(stan_model, stan_data, point, tau):
    delta = np.asarray(point["delta"], dtype=float)
    lp, gradient = kkt.stan_gradient(stan_model, stan_data, point["k"], point["m"], delta,
                                     point["sigma_obs"], point["beta"])
    relaxed = kkt.summarise(kkt.certificate(gradient, delta, tau), delta)["norm_relaxed"]
    return lp, relaxed, int(np.sum(delta == 0.0))


def one_series(job):
    from analytic_prophet import AnalyticProphet

    name, frame, frequency, lib_path = job
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = frame.assign(ds=pd.to_datetime(frame["ds"])).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stan_model, stan_data, lbfgs, seconds_lbfgs = _timed_fit(df)
        try:
            _, newton_data, newton, seconds_newton = _timed_fit(df, algorithm="Newton")
            error = ""
        except RuntimeError as failure:            # Prophet re-raises a failed Newton
            newton, seconds_newton, error = None, float("nan"), str(failure).splitlines()[0]
    if newton is not None:
        for key in ("y", "t", "X", "t_change"):
            assert np.array_equal(np.asarray(newton_data[key]), np.asarray(stan_data[key])), key

    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    ours = AnalyticProphet(**harness.PROPHET_KWARGS)
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=lib_path)

    def flat(point):
        return {key: (float(np.ravel(value)[0]) if key in ("k", "m", "sigma_obs")
                      else np.ravel(value)) for key, value in point.items()}

    optimum = {"k": ours.params["k"][0][0], "m": ours.params["m"][0][0],
               "delta": ours.params["delta"][0], "sigma_obs": ours.sigma_obs,
               "beta": ours.params["beta"][0]}
    tau = float(stan_data["tau"])
    lp_optimum, _, _ = _graded(stan_model, stan_data, flat(optimum), tau)
    lp_lbfgs, relaxed_lbfgs, _ = _graded(stan_model, stan_data, flat(lbfgs), tau)
    if newton is None:
        lp_newton = relaxed_newton = float("nan")
        zeros_newton = -1
    else:
        lp_newton, relaxed_newton, zeros_newton = _graded(stan_model, stan_data, flat(newton), tau)
    return {"series": name, "frequency": frequency, "observations": len(df),
            "lp_optimum": lp_optimum, "lp_lbfgs": lp_lbfgs, "lp_newton": lp_newton,
            "shortfall_lbfgs": lp_optimum - lp_lbfgs, "shortfall_newton": lp_optimum - lp_newton,
            "relaxed_lbfgs": relaxed_lbfgs, "relaxed_newton": relaxed_newton,
            "exact_zeros_newton": zeros_newton,
            "seconds_lbfgs": seconds_lbfgs, "seconds_newton": seconds_newton,
            "newton_error": error}


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("observations", "exact_zeros_newton"):
            row[key] = int(row[key])
        for key in FIELDS[3:10] + ["seconds_lbfgs", "seconds_newton"]:
            row[key] = float(row[key])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--sample", type=int, default=SAMPLE,
                        help="series to draw; anything but the default is a trial run")
    args = parser.parse_args(argv)
    manifest = corpora.load_manifest()
    sample = kkt.sample_manifest(manifest, args.sample)
    lib_path = harness.build_extension(tempfile.mkdtemp())
    jobs = [(name, frame, frequency, lib_path) for frequency in kkt.FREQUENCIES
            for name, frame in corpora.m4_census(frequency, sample)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = sorted(pool.map(one_series, jobs), key=lambda row: row["series"])
    if args.sample != SAMPLE:
        for row in rows:
            print({key: row[key] for key in ("series", "observations", "shortfall_lbfgs",
                                             "shortfall_newton", "seconds_lbfgs",
                                             "seconds_newton", "newton_error")})
        return
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(float(value)) if isinstance(value, (float, np.floating))
                                   else value) for key, value in row.items()})
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"sample": SAMPLE, "digest": manifest["digest"],
                                        "workers": args.workers}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} series)")


if __name__ == "__main__":
    main()
