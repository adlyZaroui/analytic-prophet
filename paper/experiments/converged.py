"""How far this implementation's default fit stops from the converged optimum (#240).

#208's census run found Prophet's own Newton scoring above the corrected optimum
on 15 of 200 series, by up to 0.0088 nats. The corrected optimum is this
implementation's default fit: the compiled L-BFGS-B on the split problem,
stopped by Stan's own tests. This measures what those tests leave on the table,
on the same random sample of the census (`kkt.py --sample`'s rule), and what it
would cost to take.

Per series, on the full history, in Prophet's default configuration and on
Prophet's changepoints:

    default    `fit(df)` as a user runs it, with its wall time and iterations;
    converged  the compiled projected Newton, started from the default fit's
               point (`fit(df, algorithm="Newton", initial_params=...)`), with
               its own wall time and the Newton steps it accepted.

The steps are counted from the loss trace, which gains one entry per accepted
step and one per saddle escape. The core's `n_iterations` leaves out a last
step that Stan's objective or parameter tests end the run on, so it would read
0 where one step was taken.

Both points are scored by Stan's log density of Prophet's compiled program and
graded by #170's relaxed residual under Stan's gradient, with how many rates
are exactly zero.

    python paper/experiments/converged.py

Writes `paper/results/converged.csv` and its `.meta.json`.
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

RESULTS = PAPER / "results" / "converged.csv"
SAMPLE = 200
FIELDS = ["series", "frequency", "observations",
          "lp_default", "lp_converged", "gain",
          "relaxed_default", "relaxed_converged",
          "exact_zeros_default", "exact_zeros_converged",
          "iterations_default", "steps_converged", "escapes_converged",
          "seconds_default", "seconds_converged", "status_converged"]


def _point(model):
    return {"k": float(model.params["k"][0][0]), "m": float(model.params["m"][0][0]),
            "delta": np.asarray(model.params["delta"][0], dtype=float),
            "sigma_obs": float(model.sigma_obs),
            "beta": np.asarray(model.params["beta"][0], dtype=float)}


def _graded(stan_model, stan_data, point):
    delta = point["delta"]
    lp, gradient = kkt.stan_gradient(stan_model, stan_data, point["k"], point["m"], delta,
                                     point["sigma_obs"], point["beta"])
    relaxed = kkt.summarise(kkt.certificate(gradient, delta, float(stan_data["tau"])),
                            delta)["norm_relaxed"]
    return lp, relaxed, int(np.sum(delta == 0.0))


def one_series(job):
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    name, frame, frequency, lib_path = job
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = frame.assign(ds=pd.to_datetime(frame["ds"])).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stan_model, stan_data, _ = harness.capture_stan_model(
            Prophet(**harness.PROPHET_KWARGS), df, sig_figs=18)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    def fitted(**fit_kwargs):
        model = AnalyticProphet(**harness.PROPHET_KWARGS)
        model.set_changepoints = lambda: setattr(model, "changepoints_t", changepoints_t.copy())
        start = time.perf_counter()
        model.fit(df, lib_path=lib_path, **fit_kwargs)
        return model, time.perf_counter() - start

    default, seconds_default = fitted()
    start = _point(default)
    converged, seconds_converged = fitted(algorithm="Newton", initial_params=start)

    lp_default, relaxed_default, zeros_default = _graded(stan_model, stan_data, start)
    lp_converged, relaxed_converged, zeros_converged = _graded(stan_model, stan_data,
                                                               _point(converged))
    return {"series": name, "frequency": frequency, "observations": len(df),
            "lp_default": lp_default, "lp_converged": lp_converged,
            "gain": lp_converged - lp_default,
            "relaxed_default": relaxed_default, "relaxed_converged": relaxed_converged,
            "exact_zeros_default": zeros_default, "exact_zeros_converged": zeros_converged,
            "iterations_default": int(default.opt.n_iterations),
            "steps_converged": (len(converged.opt.loss_trace) - 1
                                - int(converged.opt.saddle_escapes)),
            "escapes_converged": int(converged.opt.saddle_escapes),
            "seconds_default": seconds_default, "seconds_converged": seconds_converged,
            "status_converged": converged.opt_status_message}


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("observations", "exact_zeros_default", "exact_zeros_converged",
                    "iterations_default", "steps_converged", "escapes_converged"):
            row[key] = int(row[key])
        for key in ("lp_default", "lp_converged", "gain", "relaxed_default",
                    "relaxed_converged", "seconds_default", "seconds_converged"):
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
            print({key: row[key] for key in ("series", "observations", "gain", "relaxed_default",
                                             "relaxed_converged", "iterations_default",
                                             "steps_converged", "seconds_default",
                                             "seconds_converged", "status_converged")})
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
