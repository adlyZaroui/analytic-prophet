"""What CmdStan's printed precision does to the certificate (#175).

CmdStan writes the optimum with eight significant digits unless asked for
more, and Prophet does not ask, so `Prophet().params` is a rounded copy of
where L-BFGS stopped. The certificate (#170) reads Prophet's point at full
precision for that reason; this measures how much the rounding would have
moved it.

**The same run, printed twice.** Prophet is fitted on each series as users fit
it, and again with `sig_figs=18`. Prophet's initialization is deterministic
and so is Stan's L-BFGS, so the two runs take the same steps and differ only in
the output format -- which is checked rather than assumed: the default
printout must equal the full-precision point rounded to eight significant
digits. Both points are then graded by Stan's gradient, strict and relaxed.

A random sample of the census (#164), drawn as `kkt.py --sample` draws it.

    python paper/experiments/rounding.py

Writes `paper/results/rounding.csv` and its `.meta.json`.
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

RESULTS = PAPER / "results" / "rounding.csv"
SAMPLE = 200
PRINTED_DIGITS = 8
FIELDS = ["series", "frequency", "observations", "printed_is_rounded",
          "strict_full", "strict_printed", "relaxed_full", "relaxed_printed",
          "strict_change", "relaxed_change", "gradient_change"]


def rounded(values, digits=PRINTED_DIGITS):
    return np.array([float(f"{float(v):.{digits}g}") for v in np.ravel(values)])


def one_series(job):
    from prophet import Prophet

    name, frame, frequency = job
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = frame.assign(ds=pd.to_datetime(frame["ds"])).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stan_model, stan_data, printed = harness.capture_stan_model(
            Prophet(**harness.PROPHET_KWARGS), df)
        _, _, full = harness.capture_stan_model(
            Prophet(**harness.PROPHET_KWARGS), df, sig_figs=18)
    keys = ("k", "m", "delta", "sigma_obs", "beta")
    printed_is_rounded = all(np.array_equal(np.ravel(printed[key]), rounded(full[key]))
                             for key in keys)
    tau = float(stan_data["tau"])
    graded = {}
    for label, point in (("full", full), ("printed", printed)):
        delta = np.asarray(point["delta"], dtype=float)
        _, gradient = kkt.stan_gradient(stan_model, stan_data, point["k"][0], point["m"][0],
                                        delta, point["sigma_obs"][0], point["beta"])
        graded[label] = (kkt.summarise(kkt.certificate(gradient, delta, tau), delta), gradient)
    flat = {label: np.concatenate([[g["k"], g["m"]], g["delta"], g["beta"]])
            for label, (_, g) in graded.items()}
    return {"series": name, "frequency": frequency, "observations": len(df),
            "printed_is_rounded": printed_is_rounded,
            "strict_full": graded["full"][0]["norm_strict"],
            "strict_printed": graded["printed"][0]["norm_strict"],
            "relaxed_full": graded["full"][0]["norm_relaxed"],
            "relaxed_printed": graded["printed"][0]["norm_relaxed"],
            "strict_change": abs(graded["printed"][0]["norm_strict"]
                                 - graded["full"][0]["norm_strict"]),
            "relaxed_change": abs(graded["printed"][0]["norm_relaxed"]
                                  - graded["full"][0]["norm_relaxed"]),
            "gradient_change": float(np.max(np.abs(flat["printed"] - flat["full"])))}


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in FIELDS[4:]:
            row[key] = float(row[key])
        row["observations"] = int(row["observations"])
        row["printed_is_rounded"] = row["printed_is_rounded"] == "True"
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    manifest = corpora.load_manifest()
    sample = kkt.sample_manifest(manifest, SAMPLE)
    jobs = [(name, frame, frequency) for frequency in kkt.FREQUENCIES
            for name, frame in corpora.m4_census(frequency, sample)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = sorted(pool.map(one_series, jobs), key=lambda row: row["series"])
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(float(value)) if isinstance(value, (float, np.floating))
                                   else value) for key, value in row.items()})
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"sample": SAMPLE, "digest": manifest["digest"],
                                        "printed_digits": PRINTED_DIGITS}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} series)")


if __name__ == "__main__":
    main()
