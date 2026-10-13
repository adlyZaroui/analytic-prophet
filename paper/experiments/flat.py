"""The negative control: flat growth, where both implementations agree (#175).

Under `growth="flat"` the trend is the constant m, so the likelihood never sees
k or delta. They remain parameters and keep their priors, nothing pulls against
the prior, and every changepoint rate goes to zero. That leaves the objective
in delta as sum_j |delta_j| / tau alone, which every method minimises exactly,
so the two implementations should reach one point. If they do, they fit the
same model, and the margin under linear growth is a matter of optimizers, not
of specifications. It does not isolate the kink (docs/non-smooth-objective.md,
#149): removing the trend also removes the ill-conditioned k--delta directions.

Tier 0's protocol, as for #173: Prophet fitted on the first T rows of Peyton
Manning, this implementation on Prophet's own changepoints with its default
fit, both scored by Stan's log density of Prophet's compiled program, and
Prophet's optimum read at full precision (`sig_figs=18`).

    python paper/experiments/flat.py

Writes `paper/results/flat.csv` and its `.meta.json`.
"""
import argparse
import csv
import json
import logging
import sys
import tempfile
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

RESULTS = PAPER / "results" / "flat.csv"
LENGTHS = (300, 1000, 2905)
FIELDS = ["observations", "implementation", "lp", "exact_zeros", "changepoints",
          "max_abs_delta", "k", "m", "sigma_obs", "lp_gap", "m_gap", "beta_gap"]


def measure(size, lib_path):
    """Both implementations at one length, as two rows."""
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    kwargs = {**harness.PROPHET_KWARGS, "growth": "flat"}
    df = corpora.peyton_manning(size)
    df = df.assign(ds=pd.to_datetime(df["ds"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stan_model, stan_data, theirs = harness.capture_stan_model(
            Prophet(**kwargs), df, sig_figs=18)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    ours = AnalyticProphet(**kwargs)
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=lib_path)

    points = {
        "prophet": (theirs["k"][0], theirs["m"][0], np.asarray(theirs["delta"], dtype=float),
                    theirs["sigma_obs"][0], np.asarray(theirs["beta"], dtype=float)),
        "analytic_prophet": (ours.params["k"][0][0], ours.params["m"][0][0],
                             np.asarray(ours.params["delta"][0], dtype=float),
                             float(ours.sigma_obs), np.asarray(ours.params["beta"][0], dtype=float)),
    }
    lp = {name: kkt.stan_gradient(stan_model, stan_data, *point)[0]
          for name, point in points.items()}
    gaps = {"lp_gap": abs(lp["analytic_prophet"] - lp["prophet"]),
            "m_gap": abs(float(points["analytic_prophet"][1]) - float(points["prophet"][1])),
            "beta_gap": float(np.max(np.abs(points["analytic_prophet"][4] - points["prophet"][4])))}
    rows = []
    for name, (k, m, delta, sigma_obs, beta) in points.items():
        rows.append({"observations": size, "implementation": name, "lp": lp[name],
                     "exact_zeros": int(np.sum(delta == 0.0)), "changepoints": len(delta),
                     "max_abs_delta": float(np.max(np.abs(delta))), "k": float(k),
                     "m": float(m), "sigma_obs": float(sigma_obs), **gaps})
    return rows


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("lp", "max_abs_delta", "k", "m", "sigma_obs", "lp_gap", "m_gap", "beta_gap"):
            row[key] = float(row[key])
        for key in ("observations", "exact_zeros", "changepoints"):
            row[key] = int(row[key])
    return rows


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    lib_path = harness.build_extension(tempfile.mkdtemp())
    rows = [row for size in LENGTHS for row in measure(size, lib_path)]
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
