"""How closely this implementation's objective and gradient match Stan's (#201).

Two figures the paper and the README lean on were not measured at the
precision they were quoted: that this implementation's objective is Prophet's
density "to 1e-12", and that its gradient agrees with Stan's "to 1e-7". This
measures both, at full precision, on Prophet's own compiled program.

**What is compared.** Stan's `lp__` and its gradient from
`CmdStanModel.log_prob` (`jacobian=False`, which is what `optimize` maximizes;
`sig_figs=18`, so the comparison is limited by the arithmetic rather than by
the printout), against this implementation's objective F and its hand-derived
gradient, at the same point, on the same data:

    offset      lp__ + F. Stan's sampling statements drop exactly the constants
                F omits, so the offset should be zero, not merely constant.
    gradient    Stan differentiates on its unconstrained scale. sigma_obs is
                declared `<lower=0>`, so its component is d lp__ / d log sigma
                = sigma * d lp__ / d sigma (`jacobian=False` adds no log-Jacobian
                term); every other parameter is unconstrained.

**The points.** At each length of the lp tables: Prophet's solution, read at
full precision; this implementation's, fitted on Prophet's changepoints; and
seeded random points -- #169's draws -- where the gradient is far from zero in
every coordinate. A solution with a rate exactly at zero exercises sign(0) = 0
on both sides (#170).

    python paper/experiments/agreement.py

Writes `paper/results/agreement.csv` and its `.meta.json`.
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
import multistart                                   # noqa: E402

RESULTS = PAPER / "results" / "agreement.csv"
LENGTHS = (300, 1000, 2905)
RANDOM = 5
FIELDS = ["observations", "point", "lp_stan", "objective_ours", "offset", "offset_relative",
          "gradient_max_abs", "gradient_max_relative", "gradient_sigma_abs", "exact_zeros",
          "design_gap"]


def stan_lp_and_gradient(stan_model, stan_data, point):
    """(lp__, gradient over k, m, delta, beta, gradient in log sigma), at full precision."""
    frame = stan_model.log_prob(
        params={"k": float(point["k"]), "m": float(point["m"]),
                "delta": [float(v) for v in point["delta"]],
                "sigma_obs": float(point["sigma_obs"]),
                "beta": [float(v) for v in point["beta"]]},
        data=stan_data, jacobian=False, sig_figs=18)
    row = frame.iloc[0]
    S, K = len(point["delta"]), len(point["beta"])
    gradient = np.concatenate([[row["g_k"], row["g_m"]],
                               [row[f"g_delta.{j + 1}"] for j in range(S)],
                               [row[f"g_beta.{j + 1}"] for j in range(K)]]).astype(float)
    return float(row["lp__"]), gradient, float(row["g_sigma_obs"])


def ours_objective_and_gradient(model, design, point):
    """(F, -dF over k, m, delta, beta, -sigma * dF/dsigma): ours, on Stan's scale."""
    from analytic_prophet.layout import from_dict_to_array

    layout = model.layout
    x = from_dict_to_array({key: point[key] for key in ("k", "m", "delta", "sigma_obs", "beta")},
                           layout)
    objective = float(model._minus_log_posterior(x, design=design))
    grad = model._gradient(x, design=design)
    gradient = -np.concatenate([grad[[0, 1]], grad[layout.delta], grad[layout.beta]])
    return objective, gradient, float(-grad[layout.sigma_obs_idx] * point["sigma_obs"])


def measure(size, lib_path):
    """Every point at one length, as rows."""
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    df = corpora.peyton_manning(size)
    df = df.assign(ds=pd.to_datetime(df["ds"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stan_model, stan_data, theirs = harness.capture_stan_model(
            Prophet(**harness.PROPHET_KWARGS), df, sig_figs=18)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    ours = AnalyticProphet(**harness.PROPHET_KWARGS)
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=lib_path)
    design = ours._design_matrices()

    X_ours = np.asarray(design[1], dtype=float)
    X_stan = np.asarray(stan_data["X"], dtype=float).reshape(X_ours.shape[0], -1)
    design_gap = (float(np.max(np.abs(X_ours - X_stan), initial=0.0))
                  if X_ours.shape == X_stan.shape else float("inf"))

    points = {
        "prophet": {"k": theirs["k"][0], "m": theirs["m"][0], "delta": theirs["delta"],
                    "sigma_obs": theirs["sigma_obs"][0], "beta": theirs["beta"]},
        "analytic_prophet": {"k": ours.params["k"][0][0], "m": ours.params["m"][0][0],
                             "delta": ours.params["delta"][0], "sigma_obs": ours.sigma_obs,
                             "beta": ours.params["beta"][0]},
    }
    draws = multistart.starts(size, len(changepoints_t), int(stan_data["K"]), RANDOM)
    points.update({f"random {i}": draw for i, draw in enumerate(draws)})

    rows = []
    for name, point in points.items():
        point = {key: (np.asarray(value, dtype=float) if np.ndim(value) else float(value))
                 for key, value in point.items()}
        lp, g_stan, g_sigma_stan = stan_lp_and_gradient(stan_model, stan_data, point)
        objective, g_ours, g_sigma_ours = ours_objective_and_gradient(ours, design, point)
        difference = np.abs(g_stan - g_ours)
        offset = lp + objective
        rows.append({
            "observations": size, "point": name, "lp_stan": lp, "objective_ours": objective,
            "offset": offset, "offset_relative": abs(offset) / abs(lp),
            "gradient_max_abs": float(difference.max()),
            "gradient_max_relative": float(difference.max() / np.abs(g_stan).max()),
            "gradient_sigma_abs": abs(g_sigma_stan - g_sigma_ours),
            "exact_zeros": int(np.sum(point["delta"] == 0.0)),
            "design_gap": design_gap,
        })
    return rows


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in FIELDS[2:]:
            row[key] = float(row[key])
        row["observations"] = int(row["observations"])
        row["exact_zeros"] = int(row["exact_zeros"])
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
        json.dump(harness.run_metadata({"lengths": list(LENGTHS), "random": RANDOM}), handle,
                  indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
