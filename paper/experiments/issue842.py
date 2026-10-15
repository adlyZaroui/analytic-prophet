"""The prophet#842 reproducer, rerun on current software (#172).

Issue #842 reported a ten-point weekly series on which Prophet's L-BFGS stalled
indefinitely (Prophet 0.4, rstan 2.17). The maintainer diagnosed it as
under-identification -- "the model parameters are not very well specified by so
little data" -- and added the rule that runs Newton below 100 observations. The
reproducer is ported here exactly: the same dates, values, regressor at full
precision, `n_changepoints=2` and multiplicative seasonality. With weekly data
Prophet's own rules disable every seasonality, so the regressor is the only
multiplicative column.

Recorded, all scored by Stan's density of Prophet's compiled program:

  * **current Prophet** -- its default rule (Newton, since T = 10) and L-BFGS
    forced, with iterations, CmdStan's stated reason for stopping, wall time,
    and a timeout that would catch a stall;
  * **this implementation** on the same data and changepoints, by its default
    rule and by L-BFGS, and from 40 random starts -- the model is not convex,
    so the best of those is the reference optimum and their spread maps the
    local optima -- with how many saddle points its Newton path stepped off on
    the way (#193);
  * **the smoothed program** (#168) fitted by Stan's L-BFGS across eps -- the
    control that separates the kink from the flat direction;
  * at each point, the minimum-norm subgradient over every parameter, sigma
    included (strict and, as in #170, relaxed), the smallest eigenvalue of the
    curvature in (k, m, beta, sigma) -- negative at a saddle -- and the angle
    between its shortfall and the softest direction of the curvature at the
    optimum, which is under-identification's signature;
  * **the variants the thread reports as stopping the stall**, each under both
    of Prophet's algorithms.

    python paper/experiments/issue842.py

Writes `paper/results/issue842.csv`, `paper/results/issue842_variants.csv` and
`paper/results/issue842.meta.json`.
"""
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
import harness                                      # noqa: E402
import smoothing                                    # noqa: E402
import stan_runner as runner                        # noqa: E402

RESULTS = PAPER / "results" / "issue842.csv"
VARIANTS_RESULTS = PAPER / "results" / "issue842_variants.csv"

# The reprex of prophet#842, verbatim: R parses these literals to the same
# doubles Python does.
XREG = [-0.76537379485893120012, -0.74533153677032704109, -0.62494542934172692127,
        -0.39778106465657125934, -0.06976829815746909968, 0.34070346369915599504,
        0.80422096949408772292, 1.28317620368430906907, 1.73567381412022903042,
        2.12006230186317567998]
Y = [1, 1, 0, 1, 0, 1, 0, 2, 2, 0]
START = "2018-09-10"

EPSILONS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
STARTS = 40
TIMEOUT = 120          # seconds; the thread's stall never returned at all
CAP = 10_000           # Prophet's iteration budget

POINT_FIELDS = ["point", "algorithm", "epsilon", "iterations", "stopped_by", "seconds",
                "lp", "shortfall", "k", "m", "delta_1", "delta_2", "beta", "sigma_obs",
                "residual", "residual_relaxed", "cos_softest", "min_eigenvalue",
                "saddle_escapes"]
VARIANT_FIELDS = ["variant", "algorithm", "iterations", "stopped_by", "seconds", "lp",
                  "best_lp", "shortfall", "stalled"]


def frame(xreg=XREG, y=Y):
    return pd.DataFrame({"ds": pd.date_range(START, periods=len(y), freq="7D"),
                         "y": y, "xreg": xreg})


def prophet_model(n_changepoints=2, mode="multiplicative", regressor=True):
    from prophet import Prophet
    model = Prophet(seasonality_mode=mode, n_changepoints=n_changepoints)
    if regressor:
        model.add_regressor("xreg")
    return model


def ours_model(n_changepoints=2, mode="multiplicative", regressor=True):
    from analytic_prophet import AnalyticProphet
    model = AnalyticProphet(seasonality_mode=mode, n_changepoints=n_changepoints)
    if regressor:
        model.add_regressor("xreg")
    return model


def _stan(variant, data, inits, algorithm, epsilon=None):
    """One Stan fit through the #183 runner: (point, iterations, reason, seconds)."""
    started = time.perf_counter()
    try:
        params, result = runner.fit(variant, data, inits, epsilon=epsilon, algorithm=algorithm,
                                    sig_figs=18, seed=harness.SEED, timeout=TIMEOUT,
                                    newton_fallback=False)
    except TimeoutError:
        return None, CAP, f"timed out after {TIMEOUT} s", time.perf_counter() - started
    iterations, reason = _parse_log(result.runset.stdout_files[0])
    return params, iterations, reason, time.perf_counter() - started


def _parse_log(path):
    """(iterations, reason) from CmdStan's optimize log, for either algorithm.

    L-BFGS prints a table and states its stopping test. Newton prints
    "Iteration N. ... Improved by d." and stops, without saying why, once
    |d| <= 1e-8 (stan/services/optimize/newton.hpp), which is what is recorded.
    """
    _, iterations, _, reason = smoothing._parse_stdout(path)
    if reason:
        return iterations, reason
    newton = [line for line in Path(path).read_text().splitlines()
              if line.strip().startswith("Iteration ")]
    if newton:
        last = newton[-1].strip()
        return int(last.split()[1].rstrip(".")), "improvement in lp__ at most 1e-8"
    return iterations, "terminated"


def _ours(df, changepoints_t, lib_path, algorithm=None, initial_params=None, **model):
    fitted = ours_model(**model)
    fitted.set_changepoints = lambda: setattr(fitted, "changepoints_t", changepoints_t.copy())
    kwargs = {"lib_path": lib_path}
    if algorithm:
        kwargs["algorithm"] = algorithm
    if initial_params is not None:
        kwargs["initial_params"] = initial_params
    fitted.fit(df, **kwargs)
    return {"k": fitted.params["k"][0][0], "m": fitted.params["m"][0][0],
            "delta": np.asarray(fitted.params["delta"][0], dtype=float),
            "sigma_obs": float(fitted.sigma_obs),
            "beta": np.asarray(fitted.params["beta"][0], dtype=float)}, fitted.optimizer_used, \
        fitted.opt.saddle_escapes


def _beta_size(df, changepoints_t, lib_path, **model):
    """How many coefficients this implementation's layout has for the model."""
    return np.ravel(_ours(df, changepoints_t, lib_path, **model)[0]["beta"]).size


def _random_starts(count, n_changepoints, n_beta, seed=harness.SEED):
    rng = np.random.default_rng(seed)
    return [{"k": float(rng.normal(0, 1)), "m": float(rng.normal(0.5, 0.5)),
             "delta": rng.normal(0, 0.1, n_changepoints), "beta": rng.normal(0, 1, n_beta),
             "sigma_obs": float(rng.uniform(0.1, 1.0))} for _ in range(count)]


def _padded(data, point):
    """`point` with beta padded to Prophet's K. With no seasonal column Prophet
    adds a dummy all-zero one and this implementation adds none; a zero
    coefficient on a zero column changes nothing in the density."""
    beta = np.ravel(point["beta"])
    missing = int(data["K"]) - beta.size
    return dict(point, beta=np.concatenate([beta, np.zeros(missing)]) if missing > 0 else beta)


def _score(data, point, relaxed=False):
    """(lp__, minimum-norm subgradient norm over every parameter, sigma included).

    `relaxed` grants every rate the zero reading, as in #170: points that leave
    a rate at 1e-10 rather than 0 are not then charged 1/tau for it."""
    point = _padded(data, point)
    lp, frame_ = runner.log_prob("original", data, point)
    row = frame_.iloc[0]
    delta = np.asarray(point["delta"], dtype=float)
    gradient = {"k": row["g_k"], "m": row["g_m"],
                "delta": np.array([row[f"g_delta.{j + 1}"] for j in range(len(delta))]),
                "beta": np.array([row[c] for c in frame_.columns if c.startswith("g_beta")])}
    cert = kkt.certificate(gradient, delta, float(data["tau"]))
    full = np.concatenate([cert["k"], cert["m"], cert["relaxed" if relaxed else "strict"],
                           cert["beta"], [-row["g_sigma_obs"]]])
    return lp, float(np.linalg.norm(full))


def _softest(data, point, step=1e-5):
    """The softest direction of the curvature in (k, m, beta, sigma) at `point`,
    with delta held where it is: central differences of Stan's gradient."""
    def gradient(v):
        q = dict(point, k=v[0], m=v[1], beta=[v[2]], sigma_obs=v[3])
        row = runner.log_prob("original", data, q)[1].iloc[0]
        return -np.array([row["g_k"], row["g_m"], row["g_beta.1"], row["g_sigma_obs"]])
    v = np.array([point["k"], point["m"], np.ravel(point["beta"])[0], point["sigma_obs"]])
    H = np.empty((4, 4))
    for i in range(4):
        e = np.zeros(4)
        e[i] = step
        H[:, i] = (gradient(v + e) - gradient(v - e)) / (2 * step)
    values, vectors = np.linalg.eigh(0.5 * (H + H.T))
    return values, vectors[:, 0]


def reproducer(lib_path):
    """Every point on the reprex itself, as rows, plus the curvature summary."""
    df = frame()
    data, inits = runner.prophet_inputs(prophet_model(), df, sig_figs=18)
    changepoints_t = np.asarray(data["t_change"], dtype=float)

    started = time.perf_counter()
    default_model = prophet_model()
    default_model.fit(df)
    default_seconds = time.perf_counter() - started

    points = []
    for algorithm in ("Newton", "LBFGS"):
        point, iterations, reason, seconds = _stan("original", data, inits, algorithm)
        points.append((f"prophet {algorithm}", algorithm, "", iterations, reason,
                       default_seconds if algorithm == "Newton" else seconds, point, ""))
    for algorithm in (None, "LBFGS"):
        point, used, escapes = _ours(df, changepoints_t, lib_path, algorithm)
        points.append((f"ours {'default' if algorithm is None else algorithm}", used, "",
                       "", "", "", point, escapes))
    for epsilon in EPSILONS:
        point, iterations, reason, seconds = _stan("smooth", data, inits, "LBFGS", epsilon)
        points.append((f"smoothed eps={epsilon:g}", "LBFGS", epsilon, iterations, reason,
                       seconds, point, ""))

    starts = _random_starts(STARTS, int(data["S"]), _beta_size(df, changepoints_t, lib_path))
    local = []
    for start in starts:
        point, _, _ = _ours(df, changepoints_t, lib_path, "LBFGS", initial_params=start)
        local.append((_score(data, point)[0], point))
    best_lp, best = max(local, key=lambda item: item[0])
    points.append(("ours best of 40 starts", "LBFGS", "", "", "", "", best, 0))

    eigenvalues, softest = _softest(data, best)
    rows = []
    for name, algorithm, epsilon, iterations, reason, seconds, point, escapes in points:
        lp, residual = _score(data, point)
        relaxed = _score(data, point, relaxed=True)[1]
        shift = np.array([point["k"] - best["k"], point["m"] - best["m"],
                          np.ravel(point["beta"])[0] - np.ravel(best["beta"])[0]])
        cosine = (abs(shift @ softest[:3]) / (np.linalg.norm(shift) * np.linalg.norm(softest[:3]))
                  if np.linalg.norm(shift) > 1e-9 else float("nan"))
        rows.append({"point": name, "algorithm": algorithm, "epsilon": epsilon,
                     "iterations": iterations, "stopped_by": reason,
                     "seconds": seconds, "lp": lp, "shortfall": best_lp - lp,
                     "k": float(point["k"]), "m": float(point["m"]),
                     "delta_1": float(point["delta"][0]), "delta_2": float(point["delta"][1]),
                     "beta": float(np.ravel(point["beta"])[0]),
                     "sigma_obs": float(point["sigma_obs"]), "residual": residual,
                     "residual_relaxed": relaxed, "cos_softest": cosine,
                     "min_eigenvalue": float(_softest(data, point)[0][0]),
                     "saddle_escapes": escapes})
    landscape = sorted({round(lp, 4) for lp, _ in local}, reverse=True)
    counts = {value: sum(1 for lp, _ in local if round(lp, 4) == value) for value in landscape}
    summary = {"eigenvalues": eigenvalues.tolist(), "softest": softest.tolist(),
               "local_optima": [[value, counts[value]] for value in landscape],
               "default_fit_seconds": default_seconds}
    return rows, summary


VARIANTS = {
    "as reported": {},
    "regressor rounded to 15 digits": {"xreg": [round(x, 15) for x in XREG]},
    "regressor rounded to 16 digits": {"xreg": [round(x, 16) for x in XREG]},
    "last y 1 instead of 0": {"y": Y[:-1] + [1]},
    "1 changepoint": {"n_changepoints": 1},
    "3 changepoints": {"n_changepoints": 3},
    "6 changepoints": {"n_changepoints": 6},
    "additive seasonality": {"mode": "additive"},
    "no regressor": {"regressor": False},
}


def variants(lib_path):
    rows = []
    for name, change in VARIANTS.items():
        df = frame(change.get("xreg", XREG), change.get("y", Y))
        model = {key: change[key] for key in ("n_changepoints", "mode", "regressor") if key in change}
        data, inits = runner.prophet_inputs(prophet_model(**model), df, sig_figs=18)
        changepoints_t = np.asarray(data["t_change"], dtype=float)
        size = _beta_size(df, changepoints_t, lib_path, **model)
        best_lp = max(_score(data, _ours(df, changepoints_t, lib_path, "LBFGS",
                                         initial_params=start, **model)[0])[0]
                      for start in _random_starts(10, int(data["S"]), size))
        for algorithm in ("LBFGS", "Newton"):
            point, iterations, reason, seconds = _stan("original", data, inits, algorithm)
            lp = _score(data, point)[0] if point is not None else float("nan")
            rows.append({"variant": name, "algorithm": algorithm, "iterations": iterations,
                         "stopped_by": reason, "seconds": seconds, "lp": lp,
                         "best_lp": best_lp, "shortfall": best_lp - lp,
                         "stalled": point is None or iterations >= CAP})
    return rows


def _write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(float(value)) if isinstance(value, float) else value)
                             for key, value in row.items()})


def main():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    warnings.simplefilter("ignore")
    runner.cmdstan_path()
    lib_path = harness.build_extension(tempfile.mkdtemp())
    rows, summary = reproducer(lib_path)
    _write(RESULTS, POINT_FIELDS, rows)
    _write(VARIANTS_RESULTS, VARIANT_FIELDS, variants(lib_path))
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"summary": summary, "starts": STARTS,
                                        "epsilons": list(EPSILONS), "timeout": TIMEOUT}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    for row in rows:
        print(f"{row['point']:26s} lp {row['lp']:.6f} short {row['shortfall']:.2e} "
              f"relaxed {row['residual_relaxed']:.2e} cos {row['cos_softest']:.3f} "
              f"min-eig {row['min_eigenvalue']:+.3f} "
              f"{row['iterations']} {row['stopped_by']}")
    print("eigenvalues", np.round(summary["eigenvalues"], 4), "local optima", summary["local_optima"])
    print(f"wrote {RESULTS.relative_to(REPO)} and {VARIANTS_RESULTS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
