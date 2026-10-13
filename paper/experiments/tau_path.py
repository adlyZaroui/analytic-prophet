"""The solution path in tau, for both implementations (#171).

For a convex l1 problem the solution is continuous in the penalty, and for a
quadratic loss with an l1 penalty -- extra quadratic penalties included -- it is
piecewise linear in lambda = 1/tau. Prophet's MAP problem is exactly that once
sigma_obs is fixed (linear growth, additive seasonality). So a fine sweep of
tau separates two things a single fit cannot: steep but continuous motion of
the optimum, and an optimizer that lands somewhere different each time.

Peyton Manning in full, Prophet's default configuration, two settings:

    fixed   sigma_obs held at this implementation's optimum at tau = 0.05: the
            convex problem, where continuity is guaranteed
    free    sigma_obs estimated, as Prophet's users get it -- continuity is
            then not guaranteed, but it is what prophet#1280 is about

and these paths through each:

    prophet        Stan's L-BFGS from Prophet's own initialization at every tau,
                   as a user refitting with a new changepoint_prior_scale gets
                   (`paper/stan/prophet_fixed_sigma.stan` for the fixed setting)
    ours_exact     the exact solution path: this implementation's Newton on the
                   split reformulation, each tau warm-started from its neighbour
                   and certified by Stan's gradient (#170's relaxed residual)
    ours_default   this implementation's default fit from a cold start (free
                   setting), to say how far it is from the exact path

The grids: lambda from 2 to 200 (`GLOBAL`, 201 points, 2.3% apart); a zoom
around the default lambda = 20 (`ZOOM`, 41 points 0.025 apart); and the largest
step of each path in the zoom bisected four times (`refine`), which is what
tells a jump -- a step that does not shrink -- from steep continuous motion.
Then prophet#1280's change, tau against float32(tau), at tau = 0.01 and 0.05.
At every point it records the rates, lp__, the relaxed residual and the
forecast a year past the end of the series.

    python paper/experiments/tau_path.py

Writes `paper/results/tau_path.csv`, `tau_refine.csv`, `tau_pairs.csv`, the
`.meta.json`, and `paper/figures/tau_path.pdf`.
"""
import argparse
import csv
import json
import logging
import os
import sys
import tempfile
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
import stan_runner as runner                        # noqa: E402

RESULTS = PAPER / "results"
PATH = RESULTS / "tau_path.csv"
REFINE = RESULTS / "tau_refine.csv"
PAIRS = RESULTS / "tau_pairs.csv"
FIGURE = PAPER / "figures" / "tau_path.pdf"

GLOBAL = tuple(float(v) for v in np.geomspace(2.0, 200.0, 201))
ZOOM = tuple(float(v) for v in np.linspace(19.5, 20.5, 41))
LEVELS = 4
PAIR_TAUS = (0.01, 0.05)
HORIZON_DAYS = 365
S = 25

PATH_FIELDS = (["grid", "setting", "path", "lam", "tau", "lp", "residual", "yhat_h", "k",
                "sigma_obs"] + [f"delta_{j + 1}" for j in range(S)])
REFINE_FIELDS = ["setting", "path", "level", "lam_lo", "lam_hi", "width", "step"]
PAIR_FIELDS = ["tau", "tau_float32", "setting", "path", "delta_change", "yhat_change",
               "yhat_relative", "lp_change"]


class Problem:
    """Prophet's data and inits on the series, and the pieces every fit shares."""

    def __init__(self):
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        warnings.simplefilter("ignore")
        self.df = corpora.peyton_manning()
        self.df = self.df.assign(ds=pd.to_datetime(self.df["ds"]))
        self.model = Prophet(**harness.PROPHET_KWARGS)
        self.data, self.inits = runner.prophet_inputs(self.model, self.df, sig_figs=18)
        self.model.uncertainty_samples = 0
        self.changepoints_t = np.asarray(self.data["t_change"], dtype=float)
        self.future = pd.DataFrame({"ds": [self.df["ds"].iloc[-1]
                                           + pd.Timedelta(days=HORIZON_DAYS)]})

    def data_at(self, tau):
        return dict(self.data, tau=float(tau))

    def forecast(self, point):
        """yhat a year past the end, from Prophet's own predict with `point`."""
        self.model.params = {"k": np.array([[point["k"]]]), "m": np.array([[point["m"]]]),
                             "delta": np.asarray(point["delta"], dtype=float)[None, :],
                             "sigma_obs": np.array([[point["sigma_obs"]]]),
                             "beta": np.asarray(point["beta"], dtype=float)[None, :]}
        return float(self.model.predict(self.future)["yhat"].iloc[0])

    def score(self, point, tau):
        """(lp__, relaxed residual) under the original program at this tau."""
        lp, frame = runner.log_prob("original", self.data_at(tau), point)
        row = frame.iloc[0]
        delta = np.asarray(point["delta"], dtype=float)
        gradient = {"k": row["g_k"], "m": row["g_m"],
                    "delta": np.array([row[f"g_delta.{j + 1}"] for j in range(len(delta))]),
                    "beta": np.array([row[c] for c in frame.columns if c.startswith("g_beta")])}
        summary = kkt.summarise(kkt.certificate(gradient, delta, tau), delta)
        return lp, summary["norm_relaxed"]

    def prophet(self, tau, sigma=None):
        variant = "original" if sigma is None else "fixed_sigma"
        params, _ = runner.fit(variant, self.data_at(tau), self.inits, sigma_obs=sigma,
                               sig_figs=18, seed=harness.SEED)
        return params

    def ours(self, tau, sigma=None, start=None, exact=True, lib_path=None):
        from analytic_prophet import AnalyticProphet

        fitted = AnalyticProphet(**{**harness.PROPHET_KWARGS, "changepoint_prior_scale": tau})
        fitted.set_changepoints = lambda: setattr(fitted, "changepoints_t", self.changepoints_t.copy())
        if exact:
            kwargs = {"backend": "python", "algorithm": "Newton"}
            if sigma is not None:
                kwargs["fixed_sigma_obs"] = sigma
            if start is not None:
                kwargs["initial_params"] = {key: (np.array(value) if np.ndim(value) else value)
                                            for key, value in start.items()}
        else:
            kwargs = {"lib_path": lib_path}
        fitted.fit(self.df, **kwargs)
        return {"k": float(fitted.params["k"][0][0]), "m": float(fitted.params["m"][0][0]),
                "delta": np.asarray(fitted.params["delta"][0], dtype=float),
                "sigma_obs": float(fitted.sigma_obs),
                "beta": np.asarray(fitted.params["beta"][0], dtype=float)}


def _row(problem, grid, setting, path, lam, point):
    tau = 1.0 / lam
    lp, residual = problem.score(point, tau)
    row = {"grid": grid, "setting": setting, "path": path, "lam": lam, "tau": tau, "lp": lp,
           "residual": residual, "yhat_h": problem.forecast(point), "k": point["k"],
           "sigma_obs": point["sigma_obs"]}
    row.update({f"delta_{j + 1}": float(point["delta"][j]) for j in range(S)})
    return row


_PROBLEM = None


def _worker_problem():
    global _PROBLEM
    if _PROBLEM is None:
        _PROBLEM = Problem()
    return _PROBLEM


def _cold(job):
    """One independent fit, in a worker: Prophet's, or ours from a cold start."""
    grid, setting, path, lam, sigma, lib_path = job
    problem = _worker_problem()
    if path == "prophet":
        point = problem.prophet(1.0 / lam, sigma)
    else:
        point = problem.ours(1.0 / lam, exact=False, lib_path=lib_path)
    return _row(problem, grid, setting, path, lam, point)


def _walk(problem, lambdas, anchor, sigma):
    """The exact path through `lambdas`, warm-started outward from `anchor`."""
    order = sorted(lambdas)
    start = min(range(len(order)), key=lambda i: abs(order[i] - anchor[0]))
    points = {}
    for direction in (range(start, len(order)), range(start - 1, -1, -1)):
        previous = anchor[1]
        for i in direction:
            point = problem.ours(1.0 / order[i], sigma=sigma, start=previous)
            points[order[i]] = point
            previous = point
    return points


def refine(problem, setting, path, rows, sigma, exact_points=None):
    """Bisect the largest zoom step LEVELS times, following the larger half."""
    def fit(lam, near=None):
        if path == "prophet":
            return problem.prophet(1.0 / lam, sigma)
        return problem.ours(1.0 / lam, sigma=sigma, start=near)

    zoom = sorted((r for r in rows if r["grid"] == "zoom" and r["setting"] == setting
                   and r["path"] == path), key=lambda r: r["lam"])
    deltas = [np.array([r[f"delta_{j + 1}"] for j in range(S)]) for r in zoom]
    steps = [np.max(np.abs(b - a)) for a, b in zip(deltas, deltas[1:])]
    i = int(np.argmax(steps))
    lo, hi = (zoom[i]["lam"], deltas[i]), (zoom[i + 1]["lam"], deltas[i + 1])
    lo_point = exact_points.get(lo[0]) if exact_points else None
    out = [{"setting": setting, "path": path, "level": 0, "lam_lo": lo[0], "lam_hi": hi[0],
            "width": hi[0] - lo[0], "step": float(steps[i])}]
    for level in range(1, LEVELS + 1):
        mid = 0.5 * (lo[0] + hi[0])
        point = fit(mid, near=lo_point)
        mid_delta = point["delta"]
        left, right = np.max(np.abs(mid_delta - lo[1])), np.max(np.abs(hi[1] - mid_delta))
        if left >= right:
            hi = (mid, mid_delta)
        else:
            lo, lo_point = (mid, mid_delta), point
        out.append({"setting": setting, "path": path, "level": level, "lam_lo": lo[0],
                    "lam_hi": hi[0], "width": hi[0] - lo[0], "step": float(max(left, right))})
    return out


def pairs(problem, sigma0, exact_anchor):
    """prophet#1280's change: tau against float32(tau)."""
    out = []
    for tau in PAIR_TAUS:
        tau32 = float(np.float32(tau))
        for setting, sigma in (("free", None), ("fixed", sigma0)):
            for path in ("prophet", "ours_exact"):
                points = []
                for value in (tau, tau32):
                    if path == "prophet":
                        points.append(problem.prophet(value, sigma))
                    else:
                        points.append(problem.ours(value, sigma=sigma, start=exact_anchor))
                y0, y1 = problem.forecast(points[0]), problem.forecast(points[1])
                lp0 = problem.score(points[0], tau)[0]
                lp1 = problem.score(points[1], tau)[0]
                out.append({"tau": tau, "tau_float32": tau32, "setting": setting, "path": path,
                            "delta_change": float(np.max(np.abs(points[1]["delta"] - points[0]["delta"]))),
                            "yhat_change": y1 - y0, "yhat_relative": (y1 - y0) / abs(y0),
                            "lp_change": lp1 - lp0})
    return out


def read(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key not in ("grid", "setting", "path"):
                row[key] = float(value)
    return rows


def _write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(float(value)) if isinstance(value, (float, np.floating))
                                   else value) for key, value in row.items()})


def draw(rows, out=FIGURE):
    """Rates against lambda for both paths and both settings, and the forecast.

    Columns: Prophet's path, the exact path, the zoom around the default with
    both, and the forecast a year ahead. Rows: sigma fixed, sigma free.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 4.0))
    colours = {"prophet": "#c44e52", "ours_exact": "#4c72b0", "ours_default": "#8c8c8c"}
    for r, setting in enumerate(("fixed", "free")):
        def series(grid, path):
            cells = sorted((x for x in rows if x["grid"] == grid and x["setting"] == setting
                            and x["path"] == path), key=lambda x: x["lam"])
            lam = np.array([x["lam"] for x in cells])
            delta = np.array([[x[f"delta_{j + 1}"] for j in range(S)] for x in cells])
            return lam, delta, np.array([x["yhat_h"] for x in cells])

        for c, path, title in ((0, "prophet", "Prophet (Stan L-BFGS)"),
                               (1, "ours_exact", "exact path (this implementation)")):
            lam, delta, _ = series("global", path)
            axis = axes[r, c]
            axis.plot(lam, delta, color=colours[path], linewidth=0.5)
            axis.set_xscale("log")
            axis.axvline(20, color="0.6", linestyle=":", linewidth=0.7)
            axis.set_title(f"{title}, $\\\\sigma$ {setting}", fontsize=7)
            axis.set_xlabel("$\\\\lambda = 1/\\\\tau$")
            if c == 0:
                axis.set_ylabel("changepoint rates $\\\\delta_j$")
        axis = axes[r, 2]
        lam_p, delta_p, _ = series("zoom", "prophet")
        lam_o, delta_o, _ = series("zoom", "ours_exact")
        moving = np.argsort(np.ptp(delta_p, axis=0))[-3:]
        for j in moving:
            axis.plot(lam_p, delta_p[:, j], color=colours["prophet"], linewidth=0.7,
                      marker=".", markersize=2.5)
            axis.plot(lam_o, delta_o[:, j], color=colours["ours_exact"], linewidth=0.9)
        axis.set_title(f"zoom near $\\\\lambda = 20$, $\\\\sigma$ {setting}", fontsize=7)
        axis.set_xlabel("$\\\\lambda$ (linear)")
        axis = axes[r, 3]
        for path, label in (("prophet", "Prophet"), ("ours_exact", "exact path"),
                            ("ours_default", "this implementation, default fit")):
            if not any(x["path"] == path and x["setting"] == setting for x in rows):
                continue
            lam, _, yhat = series("global", path)
            axis.plot(lam, yhat, color=colours[path], linewidth=0.7, label=label)
        axis.set_xscale("log")
        axis.set_title(f"forecast {HORIZON_DAYS} days ahead, $\\\\sigma$ {setting}", fontsize=7)
        axis.set_xlabel("$\\\\lambda = 1/\\\\tau$")
        if r == 1:
            axis.legend(fontsize=5.5, frameon=False)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args(argv)
    if args.draw:
        print(f"wrote {draw(read(PATH)).relative_to(REPO)}")
        return

    runner.cmdstan_path()
    for variant in ("original", "fixed_sigma"):
        runner.model(variant)
    lib_path = harness.build_extension(tempfile.mkdtemp())
    problem = Problem()

    # The anchor: the exact optimum at the default tau, from a cold start.
    anchor = problem.ours(0.05)
    sigma0 = anchor["sigma_obs"]
    anchor_fixed = problem.ours(0.05, sigma=sigma0, start=anchor)

    jobs = []
    for grid, lambdas in (("global", GLOBAL), ("zoom", ZOOM)):
        for lam in lambdas:
            jobs.append((grid, "free", "prophet", lam, None, lib_path))
            jobs.append((grid, "fixed", "prophet", lam, sigma0, lib_path))
            if grid == "global":
                jobs.append((grid, "free", "ours_default", lam, None, lib_path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_cold, jobs, chunksize=4))

    exact = {}
    for setting, sigma, start in (("free", None, anchor), ("fixed", sigma0, anchor_fixed)):
        exact[setting] = {}
        for grid, lambdas in (("global", GLOBAL), ("zoom", ZOOM)):
            points = _walk(problem, lambdas, (20.0, start), sigma)
            exact[setting].update(points)
            rows += [_row(problem, grid, setting, "ours_exact", lam, points[lam])
                     for lam in sorted(points)]
    rows.sort(key=lambda r: (r["grid"], r["setting"], r["path"], r["lam"]))

    refined = []
    for setting, sigma in (("free", None), ("fixed", sigma0)):
        for path in ("prophet", "ours_exact"):
            refined += refine(problem, setting, path, rows, sigma,
                              exact[setting] if path == "ours_exact" else None)

    _write(PATH, PATH_FIELDS, rows)
    _write(REFINE, REFINE_FIELDS, refined)
    _write(PAIRS, PAIR_FIELDS, pairs(problem, sigma0, anchor))
    with open(PATH.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"sigma0": sigma0, "global": [GLOBAL[0], GLOBAL[-1], len(GLOBAL)],
                                        "zoom": [ZOOM[0], ZOOM[-1], len(ZOOM)], "levels": LEVELS,
                                        "horizon_days": HORIZON_DAYS}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {PATH.relative_to(REPO)} ({len(rows)} rows), {REFINE.relative_to(REPO)}, "
          f"{PAIRS.relative_to(REPO)}")
    print(f"wrote {draw(read(PATH)).relative_to(REPO)}")


if __name__ == "__main__":
    main()
