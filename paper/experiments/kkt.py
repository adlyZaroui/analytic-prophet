"""A first-order optimality certificate at both solutions, per series (#170).

"Our `lp__` is higher" is a comparison between two points. This is a property
of each point on its own: does it satisfy the first-order conditions of the
problem Prophet specifies? Under the convex configuration -- linear growth,
additive seasonality, `sigma_obs` held fixed -- those conditions are necessary
and sufficient, so a residual near zero certifies an optimum and a large one
certifies its absence, with no second optimizer to agree with.

**The instrument is Stan's own gradient.** Neither implementation's code is
trusted to grade itself: `CmdStanModel.log_prob` evaluates the density Prophet
compiled, and its gradient, at each point. Writing F = -lp__ as a smooth part
s plus sum_j |delta_j| / tau, the minimum-norm element of the subdifferential
of F is

    k, m, beta        ds/dx                                      (differentiable)
    delta_j != 0      ds/ddelta_j + sign(delta_j) / tau          (strict)
    delta_j == 0      soft_threshold(ds/ddelta_j, 1 / tau)

and Stan's gradient of lp__ at delta_j carries -sign(delta_j) / tau, with
sign(0) = 0 -- which `tests/test_paper_kkt.py` checks, because the exact-zero
case depends on it.

**sigma_obs is held at each solution's own value**, not optimized over. With it
free the problem is not jointly convex and the conditions are necessary only.
That costs the argument nothing in one direction: a point optimal for the full
problem is stationary in (k, m, delta, beta) at its own sigma, and with sigma
fixed that block is convex, so a non-zero residual proves the point is not
optimal even holding its own sigma.

**Prophet's point is read at full precision.** CmdStan writes the optimum with
eight significant digits unless told otherwise, and Prophet does not tell it
otherwise, so `Prophet().params` is a rounded copy of where L-BFGS stopped.
Rounding alone perturbs the gradient -- by up to about 0.2 on the series tried,
against violations of 2 to 10 -- so the fit here asks for all eighteen, and
what is graded is the optimizer rather than its output format.

**A relaxed residual answers the obvious objection.** Prophet's rates are
never exactly zero under linear growth, so wherever one is tiny the strict
condition demands ds/ddelta_j = -sign(delta_j) / tau exactly, and a residual of
order 1/tau = 20 can be read as an artifact of a rate that "should have been"
zero. The relaxed residual gives every coordinate the zero reading at once:
soft_threshold(ds/ddelta_j, 1/tau) for all j. Since sign(delta_j) / tau is an
endpoint of [-1/tau, 1/tau], it is never larger than the strict one, it needs
no threshold for what counts as zero, and it is still a lower bound on the
distance from 0 to the subdifferential at the point. Whatever survives it is
not about the kink.

    python paper/experiments/kkt.py                 # the census, resumable
    python paper/experiments/kkt.py --sample 40     # a random sample, for timing
    python paper/experiments/kkt.py --detail        # Peyton Manning, per coordinate

Writes `paper/results/kkt_census.csv` (one row per series and implementation),
its `.meta.json`, and `paper/results/kkt_peyton_manning.csv`.
"""
import argparse
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
RESULTS = PAPER / "results"
for path in (REPO, REPO / "evaluation", REPO / "benchmark"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import corpora                                      # noqa: E402
import harness                                      # noqa: E402

FREQUENCIES = ("Weekly", "Daily")
IMPLEMENTATIONS = ("analytic_prophet", "prophet")
FIELDS = ["series", "frequency", "implementation", "observations", "changepoints",
          "lp", "sigma_obs", "exact_zeros",
          "residual_k", "residual_m", "residual_beta",
          "residual_delta_strict", "residual_delta_relaxed",
          "norm_strict", "norm_relaxed",
          "argmax_strict", "argmax_relaxed", "delta_at_argmax_strict",
          "design_gap"]
DETAIL_FIELDS = ["coordinate", "implementation", "value", "smooth_gradient",
                 "residual_strict", "residual_relaxed"]

# The design matrices are shown identical by Tier 0's gate on Peyton Manning;
# here the gap is recorded on every series instead of assumed, because beta
# crosses from one implementation to the other untransformed, and
# tests/test_paper_kkt.py holds the committed results to this.
DESIGN_TOLERANCE = 1e-9


def soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def certificate(gradient, delta, tau):
    """The minimum-norm subgradient of F = -lp__, from Stan's gradient of lp__.

    `gradient` maps "k", "m", "delta", "beta" to Stan's d lp__ / dx at the
    point (sigma_obs is not among them: it is held fixed). Returns per-block
    arrays: "k", "m", "beta" (differentiable, so the plain gradient of F),
    "smooth_delta" (ds/ddelta, the prior's kink removed), "strict" and
    "relaxed" (the two readings of the delta block, see the module docstring).
    """
    delta = np.asarray(delta, dtype=float)
    stan_delta = np.asarray(gradient["delta"], dtype=float)
    # lp__ carries -|delta_j|/tau, whose derivative Stan takes as
    # -sign(delta_j)/tau with sign(0) = 0; removing it leaves the smooth part
    smooth_delta = -stan_delta - np.sign(delta) / tau
    strict = np.where(delta != 0.0, -stan_delta, soft_threshold(smooth_delta, 1.0 / tau))
    return {
        "k": np.array([-float(gradient["k"])]),
        "m": np.array([-float(gradient["m"])]),
        "beta": -np.asarray(gradient["beta"], dtype=float),
        "smooth_delta": smooth_delta,
        "strict": strict,
        "relaxed": soft_threshold(smooth_delta, 1.0 / tau),
    }


def _coordinates(n_delta, n_beta):
    return (["k", "m"] + [f"delta[{j + 1}]" for j in range(n_delta)]
            + [f"beta[{j + 1}]" for j in range(n_beta)])


def summarise(cert, delta):
    """One point's certificate as the numbers a row reports, located."""
    delta = np.asarray(delta, dtype=float)
    names = _coordinates(len(delta), len(cert["beta"]))
    out = {
        "exact_zeros": int(np.sum(delta == 0.0)),
        "residual_k": float(abs(cert["k"][0])),
        "residual_m": float(abs(cert["m"][0])),
        "residual_beta": float(np.max(np.abs(cert["beta"]))) if len(cert["beta"]) else 0.0,
        "residual_delta_strict": float(np.max(np.abs(cert["strict"]))),
        "residual_delta_relaxed": float(np.max(np.abs(cert["relaxed"]))),
    }
    for reading in ("strict", "relaxed"):
        full = np.concatenate([cert["k"], cert["m"], cert[reading], cert["beta"]])
        out[f"norm_{reading}"] = float(np.linalg.norm(full))
        where = int(np.argmax(np.abs(full)))
        out[f"argmax_{reading}"] = names[where]
        if reading == "strict":
            out["delta_at_argmax_strict"] = (float(delta[where - 2])
                                             if names[where].startswith("delta") else "")
    return out


def stan_gradient(stan_model, stan_data, k, m, delta, sigma_obs, beta):
    """(lp__, gradient) under Prophet's compiled density, at full precision.

    jacobian=False because that is what `optimize` maximizes. sig_figs is
    raised from CmdStan's default of eight so that the residuals reported are
    limited by the arithmetic rather than by the output format.
    """
    frame = stan_model.log_prob(
        params={"k": float(k), "m": float(m),
                "delta": [float(v) for v in delta],
                "sigma_obs": float(sigma_obs),
                "beta": [float(v) for v in beta]},
        data=stan_data, jacobian=False, sig_figs=18)
    row = frame.iloc[0]
    gradient = {
        "k": row["g_k"], "m": row["g_m"],
        "delta": np.array([row[f"g_delta.{j + 1}"] for j in range(len(delta))]),
        "beta": np.array([row[f"g_beta.{j + 1}"] for j in range(len(beta))]),
    }
    return float(row["lp__"]), gradient


def _fit_both(df, lib_path):
    """Both implementations on one series, ours on Prophet's changepoints.

    Returns (stan_model, stan_data, {implementation: point}, design_gap). The
    changepoints are handed across, as Tier 0 does, so delta indexes the same
    breakpoints in both and the comparison is of optimizers, not of
    specifications.
    """
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # sig_figs reaches `optimize` through Prophet's own args.update(kwargs).
        # It changes what CmdStan prints, not what it computes: by default the
        # optimum is written to eight significant digits, and that rounding alone
        # moves a residual by up to about 0.2 -- small beside the violations,
        # but the certificate is of where the optimizer stopped, not of its
        # printout, so it is read at full precision.
        stan_model, stan_data, theirs = harness.capture_stan_model(
            Prophet(**harness.PROPHET_KWARGS), df, sig_figs=18)
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    ours = AnalyticProphet(**harness.PROPHET_KWARGS)
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=lib_path)

    X_ours = np.asarray(ours.make_all_seasonality_features(df)[0], dtype=float)
    X_stan = np.asarray(stan_data["X"], dtype=float).reshape(X_ours.shape[0], -1)
    if X_ours.shape != X_stan.shape:
        design_gap = float("inf")
    else:
        design_gap = float(np.max(np.abs(X_ours - X_stan), initial=0.0))

    points = {
        "prophet": (theirs["k"][0], theirs["m"][0], theirs["delta"],
                    theirs["sigma_obs"][0], theirs["beta"]),
        "analytic_prophet": (ours.params["k"][0][0], ours.params["m"][0][0],
                             ours.params["delta"][0], ours.sigma_obs,
                             ours.params["beta"][0]),
    }
    return stan_model, stan_data, points, design_gap


def one_series(name, df, frequency, lib_path):
    """Both certificates for one series, as two rows."""
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = df.assign(ds=pd.to_datetime(df["ds"])).reset_index(drop=True)
    stan_model, stan_data, points, design_gap = _fit_both(df, lib_path)
    tau = float(stan_data["tau"])
    rows = []
    for implementation in IMPLEMENTATIONS:
        k, m, delta, sigma_obs, beta = points[implementation]
        lp, gradient = stan_gradient(stan_model, stan_data, k, m, delta, sigma_obs, beta)
        rows.append({
            "series": name, "frequency": frequency, "implementation": implementation,
            "observations": len(df), "changepoints": len(delta),
            "lp": lp, "sigma_obs": float(sigma_obs),
            **summarise(certificate(gradient, delta, tau), delta),
            "design_gap": design_gap,
        })
    return rows


def _measure(job):
    name, frame, frequency, lib_path = job
    return name, one_series(name, frame, frequency, lib_path)


def _jobs(manifest, lib_path):
    for frequency in FREQUENCIES:
        for name, frame in corpora.m4_census(frequency, manifest):
            yield name, frame, frequency, lib_path


def sample_manifest(manifest, size, seed=harness.SEED):
    """A uniformly random subset of the census, for timing a run before it.

    Random rather than the head of the file: the census is sorted by
    identifier, and timing its first series once under-estimated Tier 2's run
    by a factor of seven, because they were the short ones (evaluation/README).
    """
    everything = sorted((frequency, identifier)
                        for frequency, identifiers in manifest["series"].items()
                        for identifier in identifiers)
    chosen = random.Random(seed).sample(everything, size)
    series = {}
    for frequency, identifier in sorted(chosen):
        series.setdefault(frequency, []).append(identifier)
    return {"series": series}


def collect(manifest, lib_path, workers, checkpoint=None):
    """Every series in `manifest`, in parallel, resumably; rows sorted."""
    done = {}
    if checkpoint:
        Path(checkpoint).mkdir(parents=True, exist_ok=True)

    def record(name, rows):
        done[name] = rows
        if checkpoint:
            (Path(checkpoint) / f"{name}.json").write_text(json.dumps(rows))
        if len(done) % 100 == 0:
            print(f"    {len(done)} series done", flush=True)

    pending = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job in _jobs(manifest, lib_path):
            cached = Path(checkpoint) / f"{job[0]}.json" if checkpoint else None
            if cached and cached.exists():
                done[job[0]] = json.loads(cached.read_text())
                continue
            while len(pending) >= workers * 2:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    record(*future.result())
            pending.add(pool.submit(_measure, job))
        for future in as_completed(pending):
            record(*future.result())
    return [row for name in sorted(done) for row in done[name]]


def detail(lib_path):
    """Peyton Manning, every coordinate of both certificates."""
    df = corpora.peyton_manning()
    stan_model, stan_data, points, _ = _fit_both(df, lib_path)
    tau = float(stan_data["tau"])
    rows = []
    for implementation in IMPLEMENTATIONS:
        k, m, delta, sigma_obs, beta = points[implementation]
        _, gradient = stan_gradient(stan_model, stan_data, k, m, delta, sigma_obs, beta)
        cert = certificate(gradient, delta, tau)
        values = np.concatenate([[k, m], delta, beta])
        smooth = np.concatenate([cert["k"], cert["m"], cert["smooth_delta"], cert["beta"]])
        strict = np.concatenate([cert["k"], cert["m"], cert["strict"], cert["beta"]])
        relaxed = np.concatenate([cert["k"], cert["m"], cert["relaxed"], cert["beta"]])
        for i, name in enumerate(_coordinates(len(delta), len(beta))):
            rows.append({"coordinate": name, "implementation": implementation,
                         "value": float(values[i]), "smooth_gradient": float(smooth[i]),
                         "residual_strict": float(strict[i]),
                         "residual_relaxed": float(relaxed[i])})
    return rows


def _write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(value) if isinstance(value, float) else value)
                             for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--sample", type=int, default=None,
                        help="a random sample of this many series, timed and not written")
    parser.add_argument("--detail", action="store_true",
                        help="only the Peyton Manning per-coordinate table")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    from analytic_prophet.build import cache_root

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    lib_path = harness.build_extension(tempfile.mkdtemp())

    if args.detail:
        _write(RESULTS / "kkt_peyton_manning.csv", DETAIL_FIELDS, detail(lib_path))
        print(f"wrote {(RESULTS / 'kkt_peyton_manning.csv').relative_to(REPO)}")
        return

    manifest = corpora.load_manifest()
    if args.sample:
        manifest = sample_manifest(manifest, args.sample)
        started = time.perf_counter()
        rows = collect(manifest, lib_path, args.workers)
        elapsed = time.perf_counter() - started
        total = sum(len(v) for v in corpora.load_manifest()["series"].values())
        print(f"{args.sample} random series in {elapsed:.0f}s on {args.workers} workers: "
              f"{elapsed / args.sample * args.workers:.1f} worker-seconds per series, "
              f"so the census of {total} is about "
              f"{elapsed / args.sample * total / 60:.0f} minutes at this width")
        return

    commit = (harness.run_metadata().get("commit") or "uncommitted")[:12]
    checkpoint = None if args.no_resume else cache_root() / "checkpoints" / "kkt" / commit
    rows = collect(manifest, lib_path, args.workers, checkpoint)
    _write(RESULTS / "kkt_census.csv", FIELDS, rows)
    with open(RESULTS / "kkt_census.meta.json", "w") as handle:
        json.dump(harness.run_metadata({"protocol": manifest["protocol"],
                                        "digest": manifest["digest"],
                                        "workers": args.workers}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {(RESULTS / 'kkt_census.csv').relative_to(REPO)} "
          f"({len(rows) // 2} series)")


if __name__ == "__main__":
    main()
