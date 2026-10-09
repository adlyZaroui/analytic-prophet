"""The lp__ margin against series length, on a fine grid of T (#173).

Tier 0 measures the margin at three lengths, and #146 found those three points
misleading on their own: the absolute margin is not monotone in T, and the two
widths of model measured disagree on its ordering. This measures the same
quantity on a grid fine enough to show its shape, and draws it.

**The measurement is Tier 0's**: Prophet fitted on the first T rows of Peyton
Manning, this implementation fitted on Prophet's own changepoints, and both
scored by Stan's log density of Prophet's compiled program. Prophet's optimum
is read at full precision (`sig_figs=18`), as for the certificate (#170).

**Two configurations**:

    default       Prophet's own seasonality selection. Weekly throughout;
                  yearly switches on at T = 693, where the history first spans
                  730 days, so the model widens there.
    yearly_only   yearly forced on, weekly off -- measured only from T = 693,
                  because below 730 days forcing yearly on is the configuration
                  docs/non-smooth-objective.md withdrew a claim over.

**Which optimizer ran** is recorded per cell, for both: Prophet runs Newton
below T = 100 because of prophet#842 and falls back to Newton when L-BFGS exits
abnormally, and this implementation mirrors both rules. The margin below 100 is
therefore partly a statement about which method ran, and the figure marks it.

    python paper/experiments/margin.py           # fit the grid, then draw
    python paper/experiments/margin.py --draw    # redraw from the committed CSV

Writes `paper/results/margin.csv`, its `.meta.json`, and
`paper/figures/margin_vs_length.pdf`.
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

RESULTS = PAPER / "results" / "margin.csv"
FIGURE = PAPER / "figures" / "margin_vs_length.pdf"

NEWTON_BELOW = 100
# Rows of Peyton Manning at which the history first spans 730 days, so that
# Prophet's own rule turns yearly seasonality on. Checked against the data by
# tests/test_paper_margin.py rather than trusted.
YEARLY_FROM = 693

CONFIGURATIONS = {
    "default": {},
    "yearly_only": {"yearly_seasonality": True, "weekly_seasonality": False,
                    "daily_seasonality": False},
}

# Dense where the method changes (T = 100) and where the default model widens
# (T = 693); every 50 rows elsewhere; Tier 0's three lengths included, so the
# grid can be checked against it.
GRID = sorted({20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100, 101, 105, 110, 120,
               130, 140, 150, 175, 200, 225, 250, 275, 300,
               *range(350, 2901, 50), YEARLY_FROM - 1, YEARLY_FROM, 2905})

FIELDS = ["configuration", "observations", "days", "seasonalities", "columns",
          "changepoints", "prophet_algorithm", "ours_algorithm",
          "lp_prophet", "lp_ours", "margin", "relative_margin_percent",
          "sigma_obs_prophet", "sigma_obs_ours", "design_gap"]


def cells():
    """Every (configuration, T) the grid measures."""
    for configuration in CONFIGURATIONS:
        for size in GRID:
            if configuration == "yearly_only" and size < YEARLY_FROM:
                continue
            yield configuration, size


class _FallbackSeen(logging.Handler):
    """Notices Prophet's 'Falling back to Newton' warning, which is the only
    trace a fall-back leaves on a fitted Prophet."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.seen = False

    def emit(self, record):
        if "Falling back to Newton" in record.getMessage():
            self.seen = True


def measure(cell, lib_path):
    """One (configuration, T): both fits, both scored, as a row."""
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    configuration, size = cell
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    kwargs = {**harness.PROPHET_KWARGS, **CONFIGURATIONS[configuration]}
    df = corpora.peyton_manning(size)
    df = df.assign(ds=pd.to_datetime(df["ds"]))

    fallback = _FallbackSeen()
    logging.getLogger("prophet").addHandler(fallback)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prophet_model = Prophet(**kwargs)
            stan_model, stan_data, theirs = harness.capture_stan_model(
                prophet_model, df, sig_figs=18)
    finally:
        logging.getLogger("prophet").removeHandler(fallback)

    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    ours = AnalyticProphet(**kwargs)
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit(df, lib_path=lib_path)

    X_ours = np.asarray(ours.make_all_seasonality_features(df)[0], dtype=float)
    X_stan = np.asarray(stan_data["X"], dtype=float).reshape(X_ours.shape[0], -1)
    design_gap = (float(np.max(np.abs(X_ours - X_stan), initial=0.0))
                  if X_ours.shape == X_stan.shape else float("inf"))

    lp_prophet, _ = kkt.stan_gradient(stan_model, stan_data, theirs["k"][0], theirs["m"][0],
                                      theirs["delta"], theirs["sigma_obs"][0], theirs["beta"])
    lp_ours, _ = kkt.stan_gradient(stan_model, stan_data, ours.params["k"][0][0],
                                   ours.params["m"][0][0], ours.params["delta"][0],
                                   ours.sigma_obs, ours.params["beta"][0])

    prophet_algorithm = "Newton" if size < NEWTON_BELOW else "LBFGS"
    if fallback.seen:
        prophet_algorithm = "LBFGS, then Newton"
    margin = lp_ours - lp_prophet
    return {
        "configuration": configuration,
        "observations": size,
        "days": int((df["ds"].iloc[-1] - df["ds"].iloc[0]).days),
        "seasonalities": "+".join(sorted(prophet_model.seasonalities)),
        "columns": int(stan_data["K"]),
        "changepoints": int(stan_data["S"]),
        "prophet_algorithm": prophet_algorithm,
        "ours_algorithm": ours.optimizer_used,
        "lp_prophet": lp_prophet,
        "lp_ours": lp_ours,
        "margin": margin,
        "relative_margin_percent": margin / abs(lp_prophet) * 100,
        "sigma_obs_prophet": float(theirs["sigma_obs"][0]),
        "sigma_obs_ours": float(ours.sigma_obs),
        "design_gap": design_gap,
    }


def _measure(job):
    cell, lib_path = job
    return measure(cell, lib_path)


def collect(workers, lib_path):
    """The whole grid, in parallel; rows in (configuration, T) order."""
    jobs = [(cell, lib_path) for cell in cells()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_measure, jobs))
    return sorted(rows, key=lambda row: (row["configuration"], row["observations"]))


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("observations", "days", "columns", "changepoints"):
            row[key] = int(row[key])
        for key in ("lp_prophet", "lp_ours", "margin", "relative_margin_percent",
                    "sigma_obs_prophet", "sigma_obs_ours", "design_gap"):
            row[key] = float(row[key])
    return rows


def draw(rows, out=FIGURE):
    """Absolute and relative margin against T, both configurations.

    Log axes on both: T spans two orders of magnitude and the margin three,
    and on linear axes everything below T = 300 is a smudge at the origin. A
    log scale cannot show a margin of zero or less; `draw` refuses rather than
    dropping such a cell silently.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if any(row["margin"] <= 0 for row in rows):
        raise ValueError("a non-positive margin cannot be drawn on a log axis; "
                         "the figure needs a different scale before it can show it")

    style = {"default": ("#4c72b0", "o", "default (Prophet's own seasonalities)"),
             "yearly_only": ("#dd8452", "s", "yearly only")}
    matplotlib.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    panels = (("margin", "lp$\\_\\_$ margin, ours $-$ Prophet's (nats)"),
              ("relative_margin_percent", "relative margin (% of Prophet's lp$\\_\\_$)"))
    for axis, (column, label) in zip(axes, panels):
        for configuration, (colour, marker, name) in style.items():
            cells_ = [row for row in rows if row["configuration"] == configuration]
            axis.plot([row["observations"] for row in cells_],
                      [row[column] for row in cells_],
                      color=colour, marker=marker, markersize=2.6, linewidth=0.9,
                      label=name)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("series length $T$ (rows of Peyton Manning)")
        axis.set_ylabel(label)
        axis.axvline(NEWTON_BELOW, color="0.35", linestyle="--", linewidth=0.8)
        axis.axvline(YEARLY_FROM, color="0.35", linestyle=":", linewidth=0.8)
        axis.grid(True, which="major", color="0.9", linewidth=0.5)
        bottom, top = axis.get_ylim()
        axis.text(NEWTON_BELOW * 0.93, top, "Newton | L-BFGS", rotation=90,
                  ha="right", va="top", fontsize=6.5, color="0.3")
        axis.text(YEARLY_FROM * 0.93, top, "yearly on (730 days)", rotation=90,
                  ha="right", va="top", fontsize=6.5, color="0.3")
    axes[0].legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    # no timestamp, so redrawing unchanged results gives an unchanged file
    figure.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)
    return out


def _write(rows, path=RESULTS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(value) if isinstance(value, float) else value)
                             for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--draw", action="store_true",
                        help="only redraw the figure from the committed results")
    args = parser.parse_args(argv)

    if not args.draw:
        lib_path = harness.build_extension(tempfile.mkdtemp())
        rows = collect(args.workers, lib_path)
        _write(rows)
        with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
            json.dump(harness.run_metadata({"workers": args.workers, "grid": GRID}),
                      handle, indent=1, sort_keys=True)
            handle.write("\n")
        print(f"wrote {RESULTS.relative_to(REPO)} ({len(rows)} cells)")
    print(f"wrote {draw(read()).relative_to(REPO)}")


if __name__ == "__main__":
    main()
