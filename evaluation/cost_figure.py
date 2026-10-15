#!/usr/bin/env python
"""The README's cost figure: speed-up and memory gain against series length (#231).

The README gave its costs as a table: fit time at three lengths, prediction and
memory at one. Plotted against T, the shape shows instead -- the fit is
fastest where Prophet runs Stan's Newton method, below T = 100, and narrows to
about 1.4x from T = 1000 up, while the memory gain grows with T.

Every curve is Prophet's measurement over this package's, from the committed
Tier 3 results (`results/tier3_cost.csv`), so above 1 this package is ahead.
Nothing is remeasured: the figure is a view of the file, and drawing it twice
gives the same bytes.

    python evaluation/cost_figure.py

Writes `results/figures/cost_ratios.png`.
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIER3 = HERE / "results" / "tier3_cost.csv"
FIGURE = HERE / "results" / "figures" / "cost_ratios.png"

# Prophet's rule ([fc] CmdStanPyBackend.fit): Newton below this many
# observations, L-BFGS from it. analytic_prophet.optimizer.NEWTON_BELOW.
NEWTON_BELOW = 100

# the evaluation report's palette (report.COLOURS), so the figures read as one set
COLOURS = {"ours": "#4c72b0", "exact": "#55a868"}

# (panel title, implementations whose ratio it plots, metric)
PANELS = (
    ("fit time", (("prophet", "compiled", "ours"),), "fit_wall"),
    ("prediction time", (("prophet", "compiled", "approximate sampler"),
                         ("prophet(exact)", "compiled(exact)", "exact sampler")), "predict_wall"),
    ("memory a fit adds", (("prophet", "compiled", "ours"),), "fit_peak_rss_added"),
)


def ratios(path=TIER3):
    """`{metric: {label: {T: Prophet's / this package's}}}` from Tier 3."""
    values = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["series"].startswith("peyton_manning[:"):
                size = int(row["series"].split("[:")[1].rstrip("]"))
                values[(row["implementation"], row["metric"], size)] = float(row["value"])
    sizes = sorted({size for _, _, size in values})
    out = {}
    for _title, pairs, metric in PANELS:
        out[metric] = {label: {size: values[(theirs, metric, size)] / values[(ours, metric, size)]
                               for size in sizes}
                       for theirs, ours, label in pairs}
    return out


def draw(data, out=FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True)
    for axis, (title, pairs, metric) in zip(axes, PANELS):
        for _theirs, _ours, label in pairs:
            curve = data[metric][label]
            sizes = sorted(curve)
            colour = COLOURS["exact"] if label == "exact sampler" else COLOURS["ours"]
            axis.plot(sizes, [curve[size] for size in sizes], marker="o", color=colour,
                      linewidth=1.6, markersize=4.5, label=label)
            for size in sizes:
                axis.annotate(f"{curve[size]:.1f}×", (size, curve[size]),
                              textcoords="offset points", xytext=(0, 6), ha="center",
                              fontsize=7, color=colour)
        axis.axhline(1.0, color="0.55", linewidth=0.9)
        axis.axvline(NEWTON_BELOW, color="0.65", linestyle=":", linewidth=1.0)
        axis.set_xscale("log")
        sizes = sorted(next(iter(data[metric].values())))
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(size) for size in sizes])
        axis.minorticks_off()
        axis.set_title(f"{title}: Prophet's ÷ this package's", fontsize=9, loc="left")
        axis.set_xlabel("series length T (observations)", fontsize=8)
        axis.tick_params(labelsize=8)
        top = max(value for _t, _o, label in pairs for value in data[metric][label].values())
        axis.set_ylim(0, top * 1.22)
        if len(pairs) > 1:
            axis.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0].set_ylabel("× in this package's favour", fontsize=8)
    # Prophet's own rule: Newton below T = 100, L-BFGS from it
    for text, side, offset in (("Prophet: Newton", "right", -4), ("L-BFGS", "left", 4)):
        axes[0].annotate(text, (NEWTON_BELOW, 1.0), xycoords=("data", "axes fraction"),
                         textcoords="offset points", xytext=(offset, -10), ha=side,
                         fontsize=7, color="0.4")
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140, metadata={"Software": None})
    plt.close(figure)
    return out


def main():
    written = draw(ratios())
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
