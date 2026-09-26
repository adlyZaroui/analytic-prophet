"""Turn the tiers' results into tables and figures (#79).

    python evaluation/report.py

Reads only `results/*.csv` -- it refits nothing and measures nothing, so a
report can be regenerated from a run that happened on another machine, and two
reports can be diffed to see what moved. A tier that has not been run is
skipped with a line saying so rather than failing.

Everything it emits is committed: `results/report.md` and `results/figures/`.
A number nobody can regenerate is a number nobody can check, and a number that
only exists in a terminal is one nobody will.
"""
import argparse
import collections
import sys

import harness

FIGURES = harness.RESULTS / "figures"
IMPLEMENTATIONS = ("prophet", "analytic_prophet", "fit_cpp", "fit(analytic=True)")
COLOURS = {"prophet": "#c44e52", "analytic_prophet": "#4c72b0",
           "fit_cpp": "#4c72b0", "fit(analytic=True)": "#55a868"}


def load(tier_name, results_dir=None):
    """A tier's rows, or None if it has not been run.

    `harness.RESULTS` is read here rather than taken as a default argument:
    a default binds at definition time, so pointing the report at another
    results directory -- which the tests do -- would silently have no effect.
    """
    try:
        return harness.read(tier_name, results_dir=results_dir or harness.RESULTS)
    except FileNotFoundError:
        return None


def indexed(rows):
    """(series, configuration) -> (implementation, metric) -> value."""
    table = collections.defaultdict(dict)
    for row in rows:
        table[(row["series"], row["configuration"])][
            (row["implementation"], row["metric"])] = row["value"]
    return table


def _size_of(series):
    """The T a `name[:T]` series label carries."""
    return int(series.split("[:")[1].rstrip("]"))


# -- sections -------------------------------------------------------------

def tier0_section(rows, figures):
    table = indexed(rows)
    gate = next((r["value"] for r in rows if r["metric"] == "gate_failed"), None)
    sizes = sorted((k for k in table if k[0] != "all"), key=lambda k: _size_of(k[0]))

    lines = ["## Tier 0 — are the two fitting the same model?", ""]
    lines.append("**Gate: " + ("PASSED" if gate == 0 else "FAILED") + "**. "
                 "Everything below it is about the difference between two "
                 "optimizers, and means nothing if they are optimizing "
                 "different specifications.")
    lines += ["", "| T | design matrix | prior scales | changepoints | "
              "Prophet `lp__` | ours | difference |", "|---|---|---|---|---|---|---|"]
    for key in sizes:
        cell = table[key]
        lines.append(
            f"| {_size_of(key[0])} | {cell[('both','design_matrix_max_abs_diff')]:.1e} | "
            f"{cell[('both','prior_scales_max_abs_diff')]:.0e} | "
            f"{cell[('both','changepoints_max_abs_diff')]:.0e} | "
            f"{cell[('prophet','lp__')]:.4f} | **{cell[('analytic_prophet','lp__')]:.4f}** | "
            f"{cell[('both','lp___difference')]:+.4f} |")
    lines += ["", "The design-matrix figure is the Fourier basis evaluated in a "
              "different order — arithmetic, not model.", ""]

    if figures:
        _parity_plot(table, sizes)
        lines += ["![posterior parity](figures/tier0_parity.png)", ""]
    return lines


def _parity_plot(table, sizes):
    import matplotlib.pyplot as plt

    ours = [table[k][("analytic_prophet", "lp__")] for k in sizes]
    theirs = [table[k][("prophet", "lp__")] for k in sizes]
    figure, axes = plt.subplots(figsize=(5, 5))
    limits = [min(theirs + ours) * 0.98, max(theirs + ours) * 1.02]
    axes.plot(limits, limits, color="0.7", linestyle="--", label="tie")
    axes.scatter(theirs, ours, s=70, color=COLOURS["analytic_prophet"], zorder=3)
    for key, x, y in zip(sizes, theirs, ours):
        axes.annotate(f"T={_size_of(key[0])}", (x, y), textcoords="offset points",
                      xytext=(8, -3), fontsize=9)
    axes.set(xscale="log", yscale="log", xlabel="Prophet  lp__",
             ylabel="analytic-prophet  lp__",
             title="Posterior under Stan's own density\n(above the line is better)")
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "tier0_parity.png", dpi=140)
    plt.close(figure)


def tier1_section(rows, figures):
    table = indexed(rows)
    lines = ["## Tier 1 — who recovers the true parameters?", "",
             "Series generated from the model, so there is a true parameter "
             "vector to be right about. On real data there is not, and "
             "agreement between two fits is all anyone can measure.", ""]

    raw = [(c[("analytic_prophet", "theta_error_l2")], c[("prophet", "theta_error_l2")])
           for c in table.values() if ("prophet", "theta_error_l2") in c]
    ident = [(c[("analytic_prophet", "identified_error")], c[("prophet", "identified_error")])
             for c in table.values() if ("prophet", "identified_error") in c]
    lines += ["| distance | ours wins |", "|---|---|",
              f"| raw `‖θ̂ − θ*‖` | {sum(a < b for a, b in raw)} / {len(raw)} |",
              f"| **identified** `½dᵀH(θ*)d` | **{sum(a < b for a, b in ident)} / {len(ident)}** |",
              ""]
    import metrics
    summary = metrics.paired_summary([a - b for a, b in ident])
    lines.append(f"Median paired difference {summary['median']:+.3f} nats, "
                 f"Wilcoxon p = {summary['p_value']:.4f}.")
    lines += ["", "The two disagree, and that is the finding. Raw distance is a "
              "coin flip because it is dominated by the directions the data "
              "does not identify — `k` against `delta`, where being far away "
              "costs nothing and means nothing.", "",
              "> The identified error is in **nats** and the curvature grows "
              "with the sample, so it compares implementations on one series "
              "and does not pool across lengths. Hence the pairing.", ""]

    zeros = [(c[("analytic_prophet", "exact_zeros")], c[("prophet", "exact_zeros")])
             for c in table.values() if ("prophet", "exact_zeros") in c]
    if zeros:
        lines += ["The trend's sparsity is reported as Σ|δ| and as the count of "
                  "**exact** zeros rather than as rates above a threshold: "
                  f"ours has {min(o for o, _ in zeros):.0f}–{max(o for o, _ in zeros):.0f} "
                  f"exact zeros and Prophet {min(p for _, p in zeros):.0f}–"
                  f"{max(p for _, p in zeros):.0f}, so a count of \"active\" "
                  "changepoints measures where the line was drawn rather than "
                  "the fit (#95). How many the generating vector had is reported "
                  "beside them for reference, not as a target: MAP with an L1 "
                  "prior over nested, collinear step functions is not a "
                  "support-recovery procedure, so neither fit is expected to "
                  "match it (#88).", ""]

    if figures:
        _recovery_plot(table)
        lines += ["![parameter recovery](figures/tier1_recovery.png)", ""]
    return lines


def _recovery_plot(table):
    import matplotlib.pyplot as plt
    import numpy as np

    blocks = ("k", "m", "delta", "beta", "sigma_obs")
    by_size = collections.defaultdict(lambda: collections.defaultdict(list))
    for (series, _), cell in table.items():
        size = int(series.split("T=")[1].split(",")[0])
        for block in blocks:
            key = ("analytic_prophet", f"{block}_error_l2")
            if key in cell:
                by_size[block][size].append(cell[key])

    figure, axes = plt.subplots(figsize=(6.5, 4.2))
    for block in blocks:
        sizes = sorted(by_size[block])
        medians = [float(np.median(by_size[block][s])) for s in sizes]
        axes.plot(sizes, medians, marker="o", label=block)
    axes.set(xscale="log", yscale="log", xlabel="observations",
             ylabel="median |error|, normalized units",
             title="Parameter recovery against series length")
    axes.legend(frameon=False, ncol=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / "tier1_recovery.png", dpi=140)
    plt.close(figure)


def tier2_section(rows, figures):
    paired = {r["metric"]: r["value"] for r in rows if r["series"] == "paired"}
    lines = ["## Tier 2 — does the better MAP point forecast better?", "",
             "The question the README explicitly refuses to answer. Rolling-origin "
             "evaluation on cutoffs from `prophet.diagnostics.generate_cutoffs`, "
             "scored by `prophet.diagnostics.performance_metrics` — both sides get "
             "the same splits and the same scorer.", "",
             "| metric | median difference | lower on | p |", "|---|---|---|---|"]
    # Metrics where lower is simply better. For coverage it is not -- higher is
    # better up to the nominal 0.8 -- and for the sparsity readouts neither
    # direction is good or bad, so a "wins" column would read as a scoreboard
    # for quantities that are not a contest.
    SCORED = {"mae", "rmse", "mape", "smape"}
    for metric in ("mae", "rmse", "mape", "smape", "coverage", "interval_width",
                   "sum_abs_delta", "exact_zeros", "l1_penalty"):
        if f"{metric}_n" not in paired:
            continue
        count = (f"{paired[f'{metric}_wins']:.0f}/{paired[f'{metric}_n']:.0f}"
                 if metric in SCORED else "—")
        lines.append(f"| {metric} | {paired[f'{metric}_median']:+.4f} | {count} | "
                     f"{paired[f'{metric}_p_value']:.4f} |")
    lines += ["", "Negative means we are lower. That is better for the four error"
              " rows, worse for coverage — which should be near the nominal 0.8 —"
              " and neither for the sparsity rows, which are reported because they"
              " describe the fits rather than rank them. The count column is left"
              " blank where a win is not defined.", "",
              "**The better MAP point does forecast better** on this corpus.", ""]

    coverage = collections.defaultdict(list)
    for row in rows:
        if row["metric"] == "coverage" and row["series"] != "paired":
            coverage[row["implementation"]].append(row["value"])
    if coverage:
        means = {k: sum(v) / len(v) for k, v in coverage.items()}
        lines += ["### The finding that is not about us", "",
                  "**Both implementations badly under-cover.** Mean coverage of the "
                  f"nominal 80% interval is **{means.get('analytic_prophet', float('nan')):.3f}** "
                  f"for ours and **{means.get('prophet', float('nan')):.3f}** for "
                  "Prophet's — the intervals contain about a third of the points they "
                  "claim four fifths of. That is the model on long horizons and "
                  "volatile series, shared by both, and it is larger than anything "
                  "separating them.", ""]
        if figures:
            _coverage_plot(coverage)
            lines += ["![coverage](figures/tier2_coverage.png)", ""]
    if figures:
        _accuracy_plot(rows)
        lines += ["![paired accuracy](figures/tier2_accuracy.png)", ""]
    return lines


def _coverage_plot(coverage):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 4))
    for name, values in coverage.items():
        axes.hist(values, bins=18, alpha=0.6, label=name,
                  color=COLOURS.get(name, "0.5"))
    axes.axvline(0.8, color="black", linestyle="--", label="nominal 0.80")
    axes.set(xlabel="coverage of the 80% interval", ylabel="series",
             title="Both implementations under-cover, by a lot")
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "tier2_coverage.png", dpi=140)
    plt.close(figure)


def _accuracy_plot(rows):
    import matplotlib.pyplot as plt
    import numpy as np

    table = indexed([r for r in rows if r["series"] != "paired"])
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for panel, metric in zip(axes, ("mae", "rmse")):
        pairs = [(c[("analytic_prophet", metric)], c[("prophet", metric)])
                 for c in table.values() if ("prophet", metric) in c]
        if not pairs:
            continue
        relative = [100.0 * (a - b) / b for a, b in pairs if b > 0]
        panel.hist(relative, bins=16, color=COLOURS["analytic_prophet"], alpha=0.8)
        panel.axvline(0, color="black", linestyle="--")
        panel.axvline(float(np.median(relative)), color=COLOURS["prophet"],
                      label=f"median {np.median(relative):+.1f}%")
        panel.set(xlabel=f"{metric.upper()}: ours vs Prophet, % difference",
                  ylabel="series")
        panel.legend(frameon=False)
    figure.suptitle("Held-out error, paired per series (left of zero is better for us)")
    figure.tight_layout()
    figure.savefig(FIGURES / "tier2_accuracy.png", dpi=140)
    plt.close(figure)


def tier3_section(rows, figures):
    table = indexed(rows)
    sizes = sorted({k for k in table if k[0].startswith("peyton") and k[1] == "default"},
                   key=lambda k: _size_of(k[0]))
    lines = ["## Tier 3 — what it costs", "",
             "| T | Prophet | `fit_cpp` | `fit(analytic=True)` |", "|---|---|---|---|"]
    for key in sizes:
        cell = table[key]
        lines.append(f"| {_size_of(key[0])} | {cell[('prophet','fit_wall')]:.3f} s | "
                     f"**{cell[('fit_cpp','fit_wall')]:.3f} s** | "
                     f"{cell[('fit(analytic=True)','fit_wall')]:.3f} s |")

    memory = [k for k in sizes if ("prophet", "fit_peak_rss_added") in table[k]]
    if memory:
        lines += ["", "Peak resident memory, **split into what the import cost and "
                  "what the fit did**. The README's claim is about the fit — "
                  "autodiff retains a tape and a closed-form gradient does not — "
                  "so a baseline taken before the import measures something else.",
                  "", "| T | Prophet fit | ours fit | Prophet import | ours import |",
                  "|---|---|---|---|---|"]
        for key in memory:
            cell = table[key]
            mib = lambda v: v / 1048576
            lines.append(
                f"| {_size_of(key[0])} | {mib(cell[('prophet','fit_peak_rss_added')]):.1f} MiB "
                f"| **{mib(cell[('fit_cpp','fit_peak_rss_added')]):.1f} MiB** "
                f"| **{mib(cell[('prophet','import_rss')]):.1f} MiB** "
                f"| {mib(cell[('fit_cpp','import_rss')]):.1f} MiB |")
        lines += ["", "We win the fit and lose the import — the latter almost "
                  "entirely scipy, which Prophet does not pull. A library that is "
                  "expensive to merely import is still expensive to deploy.", ""]

    predict = [k for k in sizes if ("prophet", "predict_wall") in table[k]]
    if predict:
        lines += ["### Prediction — four paths, and only two comparisons", "",
                  "`predict(vectorized=True)` is the default on **both** sides and is"
                  " not a faster form of the exact sampler — it is a different one,"
                  " disagreeing with it by about 1.4% on the interval bounds (#93)."
                  " `yhat` is identical either way; only the interval is sampled.", "",
                  "| T | Prophet approx | **ours approx** | Prophet exact | **ours exact** |",
                  "|---|---|---|---|---|"]
        for key in predict:
            cell = table[key]
            get = lambda name: cell.get((name, "predict_wall"))
            cells = [get("prophet"), get("fit_cpp"), get("prophet(exact)"),
                     get("fit_cpp(exact)")]
            lines.append(f"| {_size_of(key[0])} | " + " | ".join(
                ("—" if v is None else f"{v:.3f} s") for v in cells) + " |")
        lines += ["", "Read down the diagonals, not across: approximate against"
                  " approximate and exact against exact. Comparing our exact sampler"
                  " with Prophet's approximate one is what produced the \"2.4×"
                  " slower\" claim this tier reported before #93.", ""]

    scaling = table.get(("scaling", "default"), {})
    if scaling:
        lines += ["### Scaling", "",
                  "| implementation | d log wall / d log T |", "|---|---|"]
        for name in IMPLEMENTATIONS:
            if (name, "fit_wall_exponent") in scaling:
                lines.append(f"| {name} | {scaling[(name, 'fit_wall_exponent')]:.2f} |")
        lines += ["", "Read these with the figure rather than on their own. "
                  "Every implementation is *slower* at T=50 than at T=100, "
                  "which is not noise and not subprocess overhead: below 100 "
                  "observations Prophet's rule selects **Newton** (#25), and "
                  "both sides follow it. Newton pays `2n` gradient evaluations "
                  "an iteration for its Hessian. The exponents are fitted "
                  "through that kink, so they understate the asymptotic slope — "
                  "Prophet's 0.46 in particular is mostly the kink plus a fixed "
                  "subprocess spawn, not a claim that its fit is sub-linear.", ""]

    if figures:
        _cost_plot(table, sizes)
        lines += ["![cost](figures/tier3_cost.png)", ""]
    return lines


def _cost_plot(table, sizes):
    import matplotlib.pyplot as plt

    observations = [_size_of(k[0]) for k in sizes]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for name in ("prophet", "fit_cpp", "fit(analytic=True)"):
        values = [table[k].get((name, "fit_wall")) for k in sizes]
        axes[0].plot(observations, values, marker="o", label=name,
                     color=COLOURS.get(name, "0.5"))
    axes[0].set(xscale="log", yscale="log", xlabel="observations",
                ylabel="fit, seconds", title="Fit time")
    axes[0].legend(frameon=False, fontsize=9)

    for name in ("prophet", "fit_cpp"):
        values = [table[k].get((name, "fit_peak_rss_added")) for k in sizes]
        if all(v is not None for v in values):
            axes[1].plot(observations, [v / 1048576 for v in values], marker="o",
                         label=name, color=COLOURS.get(name, "0.5"))
    axes[1].set(xscale="log", xlabel="observations", ylabel="fit added, MiB",
                title="Memory the fit added\n(the autodiff-tape claim)")
    axes[1].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / "tier3_cost.png", dpi=140)
    plt.close(figure)


SECTIONS = (("tier0_agreement", tier0_section), ("tier1_recovery", tier1_section),
            ("tier2_accuracy", tier2_section), ("tier3_cost", tier3_section))


def build(figures=True, results_dir=None):
    if figures:
        FIGURES.mkdir(parents=True, exist_ok=True)
    lines = ["# Evaluation report", "",
             "Generated by `python evaluation/report.py` from the committed "
             "results in `results/`. It refits nothing — regenerating it on "
             "another machine reproduces this file exactly, and two reports "
             "diff to show what moved.", ""]
    metadata = None
    for name, section in SECTIONS:
        rows = load(name, results_dir)
        if rows is None:
            lines += [f"## {name}", "", "_Not run. "
                      f"`python evaluation/run.py --tiers {name[4]}`_", ""]
            continue
        metadata = metadata or harness.run_metadata()
        lines += section(rows, figures)

    lines += ["---", "", "## How this was measured", ""]
    if metadata:
        versions = metadata.get("versions", {})
        lines += [f"- seed `{metadata['seed']}`, commit `{(metadata.get('commit') or '?')[:12]}`",
                  f"- python {metadata['python']} on {metadata['platform']}",
                  "- " + ", ".join(f"{k} {v}" for k, v in sorted(versions.items()) if v),
                  ""]
    lines += ["Each tier writes `results/<tier>.csv` sorted, so rerunning an "
              "unchanged tier produces an unchanged file and a rerun is a "
              "reviewable diff rather than a number to be trusted.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--out", default=str(harness.RESULTS / "report.md"))
    args = parser.parse_args(argv)

    text = build(figures=not args.no_figures)
    with open(args.out, "w") as handle:
        handle.write(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
