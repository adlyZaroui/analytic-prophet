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


def captioned(path, alt, caption):
    """A figure with a caption under it, as markdown lines.

    [#99] Every plot needs the corpus, the configuration, the quantity and
    which direction is good. The prose around a figure used to carry some of
    that and the figure none of it, so a figure lifted out of the report said
    nothing. This is the one place that gets it right or wrong.
    """
    return [f"![{alt}](figures/{path})", "",
            " ".join(caption.split()), ""]
IMPLEMENTATIONS = ("prophet", "analytic_prophet", "compiled", "python")
COLOURS = {"prophet": "#c44e52", "analytic_prophet": "#4c72b0",
           "compiled": "#4c72b0", "python": "#55a868"}


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


def _linear_in_t(points):
    """`(fixed, slope per 1000 observations, R²)` for `memory = fixed + slope·T`.

    [#125] The README attributes the memory gap to a retained autodiff tape, and
    a single peak-RSS number cannot support that: a constant difference in
    interpreter or library overhead would look identical. Splitting the measured
    cost into the part that does not depend on the series and the part that
    scales with it is what distinguishes the two, since a tape is O(T) in the
    operations it records and so has to live entirely in the second.
    """
    import numpy as np

    sizes = np.array([t for t, _ in points], dtype=float)
    values = np.array([v for _, v in points], dtype=float)
    design = np.column_stack([np.ones_like(sizes), sizes / 1000.0])
    fixed, slope = np.linalg.lstsq(design, values, rcond=None)[0]
    spread = float(((values - values.mean()) ** 2).sum())
    residual = float(((values - design @ [fixed, slope]) ** 2).sum())
    return float(fixed), float(slope), (1.0 - residual / spread if spread else float("nan"))


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
        lines += captioned(
            "tier0_parity.png", "posterior parity",
            """**What this is.** The Peyton Manning series, first *T* rows, default configuration.
Both implementations are fitted on **Prophet's own changepoints**, so the only thing
that differs is the optimizer. Each is then scored by `CmdStanModel.log_prob` — Prophet's
own Stan density, not ours — and the bar is the difference in nats.
**Above zero is better for us.** The margin is small relative to the level (813 to 8005
nats), which is why the difference is plotted rather than the two values: on a parity
scatter the three points sit exactly on the diagonal and the figure says nothing.""")
    return lines


def _parity_plot(table, sizes):
    """The *difference*, not the pair.

    This was a log-log scatter of Prophet's lp__ against ours with a dashed
    diagonal and the title "above the line is better", and the three points
    sat exactly on the line -- because the margin is +0.36 to +2.76 nats on
    values of 813 to 8005, between 0.004% and 0.24%. A parity scatter cannot
    resolve that, so the plot asserted a claim and displayed its negation: a
    reader concluded the two were identical (#99).

    A difference of log densities is meaningful in absolute nats, so there is
    no reason to draw the two levels at all. Bars, because three points is
    too thin to read as a line and pretending otherwise is its own small lie.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    observations = [_size_of(k[0]) for k in sizes]
    margin = [table[k][("analytic_prophet", "lp__")] - table[k][("prophet", "lp__")]
              for k in sizes]

    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    positions = np.arange(len(sizes))
    bars = axes.bar(positions, margin, width=0.55,
                    color=[COLOURS["analytic_prophet"] if m >= 0 else COLOURS["prophet"]
                           for m in margin])
    axes.axhline(0.0, color="black", linewidth=1.0)
    for position, value in zip(positions, margin):
        axes.annotate(f"{value:+.2f}", (position, value), ha="center",
                      va="bottom" if value >= 0 else "top",
                      xytext=(0, 4 if value >= 0 else -4),
                      textcoords="offset points", fontsize=9)

    axes.set_xticks(positions)
    axes.set_xticklabels([f"T = {n}" for n in observations])
    axes.set_ylabel("lp__(ours) \u2212 lp__(Prophet), nats")
    axes.set_title("How much better our optimum is, by Prophet's own density\n"
                   "above zero is better for us", fontsize=10, loc="left")
    axes.margins(y=0.22)
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
        lines += captioned(
            "tier1_recovery.png", "parameter recovery",
            """**What this is.** 18 synthetic series generated from the model itself — *T* ∈ {100, 300,
1000} × noise ∈ {0.05, 0.2} × 3 seeds — so the true parameters are known.
**Left:** the identified error, a Hessian-weighted distance from the generating
parameters, in nats; one point per series, ours against Prophet's. **Below the diagonal
is better for us.** **Right:** the same recovery for this implementation alone, per
parameter block, in the normalized units the model fits in (`y / max|y|`), median over
the series at each length — a generator sanity check, where falling with *T* is the
point, not a comparison with anybody.
*The identified error is in nats and does not pool across series lengths*, which is why
the left panel is paired per series rather than averaged.""")
    return lines


def _recovery_plot(table):
    """Two panels: who recovers the truth better, and how recovery scales.

    The single panel this replaces plotted "median |error|, normalized units"
    without saying what the error was *against* -- a reader's first guess is
    ours minus Prophet's, and it is neither: it is this implementation's fit
    minus the parameters that generated the series. Prophet was not on the
    figure at all, so a plot in the tier whose subject is "who recovers the
    truth better" compared nobody to anybody (#99).

    The left panel is what the tier concludes on: the paired identified error,
    one point per synthetic series, ours against Prophet's. Points below the
    diagonal are series where our fit is closer to the generating parameters.
    The right panel is the old content, kept and labelled honestly as a
    generator sanity check -- the error falls as T grows, which is what it is
    there to show.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    blocks = ("k", "m", "delta", "beta", "sigma_obs")
    by_size = collections.defaultdict(lambda: collections.defaultdict(list))
    paired = []
    for (series, _), cell in table.items():
        size = int(series.split("T=")[1].split(",")[0])
        ours = cell.get(("analytic_prophet", "identified_error"))
        theirs = cell.get(("prophet", "identified_error"))
        if ours is not None and theirs is not None:
            paired.append((theirs, ours, size))
        for block in blocks:
            key = ("analytic_prophet", f"{block}_error_l2")
            if key in cell:
                by_size[block][size].append(cell[key])

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    if paired:
        theirs, ours, sizes_of = zip(*paired)
        wins = sum(1 for t, o in zip(theirs, ours) if o < t)
        limits = [min(theirs + ours) * 0.8, max(theirs + ours) * 1.2]
        axes[0].plot(limits, limits, color="0.7", linestyle="--", zorder=1)
        for size, marker in zip(sorted(set(sizes_of)), ("o", "s", "^")):
            selected = [(t, o) for t, o, n in paired if n == size]
            axes[0].scatter([t for t, _ in selected], [o for _, o in selected],
                            s=46, marker=marker, alpha=0.85,
                            color=COLOURS["analytic_prophet"], label=f"T = {size}")
        axes[0].set(xscale="log", yscale="log",
                    xlabel="Prophet: distance from the generating parameters (nats)",
                    ylabel="ours: same distance (nats)")
        # a log axis spanning one decade labels its minor ticks too, and they
        # collide into an unreadable row
        for axis in (axes[0].xaxis, axes[0].yaxis):
            axis.set_minor_formatter(plt.NullFormatter())
        axes[0].set_title(f"Who recovers the truth\nbelow the line is better for us "
                          f"({wins} of {len(paired)})", fontsize=10, loc="left")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    for block in blocks:
        sizes = sorted(by_size[block])
        medians = [float(np.median(by_size[block][s])) for s in sizes]
        axes[1].plot(sizes, medians, marker="o", label=block)
    axes[1].set(xscale="log", yscale="log", xlabel="observations",
                ylabel="median |fitted \u2212 generating|, normalized units")
    axes[1].set_title("Recovery against the generating parameters\n"
                      "ours only; falling is the point", fontsize=10, loc="left")
    # outside the axes: it used to sit on top of the data
    axes[1].legend(frameon=False, ncol=5, fontsize=8,
                   loc="upper center", bbox_to_anchor=(0.5, -0.16))
    figure.tight_layout()
    figure.savefig(FIGURES / "tier1_recovery.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


# The rows that are comparisons, in the order the table prints them. The
# sparsity readouts are deliberately not here: they describe the fits rather
# than ranking them, so counting them in a multiple-comparison family would
# make the correction harsher on the strength of tests nobody is using to
# claim anything (#124).
TIER2_COMPARISONS = ("mae", "rmse", "mape", "smape", "coverage", "interval_width")


def holm(p_values):
    """Holm-Bonferroni adjusted p-values.

    Holm rather than Bonferroni because these metrics are computed on the same
    forecasts and are heavily correlated, and plain Bonferroni assumes an
    independence nobody believes -- it is valid but needlessly blunt. Holm is
    valid under *arbitrary* dependence and uniformly more powerful, so there is
    no case for the cruder one.

    Step-down: sort ascending, scale the i-th by (m - i), and keep the running
    maximum so the sequence cannot decrease.
    """
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count, running, adjusted = len(ordered), 0.0, {}
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def _paired_rows(rows, stratum="all"):
    """The paired summary for one stratum.

    **Filtering on `configuration` is load-bearing.** This read every row whose
    series was `paired` and keyed only on the metric, which was correct while
    the tier produced one summary. #164 added a per-stratum summary under the
    same series name, and without this filter the two sets collide in the dict
    comprehension -- whichever the CSV happened to sort last would silently
    become the pooled table.
    """
    return {r["metric"]: r["value"] for r in rows
            if r["series"] == "paired" and r["configuration"] == stratum}


def _corpus(rows):
    """`{(configuration, metric): value}` for the corpus rows, empty if absent.

    Absent in results produced before #164, which still have to render.
    """
    return {(r["configuration"], r["metric"]): r["value"]
            for r in rows if r["series"] == "corpus"}


def _corpus_note(rows):
    """Where the corpus came from, and that it was fixed before these numbers.

    [#164] A seeded sample of 36 invites one question -- what else was tried --
    and the only answer that does not rest on trust is a membership written
    down beforehand. Read straight from the manifest rather than through
    `corpora`, so the report stays free of the package's imports.
    """
    import json

    corpus = _corpus(rows)
    manifest_path = (harness.EVALUATION / "corpus" / "m4_census_v1.json")
    if not corpus or not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())

    measured = corpus.get(("all", "series_measured"))
    frozen = manifest["total"]
    strata = ", ".join(
        f"{frequency} {int(corpus[(frequency, 'series_measured')])}"
        f"/{int(corpus[(frequency, 'series_frozen')])}"
        for frequency in sorted(manifest["series"])
        if (frequency, "series_measured") in corpus)

    lines = [
        f"**The corpus is a census, and it was frozen before any of this was run.**"
        f" `evaluation/corpus/{manifest_path.name}` lists every series measured —"
        f" {frozen} of them, {strata} — chosen by a rule with no sampling step:"
        f" every M4 series of these frequencies with at least"
        f" {manifest['min_length']} observations, a parseable start date, and at"
        f" least one rolling-origin cutoff under the horizons below. There is"
        f" nothing to have selected. The file carries a digest of its own"
        f" membership (`{manifest['digest'][:12]}`), the generator re-derives it"
        f" from the M4 files on demand, and a test fails if the measured series"
        f" are not exactly the frozen ones ([#164]).", ""]
    if measured is not None and measured < frozen:
        lines[0:1] = [lines[0] + f" **This run measured {int(measured)} of the"
                      f" {frozen}**, so it is a partial pass over the frozen"
                      " corpus rather than the whole of it."]
    return lines


def _stratum_table(rows, manifest_strata):
    """Per-stratum effect sizes, which is what a census is for.

    **At three thousand series the pooled p-value stops being the finding.**
    Any consistent difference clears every threshold at that n, so a bigger
    corpus that still leads with significance has learned nothing from being
    bigger. What it buys is the right to ask *where* the difference holds.
    """
    strata = [s for s in manifest_strata if _paired_rows(rows, s)]
    if not strata:
        return []

    lines = ["### Where it holds, by stratum", "",
             "The pooled row is the first; the rest are the frozen corpus's own"
             " strata. Read the medians and the counts rather than the p-values:"
             " at this n a p-value distinguishes nothing, which is the point of"
             " reporting the breakdown instead.", "",
             "| stratum | metric | median difference | lower on | n |",
             "|---|---|---|---|---|"]
    for stratum in ["all"] + list(strata):
        summary = _paired_rows(rows, stratum)
        for metric in ("mae", "rmse", "mape", "smape"):
            if f"{metric}_n" not in summary:
                continue
            lines.append(
                f"| {stratum} | {metric} | {summary[f'{metric}_median']:+.4f} |"
                f" {summary[f'{metric}_wins']:.0f}/{summary[f'{metric}_n']:.0f} |"
                f" {summary[f'{metric}_n']:.0f} |")
    return lines + [""]


def tier2_section(rows, figures):
    paired = _paired_rows(rows)
    lines = ["## Tier 2 — does the better MAP point forecast better?", "",
             "The question the README explicitly refuses to answer. Rolling-origin "
             "evaluation on cutoffs from `prophet.diagnostics.generate_cutoffs`, "
             "scored by `prophet.diagnostics.performance_metrics` — both sides get "
             "the same splits and the same scorer.", "",
             "Differences are paired per series and tested with the **Wilcoxon "
             "signed-rank test** (`scipy.stats.wilcoxon`), which is why the summary "
             "is a median rather than a mean: forecast errors across series are "
             "heavy-tailed, and one series here differs by about 35% while most "
             "differ by under 2%.", "",
             "| metric | median difference | lower on | p | Holm p |",
             "|---|---|---|---|---|"]
    # Metrics where lower is simply better. For coverage it is not -- higher is
    # better up to the nominal 0.8 -- and for the sparsity readouts neither
    # direction is good or bad, so a "wins" column would read as a scoreboard
    # for quantities that are not a contest.
    SCORED = {"mae", "rmse", "mape", "smape"}
    # Raw-unit rows: medians of differences across series whose levels differ by
    # orders of magnitude. They rank direction; they are not a pooled effect
    # size, because a unit of MAE means something different on every series
    # (#124). The scale-free rows beside them are what magnitude should be read
    # from.
    RAW_UNITS = {"mae", "rmse", "interval_width", "sum_abs_delta", "l1_penalty"}

    adjusted = holm({metric: paired[f"{metric}_p_value"]
                     for metric in TIER2_COMPARISONS
                     if f"{metric}_p_value" in paired})

    for metric in ("mae", "rmse", "mape", "smape", "coverage", "interval_width",
                   "sum_abs_delta", "exact_zeros", "l1_penalty"):
        if f"{metric}_n" not in paired:
            continue
        count = (f"{paired[f'{metric}_wins']:.0f}/{paired[f'{metric}_n']:.0f}"
                 if metric in SCORED else "—")
        label = f"{metric} ⁑" if metric in RAW_UNITS else metric
        holm_cell = (f"{adjusted[metric]:.4f}" if metric in adjusted else "—")
        lines.append(f"| {label} | {paired[f'{metric}_median']:+.4f} | {count} | "
                     f"{paired[f'{metric}_p_value']:.4f} | {holm_cell} |")
    survivors = [m for m in ("mae", "rmse", "mape", "smape", "coverage")
                 if adjusted.get(m, 1.0) < 0.05]
    lines += ["", "Negative means we are lower. That is better for the four error"
              " rows, worse for coverage — which should be near the nominal 0.8 —"
              " and neither for the sparsity rows, which are reported because they"
              " describe the fits rather than rank them. The count column is left"
              " blank where a win is not defined.", "",
              "**⁑ These rows are in the series' own units**, and the 36 M4 series"
              " differ in level by orders of magnitude. The median of a raw"
              " difference across them ranks direction and is *not* a pooled effect"
              " size — a unit of MAE means something different on every series. Read"
              " magnitude from the scale-free rows: `mape`, `smape` and `coverage`."
              " (MASE, the M4 standard, would be the better answer and needs the"
              " tier re-run with a new metric; it is not here.)", "",
              "**On testing nine things at once.** The comparisons are the six rows"
              f" above the sparsity readouts, and the `Holm p` column adjusts across"
              f" them. Holm rather than Bonferroni because these metrics are computed"
              f" on the same forecasts and are far from independent: Bonferroni is"
              f" valid but needlessly blunt, and Holm is valid under arbitrary"
              f" dependence and uniformly more powerful."
              f" **{len(survivors)} of the 5 comparative metrics survive"
              f" adjustment** — {', '.join(f'`{m}`' for m in survivors)} — and"
              " `interval_width` does not, which claims nothing it was not already"
              " declining to claim at p = 0.47. Plain Bonferroni would reject"
              " `rmse` and `smape`; including the three sparsity rows in the family"
              " changes none of it.", "",
              "**The better MAP point does forecast better** on this corpus.", ""]

    lines += _corpus_note(rows)
    import json as _json
    _manifest = harness.EVALUATION / "corpus" / "m4_census_v1.json"
    _strata = (sorted(_json.loads(_manifest.read_text())["series"])
               if _manifest.exists() else [])
    lines += _stratum_table(rows, _strata)

    coverage = collections.defaultdict(list)
    for row in rows:
        if row["metric"] == "coverage" and row["series"] != "paired":
            coverage[row["implementation"]].append(row["value"])
    if coverage:
        means = {k: sum(v) / len(v) for k, v in coverage.items()}
        # the shape behind the gap between the median difference and the
        # difference of means, counted rather than described (#124)
        # paired by series name, not by position: the two lists happen to be
        # in the same order because the results file is sorted, which is not a
        # property worth depending on
        _by_series = collections.defaultdict(dict)
        for row in rows:
            if row["metric"] == "coverage" and row["series"] != "paired":
                _by_series[row["series"]][row["implementation"]] = row["value"]
        _differences = sorted(
            pair["analytic_prophet"] - pair["prophet"]
            for pair in _by_series.values()
            if {"analytic_prophet", "prophet"} <= set(pair))
        _paired_count = len(_differences)
        _near = sum(1 for d in _differences if abs(d) < 0.02)
        _above = sum(1 for d in _differences if d > 0.05)
        _largest = _differences[-1] if _differences else float("nan")
        lines += ["### The finding that is not about us", "",
                  "**Both implementations badly under-cover.** Mean coverage of the "
                  f"nominal 80% interval is **{means.get('analytic_prophet', float('nan')):.3f}** "
                  f"for ours and **{means.get('prophet', float('nan')):.3f}** for "
                  "Prophet's — the intervals contain about a third of the points they "
                  "claim four fifths of. That is the model on long horizons and "
                  "volatile series, shared by both, and it is larger than anything "
                  "separating them.", "",
                  "*These two coverage numbers are different quantities, and the gap "
                  "between them is informative rather than a discrepancy.* The table "
                  f"reports **{paired.get('coverage_median', float('nan')):+.4f}**, the "
                  "median of the per-series *differences*; here it is the difference "
                  f"of the *means*, "
                  f"**{means.get('analytic_prophet', float('nan')) - means.get('prophet', float('nan')):+.3f}** "
                  "— about six times larger. A median of paired differences is not the "
                  "difference of means, and the two separating this far says the "
                  "per-series differences are skewed — which they are: "
                  f"**{_near} of the {_paired_count} series differ by less than 0.02**, "
                  f"{_above} differ by more than +0.05 (the largest, {_largest:+.2f}), "
                  "and none differs by more than 0.05 the other way. The mean is "
                  "carried by those few; the median is what the other thirty-odd "
                  "look like.", ""]
        if figures:
            _coverage_plot(coverage)
            lines += captioned(
                "tier2_coverage.png", "coverage",
                """**What this is.** 36 M4 series (20 weekly, 20 daily), rolling-origin cutoffs from
`prophet.diagnostics.generate_cutoffs`, with coverage of the nominal 80% interval
computed by `prophet.diagnostics.performance_metrics` for **both** sides, so neither is
scored by its own ruler. One dot per series per implementation, joined, sorted by ours;
dotted lines are the two means.
**Nearer 0.80 is better, and neither is near it.** This is the largest number in the
suite and it is shared: the gap between the dots is small, and the gap between both and
the dashed line is not. Since [#93] the default sampler on both sides is the approximate
one.""")
    if figures:
        _accuracy_plot(rows)
        lines += captioned(
            "tier2_accuracy.png", "paired accuracy",
            """**What this is.** The same corpus and protocol as above, scored by Prophet's own
`performance_metrics`. Each panel is the per-series relative difference,
(ours − Prophet) / Prophet, so the pairing is preserved rather than averaged away.
**Left of zero is better for us**, and the shading says so.
The summary is a median with an IQR rather than a mean, because forecast errors across
series are heavy-tailed — one series differs by about 35% while most differ by under 2%.
The axis is bounded by a robust range for that reason, and the panel counts what falls
outside it rather than cropping it silently. `n` and the paired p-value are in each
panel.""")
    return lines


def _coverage_plot(coverage):
    """Per series, paired, sorted -- not two translucent histograms.

    This is the largest number the suite produces: both implementations cover
    about a third of the points they claim four fifths of. Two overlaid
    histograms of 36 values each gave no sense of that, and at that sample
    size a histogram is a smoothed impression rather than the data (#99).

    A dot per series with a line joining the two implementations shows the
    shared miss *and* the pairing, which is how the tier actually analyses it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ours = coverage.get("analytic_prophet", [])
    theirs = coverage.get("prophet", [])
    if not ours or len(ours) != len(theirs):
        return

    order = np.argsort(ours)
    ours = np.asarray(ours)[order]
    theirs = np.asarray(theirs)[order]
    positions = np.arange(len(ours))

    figure, axes = plt.subplots(figsize=(9.5, 4.3))
    axes.vlines(positions, np.minimum(ours, theirs), np.maximum(ours, theirs),
                color="0.8", linewidth=1.2, zorder=1)
    axes.scatter(positions, theirs, s=26, color=COLOURS["prophet"],
                 label="Prophet", zorder=2)
    axes.scatter(positions, ours, s=26, color=COLOURS["analytic_prophet"],
                 label="analytic-prophet", zorder=2)
    axes.axhline(0.8, color="black", linestyle="--", linewidth=1.1,
                 label="nominal 0.80")
    axes.axhline(float(np.mean(ours)), color=COLOURS["analytic_prophet"],
                 linestyle=":", linewidth=1.0)
    axes.axhline(float(np.mean(theirs)), color=COLOURS["prophet"],
                 linestyle=":", linewidth=1.0)

    axes.set(xlabel="the 36 M4 series, sorted by our coverage",
             ylabel="fraction of held-out points inside the 80% interval",
             ylim=(0, 1))
    axes.set_title("Both implementations under-cover, by about the same amount\n"
                   f"means: ours {np.mean(ours):.3f}, Prophet {np.mean(theirs):.3f} "
                   f"\u2014 against a nominal 0.80", fontsize=10, loc="left")
    axes.set_xticks([])
    axes.legend(frameon=False, fontsize=9, loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES / "tier2_coverage.png", dpi=140)
    plt.close(figure)


def _accuracy_plot(rows):
    """All four error metrics, paired, with the sign spelled out.

    The headline claim of the project -- better held-out forecasts -- was
    drawn as two unlabelled histograms with no indication of which side was
    good, no n and no p-value, under an axis label reading "ours vs Prophet,
    % difference" (#99).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    per_series = indexed([r for r in rows if r["series"] != "paired"])
    summary = {(r["metric"], r["implementation"]): r["value"]
               for r in rows if r["series"] == "paired"}

    metrics = ("mae", "rmse", "smape", "mape")
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for panel, metric in zip(axes.ravel(), metrics):
        pairs = [(c[("analytic_prophet", metric)], c[("prophet", metric)])
                 for c in per_series.values() if ("prophet", metric) in c]
        relative = [100.0 * (a - b) / b for a, b in pairs if b > 0]
        if not relative:
            panel.axis("off")
            continue

        # Symmetric about zero, so the sign is readable at a glance -- but
        # bounded by a robust range rather than by the extreme. One series
        # sits near -35% while 34 of 36 are inside a few percent, and letting
        # it set the axis put the whole distribution in one bar. What is
        # outside is counted in the panel rather than quietly cropped.
        values = np.asarray(relative)
        limit = max(float(np.percentile(np.abs(values), 90)) * 1.8, 1.0)
        outside = int(np.sum(np.abs(values) > limit))

        panel.axvspan(-limit, 0, color=COLOURS["analytic_prophet"], alpha=0.07)
        panel.axvspan(0, limit, color=COLOURS["prophet"], alpha=0.07)
        panel.hist(np.clip(values, -limit, limit), bins=24, color="0.45", alpha=0.85)

        median = float(np.median(values))
        low, high = (float(np.percentile(values, 25)),
                     float(np.percentile(values, 75)))
        panel.axvspan(low, high, color=COLOURS["analytic_prophet"], alpha=0.16,
                      zorder=0)
        panel.axvline(0, color="black", linestyle="--", linewidth=1.0)
        panel.axvline(median, color=COLOURS["analytic_prophet"], linewidth=1.8)

        p_value = summary.get((f"{metric}_p_value", "difference"))
        n = summary.get((f"{metric}_n", "difference"))
        caption = f"median {median:+.2f}%   IQR [{low:+.2f}, {high:+.2f}]"
        if n is not None:
            caption += f"   n = {int(n)}"
        if p_value is not None:
            caption += f"   p = {p_value:.4f}"
        if outside:
            caption += f"\n{outside} series beyond \u00b1{limit:.0f}%, drawn at the edge"
        panel.set_title(f"{metric.upper()}   {caption}", fontsize=9, loc="left")
        panel.set(xlabel=f"{metric.upper()}: (ours \u2212 Prophet) / Prophet, %",
                  ylabel="series", xlim=(-limit, limit))
        panel.text(0.02, 0.95, "we are better", transform=panel.transAxes,
                   fontsize=8, va="top", color=COLOURS["analytic_prophet"])
        panel.text(0.98, 0.95, "Prophet is better", transform=panel.transAxes,
                   fontsize=8, va="top", ha="right", color=COLOURS["prophet"])

    figure.suptitle("Held-out forecast error, paired per series \u2014 "
                    "left of zero is better for us", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "tier2_accuracy.png", dpi=140)
    plt.close(figure)


def tier3_section(rows, figures):
    table = indexed(rows)
    sizes = sorted({k for k in table if k[0].startswith("peyton") and k[1] == "default"},
                   key=lambda k: _size_of(k[0]))
    lines = ["## Tier 3 — what it costs", "",
             "| T | Prophet | `compiled` | `python` |", "|---|---|---|---|"]
    for key in sizes:
        cell = table[key]
        lines.append(f"| {_size_of(key[0])} | {cell[('prophet','fit_wall')]:.3f} s | "
                     f"**{cell[('compiled','fit_wall')]:.3f} s** | "
                     f"{cell[('python','fit_wall')]:.3f} s |")

    memory = [k for k in sizes if ("prophet", "fit_peak_rss_added") in table[k]]
    if memory:
        mib = lambda v: v / 1048576
        lines += ["", "Peak resident memory **attributable to the fit**, over and "
                  "above what was already resident. The README's claim is about "
                  "the fit — autodiff retains a tape and a closed-form gradient "
                  "does not — so a baseline taken before the import would measure "
                  "something else.",
                  "", "| T | Prophet fit | ours fit |", "|---|---|---|"]
        for key in memory:
            cell = table[key]
            lines.append(
                f"| {_size_of(key[0])} | {mib(cell[('prophet','fit_peak_rss_added')]):.1f} MiB "
                f"| **{mib(cell[('compiled','fit_peak_rss_added')]):.1f} MiB** |")
        lines += [""]

        imports = table.get(("imports", "default"), {})
        if imports:
            levels = {name: value for (name, metric), value in imports.items()
                      if metric == "import_peak_rss"}
            reference = levels.get("numpy+pandas")
            lines += ["And what each library costs to merely import — one fresh "
                      "interpreter per row, importing that and nothing else, so "
                      "each number is a level rather than a difference:",
                      "", "| a fresh interpreter importing | peak RSS | over numpy+pandas |",
                      "|---|---|---|"]
            for name in ("bare", "numpy+pandas", "prophet", "analytic_prophet"):
                if name not in levels:
                    continue
                over = ("—" if reference is None or name in ("bare", "numpy+pandas")
                        else f"+{mib(levels[name] - reference):.1f} MiB")
                label = name if name in ("bare", "numpy+pandas") else f"`{name}`"
                lines.append(f"| {label} | {mib(levels[name]):.1f} MiB | {over} |")
            ours = levels.get("analytic_prophet")
            theirs = levels.get("prophet")
            if ours is not None and theirs is not None and reference is not None:
                verdict = (
                    f"**We win the import as well as the fit**, by "
                    f"{mib(theirs - ours):.1f} MiB. Until [#130] we lost it, and "
                    "by about as much in the other direction: scipy was imported "
                    "whatever backend you asked for, and it was 57 of the 59 MiB "
                    "this package added. It serves one thing — `_fit_python`, the "
                    "reference backend kept for reading against the C++ core — so "
                    "it is now imported when that runs and not before. Prophet "
                    "does not pull scipy at all, which is why this was the one "
                    "cost comparison here that it won."
                    if ours < theirs else
                    f"**We win the fit and lose the import** by "
                    f"{mib(ours - theirs):.1f} MiB — a library that is expensive "
                    "to merely import is still expensive to deploy.")
            else:
                verdict = ("Measured against a numpy+pandas interpreter, since "
                           "both libraries require both.")
            lines += ["", verdict, "",
                      "**Levels rather than a before-and-after, and `VmHWM` "
                      "rather than `ru_maxrss`.** Measuring this as two marks "
                      "inside one interpreter reported 58 MiB on macOS/arm64 and "
                      "**exactly zero** in all seven Linux CI jobs. `ru_maxrss` "
                      "is inherited across `fork` and `exec` does not reset it, "
                      "so a subprocess launched from a large parent reports the "
                      "*parent's* peak as its own — a bare `python -c` child of a "
                      "413 MiB parent reports 413 MiB. Every measurement in this "
                      "tier runs in a subprocess and the runner holds numpy, "
                      "pandas, scipy and this package, so on Linux the call "
                      "returned a constant, both marks read it, and every delta "
                      "came out zero. That is the \"reads exactly 0 in every CI "
                      "job on Linux\" quirk [#103] recorded without a cause. The "
                      "parent side now reads `VmHWM` from `/proc/self/status`, "
                      "which lives in the mm that `exec` replaces and reports "
                      "7 MiB for that same child; a Linux container now "
                      "reproduces these levels to within a megabyte. Recorded "
                      "once rather than per series length, because it does not "
                      "depend on T ([#125]).", ""]
        lines += _memory_attribution(table, memory)

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
            cells = [get("prophet"), get("compiled"), get("prophet(exact)"),
                     get("compiled(exact)")]
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
        lines += captioned(
            "tier3_cost.png", "cost",
            """**What this is.** The Peyton Manning series, default configuration, wall clock, best of
3 runs. **Lower is better everywhere.**
**Left:** fit time. Below *T* = 100 both implementations run Newton rather than L-BFGS,
which is Prophet's own algorithm rule ([#25]) and is what the kink at *T* = 50 is.
**Middle:** peak resident memory attributable to the fit, *over and above what importing
the library cost* — that separation is the measurement, since the claim is about the
autodiff tape and not about import weight. Measured in a fresh subprocess per point,
**summing** `RUSAGE_SELF` and `RUSAGE_CHILDREN` so that Prophet's cmdstan child is
counted; the sum is an upper bound on simultaneous residency, since the two
high-water marks need not coincide, and taking the maximum instead would be a lower
bound that here counts the child not at all. Neither convention changes the ordering
([#125]).
**Right:** inference, all four paths. Prophet's default is an approximation and so is
ours since [#94]; solid lines are the two approximate samplers and dashed the two exact
ones, so the honest comparisons are **down** each style rather than across.""")
    return lines


def _memory_attribution(table, memory):
    """[#125] The memory gap, decomposed -- and what the decomposition can show.

    The README used to read the ~3x memory ratio as evidence of a retained
    autodiff tape. Peak RSS cannot establish that on its own, so this reports
    the two things it *can* show: that the gap is in the part which scales with
    the series rather than in fixed overhead, and that the part which scales
    sits on the side of the fork where Stan's autodiff actually runs.
    """
    mib = lambda v: v / 1048576
    fits = {}
    for name in ("prophet", "compiled", "python"):
        points = [(_size_of(k[0]), mib(table[k][(name, "fit_peak_rss_added")]))
                  for k in memory if (name, "fit_peak_rss_added") in table[k]]
        if len(points) >= 3:
            fits[name] = _linear_in_t(points)
    if "prophet" not in fits or "compiled" not in fits:
        return []

    lines = ["### Where the memory gap is",
             "",
             "A ratio of two peaks is compatible with a constant difference in "
             "overhead, which is not what the README claims. Fitting "
             "`memory = fixed + slope·T` over the five lengths separates the two:",
             "", "| implementation | fixed | per 1000 obs | R² |", "|---|---|---|---|"]
    for name, (fixed, slope, r2) in fits.items():
        lines.append(f"| {name} | {fixed:.2f} MiB | {slope:.3f} MiB | {r2:.4f} |")

    fixed_gap = abs(fits["prophet"][0] - fits["compiled"][0])
    slope_ratio = fits["prophet"][1] / fits["compiled"][1]
    lines += ["",
              f"**The fixed costs are within {fixed_gap:.2f} MiB of each other "
              f"while the per-observation costs differ {slope_ratio:.1f}×.** "
              "The gap is almost "
              "entirely in the term that scales with the series, which is the "
              "shape a retained tape predicts — a tape is O(T) in the operations "
              "recorded, so it must cost nothing at T = 0 and grow from there. "
              "It is equally the shape predicted by *any* other allocation in "
              "Prophet's optimizer that scales with T, and peak RSS cannot tell "
              "those apart. So this rules out the explanation that the gap is "
              "fixed overhead; it does not isolate the tape, and the README no "
              "longer says it does.", "",
              "**Both of these numbers moved when [#130] landed, and upward "
              "for us.** Deferring scipy dropped what is resident before the "
              "fit from 130 MiB to 73, and peak RSS is a high-water mark: "
              "growth that fitted under the import's own transient peak used "
              "to be invisible and now is not. Our measured fit cost roughly "
              "doubled as a result — the measurement got more sensitive, the "
              "fit did not get worse — and the slope ratio fell from about "
              "twelvefold to the figure above. The earlier number was the "
              "flattering one and it was flattering by accident.", "",
              "**The two sides are unevenly masked, and the asymmetry now runs "
              "against us.** Prophet still imports 40 MiB where we import 1.6, "
              "so more of its fit growth can hide under its own import "
              "transient than ours can. A subtraction of two high-water marks "
              "cannot see growth smaller than that headroom, and the headroom "
              "is not equal, so read the gap as a conservative one rather than "
              "as exact. It is also why the ratio is better conditioned than it "
              "was: our slope is no longer near zero, and two reruns of this "
              "tier now agree on it to a few percent where they used to "
              "disagree by a fifth.", ""]

    split = [k for k in memory
             if ("prophet", "fit_peak_rss_added_children") in table[k]]
    if split:
        lines += ["Which side of the fork the growth is on, which is the closest "
                  "this measurement gets to the mechanism — Prophet's optimizer "
                  "and therefore its tape run in the cmdstan child, while the "
                  "parent is cmdstanpy writing a data file and reading draws back:",
                  "", "| T | Prophet parent | Prophet child | ours parent | ours child |",
                  "|---|---|---|---|---|"]
        for key in split:
            cell = table[key]
            get = lambda name, side: cell.get((name, f"fit_peak_rss_added_{side}"))
            cells = [get("prophet", "self"), get("prophet", "children"),
                     get("compiled", "self"), get("compiled", "children")]
            lines.append(f"| {_size_of(key[0])} | " + " | ".join(
                ("—" if v is None else f"{mib(v):.1f} MiB") for v in cells) + " |")
        lines += ["",
                  "**The child column is left-censored at zero and cannot be "
                  "regressed.** `RUSAGE_CHILDREN` is a maximum over every child "
                  "waited on, and importing Prophet spawns one before the fit "
                  "does, so the fit's child is invisible until it exceeds that "
                  "floor — which is why the short series read 0.0 rather than a "
                  "small number. Read the column as a lower bound that becomes "
                  "informative at the top. Ours is zero throughout for a "
                  "different reason: with the extension already built it forks "
                  "nothing, so for us the sum and the maximum are the same number.",
                  ""]
        reliable = table.get(("environment", "default"), {}).get(
            ("both", "child_peak_rss_reliable"))
        if reliable is not None:
            lines += [
                ("**The child column is only meaningful on a platform where a "
                 "child's `ru_maxrss` describes the child, and these numbers were "
                 "produced on one.** There is no `/proc` repair available for this "
                 "half — a reaped child has no entry left to read and cmdstan is "
                 "not ours to instrument — so the suite measures the property "
                 "instead of assuming it, and records the answer beside the "
                 "numbers it governs. Here it held."
                 if reliable else
                 "**These numbers were produced where a child's `ru_maxrss` is "
                 "inherited from this process rather than describing the child, "
                 "so the child column above is not trustworthy and the parent "
                 "column is the one to read.** `VmHWM` repairs the parent side "
                 "only; a reaped child has no `/proc` entry left and cmdstan is "
                 "not ours to instrument. Re-run the tier where "
                 "`child_peak_rss_is_reliable()` holds to get the split."), ""]
    return lines


def _cost_plot(table, sizes):
    """Three panels: fit time, fit memory, and inference -- which had none.

    Inference was measured as a four-way comparison and plotted nowhere,
    although it is the one place the two implementations differ in kind
    rather than degree: Prophet's default sampler is an approximation, and
    since #94 this one offers the same approximation and the exact sampler
    both. Reading down the diagonals is the whole point, so the two
    approximate paths share a line style and the two exact ones another (#99).

    "fit added, MiB" is also gone. It meant "peak resident memory attributable
    to the fit, over and above what importing the library cost", and that
    distinction is the entire measurement -- the label hid it.
    """
    import matplotlib.pyplot as plt

    observations = [_size_of(k[0]) for k in sizes]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    for name in ("prophet", "compiled", "python"):
        values = [table[k].get((name, "fit_wall")) for k in sizes]
        axes[0].plot(observations, values, marker="o", label=name,
                     color=COLOURS.get(name, "0.5"))
    # the kink at T = 50 is Prophet's algorithm rule, and was unexplained on
    # the figure -- it reads as noise. In the label rather than annotated into
    # the axes, where it sat on top of the data.
    axes[0].set(xscale="log", yscale="log",
                xlabel="observations\n(below T = 100 both run Newton, #25)",
                ylabel="wall clock, seconds (best of 3)")
    axes[0].set_title("Fit time\nlower is better", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=9)

    for name in ("prophet", "compiled"):
        values = [table[k].get((name, "fit_peak_rss_added")) for k in sizes]
        if all(v is not None for v in values):
            axes[1].plot(observations, [v / 1048576 for v in values], marker="o",
                         label=name, color=COLOURS.get(name, "0.5"))
    axes[1].set(xscale="log", xlabel="observations",
                ylabel="peak resident memory added by the fit (MiB)")
    axes[1].set_title("Memory the fit itself costs\nover and above importing the "
                      "library", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=9)

    # the panel that did not exist. Approximate solid, exact dashed, so the
    # eye reads down the diagonals rather than across the four lines.
    inference = (("prophet", "approximate", "-"), ("compiled", "approximate", "-"),
                 ("prophet(exact)", "exact", "--"), ("compiled(exact)", "exact", "--"))
    for name, kind, style in inference:
        values = [table[k].get((name, "predict_wall")) for k in sizes]
        if not all(v is not None for v in values):
            continue
        colour = COLOURS["prophet"] if name.startswith("prophet") \
            else COLOURS["analytic_prophet"]
        label = ("Prophet" if name.startswith("prophet") else "ours") + f", {kind}"
        axes[2].plot(observations, values, marker="o", linestyle=style,
                     label=label, color=colour)
    axes[2].set(xscale="log", yscale="log", xlabel="observations",
                ylabel="wall clock, seconds (best of 3)")
    axes[2].set_title("Inference time, all four paths\nsolid approximate, dashed "
                      "exact; compare down, not across", fontsize=10, loc="left")
    axes[2].legend(frameon=False, fontsize=8)

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
