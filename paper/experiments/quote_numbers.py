"""Write the numbers the paper's prose quotes, as LaTeX macros (#167, #180).

Every figure a sentence in the paper states comes from here, read out of the
committed evaluation results. None is typed into a `.tex` file by hand, which is
the rule `evaluation/report.py` follows and the reason the census landing on
`main` turned the suite red: the README had been quoting the previous corpus.

    python paper/experiments/quote_numbers.py      # writes paper/generated/numbers.tex

A macro that cannot be computed from what is committed is not written, so the
document fails to compile on it rather than printing a stale value.
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
RESULTS = REPO / "evaluation" / "results"
PAPER_RESULTS = PAPER / "results"
MANIFEST = REPO / "evaluation" / "corpus" / "m4_census_v1.json"
OUT = PAPER / "generated" / "numbers.tex"


def _rows(name):
    return list(csv.DictReader(open(RESULTS / f"{name}.csv")))


def _per_series(rows, metric):
    by = defaultdict(dict)
    for row in rows:
        if row["metric"] == metric and row["series"] not in ("paired", "corpus"):
            by[row["series"]][row["implementation"]] = float(row["value"])
    return {k: v for k, v in by.items() if len(v) == 2}


def census():
    """The held-out corpus: size, and how the better optimum forecasts on it."""
    rows = _rows("tier2_accuracy")
    manifest = json.loads(MANIFEST.read_text())
    rmse = _per_series(rows, "rmse")
    relative = [(v["analytic_prophet"] - v["prophet"]) / v["prophet"]
                for v in rmse.values()]
    wins = sum(1 for d in relative if d < 0)
    return {
        "CensusSeries": f"{len(rmse):,}",
        "CensusWeekly": f"{manifest['counts']['Weekly']:,}",
        "CensusDaily": f"{manifest['counts']['Daily']:,}",
        "CensusDigest": manifest["digest"][:12],
        "RmseAdvantage": f"{-statistics.median(relative) * 100:.2f}",
        "RmseWins": f"{wins:,}",
        "RmseWinRate": f"{wins / len(relative) * 100:.0f}",
    }


def posterior():
    """Tier 0: the posterior margin at each length, under Stan's own density."""
    by = defaultdict(dict)
    for row in _rows("tier0_agreement"):
        if row["metric"] == "lp__" and row["series"].startswith("peyton"):
            size = int(row["series"].split("[:")[1].rstrip("]"))
            by[size][row["implementation"]] = float(row["value"])
    out = {}
    for size, cell in sorted(by.items()):
        margin = cell["analytic_prophet"] - cell["prophet"]
        tag = {300: "Short", 1000: "Mid", 2905: "Long"}.get(size)
        if tag:
            out[f"LpMargin{tag}"] = f"{margin:+.3f}"
            out[f"LpPercent{tag}"] = f"{margin / cell['prophet'] * 100:.4f}"
    if {"LpPercentShort", "LpPercentLong"} <= out.keys():
        out["LpDecline"] = (f"{float(out['LpPercentShort']) / float(out['LpPercentLong']):.0f}")
    return out


def _word(count):
    """Small counts as words, as prose writes them: six, not 6."""
    words = "zero one two three four five six seven eight nine ten".split()
    return words[count] if 0 <= count < len(words) else f"{count:,}"


def _scientific(value):
    """A magnitude for math mode: 5.9\\times10^{-8}."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def certificate():
    """#170: the first-order certificate at both solutions, over the census."""
    rows = list(csv.DictReader(open(PAPER_RESULTS / "kkt_census.csv")))
    by = defaultdict(dict)
    for row in rows:
        by[row["series"]][row["implementation"]] = row
    pairs = [(cell["prophet"], cell["analytic_prophet"]) for _, cell in sorted(by.items())]
    n = len(pairs)

    def values(column, side):
        return [float(pair[side][column]) for pair in pairs]

    ours = values("norm_strict", 1)
    strict = values("norm_strict", 0)
    relaxed = values("norm_relaxed", 0)
    worse = sum(1 for theirs, mine in zip(relaxed, ours) if theirs > mine)
    ratios = [theirs / mine for theirs, mine in zip(relaxed, ours)]

    blocks = [pair[0]["argmax_relaxed"].split("[")[0] for pair in pairs]
    at_kink = [abs(float(pair[0]["delta_at_argmax_strict"])) for pair in pairs
               if pair[0]["argmax_strict"].startswith("delta")]
    margins = [float(mine["lp"]) - float(theirs["lp"]) for theirs, mine in pairs]

    detail = list(csv.DictReader(open(PAPER_RESULTS / "kkt_peyton_manning.csv")))

    def peyton(implementation, column):
        return max(abs(float(row[column])) for row in detail
                   if row["implementation"] == implementation)

    return {
        "KktSeries": f"{n:,}",
        "KktOursMedian": f"{statistics.median(ours):.2f}",
        "KktOursWorst": f"{max(ours):.1f}",
        "KktProphetStrict": f"{statistics.median(strict):.0f}",
        "KktProphetRelaxed": f"{statistics.median(relaxed):.1f}",
        "KktProphetRelaxedMin": f"{min(relaxed):.1f}",
        "KktWorse": f"{worse:,}",
        "KktRatio": f"{statistics.median(ratios):.0f}",
        "KktRatioMin": f"{min(ratios):.1f}",
        "KktStrictAtRate": f"{sum(1 for p in pairs if p[0]['argmax_strict'].startswith('delta')) / n * 100:.0f}",
        "KktKinkRate": _scientific(statistics.median(at_kink)),
        "KktRelaxedAtRate": f"{blocks.count('delta') / n * 100:.0f}",
        "KktRelaxedAtTrend": f"{(blocks.count('k') + blocks.count('m')) / n * 100:.0f}",
        "KktLpWins": f"{sum(1 for margin in margins if margin > 0):,}",
        "KktLpMedian": f"{statistics.median(margins):.1f}",
        "KktLpMin": f"{min(margins):.2f}",
        "KktPeytonProphetStrict": f"{peyton('prophet', 'residual_strict'):.1f}",
        "KktPeytonProphetRelaxed": f"{peyton('prophet', 'residual_relaxed'):.1f}",
        "KktPeytonOurs": f"{peyton('analytic_prophet', 'residual_strict'):.2f}",
    }


def margin():
    """#173: the lp__ margin on the fine grid of T, both configurations."""
    rows = list(csv.DictReader(open(PAPER_RESULTS / "margin.csv")))
    cell = {(row["configuration"], int(row["observations"])): float(row["margin"])
            for row in rows}
    lbfgs = [float(row["margin"]) for row in rows if row["prophet_algorithm"] == "LBFGS"]
    newton = [float(row["margin"]) for row in rows if row["prophet_algorithm"] == "Newton"]
    yearly_from = min(size for configuration, size in cell if configuration == "yearly_only")

    def figure(value):
        return f"{value:.2g}"

    return {
        "MarginCells": f"{len(rows):,}",
        "MarginLengths": f"{sum(1 for configuration, _ in cell if configuration == 'default')}",
        "MarginLbfgsMin": figure(min(lbfgs)),
        "MarginLbfgsMax": figure(max(lbfgs)),
        "MarginLbfgsMedian": figure(statistics.median(lbfgs)),
        "MarginNewtonMedian": figure(statistics.median(newton)),
        "MarginAtNinetyNine": figure(cell[("default", 99)]),
        "MarginAtHundred": figure(cell[("default", 100)]),
        "MarginBeforeYearly": figure(cell[("default", yearly_from - 1)]),
        "MarginAfterYearly": figure(cell[("default", yearly_from)]),
        "MarginYearlyFrom": f"{yearly_from}",
        "MarginYearlyLong": f"{cell[('yearly_only', 2905)]:+.2f}",
    }


def newton():
    """#174: one Newton method on two parameterizations, and Stan's beside it."""
    rows = list(csv.DictReader(open(PAPER_RESULTS / "newton.csv")))

    def column(arm, name, kind=float):
        return [kind(row[name]) for row in rows if row["arm"] == arm]

    def span(values, form):
        return form(min(values)), form(max(values))

    lengths = sorted(set(column("split", "observations", int)))
    out = {"NewtonLengths": _word(len(lengths)),
           "NewtonLengthMin": f"{lengths[0]}", "NewtonLengthMax": f"{lengths[-1]}"}
    for prefix, arm, name, form in (
            ("NewtonSplitIter", "split", "iterations", lambda v: f"{v:.0f}"),
            ("NewtonSplitZeros", "split", "exact_zeros", lambda v: f"{v:.0f}"),
            ("NewtonIter", "natural", "iterations", lambda v: f"{v:,.0f}"),
            ("NewtonShort", "natural", "shortfall", lambda v: f"{v:.1f}"),
            ("NewtonFlips", "natural", "sign_flips", lambda v: f"{v:,.0f}"),
            ("NewtonProgress", "natural", "median_progress_tail", _scientific),
            ("NewtonStan", "stan_newton", "shortfall", lambda v: f"{v:.2g}")):
        out[f"{prefix}Min"], out[f"{prefix}Max"] = span(column(arm, name), form)
    return out


def _evaluation():
    """The report and showcase modules, so the paper reuses their definitions
    -- Holm's adjustment, the showcase's pool -- rather than restating them."""
    import sys
    for path in (REPO, REPO / "evaluation", REPO / "benchmark"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import report
    import showcase
    return report, showcase


def limitations():
    """#178: every figure the limitations section states, from committed results."""
    report, showcase = _evaluation()
    rows = _rows("tier2_accuracy")
    paired = {row["metric"]: float(row["value"]) for row in rows
              if row["series"] == "paired" and row["configuration"] == "all"}

    coverage = defaultdict(list)
    for row in rows:
        if row["metric"] == "coverage" and row["series"] not in ("paired", "corpus"):
            coverage[row["implementation"]].append(float(row["value"]))

    rmse = _per_series(rows, "rmse")
    relative = [(v["analytic_prophet"] - v["prophet"]) / v["prophet"] for v in rmse.values()]
    pool = showcase._advantages(RESULTS / "tier2_accuracy.csv")

    adjusted = report.holm({metric: paired[f"{metric}_p_value"]
                            for metric in report.TIER2_COMPARISONS})
    survivors = sum(1 for value in adjusted.values() if value < 0.05)

    calendar = {row["frequency"]: row for row in
                csv.DictReader(open(PAPER_RESULTS / "m4_calendar.csv"))}
    daily = calendar["Daily"]

    overhead = list(csv.DictReader(open(PAPER_RESULTS / "overhead.csv")))
    lbfgs = [row for row in overhead if row["prophet_algorithm"] == "LBFGS"]
    newton = min((row for row in overhead if row["prophet_algorithm"] == "Newton"),
                 key=lambda row: int(row["observations"]))

    def share(row):
        return float(row["one_iteration_call"]) / float(row["prophet_fit"])

    def parity(row):
        return (float(row["prophet_fit"]) - float(row["one_iteration_call"])) / float(row["ours_fit"])

    return {
        "LimCoverageOurs": f"{statistics.fmean(coverage['analytic_prophet']):.3f}",
        "LimCoverageProphet": f"{statistics.fmean(coverage['prophet']):.3f}",
        "LimWidthMedian": f"{paired['interval_width_median']:.2f}",
        "LimWidthWider": f"{paired['interval_width_losses']:,.0f}",
        "LimWorstCensus": f"{max(relative) * 100:.0f}",
        "LimBestCensus": f"{-min(relative) * 100:.1f}",
        "LimWorstPool": f"{-pool[-1][0] * 100:.1f}",
        "LimPoolSize": f"{len(pool):,}",
        "LimPoolThreshold": f"{showcase.FITS_AT_ALL:.2f}",
        "LimHolmSurvivors": _word(survivors),
        "LimHolmComparisons": _word(len(report.TIER2_COMPARISONS)),
        "LimHolmMin": _scientific(min(adjusted.values())),
        "LimHolmMax": _scientific(max(adjusted.values())),
        "LimMfourSeries": f"{sum(int(row['m4_series']) for row in calendar.values()):,}",
        "LimMfourFrequencies": _word(len(calendar)),
        "LimDailyModalShare": f"{int(daily['modal_count']) / int(daily['census_series']) * 100:.0f}",
        "LimDailyStarts": f"{int(daily['distinct_starts'])}",
        "LimWeeklyFutureStarts": f"{int(calendar['Weekly']['starts_after_run_year'])}",
        "LimOverheadShareMin": f"{min(map(share, lbfgs)) * 100:.0f}",
        "LimOverheadShareMax": f"{max(map(share, lbfgs)) * 100:.0f}",
        "LimParityMin": f"{min(map(parity, lbfgs)):.1f}",
        "LimParityMax": f"{max(map(parity, lbfgs)):.1f}",
        "LimNewtonLength": f"{int(newton['observations'])}",
        "LimNewtonSpeedup": f"{float(newton['prophet_fit']) / float(newton['ours_fit']):.1f}",
        "LimNewtonShare": f"{share(newton) * 100:.0f}",
    }


def _log_crossing(epsilon, gap, bias):
    """Where the falling gap meets the rising bias, interpolated in log space."""
    import math
    for i in range(1, len(epsilon)):
        if bias[i] > gap[i] and bias[i - 1] <= gap[i - 1]:
            x0, x1 = math.log10(epsilon[i - 1]), math.log10(epsilon[i])
            r0, r1 = math.log10(gap[i - 1] / bias[i - 1]), math.log10(gap[i] / bias[i])
            return 10 ** (x0 + (x1 - x0) * r0 / (r0 - r1))
    raise ValueError("the curves do not cross on this grid")


def _two_figures(value):
    """Two significant figures, in LaTeX's notation when small: for math mode."""
    return _scientific(value) if abs(value) < 1e-3 else f"{value:.2g}"


def smoothing():
    """#168: Prophet's program with the prior smoothed, fitted by Stan's L-BFGS."""
    rows = list(csv.DictReader(open(PAPER_RESULTS / "smoothing.csv")))
    for row in rows:
        for key, value in row.items():
            if key not in ("stan_algorithm", "stan_stopped_by"):
                row[key] = float(value)
    by = defaultdict(list)
    for row in rows:
        by[int(row["observations"])].append(row)
    tags = {300: "Short", 1000: "Mid", 2905: "Long"}
    epsilons = sorted({row["epsilon"] for row in rows})
    out = {"SmoothValues": f"{len(epsilons)}",
           "SmoothEpsMin": _scientific(epsilons[0]).replace("1.0\\times", ""),
           "SmoothRuns": f"{len(rows)}",
           "SmoothStoppedGradient": f"{sum(1 for r in rows if 'relative gradient' in r['stan_stopped_by'])}",
           "SmoothDecrementMax": _scientific(max(r["newton_decrement"] for r in rows)),
           "SmoothIterMin": f"{min(r['stan_iterations'] for r in rows):,.0f}",
           "SmoothIterMax": f"{max(r['stan_iterations'] for r in rows):,.0f}"}
    for size, cells in sorted(by.items()):
        tag = tags[size]
        cells.sort(key=lambda r: r["epsilon"])
        first = cells[0]
        prophet = first["lp_l1_L1"] - first["lp_l1_prophet"]
        best = min(cells, key=lambda r: r["total_l1"])
        crossing = _log_crossing([r["epsilon"] for r in cells], [r["gap_smooth"] for r in cells],
                                 [r["bias_l1"] for r in cells])
        plateau = [r["gap_smooth"] for r in cells if r["epsilon"] >= 1e-4]
        drop = first["gap_smooth"] / min(r["gap_smooth"] for r in cells)
        out.update({
            f"SmoothDrop{tag}": f"{float(f'{drop:.2g}'):,.0f}",
            f"SmoothGapAtMin{tag}": f"{first['gap_smooth']:.2f}",
            f"SmoothBest{tag}": _two_figures(best["total_l1"]),
            f"SmoothBestEps{tag}": _scientific(best["epsilon"]).replace("1.0\\times", ""),
            f"SmoothFactor{tag}": f"{prophet / best['total_l1']:,.0f}",
            f"SmoothCrossing{tag}": _scientific(crossing),
            f"SmoothPlateauMin{tag}": _two_figures(min(plateau)),
            f"SmoothPlateauMax{tag}": _two_figures(max(plateau)),
        })
    return out


def stall():
    """#172: the prophet#842 reproducer on current Prophet and this implementation."""
    points = {row["point"]: row for row in csv.DictReader(open(PAPER_RESULTS / "issue842.csv"))}
    variants = list(csv.DictReader(open(PAPER_RESULTS / "issue842_variants.csv")))
    summary = json.loads((PAPER_RESULTS / "issue842.meta.json").read_text())["summary"]

    def value(name, column):
        return float(points[name][column])

    eigenvalues = summary["eigenvalues"]
    smoothed = {float(row["epsilon"]): row for name, row in points.items()
                if name.startswith("smoothed")}
    best_smoothed = min(smoothed.values(), key=lambda row: float(row["shortfall"]))
    lbfgs = [row for row in variants if row["algorithm"] == "LBFGS"]
    newton = [row for row in variants if row["algorithm"] == "Newton"]
    optima = summary["local_optima"]
    return {
        "StallLbfgsIter": f"{int(points['prophet LBFGS']['iterations'])}",
        "StallLbfgsShort": f"{value('prophet LBFGS', 'shortfall'):.3f}",
        "StallLbfgsRelaxed": f"{value('prophet LBFGS', 'residual_relaxed'):.2f}",
        "StallLbfgsCosine": f"{value('prophet LBFGS', 'cos_softest'):.3f}",
        "StallNewtonShort": _scientific(value("prophet Newton", "shortfall")),
        "StallNewtonIter": f"{int(points['prophet Newton']['iterations'])}",
        "StallSmoothShort": _scientific(float(best_smoothed["shortfall"])),
        "StallSmoothEps": _scientific(float(best_smoothed["epsilon"])).replace("1.0\\times", ""),
        "StallSmoothIdentical": f"{value('smoothed eps=1e-08', 'shortfall'):.3f}",
        "StallCondition": f"{eigenvalues[-1] / eigenvalues[0]:.0f}",
        "StallStarts": f"{sum(count for _, count in optima)}",
        "StallOptima": _word(len(optima)),
        "StallBestReach": f"{optima[0][1]}",
        "StallOursShort": f"{value('ours default', 'shortfall'):.2f}",
        "StallOursEigen": f"{value('ours default', 'min_eigenvalue'):.2f}",
        "StallVariants": _word(len(lbfgs)),
        "StallVariantShortMin": f"{min(float(r['shortfall']) for r in lbfgs):.3f}",
        "StallVariantShortMax": f"{max(float(r['shortfall']) for r in lbfgs):.2f}",
        "StallVariantIterMin": f"{min(int(r['iterations']) for r in lbfgs)}",
        "StallVariantIterMax": f"{max(int(r['iterations']) for r in lbfgs)}",
        "StallVariantNewtonMax": _scientific(max(float(r["shortfall"]) for r in newton)),
        "StallSecondsMax": f"{max(float(r['seconds']) for r in variants):.2f}",
    }


def tau_path():
    """#171: the solution path in lambda = 1/tau, Prophet's and the exact one."""
    import numpy as np
    rows = list(csv.DictReader(open(PAPER_RESULTS / "tau_path.csv")))
    refine = list(csv.DictReader(open(PAPER_RESULTS / "tau_refine.csv")))
    pairs = list(csv.DictReader(open(PAPER_RESULTS / "tau_pairs.csv")))
    for row in rows:
        for key, value in row.items():
            if key not in ("grid", "setting", "path"):
                row[key] = float(value)

    def path(grid, setting, name):
        cells = sorted((r for r in rows if r["grid"] == grid and r["setting"] == setting
                        and r["path"] == name), key=lambda r: r["lam"])
        return cells, np.array([[r[f"delta_{j + 1}"] for j in range(25)] for r in cells])

    def steps(grid, setting, name):
        _, delta = path(grid, setting, name)
        return np.max(np.abs(np.diff(delta, axis=0)), axis=1)

    def final(setting, name):
        level = max(int(r["level"]) for r in refine)
        return next(float(r["step"]) for r in refine if r["setting"] == setting
                    and r["path"] == name and int(r["level"]) == level)

    settings = ("fixed", "free")
    exact_residual = max(r["residual"] for r in rows if r["path"] == "ours_exact")
    prophet_residual = statistics.median(r["residual"] for r in rows
                                         if r["path"] == "prophet" and r["grid"] == "global")
    gap = {}
    for setting in settings:
        prophet_cells, _ = path("global", setting, "prophet")
        exact_cells, _ = path("global", setting, "ours_exact")
        gap[setting] = [e["lp"] - q["lp"] for e, q in zip(exact_cells, prophet_cells)]
    default_cells, default_delta = path("global", "free", "ours_default")
    _, exact_delta = path("global", "free", "ours_exact")
    widths = [float(r["width"]) for r in refine if int(r["level"]) == max(int(x["level"]) for x in refine)]
    zoom_cells, _ = path("zoom", "free", "ours_exact")
    global_cells, _ = path("global", "free", "ours_exact")
    at = {(float(r["tau"]), r["setting"], r["path"]): r for r in pairs}

    def pair(tau, setting, name, column):
        return abs(float(at[(tau, setting, name)][column]))

    return {
        "TauGlobalPoints": f"{len(global_cells)}",
        "TauLamMin": f"{global_cells[0]['lam']:.0f}",
        "TauLamMax": f"{global_cells[-1]['lam']:.0f}",
        "TauZoomPoints": f"{len(zoom_cells)}",
        "TauZoomStep": f"{zoom_cells[1]['lam'] - zoom_cells[0]['lam']:.3f}",
        "TauLevels": _word(max(int(r["level"]) for r in refine)),
        "TauFinalWidth": f"{max(widths):.4f}",
        "TauProphetFinalFixed": f"{final('fixed', 'prophet'):.2g}",
        "TauProphetFinalFree": f"{final('free', 'prophet'):.2g}",
        "TauExactFinal": _scientific(max(final(s, 'ours_exact') for s in settings)),
        "TauProphetZoomMax": f"{max(steps('zoom', s, 'prophet').max() for s in settings):.2g}",
        "TauExactZoomStep": f"{max(steps('zoom', s, 'ours_exact').max() for s in settings):.2g}",
        "TauExactResidualMax": _scientific(exact_residual),
        "TauProphetResidualMedian": f"{prophet_residual:.0f}",
        "TauGapMedianFixed": f"{statistics.median(gap['fixed']):.2g}",
        "TauGapMedianFree": f"{statistics.median(gap['free']):.2g}",
        "TauGapMax": f"{max(max(v) for v in gap.values()):.2g}",
        "TauDefaultDeviation": f"{float(np.max(np.abs(default_delta - exact_delta))):.2g}",
        "TauPairProphetFree": f"{pair(0.01, 'free', 'prophet', 'delta_change'):.2g}",
        "TauPairProphetFixed": f"{pair(0.01, 'fixed', 'prophet', 'delta_change'):.2g}",
        # at prophet#1280's own value, tau = 0.01 against float32(0.01)
        "TauPairExact": _scientific(max(pair(0.01, s, "ours_exact", "delta_change")
                                        for s in settings)),
        "TauPairLpMax": f"{max(pair(0.01, s, 'prophet', 'lp_change') for s in settings):.2g}",
        "TauPairForecastMax": f"{max(pair(0.01, s, 'prophet', 'yhat_relative') for s in settings) * 100:.2g}",
    }


def macros():
    """Every quoted number, by macro name. What the paper says, as data."""
    return {**census(), **posterior(), **certificate(), **margin(), **newton(),
            **limitations(), **smoothing(), **stall(), **tau_path()}


def render(values):
    lines = ["% Generated by paper/experiments/quote_numbers.py from the committed",
             "% evaluation results. Do not edit: rerun the script.", ""]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}"
              for name, value in sorted(values.items())]
    return "\n".join(lines) + "\n"


def main():
    values = macros()
    OUT.write_text(render(values))
    print(f"wrote {OUT.relative_to(REPO)} ({len(values)} macros)")
    for name, value in sorted(values.items()):
        print(f"  \\{name:16} {value}")


if __name__ == "__main__":
    main()
