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


def macros():
    """Every quoted number, by macro name. What the paper says, as data."""
    return {**census(), **posterior(), **certificate()}


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
