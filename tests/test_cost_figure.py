"""
Issue #231: the README's costs as a figure against series length, not a table.

`evaluation/cost_figure.py` draws fit time, prediction time (both samplers)
and the memory a fit adds, each as Prophet's over this package's, at every
length Tier 3 measures. The README shows it where the table was and says three
things about its shape. Checked here:

  * **the figure is the committed results** -- its ratios recomputed from
    `tier3_cost.csv` independently -- and it redraws byte-identically;
  * **the README shows it where the costs are**, found by its file rather than
    a heading, with no table rows left;
  * **the three sentences about the shape hold in the data**: the fit gains
    most below T = 100 and least from T = 1000 up, the exact sampler gains more
    than the approximate one at every length, and the memory gain grows with
    the series.
"""
import csv
from pathlib import Path

import pytest

import cost_figure

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
TIER3 = REPO / "evaluation" / "results" / "tier3_cost.csv"
RAW = "https://raw.githubusercontent.com/adlyZaroui/analytic-prophet/main/"


def _measured(implementation, metric):
    """`{T: value}` straight from the CSV, without cost_figure's code."""
    out = {}
    with open(TIER3, newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["series"].startswith("peyton_manning[:") and row["metric"] == metric
                    and row["implementation"] == implementation):
                out[int(row["series"][len("peyton_manning[:"):-1])] = float(row["value"])
    return out


def _ratio(theirs, ours, metric):
    a, b = _measured(theirs, metric), _measured(ours, metric)
    return {size: a[size] / b[size] for size in a}


@pytest.fixture(scope="module")
def plotted():
    return cost_figure.ratios()


def test_the_figure_plots_the_committed_ratios(plotted):
    expected = {
        "fit_wall": {"ours": _ratio("prophet", "compiled", "fit_wall")},
        "predict_wall": {"approximate sampler": _ratio("prophet", "compiled", "predict_wall"),
                         "exact sampler": _ratio("prophet(exact)", "compiled(exact)",
                                                 "predict_wall")},
        "fit_peak_rss_added": {"ours": _ratio("prophet", "compiled", "fit_peak_rss_added")},
    }
    assert plotted == expected
    assert sorted(plotted["fit_wall"]["ours"]) == [50, 100, 300, 1000, 2905]


def test_the_figure_is_committed_and_redraws_identically(plotted, tmp_path):
    """Drawn from the committed results with no remeasuring, and written with
    no timestamp, so a redraw is not a binary diff."""
    pytest.importorskip("matplotlib")
    assert cost_figure.FIGURE.exists() and cost_figure.FIGURE.stat().st_size > 10_000
    first = cost_figure.draw(plotted, tmp_path / "a.png").read_bytes()
    second = cost_figure.draw(plotted, tmp_path / "b.png").read_bytes()
    assert first == second


def test_the_readme_shows_the_figure_where_the_costs_are():
    readme = README.read_text()
    link = f"]({RAW}evaluation/results/figures/cost_ratios.png)"
    assert link in readme
    costs = readme.index("**What it costs**")
    assert costs < readme.index(link) < readme.index("**Very little of the speed is the gradient.**")
    assert "| fit, T = " not in readme and "× faster** |" not in readme


def test_what_the_readme_says_about_the_shape_holds(plotted):
    one = " ".join(README.read_text().split())
    assert ("The fit gains most where Prophet runs Stan's Newton method, below T = 100, "
            "and least from T = 1000 up") in one
    fit = plotted["fit_wall"]["ours"]
    assert max(fit, key=fit.get) < cost_figure.NEWTON_BELOW
    assert min(fit, key=fit.get) >= 1000

    assert "the exact sampler gains more than the approximate one at every length" in one
    predict = plotted["predict_wall"]
    assert all(predict["exact sampler"][size] > predict["approximate sampler"][size]
               for size in predict["exact sampler"])

    assert "the memory gain grows with the series" in one
    memory = [plotted["fit_peak_rss_added"]["ours"][size]
              for size in sorted(plotted["fit_peak_rss_added"]["ours"])]
    assert memory == sorted(memory) and memory[0] > 1


def test_the_rule_the_figure_marks_is_prophets():
    from analytic_prophet.optimizer import NEWTON_BELOW

    assert cost_figure.NEWTON_BELOW == NEWTON_BELOW
