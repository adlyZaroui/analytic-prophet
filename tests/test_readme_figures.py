"""
Issue #211: the README's figures -- the showcase's dates, and the tau path.

**The showcase drew M4's dates.** Its middle panel, a weekly series, ran from
2087 to 2094: M4's starting dates are partly placeholders and their two-digit
years put some weekly series in the future (paper/citations.md, m4methods).
The census keeps those dates on purpose -- both implementations see the same
ones -- so the figure changes, not the parsing: each axis now counts time
from the series' first observation.

**The tau path's forecast panels are the clearest picture of the problem.**
`paper/experiments/tau_path.py` draws a README version of them from the
committed CSV, with no refit.

Both PNGs are committed and deterministic. The showcase needs the M4 cache and
two fits per cutoff, so its determinism is the script's (seeded, and checked by
regenerating it before committing); what is checked here is that no calendar
date reaches its axis.
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"


def _tau_path():
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("tau_path", EXPERIMENTS / "tau_path.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _weekly_panel(start="2087-01-04", length=60, cutoff_at=48):
    """A synthetic panel in the shape `showcase.panel_data` returns, dated where
    the real middle panel was."""
    ds = pd.date_range(start, periods=length, freq="W")
    frame = pd.DataFrame({"ds": ds, "y": np.linspace(1200.0, 1400.0, length)})
    history, future = frame.iloc[:cutoff_at + 1], frame.iloc[cutoff_at + 1:]
    curve = pd.DataFrame({"ds": ds, "yhat": frame["y"],
                          "yhat_lower": frame["y"] - 50, "yhat_upper": frame["y"] + 50})
    data = {"history": history, "shown_history": history, "future": future,
            "cutoff": ds[cutoff_at], "ours": curve, "theirs": curve,
            "advantage": 0.001, "our_rmse": 122.0, "their_rmse": 122.0}
    return (("median", "rule", 0.001, "m4_weekly_W103", 122.0, 122.0), data)


def test_the_elapsed_axis_counts_from_the_first_observation():
    from showcase import elapsed_axis

    history = pd.DataFrame({"ds": pd.date_range("2087-01-04", periods=10, freq="W")})
    to_elapsed, label = elapsed_axis("m4_weekly_W103", history)
    assert label == "weeks since the series' first observation"
    first = history["ds"].iloc[0]
    assert to_elapsed(first) == 0
    assert to_elapsed(first + pd.Timedelta(weeks=52)) == 52

    daily = pd.DataFrame({"ds": pd.date_range("2015-03-30", periods=10, freq="D")})
    to_elapsed, label = elapsed_axis("m4_daily_D2040", daily)
    assert label == "days since the series' first observation"
    assert list(to_elapsed(daily["ds"])) == list(range(10))

    with pytest.raises(KeyError):
        elapsed_axis("m4_hourly_H1", daily)


def test_no_calendar_year_reaches_the_showcase_axis(tmp_path, monkeypatch):
    """Drawn from a panel dated 2087, the axis must carry elapsed weeks and
    no year. The figure is inspected when it is saved, after layout."""
    pytest.importorskip("matplotlib")
    import matplotlib.figure

    import showcase

    seen = {}

    def capture(figure, *args, **kwargs):
        figure.canvas.draw()
        axis = figure.axes[0]
        seen["ticks"] = [label.get_text() for label in axis.get_xticklabels()]
        seen["xlabel"] = axis.get_xlabel()
        seen["lines"] = [line.get_xdata() for line in axis.get_lines()]

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    showcase.draw([_weekly_panel()], tmp_path / "showcase.png")

    assert seen["xlabel"] == "weeks since the series' first observation"
    assert seen["ticks"] and not any(re.search(r"20\d\d|2\d{3}-", tick) for tick in seen["ticks"])
    # every plotted point is a number of weeks from the start, the first at zero
    starts = [float(np.min(np.asarray(x, dtype=float))) for x in seen["lines"]]
    assert min(starts) == 0.0


def test_the_committed_showcase_is_the_one_the_readme_shows():
    figure = REPO / "evaluation" / "results" / "figures" / "showcase.png"
    assert figure.exists() and figure.stat().st_size > 10_000
    readme = " ".join((REPO / "README.md").read_text().split())
    assert "counts time from the series' first observation" in readme


@pytest.fixture(scope="module")
def tau_path():
    return _tau_path()


def test_the_readme_tau_figure_is_committed_and_redraws_identically(tau_path, tmp_path):
    """Drawn from the committed CSV with no refit; redrawing unchanged results
    must give an unchanged file."""
    pytest.importorskip("matplotlib")
    figure = tau_path.README_FIGURE
    assert figure == PAPER / "figures" / "tau_path_readme.png"
    assert figure.exists() and figure.stat().st_size > 10_000
    rows = tau_path.read(tau_path.PATH)
    first = tau_path.draw_readme(rows, tmp_path / "a.png").read_bytes()
    second = tau_path.draw_readme(rows, tmp_path / "b.png").read_bytes()
    assert first == second
    assert first.startswith(b"\x89PNG")


def test_the_readme_tau_figure_shows_what_the_issue_says(tau_path):
    """'Prophet's forecast jumps irregularly, while the exact path moves in a
    straight line' -- checked on the data the figure draws."""
    rows = tau_path.read(tau_path.PATH)
    for setting in ("fixed", "free"):
        def forecasts(path):
            cells = sorted((r for r in rows if r["grid"] == "zoom" and r["setting"] == setting
                            and r["path"] == path), key=lambda r: r["tau"])
            return np.diff([r["yhat_h"] for r in cells])

        exact, prophet = forecasts("ours_exact"), forecasts("prophet")
        assert len(exact) == len(prophet) == len(tau_path.ZOOM) - 1
        # the exact path: every step the same sign and nearly the same size
        assert np.all(exact < 0) and np.ptp(exact) < 0.2 * np.abs(exact).mean()
        # Prophet's: steps of both signs, and some a hundred times the exact one
        assert np.sum(np.diff(np.sign(prophet)) != 0) > len(prophet) // 3
        assert np.abs(prophet).max() > 100 * np.abs(exact).mean()


def test_the_readme_figure_does_not_ship_with_the_paper():
    """`\\includegraphics{figures/tau_path}` takes no extension, so the README
    PNG has its own name, and the arXiv tarball packs PDFs only."""
    makefile = (PAPER / "Makefile").read_text()
    recipe = next(line for line in makefile.splitlines() if "tar czf arxiv.tar.gz" in line)
    assert recipe.split()[-1] == "figures/*.pdf"
    tex = "\n".join(path.read_text() for path in (PAPER / "sections").glob("*.tex"))
    assert "tau_path_readme" not in tex
