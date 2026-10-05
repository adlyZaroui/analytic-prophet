#!/usr/bin/env python
"""The README's figure: the difference between the two optimizers, drawn.

Issue #100. The README's case rests on tables of nats and p-values. They are
the honest evidence and they persuade nobody in the first ten seconds.

**The honest problem with a showcase here is that the two implementations
mostly agree.** Across the 36 M4 series of Tier 2 the median RMSE advantage
is 0.53%, and a figure of a typical series shows two curves on top of each
other -- a true picture and a useless one. The largest advantage in the same
corpus is 35.3%. A figure of *that* series alone, however carefully captioned,
would leave a reader with an impression the corpus does not support.

So this draws **three** panels, chosen by rule from the committed Tier 2
results: the series where the advantage is largest, the median one, and the
one where Prophet wins by the most. The range is the point. A reader sees
both that the mechanism is real and that it is not always worth much, without
having to take a caption's word for either.

Everything drawn is held out. For each series the last cross-validation
cutoff is taken, both implementations are fitted on the history up to it, and
both forecast the horizon beyond it with the actual values drawn on top.

Run it with `python evaluation/showcase.py`. It needs the M4 corpus (cached
by evaluation/corpora.py) and prophet.
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpora
import harness
from tiers import tier2

FIGURE = harness.RESULTS / "figures" / "showcase.png"

# A Prophet-shaped model does not fit every M4 series, and a panel where both
# implementations miss badly shows the difficulty of the series rather than
# the difference between two optimizers. So the ranking is taken over the
# series where the model works at all: both implementations within this sMAPE
# held out. 0.10 keeps 11 of the 36 and is stated in the caption, because a
# threshold nobody can see is a thumb on the scale.
FITS_AT_ALL = 0.10

# Panels, in the order they are drawn, each with the rule that picks it.
SELECTIONS = (
    ("largest advantage", "of those, the series where our cross-validated RMSE "
                          "beats Prophet's by the most"),
    ("median", "the series at the median of that same ranking"),
    ("Prophet wins", "the series where Prophet beats us by the most"),
)


def _advantages(results=None):
    """(relative RMSE advantage, series, ours, theirs) per series, best first.

    Read from the committed Tier 2 results rather than recomputed, so the
    selection is reproducible from the repository and a reader can check it
    against the same file the report is built from.
    """
    path = Path(results or harness.RESULTS / "tier2_accuracy.csv")
    frame = pd.read_csv(path)

    def paired(metric):
        wanted = frame[frame["metric"] == metric]
        return {series: dict(zip(group["implementation"], group["value"]))
                for series, group in wanted.groupby("series")}

    rmse, smape = paired("rmse"), paired("smape")

    rows = []
    for series, values in rmse.items():
        ours, theirs = values.get("analytic_prophet"), values.get("prophet")
        fit = smape.get(series, {})
        if ours is None or theirs is None or not theirs:
            continue
        if not {"analytic_prophet", "prophet"} <= set(fit):
            continue
        # the filter: the worse of the two has to be a usable forecast
        if max(fit["analytic_prophet"], fit["prophet"]) > FITS_AT_ALL:
            continue
        rows.append(((theirs - ours) / theirs, series, ours, theirs))
    rows.sort(reverse=True)
    return rows


def select(results=None):
    """The three series the figure draws, with the rule that chose each."""
    ranked = _advantages(results)
    if len(ranked) < 3:
        raise SystemExit(f"need at least 3 scored series, found {len(ranked)}")
    picks = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    return [(label, rule) + pick
            for (label, rule), pick in zip(SELECTIONS, picks)]


def _frequency_of(series):
    """`m4_daily_D2626` -> `Daily`, which is the key tier2 keys its horizons by."""
    return series.split("_")[1].capitalize()


def _load(series):
    """The frame for one named series, from the same sample Tier 2 drew.

    Same frequency, same count, same seed -- so the names line up with the
    committed results rather than coming from a different draw.
    """
    frequency = _frequency_of(series)
    for name, frame in corpora.m4(frequency, n_series=tier2.N_SERIES,
                                  seed=harness.SEED, download=False):
        if name == series:
            return tier2._dated(frame)
    raise SystemExit(f"{series} is not in the M4 sample; is the corpus cached?")


def _one_window(frame, cutoff, horizon, lib_path, tail):
    """Fit both implementations on the history up to `cutoff`, then predict
    across the whole drawn span -- the tail of the training data as well as
    the horizon past it.

    **Predicting back over the training tail is what removes the step at the
    cutoff.** Drawing only the forecast starts each line at the first future
    point, whose fitted value is not the last observation: at one of the
    series in the first version of this figure the gap was +401 on a series
    around 9000, which reads as a glitch in the plot and is not one. It is the
    model's residual at the end of its own training data.

    Carrying the fitted curve through the cutoff shows it for what it is. Each
    line is then continuous, the distance between a fitted line and the
    observed one is the fit, and the vertical rule marks where the model stops
    having been shown the answer.

    The RMSEs are still computed on the horizon alone. Scoring a model on the
    data it was fitted to would be a different and much kinder number.
    """
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    history = frame[frame["ds"] <= cutoff].reset_index(drop=True)
    future = frame[(frame["ds"] > cutoff) & (frame["ds"] <= cutoff + horizon)]
    future = future.reset_index(drop=True)
    if len(history) < 2 or future.empty:
        return None

    shown_history = history if tail is None else history.tail(tail)
    drawn = pd.concat([shown_history[["ds"]], future[["ds"]]], ignore_index=True)
    n_fitted = len(shown_history)

    ours = AnalyticProphet(**harness.PROPHET_KWARGS)
    ours.rng = np.random.default_rng(harness.SEED)
    ours.fit(history, lib_path=lib_path)
    our_curve = ours.predict(drawn)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Prophet draws its intervals from numpy's global generator, so the
        # figure would differ between runs without this. [fc] tier2.
        np.random.seed(harness.SEED % (2 ** 32))
        theirs = Prophet(**harness.PROPHET_KWARGS).fit(history)
        their_curve = theirs.predict(drawn)

    truth = future["y"].to_numpy()

    def rmse(curve):
        return float(np.sqrt(np.mean(
            (truth - curve["yhat"].to_numpy()[n_fitted:]) ** 2)))

    return {
        "history": history, "shown_history": shown_history,
        "cutoff": cutoff, "future": future,
        "ours": our_curve, "theirs": their_curve, "n_fitted": n_fitted,
        "our_rmse": rmse(our_curve), "their_rmse": rmse(their_curve),
    }


def panel_data(series, cross_validated_advantage, lib_path, tail=None):
    """What one panel draws: history, the cutoff, the truth, both forecasts.

    **Which window.** A series is selected by its cross-validated RMSE, which
    averages over every cutoff -- so drawing one arbitrary window can show
    something the selection number does not describe. The first version of
    this drew the last cutoff and produced exactly that: two panels where both
    implementations happened to badly undershoot a window neither was typical
    of, which says something about forecasting a trend and nothing about the
    difference between two optimizers.

    So the window drawn is the **representative** one: the cutoff whose
    relative advantage is closest to the cross-validated advantage the series
    was selected by. Matching on our RMSE alone was not enough -- it put a
    window where *we* won inside the panel labelled "Prophet wins", which is
    the label contradicting the picture. Matching on the advantage is matching
    on the quantity the label states.
    """
    from prophet.diagnostics import generate_cutoffs

    frame = _load(series)
    frequency = _frequency_of(series)
    horizon_text, initial_text, period_text = tier2.HORIZONS[frequency]
    horizon = pd.Timedelta(horizon_text)

    cutoffs = generate_cutoffs(frame, horizon, pd.Timedelta(initial_text),
                               pd.Timedelta(period_text))
    windows = [w for w in (_one_window(frame, cutoff, horizon, lib_path, tail)
                           for cutoff in cutoffs) if w is not None]
    if not windows:
        raise SystemExit(f"{series}: no usable cutoff")

    def advantage(window):
        theirs = window["their_rmse"]
        return (theirs - window["our_rmse"]) / theirs if theirs else 0.0

    chosen = min(windows,
                 key=lambda w: abs(advantage(w) - cross_validated_advantage))
    chosen["advantage"] = advantage(chosen)
    chosen.update(series=series, horizon=horizon_text, windows=len(windows))
    return chosen


def draw(panels, out_path=FIGURE, tail=None):
    """Three stacked panels, one per selected series.

    `panels` is a list of (selection, panel_data) pairs, in draw order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours_colour = "#4c72b0"
    theirs_colour = "#c44e52"

    figure, axes = plt.subplots(len(panels), 1, figsize=(9.5, 3.1 * len(panels)))
    axes = np.atleast_1d(axes)

    for axis, ((label, _rule, advantage, series, _o, _t), data) in zip(axes, panels):
        shown, future = data["shown_history"], data["future"]

        axis.plot(shown["ds"], shown["y"], color="0.45", linewidth=0.9,
                  label="observed (training)")
        axis.plot(future["ds"], future["y"], color="0.1", linewidth=1.7,
                  label="actual (held out)")

        # One continuous line per model across the whole drawn span: fitted
        # to the left of the cutoff, forecast to the right. Drawing only the
        # right-hand part leaves a step at the cutoff that looks like a
        # plotting bug and is really the model's residual (#100 review).
        for curve, colour, name in (
                (data["theirs"], theirs_colour, "Prophet"),
                (data["ours"], ours_colour, "analytic-prophet")):
            axis.fill_between(curve["ds"], curve["yhat_lower"],
                              curve["yhat_upper"], color=colour, alpha=0.09,
                              linewidth=0)
            axis.plot(curve["ds"], curve["yhat"], color=colour,
                      linewidth=1.5, label=name)

        axis.axvline(data["cutoff"], color="0.4", linestyle=":", linewidth=1.3)
        axis.set_title(
            f"{series} — {label}: {advantage:+.1%} RMSE cross-validated, "
            f"{data['advantage']:+.1%} in this window "
            f"(ours {data['our_rmse']:.0f}, Prophet {data['their_rmse']:.0f})",
            fontsize=9, loc="left")
        axis.tick_params(labelsize=8)
        axis.margins(x=0.01)

    axes[0].legend(fontsize=8, loc="upper left", ncol=4, framealpha=0.9)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(FIGURE))
    parser.add_argument("--results", default=None,
                        help="tier2 results to select from (default: committed)")
    parser.add_argument("--tail", type=int, default=400,
                        help="training points to draw before the cutoff")
    parser.add_argument("--lib-path", default=None)
    args = parser.parse_args(argv)

    if not harness.prophet_available():
        print(harness.PROPHET_INSTALL_HINT, file=sys.stderr)
        return 1

    lib_path = args.lib_path or harness.build_extension(tempfile.mkdtemp())
    chosen = select(args.results)

    panels = []
    for label, rule, advantage, series, ours, theirs in chosen:
        print(f"  {label}: {series} ({advantage:+.1%})")
        panels.append(((label, rule, advantage, series, ours, theirs),
                       panel_data(series, advantage, lib_path, tail=args.tail)))

    written = draw(panels, Path(args.out), tail=args.tail)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
