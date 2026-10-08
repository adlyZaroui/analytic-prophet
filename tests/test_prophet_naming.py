"""
Issue #54: the names, and a test that stops them drifting again.

Where this implementation and Prophet hold the same quantity they often called
it something different, and nothing checked that. The divergence grew one task
at a time, which is how it reached fourteen pairs before anyone counted.

The rule settled on: **match Prophet's name, unless the Stan model uses a
different one, in which case match Stan's.** Stan wins because every derivation
in this repository is written against `prophet.stan`, and a comment explaining
`changepoint_prior_scale`'s gradient should use the symbol the model does.

What this module is for is the third part of #54, and the part that matters:
the mapping is enumerated here, so a rename on either side fails a test instead
of quietly widening the gap again. It checks names *and* values — a name that
matches while holding something else would be worse than no match at all.

The deliberate non-matches are enumerated too, with their reasons. Without
that, the next person to notice one has no way to tell a decision from an
oversight, and "fixing" it is a plausible mistake.
"""
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytic_prophet import forecaster
from analytic_prophet import AnalyticProphet


# ours -> Prophet's, for things that now share a name
SHARED_ATTRIBUTES = ["growth", "n_changepoints", "changepoint_range", "seasonalities",
                     "changepoint_prior_scale", "changepoints_t",
                     "extra_regressors", "holidays", "holidays_prior_scale",
                     "holidays_mode", "seasonality_mode", "seasonality_prior_scale",
                     "country_holidays", "train_holiday_names", "interval_width",
                     "uncertainty_samples", "yearly_seasonality", "weekly_seasonality",
                     "daily_seasonality", "y_scale"]

SHARED_CALLABLES = ["add_seasonality", "add_regressor", "add_country_holidays",
                    "validate_column_name", "set_auto_seasonalities",
                    "parse_seasonality_args", "construct_holiday_dataframe",
                    "make_holiday_features", "fourier_series", "predict_trend",
                    "linear_growth_init", "logistic_growth_init", "flat_growth_init",
                    "fit", "predict", "make_future_dataframe"]

# ours -> (theirs, why it stays different)
#
# `fit_cpp` left this table in #98. It was the one name a ported Prophet
# script could not keep, and the compiled core is now what `fit` does -- so
# the difference was closed rather than justified. `fit` is in
# SHARED_CALLABLES above, which is the whole point.
DELIBERATE = {
    "sigma_k": (None, "[stan] the prior scale on k. Stan writes it as a literal "
                      "in `k ~ normal(0, 5)` rather than naming it in the data "
                      "block, and Prophet does not expose it at all."),
    "sigma_m": (None, "[stan] the prior scale on m, likewise a literal there and "
                      "absent from Prophet's API."),
    "T": (None, "[stan] T, the observation count. Prophet reads it off "
                "`history.shape[0]` rather than keeping an attribute."),
}


@pytest.fixture(scope="module")
def prophet_class(prophet_comparison):
    Prophet, _, _ = prophet_comparison
    return Prophet


# -- the names ----------------------------------------------------------

@pytest.mark.parametrize("name", SHARED_ATTRIBUTES)
def test_a_shared_attribute_exists_on_both(prophet_class, name):
    """Fails if either side renames it, which is the whole point of the
    module: the gap closed in #54 must not reopen silently."""
    assert hasattr(AnalyticProphet(), name), f"{name} is gone from this implementation"
    assert hasattr(prophet_class(), name), f"{name} is gone from Prophet"


@pytest.mark.parametrize("name", SHARED_CALLABLES)
def test_a_shared_callable_exists_on_both(prophet_class, name):
    ours = getattr(AnalyticProphet, name, None) or getattr(forecaster, name, None)
    theirs = getattr(prophet_class, name, None)

    assert callable(ours), f"{name} is gone from this implementation"
    assert callable(theirs), f"{name} is gone from Prophet"


@pytest.mark.parametrize("ours,expected", sorted(DELIBERATE.items()))
def test_a_deliberate_difference_is_still_different(prophet_class, ours, expected):
    """The other half. If one of these ever *does* match, either the reason
    stopped applying or someone changed it without reading why -- and a silent
    convergence is as much a surprise as a silent divergence.
    """
    theirs, _reason = expected
    assert hasattr(AnalyticProphet(), ours) or hasattr(forecaster, ours), ours

    if theirs is None or "[" in theirs:
        return
    assert not hasattr(AnalyticProphet(), theirs), (
        f"{ours} and {theirs} now both exist here; DELIBERATE says they should "
        f"not, so either the entry is stale or this was an accident")


def test_every_deliberate_difference_carries_a_reason():
    for ours, (_theirs, reason) in DELIBERATE.items():
        assert reason and len(reason) > 30, (
            f"{ours} is listed as deliberate without saying why, which makes it "
            f"indistinguishable from an oversight")


# -- the values behind them ---------------------------------------------

def test_the_renamed_quantities_hold_what_prophet_holds(prophet_comparison,
                                                        compiled_optimizer_module):
    """A matching name on a different quantity is worse than no match. Each
    pair below was verified in #54 and is re-verified here on a real fit."""
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(1000)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    _, stan_data, _ = bridge.capture_stan_model(prophet_model, df)

    ours = AnalyticProphet()
    ours.fit(df, lib_path=compiled_optimizer_module)

    # y_scale: the divisor, shared name as of #54
    assert ours.y_scale == pytest.approx(prophet_model.y_scale)

    # y_scaled: the scaled series, Prophet's history column
    np.testing.assert_allclose(ours.y_scaled,
                               prophet_model.history["y_scaled"].to_numpy())

    # t against Stan's `t`, which is Prophet's history column too
    np.testing.assert_allclose(ours.t, np.asarray(stan_data["t"], dtype=float))

    # changepoint_prior_scale against Stan's own, under Prophet's constructor name
    assert ours.changepoint_prior_scale == pytest.approx(float(stan_data["tau"]))
    assert ours.changepoint_prior_scale == pytest.approx(prophet_model.changepoint_prior_scale)


def test_fourier_series_matches_prophets_under_the_shared_name(prophet_class,
                                                               peyton_manning_df):
    """Renamed from `fourier_components` in #54. Accessed through the class on
    Prophet's side, so the names cannot shadow each other here."""
    from analytic_prophet import seasonal_time

    dates = pd.to_datetime(peyton_manning_df["ds"])
    ours = forecaster.fourier_series(seasonal_time(dates), 365.25, 10)
    theirs = prophet_class.fourier_series(dates, 365.25, 10)

    assert ours.shape == theirs.shape
    assert np.max(np.abs(ours - theirs)) < 1e-10


def test_the_signature_of_each_shared_callable_is_compatible(prophet_class):
    """Not identical -- several of ours take fewer arguments, and some are
    module functions where Prophet's are methods. What must hold is that every
    parameter *we* require, Prophet also has, so a call written against their
    documentation does not fail on an argument name."""
    for name in ("add_seasonality", "add_regressor", "add_country_holidays"):
        ours = set(inspect.signature(getattr(AnalyticProphet, name)).parameters) - {"self"}
        theirs = set(inspect.signature(getattr(prophet_class, name)).parameters) - {"self"}
        assert ours <= theirs, f"{name} takes {sorted(ours - theirs)}, which Prophet does not"


# -- the names of the files themselves (#142) ------------------------------
#
# The same idea one level up. The README's Layout block claims which of this
# package's modules carry Prophet's own filenames, in two places -- as labels
# in the tree and as a sentence under it -- and both had drifted from the
# package and from each other: the sentence named three where there are four,
# dropping `serialize.py`, which the tree two lines above labelled as Prophet's.
# It also said "the other four have no Prophet counterpart" where there are six.
# Three copies of one fact, so the test is that they agree.

LAYOUT_PACKAGE = Path(__file__).parent.parent / "analytic_prophet"
LAYOUT_README = Path(__file__).parent.parent / "README.md"


def _tree_labelled():
    """Modules the README's file tree labels as carrying a Prophet name."""
    return {match.group(1) for match in re.finditer(
        r"^\s{4}(\S+\.py)\s+Prophet's ", LAYOUT_README.read_text(), re.MULTILINE)}


def _prose_named():
    """Modules the sentence under the tree names as taking Prophet's names."""
    text = " ".join(LAYOUT_README.read_text().split())
    sentence = re.search(r"((?:`[a-z_]+\.py`(?:, | and )?)+) take Prophet's own\s*names",
                         text)
    assert sentence, "the Layout sentence has moved; this test needs updating"
    return set(re.findall(r"`([a-z_]+\.py)`", sentence.group(1)))


def test_the_tree_and_the_sentence_name_the_same_files():
    """They are two copies of one fact and disagreed for however long."""
    assert _tree_labelled() == _prose_named(), (
        f"tree labels {sorted(_tree_labelled())}, sentence names "
        f"{sorted(_prose_named())}")


def test_the_readme_names_exactly_the_modules_that_share_a_name(prophet_class):
    """The claim, against the installed package rather than against memory.

    `__init__.py` is excluded: every package has one, and nobody means to
    count it among the files named after Prophet's.
    """
    import os

    upstream = {f for f in os.listdir(os.path.dirname(inspect.getfile(prophet_class)))
                if f.endswith(".py")} - {"__init__.py"}
    ours = {p.name for p in LAYOUT_PACKAGE.glob("*.py")} - {"__init__.py"}

    assert _prose_named() == ours & upstream, (
        f"the README names {sorted(_prose_named())}; the modules that actually "
        f"share a name with Prophet's are {sorted(ours & upstream)}")


def test_the_readme_does_not_count_the_files_it_does_not_name():
    """[#142] The sentence used to end "the other four have no Prophet
    counterpart" where there are six, and the clause after it named three of
    them. A bare count of the complement goes stale whenever a module is added,
    which is twice now, so there is no longer a number there to go stale."""
    text = " ".join(LAYOUT_README.read_text().split())

    assert not re.search(r"[Tt]he other (one|two|three|four|five|six|seven|\d+)", text), (
        "the Layout paragraph counts the modules it does not name again; that "
        "count has been wrong twice")


# -- the [fc] marker in the documentation (#150) ---------------------------
#
# `[fc]` marks a decision taken from Prophet's `forecaster.py`. #132 removed it
# from the README, where it had drifted into a second meaning and in one place
# named the wrong file. The same two faults were still in docs/: the single use
# in non-smooth-objective.md pointed at `CmdStanPyBackend.fit`, which lives in
# Prophet's models.py, and deviations.md used it five times without defining
# it. The marker is only worth having if it means one thing, so these check
# both halves -- that a file using it says what it is, and that what it points
# at is actually there.

DOCS = Path(__file__).parent.parent / "docs"
FC_DEFINITION = "`python/prophet/forecaster.py`"


def _fc_uses(text):
    """`[fc]` markers paired with the first backticked symbol that follows.

    The convention is "`[fc]` then the symbol", sometimes across a line break,
    so this looks ahead a little rather than requiring them adjacent.

    The occurrence inside the definition sentence is not a use and is skipped,
    which is a distinction worth making rather than working around: a file that
    defines the marker names a *file*, and every other occurrence names a
    *symbol* in it.
    """
    found = []
    for match in re.finditer(r"\[fc\]", text):
        window = text[match.end():match.end() + 160]
        if FC_DEFINITION in window[:60]:      # the definition, not a use
            continue
        symbol = re.search(r"`([A-Za-z_][\w.]*)`", window)
        found.append(symbol.group(1) if symbol else None)
    return found


@pytest.mark.parametrize("path", sorted(DOCS.glob("*.md")), ids=lambda p: p.name)
def test_a_doc_using_fc_says_what_it_means(path):
    text = path.read_text()
    if "[fc]" not in text:
        pytest.skip("does not use the marker")

    assert FC_DEFINITION in text, (
        f"{path.name} uses `[fc]` without defining it; a reader arriving from "
        "the README has no way to resolve it")


@pytest.mark.parametrize("path", sorted(DOCS.glob("*.md")), ids=lambda p: p.name)
def test_every_fc_marker_names_a_symbol(path):
    """A marker with nothing identifiable after it cannot be checked, so the
    convention is that one always follows."""
    uses = _fc_uses(path.read_text())
    if not uses:
        pytest.skip("does not use the marker")

    assert None not in uses, (
        f"{path.name} has an `[fc]` with no backticked symbol after it")


def test_what_fc_points_at_is_in_prophets_forecaster(prophet_class):
    """The fault that got through twice: the marker naming the wrong file.

    `[fc]` means `forecaster.py` specifically. In the README it had been used
    for `model_to_json` (prophet/serialize.py, #132) and in
    non-smooth-objective.md for `CmdStanPyBackend.fit` (prophet/models.py).
    Both read as typos and were. Checked against the installed package.
    """
    import os

    source = open(os.path.join(os.path.dirname(inspect.getfile(prophet_class)),
                               "forecaster.py")).read()
    wrong = []
    for path in sorted(DOCS.glob("*.md")):
        for symbol in _fc_uses(path.read_text()):
            if symbol and symbol.split(".")[-1] not in source:
                wrong.append(f"{path.name}: {symbol}")

    assert wrong == [], (
        f"marked `[fc]` but not in Prophet's forecaster.py: {wrong}")
