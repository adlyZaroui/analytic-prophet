"""
Issue #101: how far from a drop-in, enumerated.

The README said "not yet a drop-in replacement" and named three things. That
was not the whole of it, and nobody could tell how far off it actually is
without reading both codebases. This file is the register, and
`docs/deviations.md` carries the table built from it.

**Why a hand-written registry rather than pure introspection.** Three of the
four verdicts can be derived -- whether a name exists on our class, at module
level, or nowhere. The fourth cannot: *deliberate* and *gap* are judgments
about whether something should exist, and no amount of `dir()` decides that.
So the judgments live here, reviewed, and introspection checks everything
around them: that no Prophet member is missing from the register, that no
entry names something Prophet no longer has, and that each entry's verdict
still matches where the name actually is.

That split is what keeps the table from going stale in either direction. If
Prophet adds a method, the first test fails. If we implement one of the gaps,
the third fails, because the verdict says absent and the name is now there.

MCMC is out of scope by decision rather than omission, which is why so much
of the sampling surface is "absent, deliberate" rather than "gap".
"""
import re
from pathlib import Path

import pytest

from analytic_prophet import AnalyticProphet
import analytic_prophet

REPO = Path(__file__).parent.parent
DEVIATIONS = REPO / "docs" / "deviations.md"

SAME = "same"
MOVED = "moved"
DELIBERATE = "absent, deliberate"
GAP = "absent, gap"

# Prophet's public methods -> (verdict, where it is here / why it is not)
METHODS = {
    # -- the same name in the same place ---------------------------------
    "add_country_holidays": (SAME, "—"),
    "add_regressor": (SAME, "—"),
    "add_seasonality": (SAME, "—"),
    "calculate_initial_params": (SAME, "—"),
    "construct_holiday_dataframe": (SAME, "—"),
    "fit": (SAME, "runs the compiled core; `backend=\"python\"` for the reference path (#98)"),
    "make_all_seasonality_features": (SAME, "—"),
    "make_future_dataframe": (SAME, "—"),
    "predict": (SAME, "—"),
    "preprocess": (SAME, "also does `setup_dataframe`'s and `initialize_scales`' work"),
    "set_auto_seasonalities": (SAME, "—"),
    "set_changepoints": (SAME, "—"),
    "validate_column_name": (SAME, "—"),

    # -- present, reachable differently -----------------------------------
    # Internal in Prophet too, so almost nobody calls them on the instance --
    # but `from analytic_prophet import fourier_series` works and
    # `model.fourier_series(...)` does not.
    "flat_growth_init": (MOVED, "`analytic_prophet.flat_growth_init`"),
    "fourier_series": (MOVED, "`analytic_prophet.fourier_series`"),
    "linear_growth_init": (MOVED, "`analytic_prophet.linear_growth_init`"),
    "logistic_growth_init": (MOVED, "`analytic_prophet.logistic_growth_init`"),
    "make_holiday_features": (MOVED, "`analytic_prophet.make_holiday_features`"),
    "make_seasonality_features": (MOVED, "`analytic_prophet.make_seasonality_features`"),
    "parse_seasonality_args": (MOVED, "`analytic_prophet.parse_seasonality_args`"),
    "predict_trend": (MOVED, "`analytic_prophet.predict_trend`"),

    # -- absent on purpose -------------------------------------------------
    "plot": (DELIBERATE, "no plotting; the evaluation suite draws its own figures"),
    "plot_components": (DELIBERATE, "no plotting, and no component decomposition to plot "
                                    "— see `predict_seasonal_components`"),
    "sample_posterior_predictive": (DELIBERATE, "MCMC"),
    "sample_model": (DELIBERATE, "MCMC; the MAP equivalent is private here"),
    "sample_model_vectorized": (DELIBERATE, "MCMC; the MAP equivalent is private here"),
    "sample_predictive_trend": (DELIBERATE, "private here: `_sample_trends`"),
    "sample_predictive_trend_vectorized": (DELIBERATE,
                                           "private here: the vectorized sampler of #93"),
    "percentile": (DELIBERATE, "a thin `np.percentile` wrapper that honours `mcmc_samples`"),
    "predict_uncertainty": (DELIBERATE, "private here; `predict` returns its columns"),
    "setup_dataframe": (DELIBERATE, "folded into `preprocess`"),
    "initialize_scales": (DELIBERATE, "folded into `preprocess`"),
    "validate_inputs": (DELIBERATE, "folded into the constructor's rejections (#52) "
                                    "and `_clean_history` (#102, #104)"),
    "piecewise_linear": (DELIBERATE, "inside `predict_trend`"),
    "piecewise_logistic": (DELIBERATE, "inside `logistic_trend_and_jacobian`"),
    "flat_trend": (DELIBERATE, "inside `predict_trend`"),

    # -- absent, and arguably should not be --------------------------------
    "predictive_samples": (GAP, "the documented way to get raw draws out of a MAP fit. "
                                "This computes exactly those draws and then discards "
                                "them behind quantiles"),
    "predict_seasonal_components": (GAP, "no counterpart; this is why there is no "
                                         "component decomposition"),
    "add_group_component": (GAP, "builds the column groupings the decomposition needs"),
    "regressor_column_matrix": (GAP, "produces `train_component_cols`, same reason"),
}

# What `Prophet.fit` leaves on the instance -> (verdict, where it is here / why not)
ATTRIBUTES = {
    "history_dates": (MOVED, "`model.ds`"),
    "history": (GAP, "the fitted frame is not kept; a ported script reading "
                     "`model.history` gets an `AttributeError`"),
    "component_modes": (GAP, "additive/multiplicative groupings, for the decomposition"),
    "train_component_cols": (GAP, "which columns belong to which component"),
    "start": (DELIBERATE, "derived from `model.ds` where it is needed"),
    "t_scale": (DELIBERATE, "derived from `model.ds` where it is needed"),
    "y_min": (DELIBERATE, "0 by construction: `scaling='absmax'` is the only mode here"),
    "scaling": (DELIBERATE, "`'absmax'` only; `'minmax'` is rejected, not ignored (#52)"),
    "mcmc_samples": (DELIBERATE, "rejected by the constructor rather than ignored (#52)"),
    "stan_backend": (DELIBERATE, "there is no Stan; rejected by the constructor"),
    "stan_fit": (DELIBERATE, "there is no Stan fit object; `model.opt` is the "
                             "optimizer's result"),
    "fit_kwargs": (DELIBERATE, "Prophet keeps them to refit in `cross_validation`; "
                               "refitting here is a fresh fit (#41)"),
}

VERDICTS = (SAME, MOVED, DELIBERATE, GAP)


@pytest.fixture(scope="module")
def prophet_class(prophet_comparison):
    Prophet, _, _ = prophet_comparison
    return Prophet


@pytest.fixture(scope="module")
def fitted_prophet_attributes(prophet_comparison):
    """The public attributes a real `Prophet.fit` leaves behind.

    Measured rather than listed, because the list is what goes stale. The
    frame is read here rather than taken from `peyton_manning_df`, which is
    function-scoped: this fit is the expensive part and is worth doing once.
    """
    import warnings

    import pandas as pd

    from conftest import DATA_PATH

    Prophet, _, _ = prophet_comparison
    frame = pd.read_csv(DATA_PATH).iloc[:200].reset_index(drop=True)

    model = Prophet()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame)
    return {name for name in vars(model) if not name.startswith("_")}


def _public(obj):
    return {name for name in dir(obj) if not name.startswith("_")}


# -- the register covers the surface --------------------------------------

def test_every_prophet_method_has_a_verdict(prophet_class):
    """Fails when Prophet grows a public method, which is the point: the
    table cannot quietly stop being a description of the gap."""
    missing = _public(prophet_class) - set(METHODS)
    assert missing == set(), f"no verdict recorded for: {sorted(missing)}"


def test_no_verdict_names_something_prophet_dropped(prophet_class):
    """The other direction: an entry for a method Prophet no longer has is a
    row describing nothing."""
    stale = set(METHODS) - _public(prophet_class)
    assert stale == set(), f"recorded but gone from Prophet: {sorted(stale)}"


def test_every_attribute_a_fit_sets_is_accounted_for(fitted_prophet_attributes):
    """A drop-in is not only about methods: a ported script reading
    `model.history` gets an AttributeError, and that belongs in the table."""
    ours = _public(AnalyticProphet())
    unaccounted = fitted_prophet_attributes - ours - set(ATTRIBUTES)
    assert unaccounted == set(), f"no verdict recorded for: {sorted(unaccounted)}"


def test_no_attribute_verdict_is_stale(fitted_prophet_attributes):
    stale = set(ATTRIBUTES) - fitted_prophet_attributes
    assert stale == set(), f"recorded but not set by Prophet.fit: {sorted(stale)}"


# -- the verdicts are true ------------------------------------------------

@pytest.mark.parametrize("name", sorted(METHODS))
def test_the_verdict_matches_where_the_name_actually_is(name):
    """Introspection checks the half of each verdict that is a fact.

    `same` must be on our class, `moved` must be at module level and not on
    the class, and either kind of `absent` must be neither -- so implementing
    one of the gaps fails this until its row is updated.
    """
    verdict, _note = METHODS[name]
    on_class = hasattr(AnalyticProphet, name)
    at_module = hasattr(analytic_prophet, name)

    if verdict == SAME:
        assert on_class, f"{name} is recorded as `same` and is not on the class"
    elif verdict == MOVED:
        assert at_module, f"{name} is recorded as `moved` and is not importable"
        assert not on_class, f"{name} is on the class now; it is no longer `moved`"
    else:
        assert not on_class, f"{name} exists now; its verdict says {verdict!r}"


@pytest.mark.parametrize("name", sorted(ATTRIBUTES))
def test_the_attribute_verdict_matches_the_instance(name):
    verdict, _note = ATTRIBUTES[name]
    present = hasattr(AnalyticProphet(), name)
    if verdict == MOVED:
        assert not present, f"{name} is on the instance now; it is no longer `moved`"
    else:
        assert not present, f"{name} exists now; its verdict says {verdict!r}"


def test_every_verdict_is_one_of_the_four():
    for table in (METHODS, ATTRIBUTES):
        for name, (verdict, note) in table.items():
            assert verdict in VERDICTS, f"{name}: unknown verdict {verdict!r}"
            assert note, f"{name}: a verdict without a reason is not a verdict"


# -- the document says what the register says ------------------------------

def _documented_rows():
    """(name, verdict) for every row of the parity table in the doc."""
    text = DEVIATIONS.read_text()
    start = text.index("<!-- parity-table:start -->")
    end = text.index("<!-- parity-table:end -->")
    section = text[start:end]
    return {match.group(1): match.group(2).strip()
            for match in re.finditer(r"^\| `([^`]+)` \| ([^|]+) \|", section,
                                     re.MULTILINE)}


def test_the_table_in_the_doc_matches_the_register():
    """The table is the deliverable, and a hand-edited table is a table that
    drifts. This is what makes the register the source and the doc a view of
    it rather than a second copy."""
    documented = _documented_rows()
    expected = {name: verdict for table in (METHODS, ATTRIBUTES)
                for name, (verdict, _note) in table.items()}

    assert set(documented) == set(expected), (
        f"only in the doc: {sorted(set(documented) - set(expected))}; "
        f"only in the register: {sorted(set(expected) - set(documented))}")
    differing = {name: (documented[name], expected[name])
                 for name in expected if documented[name] != expected[name]}
    assert differing == {}, f"verdicts disagree (doc, register): {differing}"


def test_the_doc_reports_the_counts_it_claims():
    """The prose above the table counts the groups. Counted here instead of
    trusted, because a number in prose is the first thing to rot."""
    text = DEVIATIONS.read_text()
    tally = {verdict: sum(1 for v, _ in METHODS.values() if v == verdict)
             for verdict in VERDICTS}

    for verdict, phrase in ((SAME, r"are the same"), (MOVED, r"moved"),
                            (DELIBERATE, r"are absent on purpose"),
                            (GAP, r"are gaps")):
        quoted = re.search(rf"\*\*(\d+)\*\* {phrase}", text)
        assert quoted, f"the doc no longer counts the {verdict!r} methods"
        assert int(quoted.group(1)) == tally[verdict], (
            f"the doc says {quoted.group(1)} {phrase}, the register has "
            f"{tally[verdict]}")
