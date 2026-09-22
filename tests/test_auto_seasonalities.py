"""
Issue #16 task 3: choosing the seasonal components from the history.

Prophet decides which of yearly, weekly and daily to fit from the span of the
history and the spacing between observations. This implementation registered
yearly unconditionally, which is why the acceptance criterion in #30 carried an
identifiability caveat: on a 328-day slice the benchmark forced yearly on both
sides, at a length where Prophet's own rule disables it, and the forecasts
diverged 111%. Both sides now apply the same rule and agree to 0.709% there.

The rule is small enough to restate but not to guess at, so the tests below
check it against the installed Prophet rather than against a table written from
memory -- including the spacing conditions, which the Peyton Manning series
cannot exercise on its own.
"""
import numpy as np
import pandas as pd
import pytest

from customProphet import (AUTO_SEASONALITY_RULES, CustomProphet, history_spacing,
                           parse_seasonality_args)


def model_for(ds):
    """A model with just enough state for _set_auto_seasonalities()."""
    model = CustomProphet()
    model.ds = pd.to_datetime(pd.Series(ds))
    model.y = np.arange(len(model.ds), dtype=float)
    return model


def selected(ds, **overrides):
    model = model_for(ds)
    for key, value in overrides.items():
        setattr(model, key, value)
    model._set_auto_seasonalities()
    return {name: props["fourier_order"] for name, props in model.seasonalities.items()}


def daily(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="D")


# -- the rule itself ----------------------------------------------------

@pytest.mark.parametrize("n_rows", [3, 15, 60, 300, 400, 730, 731, 1000, 2905])
def test_matches_prophet_on_the_peyton_manning_series(prophet_comparison,
                                                      peyton_manning_df, n_rows):
    """Against the installed original, at lengths that straddle the 730-day
    yearly cutoff."""
    Prophet, _, _ = prophet_comparison
    df = peyton_manning_df.iloc[:n_rows].reset_index(drop=True)

    theirs = Prophet()
    theirs.history = theirs.setup_dataframe(df[df["y"].notnull()].copy(),
                                            initialize_scales=True)
    theirs.set_auto_seasonalities()

    assert selected(df["ds"]) == {name: props["fourier_order"]
                                  for name, props in theirs.seasonalities.items()}


@pytest.mark.parametrize("freq,periods", [
    ("W", 200),      # weekly spacing: weekly unidentifiable, yearly fine
    ("h", 400),      # hourly over 16 days: weekly and daily, no yearly
    ("6h", 2000),    # 500 days: still short of two years
    ("12h", 5000),   # 2500 days: all three
    ("MS", 40),      # monthly: yearly only
])
def test_matches_prophet_on_spacings_the_series_cannot_produce(prophet_comparison,
                                                               freq, periods):
    """The spacing conditions -- `min_dt >= 7 days` disables weekly, `>= 1 day`
    disables daily -- never fire on daily data, so they need their own cases."""
    Prophet, _, _ = prophet_comparison
    ds = pd.date_range("2020-01-01", periods=periods, freq=freq)
    df = pd.DataFrame({"ds": ds, "y": np.arange(periods, dtype=float)})

    theirs = Prophet()
    theirs.history = theirs.setup_dataframe(df, initialize_scales=True)
    theirs.set_auto_seasonalities()

    assert selected(ds) == {name: props["fourier_order"]
                            for name, props in theirs.seasonalities.items()}


def test_yearly_turns_on_at_exactly_730_days():
    """`last - first < 730 days` disables, so 730 days exactly enables. An
    off-by-one here would be invisible on any series not sitting on the
    boundary."""
    assert "yearly" not in selected(daily(730))       # spans 729 days
    assert "yearly" in selected(daily(731))           # spans 730 days


def test_component_order_follows_prophets():
    """Column order in the design matrix, and so the meaning of every entry of
    `beta`. [fc] set_auto_seasonalities registers yearly, then weekly, then
    daily, and make_all_seasonality_features iterates in insertion order."""
    assert list(selected(pd.date_range("2020-01-01", periods=5000, freq="12h"))) == \
        ["yearly", "weekly", "daily"]


def test_spacing_ignores_row_order_and_duplicate_timestamps():
    """[fc] set_auto_seasonalities runs on `self.history`, which is sorted, and
    excludes zero spacings. Our fit paths do not sort, so history_spacing does
    it rather than inheriting the caller's row order."""
    ds = daily(100)
    shuffled = pd.Series(ds).sample(frac=1, random_state=0)
    with_duplicates = pd.concat([pd.Series(ds), pd.Series(ds[:10])])

    first, last, min_dt = history_spacing(ds)
    assert (first, last) == (ds[0], ds[-1])
    assert min_dt == pd.Timedelta(days=1)

    assert history_spacing(shuffled) == (first, last, min_dt)
    assert history_spacing(with_duplicates)[2] == pd.Timedelta(days=1)


# -- the explicit overrides ---------------------------------------------

@pytest.mark.parametrize("arg,expected", [
    ("auto", 10),      # history supports it
    (True, 10),        # the built-in default order
    (False, 0),        # off
    (4, 4),            # an explicit order
    (0, 0),            # explicitly none
])
def test_parse_seasonality_args_matches_prophet(prophet_comparison, arg, expected):
    Prophet, _, _ = prophet_comparison
    theirs = Prophet()
    theirs.seasonalities = {}

    assert parse_seasonality_args("yearly", arg, False, 10, {}) == expected
    assert theirs.parse_seasonality_args("yearly", arg, False, 10) == expected


@pytest.mark.parametrize("arg,expected", [("auto", 0), (True, 10), (False, 0), (4, 4)])
def test_auto_disable_only_affects_auto(prophet_comparison, arg, expected):
    """`auto_disable` is consulted only on 'auto': an explicit order overrides
    the rule, which is how a user asks for yearly on a short series."""
    Prophet, _, _ = prophet_comparison
    theirs = Prophet()
    theirs.seasonalities = {}

    assert parse_seasonality_args("yearly", arg, True, 10, {}) == expected
    assert theirs.parse_seasonality_args("yearly", arg, True, 10) == expected


def test_true_is_an_identity_check_not_a_truth_test():
    """[fc] `elif arg is True`, so weekly_seasonality=1 asks for order 1 rather
    than the default 3. Matched deliberately -- a truthiness test here would
    silently give a user order 3 when they asked for 1."""
    assert parse_seasonality_args("weekly", 1, False, 3, {}) == 1
    assert parse_seasonality_args("weekly", True, False, 3, {}) == 3


def test_a_registered_component_suppresses_its_built_in(caplog):
    """[fc] parse_seasonality_args: 'auto' yields 0 when a component of that
    name is already registered, which is how add_seasonality('weekly', ...)
    wins over the automatic one (#16 task 6)."""
    from customProphet import seasonality

    model = model_for(daily(1000))
    model.seasonalities = {"weekly": seasonality(7.0, 8)}
    model._set_auto_seasonalities()

    assert model.seasonalities["weekly"]["fourier_order"] == 8   # not overwritten
    assert list(model.seasonalities) == ["weekly", "yearly"]


def test_explicit_orders_reach_the_registry():
    assert selected(daily(1000), yearly_seasonality=4, weekly_seasonality=2) == \
        {"yearly": 4, "weekly": 2}
    assert selected(daily(1000), yearly_seasonality=False) == {"weekly": 3}


def test_forcing_yearly_on_a_short_series_warns(caplog):
    """[fc] set_auto_seasonalities warns when yearly is enabled under 730 days.
    That regime is exactly what #30 measured diverging, so the warning is the
    user-facing half of this task."""
    with caplog.at_level("WARNING", logger="customProphet"):
        assert selected(daily(300), yearly_seasonality=True) == {"yearly": 10, "weekly": 3}

    assert any("less than 730 days" in r.message for r in caplog.records)


def test_no_warning_when_the_history_supports_yearly(caplog):
    with caplog.at_level("WARNING", logger="customProphet"):
        selected(daily(1000))
    assert not [r for r in caplog.records if "730 days" in r.message]


# -- reaching the fit ---------------------------------------------------

def test_the_rule_decides_what_a_fit_estimates(peyton_manning_df,
                                               compiled_optimizer_module):
    """The selected components have to reach the parameter vector, or the rule
    is decoration. 328 days gives weekly only; the full series adds yearly."""
    short = CustomProphet()
    short.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    long = CustomProphet()
    long.fit_cpp(peyton_manning_df, lib_path=compiled_optimizer_module)

    assert list(short.seasonalities) == ["weekly"]
    assert short.layout.n_seasonality_columns == 6
    assert short.opt_params.shape == (short.layout.size,)

    assert list(long.seasonalities) == ["yearly", "weekly"]
    assert long.layout.n_seasonality_columns == 26
    assert long.opt_params.shape == (long.layout.size,)


def test_both_fit_paths_select_the_same_components(peyton_manning_df,
                                                   compiled_optimizer_module):
    """fit() and fit_cpp() must fit the same model, which now includes agreeing
    on what that model is."""
    df = peyton_manning_df.iloc[:400].reset_index(drop=True)

    python_model = CustomProphet()
    python_model.fit(df, analytic=True)

    cpp_model = CustomProphet()
    cpp_model.fit_cpp(df, lib_path=compiled_optimizer_module)

    assert python_model.seasonalities == cpp_model.seasonalities
    assert python_model.layout.size == cpp_model.layout.size


def test_selection_is_not_sticky_across_fits(peyton_manning_df, compiled_optimizer_module):
    """Prophet adds built-ins to whatever is registered, so a model refitted on
    a shorter series must not keep a component the new history cannot support.
    """
    model = CustomProphet()
    model.fit_cpp(peyton_manning_df, lib_path=compiled_optimizer_module)
    assert list(model.seasonalities) == ["yearly", "weekly"]

    model.fit_cpp(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    assert list(model.seasonalities) == ["weekly"]
    assert model.opt_params.shape == (model.layout.size,)


def test_rules_table_matches_prophets_documented_thresholds():
    """The table is the rule; a typo in it would be silent everywhere else."""
    assert AUTO_SEASONALITY_RULES == (
        ("yearly", "yearly_seasonality", 365.25, 10, pd.Timedelta(days=730), None),
        ("weekly", "weekly_seasonality", 7.0, 3, pd.Timedelta(weeks=2), pd.Timedelta(weeks=1)),
        ("daily", "daily_seasonality", 1.0, 4, pd.Timedelta(days=2), pd.Timedelta(days=1)),
    )
