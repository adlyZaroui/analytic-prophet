"""
Issue #15: where the changepoints go.

This implementation spaced them uniformly in *scaled time*:

    np.linspace(0, changepoint_range * max_t, n_changepoints + 1)[1:]

[fc] `set_changepoints` spaces `n_changepoints + 1` indexes evenly across the
first `changepoint_range` of the *rows* and takes the dates there. The two agree
on regularly-spaced data and diverge on anything gappy, because index-spacing
follows observation density and time-spacing does not.

The whole function is ported here rather than the placement alone. It does three
things, and porting one of them leaves the other two wrong:

  * **placement by row index**, above;
  * **a cap on the count** -- `n_changepoints + 1 > floor(T * range)` caps it at
    `floor(T * range) - 1`, which bites below about 32 observations: 15
    changepoints at T = 20 where this used 25, ten more rate parameters than
    Prophet fits on twenty points;
  * **an explicit list**, which the constructor rejected until now (#52).

Two consequences, both measured below rather than predicted. The changepoints
are now bit-identical to Prophet's at every size, which is the acceptance
criterion. And `fit()` stopped terminating ABNORMAL at T = 30 -- index-spaced
changepoints land on actual observations, so each has data at it, where a
time-spaced one could fall in a gap with nothing nearby.
"""
import numpy as np
import pandas as pd
import pytest

from analytic_prophet import AnalyticProphet


def changepoints_for(df, **kwargs):
    model = AnalyticProphet(**kwargs)
    model.y = df["y"].values
    model.ds = pd.to_datetime(df["ds"])
    model.t = np.array((model.ds - model.ds.min()) / (model.ds.max() - model.ds.min()))
    model.T = len(df)
    model.set_changepoints()
    return model


# -- placement ----------------------------------------------------------

@pytest.mark.parametrize("n_rows", [100, 300, 1000, 2905])
def test_the_changepoints_are_prophets_exactly(prophet_comparison,
                                               compiled_optimizer_module, n_rows):
    """The acceptance criterion. Equality, not a tolerance: both sides select
    dates from the same rows of the same frame, so anything less would mean a
    different rule rather than different arithmetic."""
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    _, stan_data, _ = bridge.capture_stan_model(prophet_model, df)

    ours = AnalyticProphet()
    ours.fit(df, lib_path=compiled_optimizer_module)

    np.testing.assert_array_equal(ours.changepoints_t,
                                  np.asarray(stan_data["t_change"], dtype=float))
    np.testing.assert_array_equal(np.asarray(ours.changepoints_t),
                                  np.asarray(prophet_model.changepoints_t))


def test_placement_follows_rows_not_time(peyton_manning_df):
    """The difference the issue is about, on a frame built to show it: a dense
    first half and a sparse second. Row-spacing puts most changepoints in the
    dense part; time-spacing spreads them evenly across the span."""
    dense = pd.date_range("2020-01-01", periods=80, freq="D")
    sparse = pd.date_range("2020-04-01", periods=20, freq="30D")
    df = pd.DataFrame({"ds": dense.append(sparse), "y": np.arange(100.0)})

    model = changepoints_for(df, n_changepoints=10)

    span = (df["ds"].max() - df["ds"].min()).days
    dense_end = (dense[-1] - df["ds"].min()).days / span
    in_dense = (model.changepoints_t <= dense_end).sum()

    # the dense block is 80 of 100 rows but only a fifth of the span, so row
    # spacing puts most changepoints there and time spacing would not
    assert in_dense >= 7, f"only {in_dense} of 10 landed in the dense region"


def test_the_first_row_is_never_a_changepoint(peyton_manning_df):
    """[fc] `.tail(-1)`: `n_changepoints + 1` indexes are taken and the first
    is dropped, so the series start is not itself a breakpoint."""
    model = changepoints_for(peyton_manning_df.iloc[:300].reset_index(drop=True))

    assert len(model.changepoints_t) == 25
    assert model.changepoints_t.min() > 0.0


def test_changepoints_land_on_observed_dates(peyton_manning_df):
    """Because they are taken from rows. This is what makes `t[i] >= t_change[j]`
    exactly true for one row per changepoint -- the boundary case that `>=`
    against `>` decides, and which never arose while the placement was
    time-spaced."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    model = changepoints_for(df)

    assert set(np.round(model.changepoints_t, 12)) <= set(np.round(model.t, 12))

    A = (model.t[:, None] >= model.changepoints_t) * 1
    boundary = sum(int(np.isclose(model.t, cp).sum()) for cp in model.changepoints_t)
    assert boundary == len(model.changepoints_t), (
        "every changepoint should sit exactly on one observation")
    assert A.shape == (len(df), 25)


# -- the cap ------------------------------------------------------------

@pytest.mark.parametrize("n_rows,expected", [(20, 15), (30, 23), (50, 25), (99, 25)])
def test_the_count_is_capped_on_short_series(prophet_comparison, peyton_manning_df,
                                             n_rows, expected):
    """[fc] `if n_changepoints + 1 > hist_size: n_changepoints = hist_size - 1`,
    with `hist_size = floor(T * changepoint_range)`.

    Checked against Prophet as well as against the formula, since the formula
    is the sort of thing a reimplementation gets off by one."""
    Prophet, common, _ = prophet_comparison
    df = peyton_manning_df.iloc[:n_rows].reset_index(drop=True)

    ours = changepoints_for(df)
    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.fit(df)

    assert len(ours.changepoints_t) == expected
    assert len(prophet_model.changepoints_t) == expected
    assert ours.n_changepoints == expected


def test_the_cap_reaches_the_parameter_vector(peyton_manning_df,
                                              compiled_optimizer_module):
    """The count is capped before the layout is built, or `delta` would be
    sized for a request the placement did not honour."""
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:20].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert len(model.changepoints_t) == 15
    assert model.layout.n_changepoints == 15
    assert model.params["delta"].shape == (1, 15)


def test_zero_changepoints_gets_prophets_dummy():
    """[fc] `changepoints_t = np.array([0])` when there are none -- a dummy, so
    the design matrix keeps a column rather than the layout collapsing."""
    df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=50), "y": np.arange(50.0)})
    model = changepoints_for(df, n_changepoints=0)

    np.testing.assert_array_equal(model.changepoints_t, [0.0])


# -- an explicit list ---------------------------------------------------

def test_an_explicit_list_is_used_as_given(peyton_manning_df, compiled_optimizer_module):
    """Rejected by the constructor until now (#52)."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    dates = ["2008-02-01", "2008-05-01", "2008-08-01"]

    model = AnalyticProphet(changepoints=dates)
    assert model.specified_changepoints
    assert model.n_changepoints == 3

    model.fit(df, lib_path=compiled_optimizer_module)
    assert len(model.changepoints_t) == 3
    assert model.layout.n_changepoints == 3


def test_an_explicit_list_must_fall_inside_the_history(peyton_manning_df,
                                                       compiled_optimizer_module):
    """[fc] "Changepoints must fall within training data.\""""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)

    with pytest.raises(ValueError, match="within training data"):
        AnalyticProphet(changepoints=["2030-01-01"]).fit(
            df, lib_path=compiled_optimizer_module)


def test_generated_changepoints_do_not_survive_a_refit(peyton_manning_df,
                                                       compiled_optimizer_module):
    """[fc] `specified_changepoints` distinguishes what the user gave from what
    was generated. Prophet refuses a second fit so never meets this; this
    implementation allows one (#41), and the previous fit's *generated* dates
    must not be read back as a user-supplied list -- they would fall outside a
    shorter history and raise."""
    model = AnalyticProphet()
    model.fit(peyton_manning_df, lib_path=compiled_optimizer_module)
    first = model.changepoints_t.copy()

    model.fit(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert not np.array_equal(first, model.changepoints_t)
    assert not model.specified_changepoints


def test_an_explicit_list_does_survive_a_refit(peyton_manning_df,
                                               compiled_optimizer_module):
    """The other half: what the user gave is theirs, and stays."""
    dates = ["2008-02-01", "2008-05-01"]
    model = AnalyticProphet(changepoints=dates)
    model.fit(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    model.fit(peyton_manning_df.iloc[:400].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert len(model.changepoints_t) == 2
