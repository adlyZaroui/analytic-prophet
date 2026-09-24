"""
Issue #24: `fit()` did not run under Stan's convergence settings.

[fc] Prophet calls `optimize(algorithm='LBFGS', iter=int(1e4))` and sets no
tolerances, so CmdStan's defaults apply: `tol_obj = 1e-12`,
`tol_rel_obj * eps = 2.22e-12`, `tol_grad = 1e-8`, `tol_param = 1e-8`. #21 put
those in the C++ core and deliberately left `fit()` out, with three overrides:
`ftol = 1e-16`, `gtol = 0.0`, `maxfun = 10 * maxiter`.

Two of the three are closed here.

**`gtol` now carries Stan's `tol_grad`.** It was disabled on the reading that
scipy tests the inf-norm of the *projected* gradient where Stan tests the 2-norm
of the full one, and that on the split problem the projected norm is far the
smaller. That is no longer measurable: enabling it changes neither the iteration
count nor the fitted point at any size tested. Prophet's changepoint placement
(#15) is the likely reason -- index-spaced changepoints land on observations, so
each has data at it, and the problem is better conditioned for it.

**`maxfun` is Stan's cap expressed the other way round.** Stan limits iterations
only; scipy's default of 15000 function evaluations would end the run before
`maxiter` does, which is a cap Stan does not impose.

**`ftol` stays tighter, and that is the deviation that remains.** The argument is
measured rather than asserted, and every number in it is pinned below.

The deciding measurement is the one #24 said it was blocked on -- what Prophet
itself reaches. Under Stan's `ftol` this path scores `lp__ = 8000.371` on the
full series against Prophet's `8004.798`: adopting the number would put the
reference path **4.43 nats below the model it reproduces**. Tightening instead
costs iterations and nothing else.
"""
import numpy as np
import pytest
from scipy.optimize import minimize as real_minimize

import customProphet
from customProphet import (SCIPY_TOL_REL_OBJ, STAN_EPS, STAN_MAX_ITERATIONS,
                           STAN_TOL_GRAD, STAN_TOL_REL_OBJ, CustomProphet)

# Stan's relative-objective threshold, the number `ftol` declines to take.
STAN_FTOL = STAN_TOL_REL_OBJ * STAN_EPS


@pytest.fixture
def scipy_options(monkeypatch):
    """Run `fit()` with scipy options overridden, and report what it reached.

    The overrides go through `minimize` rather than through `fit()`, because the
    point is to measure what the current settings buy -- an argument that stops
    being checkable the moment they become an argument.
    """
    def run(df, **overrides):
        def patched(*args, **kwargs):
            kwargs["options"] = dict(kwargs.get("options") or {}, **overrides)
            return real_minimize(*args, **kwargs)

        monkeypatch.setattr(customProphet, "minimize", patched)
        model = CustomProphet()
        model.fit(df, analytic=True, algorithm="LBFGS")
        monkeypatch.setattr(customProphet, "minimize", real_minimize)
        return model, model._minus_log_posterior(model.get_parameters())
    return run


# -- what is now Stan's ---------------------------------------------------

def test_the_settings_fit_passes_are_stans_except_for_one(peyton_manning_df, scipy_options):
    """The deviation is one option, and this is the list of it."""
    recorded = {}

    def patched(*args, **kwargs):
        recorded.update(kwargs["options"])
        return real_minimize(*args, **kwargs)

    model = CustomProphet()
    original = customProphet.minimize
    customProphet.minimize = patched
    try:
        model.fit(peyton_manning_df.iloc[:300].reset_index(drop=True),
                  analytic=True, algorithm="LBFGS")
    finally:
        customProphet.minimize = original

    assert recorded["maxiter"] == STAN_MAX_ITERATIONS   # [fc] iter=int(1e4)
    assert recorded["gtol"] == STAN_TOL_GRAD            # [fc] tol_grad
    assert recorded["ftol"] == SCIPY_TOL_REL_OBJ        # not Stan's; see below
    assert recorded["ftol"] < STAN_FTOL


@pytest.mark.parametrize("n_rows", [300, 1000, 2905])
def test_stans_gradient_tolerance_changes_nothing(peyton_manning_df, scipy_options, n_rows):
    """Half of #24, closed by measurement: `gtol` is now Stan's value, and
    turning it off again leaves the run bit-identical.

    Asserted as exact equality rather than a tolerance. If the two ever differ
    at all, the reasoning in `fit()` needs rewriting rather than loosening.
    """
    df = peyton_manning_df.iloc[:n_rows].reset_index(drop=True)

    with_stan, loss_with = scipy_options(df)
    without, loss_without = scipy_options(df, gtol=0.0)

    assert with_stan.opt.nit == without.opt.nit
    assert loss_with == loss_without
    np.testing.assert_array_equal(with_stan.get_parameters(), without.get_parameters())


def test_stans_gradient_tolerance_changes_nothing_on_short_series(peyton_manning_df,
                                                                  scipy_options):
    """The sizes where the rule would pick Newton, fitted with L-BFGS anyway --
    `gtol` is an L-BFGS option, so this is where it would show if anywhere."""
    for n_rows in (30, 50, 99, 150):
        df = peyton_manning_df.iloc[:n_rows].reset_index(drop=True)
        with_stan, loss_with = scipy_options(df)
        without, loss_without = scipy_options(df, gtol=0.0)
        assert with_stan.opt.nit == without.opt.nit, n_rows
        assert loss_with == loss_without, n_rows


# -- why ftol does not follow --------------------------------------------

def test_stans_objective_tolerance_stops_the_run_short(peyton_manning_df, scipy_options):
    """The deviation, measured on the full series.

    4.8 nats is not a rounding difference: it is larger than the entire 2.34-nat
    margin this project has over Prophet.
    """
    df = peyton_manning_df.reset_index(drop=True)

    tight, loss_tight = scipy_options(df)
    stan, loss_stan = scipy_options(df, ftol=STAN_FTOL)

    assert stan.opt.nit < tight.opt.nit / 5     # 434 against 4261
    assert loss_stan - loss_tight > 4.0, (
        f"Stan's ftol now costs {loss_stan - loss_tight:.3f} nats, not ~4.8 -- "
        f"the reasoning in fit() is out of date")


def test_the_run_stops_on_a_plateau_it_then_climbs_out_of(peyton_manning_df):
    """*Why* Stan's one-step test misfires here, rather than that it does.

    scipy's iterate sequence has single steps that barely move followed by steps
    that move a lot. The test is a one-step test, so it fires on the first of
    them -- while the run, left alone, goes on descending for thousands more
    iterations. It is not scipy's own stopping rule doing this: the quantity
    below is Stan's test evaluated by hand on the trajectory.
    """
    model = CustomProphet()
    model.fit(peyton_manning_df.reset_index(drop=True), analytic=True, algorithm="LBFGS")
    trace = np.asarray(model.loss_over_iterations)

    scale = np.maximum.reduce([np.abs(trace[:-1]), np.abs(trace[1:]), np.ones(len(trace) - 1)])
    relative_decrease = -np.diff(trace) / scale
    below = np.flatnonzero(relative_decrease <= STAN_FTOL)

    assert len(below) > 0
    first = below[0]
    # the run is nowhere near done when the first one arrives ...
    assert trace[first] - trace[-1] > 4.0
    # ... and they are pervasive rather than a single unlucky step
    assert np.mean(relative_decrease[first:] <= STAN_FTOL) > 0.25


def test_stans_objective_tolerance_would_score_below_prophet(prophet_comparison,
                                                             scipy_options):
    """The measurement #24 was blocked on, and the reason the deviation stays.

    Scored under Stan's own density on Prophet's changepoints, so only the
    optimizer differs. Taking Stan's number here would not make this path more
    faithful -- it would make it worse than the thing it reproduces.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(2905)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    stan_model, stan_data, params = bridge.capture_stan_model(prophet_model, df)
    lp_prophet = bridge.validate_bridge(
        stan_model, stan_data, params,
        float(np.asarray(prophet_model.params["lp__"]).ravel()[0]))
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    def score(**overrides):
        model = CustomProphet(n_changepoints=len(changepoints_t))
        model.set_changepoints = lambda: setattr(
            model, "changepoints_t", changepoints_t.copy())
        patched = real_minimize
        if overrides:
            def patched(*args, **kwargs):
                kwargs["options"] = dict(kwargs.get("options") or {}, **overrides)
                return real_minimize(*args, **kwargs)
        original = customProphet.minimize
        customProphet.minimize = patched
        try:
            model.fit(df, analytic=True, algorithm="LBFGS")
        finally:
            customProphet.minimize = original
        return bridge.stan_log_prob(
            stan_model, stan_data, model.params["k"][0][0], model.params["m"][0][0],
            model.params["delta"][0], model.sigma_obs, model.params["beta"][0])

    assert score() > lp_prophet, "the central claim, on this path"
    assert score(ftol=STAN_FTOL) < lp_prophet - 4.0, (
        "Stan's ftol no longer puts this path below Prophet -- if that is real, "
        "the deviation recorded in fit() should go")


def test_the_line_search_budget_is_not_the_cause(peyton_manning_df, scipy_options):
    """Ruled out rather than assumed: the C++ core gives its line search 60
    tries where scipy's default is 20, which would be the obvious explanation
    for one taking better-behaved steps than the other. It is not -- the
    trajectory is identical."""
    df = peyton_manning_df.reset_index(drop=True)

    twenty, loss_twenty = scipy_options(df, ftol=STAN_FTOL, maxls=20)
    sixty, loss_sixty = scipy_options(df, ftol=STAN_FTOL, maxls=60)

    assert twenty.opt.nit == sixty.opt.nit
    assert loss_twenty == loss_sixty


# -- the recorded trajectory ---------------------------------------------

@pytest.mark.parametrize("n_rows", [300, 1000, 2905])
def test_the_recorded_trajectory_is_the_objective_being_minimized(peyton_manning_df,
                                                                  n_rows):
    """`loss_over_iterations` used to record the *canonical* objective while
    scipy minimized the *split* one.

    They are different functions away from the optimum -- |d+ - d-| against
    d+ + d-, equal only where at most one of each pair is non-zero -- and they
    agree exactly at the answer, so the final value was right and nothing caught
    it. What it produced was a trajectory that rose (by 0.031 at T=1000) on a run
    that descends monotonically, and a trace not comparable with `fit_cpp`'s.
    """
    df = peyton_manning_df.iloc[:n_rows].reset_index(drop=True)
    model = CustomProphet()
    model.fit(df, analytic=True, algorithm="LBFGS")
    trace = np.asarray(model.loss_over_iterations)

    assert np.all(np.diff(trace) <= 0), "L-BFGS-B descends; the trace should too"
    assert trace[-1] == pytest.approx(
        model._split_minus_log_posterior(model.opt.x), rel=1e-12)
    # and at the optimum the two objectives do agree, which is why this went
    # unnoticed for as long as it did
    assert trace[-1] == pytest.approx(
        model._minus_log_posterior(model.get_parameters()), rel=1e-12)
