"""
Issue #25: Prophet switches algorithm below 100 observations, and retries with
Newton when the optimizer fails. Both paths here now do the same.

[fc] CmdStanPyBackend.fit picks `'Newton' if T < 100 else 'LBFGS'`, retries once
with Newton if the first attempt raises, and lets an explicit `algorithm=`
override the choice. The issue was filed on the reasonable assumption that
running a different algorithm from the original on short series meant we could
not expect matching parameters there, and it proposed measuring before building
anything, since a Newton optimizer needs a Hessian this project computes
nowhere.

That measurement is here, and it inverted the premise. **Prophet's Newton beats
Prophet's L-BFGS at every size measured**, which is why the switch exists.
**This implementation's L-BFGS beats Prophet's Newton at every size measured**,
including below 100 where Prophet uses Newton -- so following the rule costs
accuracy rather than buying it.

The rule is implemented anyway, because the contract is the same fit under the
same data -- and in the end it costs nothing: our Newton reaches the same
optimum as our L-BFGS at every size measured, so a default fit under the rule is
still ahead of Prophet's. That is asserted here rather than assumed, because the
first version of this Newton did not manage it. See the README, "Prophet's rule
for short series, and what it costs".

The Hessian turned out not to need deriving: central differences of the
*analytic* gradient give one at 2n gradient evaluations, and at these sizes that
is cheap. What Newton did need was the split reformulation (#23) -- on the
natural parameterization it oscillates across the Laplace kink and exhausts
Prophet's whole iteration budget 66 nats short.
"""
import types

import numpy as np
import pytest

from analytic_prophet import forecaster
from analytic_prophet import CPP_SOLVER_RAISED, SCIPY_LINE_SEARCH_FAILURE, AnalyticProphet

# Prophet's own cutoff, [fc] `'Newton' if T < 100 else 'LBFGS'`.
NEWTON_BELOW = 100

SHORT_SIZES = [20, 30, 50, 75, 99]

# Our two algorithms reach the same optimum at every size measured, so
# following the rule costs nothing. Asserted rather than assumed, because the
# first version of the Newton did not: with the damping keyed to success alone
# rather than to the line search, it stalled 1.39 nats short at T=20 and 0.05 at
# T=30. The tolerance is what "the same optimum" means here -- five decimals on
# a quantity of order 100.
NEWTON_MATCHES_LBFGS = 1e-4


def prophet_score(Prophet, bridge, common, df, algorithm):
    """Fit Prophet with a forced algorithm and return (stan handles, lp__)."""
    model = Prophet(**common.PROPHET_KWARGS)
    stan_model, stan_data, params = bridge.capture_stan_model(
        model, df, algorithm=algorithm)
    reported = float(np.asarray(model.params["lp__"]).ravel()[0])
    return stan_model, stan_data, bridge.validate_bridge(
        stan_model, stan_data, params, reported)


def our_score(bridge, stan_model, stan_data, df, lib_path, algorithm=None):
    """Our fit on Prophet's changepoints, scored under Stan's density.

    The changepoint *count* is taken from Prophet too: below about 32
    observations it caps `n_changepoints` at `floor(T * changepoint_range) - 1`,
    which this implementation does not do -- see the note on #15.

    `algorithm=None` leaves the selection rule in charge, which is what a caller
    gets by default; naming one measures that algorithm on its own.
    """
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    model = AnalyticProphet(n_changepoints=len(changepoints_t))
    model.set_changepoints = lambda: setattr(
        model, "changepoints_t", changepoints_t.copy())
    model.fit(df, lib_path=lib_path, algorithm=algorithm)
    return model, bridge.stan_log_prob(
        stan_model, stan_data, model.params["k"][0][0], model.params["m"][0][0],
        model.params["delta"][0], model.sigma_obs, model.params["beta"][0])


# -- why the switch exists, and what it costs us -------------------------

@pytest.mark.parametrize("n_rows", SHORT_SIZES)
def test_newton_beats_lbfgs_inside_prophet(prophet_comparison, n_rows):
    """Why the switch exists at all, established before asking whether we need
    it. If this ever stops holding, the rest of this module is measuring
    something that no longer matters."""
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)

    _, _, lp_newton = prophet_score(Prophet, bridge, common, df, "Newton")
    _, _, lp_lbfgs = prophet_score(Prophet, bridge, common, df, "LBFGS")

    assert lp_newton > lp_lbfgs, (
        f"Prophet's Newton no longer beats its L-BFGS at T={n_rows}")


@pytest.mark.parametrize("n_rows", SHORT_SIZES)
def test_our_lbfgs_beats_prophets_newton(prophet_comparison, compiled_optimizer_module,
                                         n_rows):
    """The measurement #25 asked for, and the reason the rule costs rather than
    buys. Below 100 observations Prophet runs Newton because its L-BFGS does
    poorly here; ours does not have that problem.

    `algorithm="LBFGS"` is explicit because at these sizes the rule now picks
    Newton -- without it this would silently stop measuring L-BFGS.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)

    stan_model, stan_data, lp_newton = prophet_score(
        Prophet, bridge, common, df, "Newton")
    _, lp_ours = our_score(bridge, stan_model, stan_data, df,
                           compiled_optimizer_module, algorithm="LBFGS")

    assert lp_ours >= lp_newton - 1e-6, (
        f"Prophet's Newton reaches {lp_newton:.5f} at T={n_rows}, ours only "
        f"{lp_ours:.5f} -- the premise of #25 would then hold after all")


@pytest.mark.parametrize("n_rows", SHORT_SIZES)
def test_following_the_rule_costs_nothing(prophet_comparison, compiled_optimizer_module,
                                          n_rows):
    """The default fit, under the rule, against both references.

    The rule picks Newton at these sizes, and Newton lands where L-BFGS lands --
    so what a caller gets by default is the better optimum anyway. This is the
    assertion that keeps that true: it is a property of the damping schedule,
    not of the objective, and it was false in the first version of this Newton.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)

    stan_model, stan_data, lp_newton = prophet_score(
        Prophet, bridge, common, df, "Newton")
    model, lp_rule = our_score(bridge, stan_model, stan_data, df,
                               compiled_optimizer_module)
    _, lp_lbfgs = our_score(bridge, stan_model, stan_data, df,
                            compiled_optimizer_module, algorithm="LBFGS")

    assert model.optimizer_used == "Newton"
    assert lp_rule == pytest.approx(lp_lbfgs, abs=NEWTON_MATCHES_LBFGS), (
        f"the rule now costs {lp_lbfgs - lp_rule:.5f} nats at T={n_rows}")
    assert lp_rule >= lp_newton - 1e-6, (
        f"the rule leaves us behind Prophet's own Newton at T={n_rows}: "
        f"{lp_rule:.5f} against {lp_newton:.5f}")


@pytest.mark.parametrize("n_rows", SHORT_SIZES)
def test_the_python_newton_also_matches_its_own_lbfgs(peyton_manning_df, n_rows):
    """The same property on the reference path, on its own objective.

    Scored with `_minus_log_posterior` rather than Stan's density: this is about
    the two optimizers agreeing, which does not need the bridge.
    """
    df = peyton_manning_df.iloc[:n_rows].reset_index(drop=True)

    newton = AnalyticProphet()
    newton.fit(df, backend="python", analytic=True, algorithm="Newton")
    lbfgs = AnalyticProphet()
    lbfgs.fit(df, backend="python", analytic=True, algorithm="LBFGS")

    assert newton._minus_log_posterior(newton.get_parameters()) == pytest.approx(
        lbfgs._minus_log_posterior(lbfgs.get_parameters()), abs=NEWTON_MATCHES_LBFGS)


# -- the selection rule --------------------------------------------------

def test_prophets_cutoff_is_strict(prophet_comparison):
    """[fc] `'Newton' if T < 100 else 'LBFGS'`. Recorded because the boundary
    is the sort of thing a reimplementation gets off by one."""
    Prophet, common, bridge = prophet_comparison
    assert forecaster.NEWTON_BELOW == NEWTON_BELOW == 100

    # at exactly 100 Prophet uses L-BFGS, so its own Newton does better there
    df = common.load_data(NEWTON_BELOW)
    _, _, lp_newton = prophet_score(Prophet, bridge, common, df, "Newton")
    _, _, lp_default = prophet_score(Prophet, bridge, common, df, "LBFGS")
    assert lp_newton > lp_default


@pytest.mark.parametrize("n_rows,expected", [(99, "Newton"), (100, "LBFGS"),
                                             (101, "LBFGS")])
def test_the_compiled_path_follows_the_rule(peyton_manning_df,
                                            compiled_optimizer_module,
                                            n_rows, expected):
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:n_rows].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)
    assert model.optimizer_used == expected


@pytest.mark.parametrize("n_rows,expected", [(99, "Newton"), (100, "LBFGS"),
                                             (101, "LBFGS")])
def test_the_python_path_follows_the_same_rule(peyton_manning_df, n_rows, expected):
    """Both paths, or a script that switches between them changes optimizer
    without asking."""
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:n_rows].reset_index(drop=True), backend="python", analytic=True)
    assert model.optimizer_used == expected


@pytest.mark.parametrize("algorithm", ["Newton", "LBFGS"])
def test_an_explicit_algorithm_overrides_the_rule(peyton_manning_df,
                                                  compiled_optimizer_module,
                                                  algorithm):
    """[fc] `args.update(kwargs)` -- an `algorithm=` passed to Prophet.fit wins
    over the length rule, in both directions."""
    short = peyton_manning_df.iloc[:50].reset_index(drop=True)
    long = peyton_manning_df.iloc[:300].reset_index(drop=True)

    for df in (short, long):
        compiled = AnalyticProphet()
        compiled.fit(df, lib_path=compiled_optimizer_module, algorithm=algorithm)
        assert compiled.optimizer_used == algorithm

        python = AnalyticProphet()
        python.fit(df, backend="python", analytic=True, algorithm=algorithm)
        assert python.optimizer_used == algorithm


# -- the fallback --------------------------------------------------------

def failing_cpp_module(real_module, status):
    """The real extension with `optimize` replaced by one reporting `status`.

    Nothing in this project currently fails, so the fallback has to be provoked
    to be tested at all -- which is the point of testing it: the branch exists
    for a failure that has not happened yet.
    """
    def optimize(**kwargs):
        result = real_module.optimize(**kwargs)
        return types.SimpleNamespace(
            params=result.params, loss_trace=result.loss_trace,
            n_iterations=result.n_iterations, status=status,
            status_message="provoked")
    return types.SimpleNamespace(optimize=optimize, newton=real_module.newton)


def test_a_failed_compiled_lbfgs_falls_back_to_newton(peyton_manning_df, cpp_module,
                                                      compiled_optimizer_module,
                                                      monkeypatch, caplog):
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    monkeypatch.setattr(forecaster, "load_cpp_module",
                        lambda _: failing_cpp_module(cpp_module, CPP_SOLVER_RAISED))

    model = AnalyticProphet()
    with caplog.at_level("WARNING", logger="analytic_prophet"):
        model.fit(df, lib_path=compiled_optimizer_module)

    assert model.optimizer_used == "Newton"
    assert model.opt_status_message == "converged"
    assert np.all(np.isfinite(model.get_parameters()))
    # [fc] the exact wording Prophet logs, so a user grepping for it finds it
    assert any("Falling back to Newton" in r.message for r in caplog.records)


def test_a_failed_python_lbfgs_falls_back_to_newton(peyton_manning_df,
                                                    monkeypatch, caplog):
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    real_minimize = forecaster.minimize

    def failing_minimize(*args, **kwargs):
        result = real_minimize(*args, **kwargs)
        result.status = SCIPY_LINE_SEARCH_FAILURE
        result.success = False
        return result

    monkeypatch.setattr(forecaster, "minimize", failing_minimize)

    model = AnalyticProphet()
    with caplog.at_level("WARNING", logger="analytic_prophet"):
        model.fit(df, backend="python", analytic=True)

    assert model.optimizer_used == "Newton"
    assert model.opt.success
    assert any("Falling back to Newton" in r.message for r in caplog.records)


def test_the_iteration_cap_is_not_a_failure(peyton_manning_df, cpp_module,
                                            compiled_optimizer_module, monkeypatch):
    """Running out of `iter` does not trigger the retry.

    [fc] cmdstanpy raises on a non-zero exit code, and CmdStan reports an
    exhausted iteration budget as a warning rather than an error -- so Prophet
    keeps that fit. Treating it as a failure here made a widened-prior fit
    measurably worse by retrying a run that had not actually failed.
    """
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    cap_status = 2  # the C++ core's "reached max_iterations", not its failure
    assert cap_status != CPP_SOLVER_RAISED
    monkeypatch.setattr(forecaster, "load_cpp_module",
                        lambda _: failing_cpp_module(cpp_module, cap_status))

    model = AnalyticProphet()
    model.fit(df, lib_path=compiled_optimizer_module)

    assert model.optimizer_used == "LBFGS"


def test_the_fallback_can_be_switched_off(peyton_manning_df, cpp_module,
                                          compiled_optimizer_module, monkeypatch):
    """[fc] `newton_fallback`, set in IStanBackend.__init__ and honoured before
    the retry."""
    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    monkeypatch.setattr(forecaster, "load_cpp_module",
                        lambda _: failing_cpp_module(cpp_module, CPP_SOLVER_RAISED))

    model = AnalyticProphet()
    model.newton_fallback = False
    model.fit(df, lib_path=compiled_optimizer_module)

    assert model.optimizer_used == "LBFGS"
    assert model.opt_status_message == "provoked"


def test_the_fallback_fires_where_the_python_path_actually_fails(peyton_manning_df):
    """Not provoked: the one configuration in this repo that really terminates
    ABNORMAL.

    Yearly forced at order 10 on 328 days, started from all zeros -- the
    configuration test_fit_cpp_parity uses to match the two paths' initial
    values. The auto rule would never select it, and scipy's line search fails
    on it (#24). It is specific to that start: from `linear_growth_init`, the
    same model converges.

    The retry converges and lands on the same objective to within 1e-9, so the
    fallback corrects the reported status without moving the answer -- which is
    the best a fallback can do.
    """
    from conftest import pin_yearly_only

    df = peyton_manning_df.iloc[:300].reset_index(drop=True)
    zero_init = {"k": 0.0, "m": 0.0,
                 "delta": np.zeros(forecaster.N_CHANGE_POINTS),
                 "beta": np.zeros(2 * forecaster.n_yearly)}

    without = pin_yearly_only(AnalyticProphet())
    without.newton_fallback = False
    without.fit(df, backend="python", analytic=True, initial_params=zero_init)
    if without.opt.status != SCIPY_LINE_SEARCH_FAILURE:
        # Which configurations provoke scipy's line search is a property of the
        # scipy build. This one does on macOS/arm64, and in no CI job on Linux,
        # where it converges cleanly -- so there is nothing here to fall back
        # from and nothing for this test to measure (#103). The *mechanism* is
        # covered either way by the tests above, which drive the status
        # directly rather than hoping for it.
        pytest.skip("scipy's line search does not fail on this build: the "
                    f"configuration converged with status {without.opt.status}, "
                    "so there is no real failure to fall back from")
    assert without.opt.status == SCIPY_LINE_SEARCH_FAILURE

    with_fallback = pin_yearly_only(AnalyticProphet())
    with_fallback.fit(df, backend="python", analytic=True, initial_params=zero_init)

    assert with_fallback.optimizer_used == "Newton"
    assert with_fallback.opt.success
    assert with_fallback._minus_log_posterior(with_fallback.get_parameters()) == \
        pytest.approx(without._minus_log_posterior(without.get_parameters()), abs=1e-9)


# -- both paths converge at every short size ------------------------------

@pytest.mark.parametrize("n_rows", [10, 20, 30, 50, 75, 99, 100])
def test_the_compiled_path_converges_on_every_short_series(peyton_manning_df,
                                                           compiled_optimizer_module,
                                                           n_rows):
    """The other half of #25: a fallback needs something to catch, and there is
    still nothing here -- the compiled path converges at every size down to ten
    observations, now under whichever algorithm the rule picks.
    """
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:n_rows].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.opt_status_message == "converged", model.opt_status_message
    assert 0 < model.opt.n_iterations < 10000
    assert np.all(np.isfinite(model.get_parameters()))


@pytest.mark.parametrize("n_rows", [10, 20, 30, 50, 75, 99, 100])
def test_the_python_path_converges_on_every_short_series(peyton_manning_df, n_rows):
    """`fit()` is the readable reference rather than the deliverable, and it
    used to terminate ABNORMAL at T = 30.

    Aligning the changepoint placement with Prophet's (#15) removed that:
    index-spaced changepoints land on actual observations, so each one has data
    at it, where a time-spaced changepoint could fall in a gap with nothing
    nearby. Better-conditioned, and it converges everywhere now.
    """
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:n_rows].reset_index(drop=True), backend="python", analytic=True)
    assert model.opt.success, f"{model.optimizer_used} reports {model.opt.message} at T={n_rows}"


# -- Newton on the natural parameterization does not work ----------------

def test_newton_runs_on_the_split_reformulation(peyton_manning_df):
    """Why Newton needed #23 as much as L-BFGS did.

    On the natural parameterization the Laplace prior leaves a kink exactly
    where the optimum sits; Newton has no mechanism to land on one and, measured,
    oscillates across it at about 1e-5 progress a step -- exhausting Prophet's
    whole 10000-iteration budget 66 nats short. Split, the L1 term is linear, so
    its curvature is zero rather than undefined.

    The check that this is what is running: at an optimum reached under the
    split bounds, every delta_pos/delta_neg pair has at most one non-zero
    member, which is the property that makes the two problems equivalent.
    """
    model = AnalyticProphet()
    model.fit(peyton_manning_df.iloc[:50].reset_index(drop=True), backend="python",
              analytic=True, algorithm="Newton")

    assert model.optimizer_used == "Newton"
    assert model.opt.success
    n_delta = model.layout.n_changepoints
    delta_pos = model.opt.x[2:2 + n_delta]
    delta_neg = model.opt.x[2 + n_delta:2 + 2 * n_delta]
    assert np.all(delta_pos >= 0) and np.all(delta_neg >= 0)
    assert np.max(np.minimum(delta_pos, delta_neg)) < 1e-8


def test_the_hessian_is_a_difference_of_the_analytic_gradient(prepared_model):
    """Newton's curvature comes from differencing the gradient this project
    derives, not from differencing the objective and not from autodiff.

    Checked against differences of the objective, which are an order less
    accurate -- agreeing to ~1e-5 is what says both describe the same curvature
    while the gradient-based one is the sharper of the two.
    """
    model = prepared_model
    z = forecaster.canonical_to_split(
        np.concatenate(([0.1], [0.2], np.zeros(model.layout.n_changepoints), [1.0],
                        np.zeros(model.layout.n_regressor_columns))),
        model.layout)

    analytic = forecaster.finite_difference_hessian(
        lambda point: model._split_gradient(point), z)

    step = 1e-4
    numeric = np.empty_like(analytic)
    for i in range(z.size):
        for j in range(z.size):
            shift_i, shift_j = np.zeros(z.size), np.zeros(z.size)
            shift_i[i] = shift_j[j] = step
            numeric[i, j] = (
                model._split_minus_log_posterior(z + shift_i + shift_j)
                - model._split_minus_log_posterior(z + shift_i - shift_j)
                - model._split_minus_log_posterior(z - shift_i + shift_j)
                + model._split_minus_log_posterior(z - shift_i - shift_j)) / (4 * step ** 2)

    assert np.allclose(analytic, analytic.T)
    assert np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic)) < 1e-5


def test_projected_newton_pins_a_coordinate_at_its_bound():
    """The active set, on a problem small enough to read.

    Minimize (x+1)^2 + (y-2)^2 with x >= 0: the unconstrained optimum is
    x = -1, so x must come to rest exactly at its bound with the gradient still
    pushing into it, while y reaches its own optimum in the interior. A Newton
    step without the projection would return x = -1 and leave the bound violated.
    """
    objective = lambda v: (v[0] + 1.0) ** 2 + (v[1] - 2.0) ** 2
    gradient = lambda v: np.array([2.0 * (v[0] + 1.0), 2.0 * (v[1] - 2.0)])

    result = forecaster.projected_newton(
        objective, gradient, np.array([3.0, -5.0]),
        np.array([0.0, -np.inf]), np.array([np.inf, np.inf]))

    assert result.success
    assert result.x[0] == pytest.approx(0.0, abs=1e-9)
    assert result.x[1] == pytest.approx(2.0, abs=1e-6)
    assert result.jac[0] > 0  # still pushing into the bound, and correctly held
