"""
Issue #25: Prophet switches algorithm below 100 observations. We do not, and
this is the measurement of whether that matters.

[fc] CmdStanPyBackend.fit picks `'Newton' if T < 100 else 'LBFGS'`, and retries
once with Newton if the first attempt raises. The issue was filed on the
reasonable assumption that running a different algorithm from the original on
short series meant we could not expect matching parameters there.

It proposed measuring before building anything, since a Newton optimizer needs
the Hessian -- which this project computes nowhere, and which is a real piece of
work to derive. That measurement is here, and it inverts the premise.

**Prophet's Newton beats Prophet's L-BFGS at every size measured**, which is why
the switch exists. **This implementation's L-BFGS-B beats Prophet's Newton at
every size measured**, including below 100 where Prophet uses it. Adopting
Newton for short series would make the fit worse, not better.

That is not a claim about Newton as a method. It is a claim about this
objective: the Laplace prior makes it non-differentiable exactly where the
optimum sits, the split reformulation (#23) removes that, and Stan's own
convergence criteria (#21) stop the run in the right place. Prophet's L-BFGS
runs on the unreformulated problem and struggles, which is what its Newton
fallback is for.
"""
import numpy as np
import pytest

from customProphet import CustomProphet

# Prophet's own cutoff, [fc] `'Newton' if T < 100 else 'LBFGS'`.
NEWTON_BELOW = 100

SHORT_SIZES = [20, 30, 50, 75, 99]


def prophet_score(Prophet, bridge, common, df, algorithm):
    """Fit Prophet with a forced algorithm and return (stan handles, lp__)."""
    model = Prophet(**common.PROPHET_KWARGS)
    stan_model, stan_data, params = bridge.capture_stan_model(
        model, df, algorithm=algorithm)
    reported = float(np.asarray(model.params["lp__"]).ravel()[0])
    return stan_model, stan_data, bridge.validate_bridge(
        stan_model, stan_data, params, reported)


def our_score(bridge, stan_model, stan_data, df, lib_path):
    """Our fit on Prophet's changepoints, scored under Stan's density.

    The changepoint *count* is taken from Prophet too: below about 32
    observations it caps `n_changepoints` at `floor(T * changepoint_range) - 1`,
    which this implementation does not do -- see the note on #15.
    """
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    model = CustomProphet(n_changepoints=len(changepoints_t))
    model._generate_change_points = lambda: setattr(
        model, "changepoints_t", changepoints_t.copy())
    model.fit_cpp(df, lib_path=lib_path)
    return model, bridge.stan_log_prob(
        stan_model, stan_data, model.params["k"][0][0], model.params["m"][0][0],
        model.params["delta"][0], model.sigma_obs, model.params["beta"][0])


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
    """The measurement #25 asked for, and the reason it does not need a Hessian.

    Below 100 observations Prophet runs Newton because its L-BFGS does poorly
    here. Ours does not have that problem, so switching would cost accuracy
    rather than buy it.
    """
    Prophet, common, bridge = prophet_comparison
    df = common.load_data(n_rows)

    stan_model, stan_data, lp_newton = prophet_score(
        Prophet, bridge, common, df, "Newton")
    _, lp_ours = our_score(bridge, stan_model, stan_data, df, compiled_optimizer_module)

    assert lp_ours >= lp_newton - 1e-6, (
        f"Prophet's Newton reaches {lp_newton:.5f} at T={n_rows}, ours only "
        f"{lp_ours:.5f} -- the premise of #25 would then hold after all")


@pytest.mark.parametrize("n_rows", [10, 20, 30, 50, 75, 99, 100])
def test_the_compiled_path_converges_on_every_short_series(peyton_manning_df,
                                                           compiled_optimizer_module,
                                                           n_rows):
    """The other half of #25: Prophet retries with Newton when the optimizer
    raises. A fallback needs something to catch, and `fit_cpp` reports
    convergence at every size down to ten observations.
    """
    model = CustomProphet()
    model.fit_cpp(peyton_manning_df.iloc[:n_rows].reset_index(drop=True),
                  lib_path=compiled_optimizer_module)

    assert model.opt_status_message == "converged", model.opt_status_message
    assert 0 < model.opt.n_iterations < 10000
    assert np.all(np.isfinite(model.get_parameters()))


def test_the_python_path_still_has_one_abnormal_termination(peyton_manning_df):
    """Pinned rather than fixed, and pinned here because #25 is where someone
    will look for it.

    `fit()` is the readable reference, not the deliverable, and its convergence
    tolerances deviate from Stan's (#24) -- which is where this belongs. At
    T = 30 scipy terminates ABNORMAL. The fit is still usable; the status is
    not clean.
    """
    model = CustomProphet()
    model.fit(peyton_manning_df.iloc[:30].reset_index(drop=True), analytic=True)

    assert not model.opt.success
    assert np.all(np.isfinite(model.get_parameters())), (
        "the fit is unusable, not merely untidy -- that would be a different bug")


def test_prophets_cutoff_is_strict(prophet_comparison):
    """[fc] `'Newton' if T < 100 else 'LBFGS'`. Recorded because the boundary
    is the sort of thing a reimplementation gets off by one."""
    Prophet, common, bridge = prophet_comparison
    assert NEWTON_BELOW == 100

    # at exactly 100 Prophet uses L-BFGS, so its own Newton does better there
    df = common.load_data(NEWTON_BELOW)
    _, _, lp_newton = prophet_score(Prophet, bridge, common, df, "Newton")
    _, _, lp_default = prophet_score(Prophet, bridge, common, df, "LBFGS")
    assert lp_newton > lp_default
