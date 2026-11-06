"""
Issue #255: the compiled Newton's `n_iterations` left out a step.

newton() in optimize.cpp counted a step only when its loop came round again, so
the step Stan's objective or parameter tests ended the run on went uncounted.
On Peyton Manning at T = 50 it reported 36 iterations for 37 steps, where the
Python twin reported 37. #240's first draft read the count and said one Newton
step where the method takes two.

Both twins now report the Newton steps they accepted. The loss trace gains one
entry per accepted step and one per saddle escape, so on either backend

    n_iterations == len(loss_trace) - 1 - saddle_escapes

however the run ends. Checked here:

  * **the identity, on default fits below 100 observations**, where Prophet's
    rule runs Newton, on both backends; those runs end on Stan's objective
    test, the exit the step went missing on, and the test says so;
  * **a restart from where a run converged**, which takes no step or one;
  * **a run with saddle escapes** (#193's reproducer): the escapes are not
    counted as steps;
  * **the two backends agree** where they take the same steps;
  * **the cap**, on the Python twin, whose `while ... else` the compiled loop
    now copies: a run that meets a test on the last step the cap allows has
    converged, and one that does not has hit the cap. The compiled cap is
    10,000 steps, out of a test's reach.
"""
import numpy as np
import pytest

from analytic_prophet import AnalyticProphet
from analytic_prophet import optimizer

STAN_RELATIVE = optimizer.STAN_TOL_REL_OBJ * optimizer.STAN_EPS


def _count(model):
    """What the model reports: `n_iterations` from the compiled core, `nit`
    from scipy's OptimizeResult on the Python path."""
    return getattr(model.opt, "n_iterations", getattr(model.opt, "nit", None))


def _steps(model):
    return len(model.opt.loss_trace) - 1 - model.opt.saddle_escapes


def _ended_on_the_objective_test(trace):
    """Whether the last step met Stan's tol_obj or tol_rel_obj: the run's last
    step is one Stan's tests were read on, and these two are visible in the
    trace."""
    before, after = trace[-2], trace[-1]
    change = before - after
    return (change < optimizer.STAN_TOL_OBJ
            or change / max(abs(before), abs(after), 1.0) < STAN_RELATIVE)


def _fit(df, backend, lib_path, **kwargs):
    model = AnalyticProphet()
    if backend == "cpp":
        kwargs["lib_path"] = lib_path
    model.fit(df, backend=backend, **kwargs)
    return model


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("n_rows", [30, 50, 75, 99])
def test_the_count_is_the_steps_taken(backend, n_rows, peyton_manning_df,
                                      compiled_optimizer_module):
    model = _fit(peyton_manning_df.iloc[:n_rows].reset_index(drop=True), backend,
                 compiled_optimizer_module)
    assert model.optimizer_used == "Newton"
    assert model.opt.status == 0
    assert _count(model) == _steps(model)


def test_the_runs_above_end_on_the_objective_test(peyton_manning_df,
                                                  compiled_optimizer_module):
    """Without this, the identity above could hold only because every run
    happened to end on the gradient test, where the old count was right too."""
    ended = [_ended_on_the_objective_test(
                 _fit(peyton_manning_df.iloc[:n_rows].reset_index(drop=True), "cpp",
                      compiled_optimizer_module).opt.loss_trace)
             for n_rows in (30, 50, 75, 99)]
    assert any(ended), "no run ended on the exit the step went missing on"


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("n_rows", [50, 99])
def test_a_restart_from_where_newton_stopped_counts_what_it_takes(
        backend, n_rows, peyton_manning_df, compiled_optimizer_module):
    """Started where a Newton run converged, the next run has at most a tiny
    step to take, and often none: Stan's gradient test ends it before any
    step, where the old count was right, or one step does, where it was not.
    Either way the count is the steps and the run has converged. Here the
    restarts at T = 50 and 99 take none; the bound of two leaves one step's
    margin for another platform's arithmetic."""
    df = peyton_manning_df.iloc[:n_rows].reset_index(drop=True)
    first = _fit(df, backend, compiled_optimizer_module)
    start = {"k": float(first.params["k"][0][0]), "m": float(first.params["m"][0][0]),
             "delta": np.asarray(first.params["delta"][0], dtype=float),
             "sigma_obs": float(first.sigma_obs),
             "beta": np.asarray(first.params["beta"][0], dtype=float)}
    again = _fit(df, backend, compiled_optimizer_module, algorithm="Newton",
                 initial_params=start)
    assert again.opt.status == 0
    assert _count(again) == _steps(again) <= 2


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_saddle_escapes_are_not_counted_as_steps(backend, compiled_optimizer_module):
    """#193's reproducer, started at the saddle: both twins step off it."""
    from test_newton_saddle import SADDLE, _frame, _model

    model = _model()
    kwargs = {"lib_path": compiled_optimizer_module} if backend == "cpp" else {}
    model.fit(_frame(), backend=backend, algorithm="Newton", initial_params=dict(SADDLE),
              **kwargs)
    assert model.opt.saddle_escapes >= 1
    assert _count(model) == _steps(model)


def test_the_backends_agree_where_they_take_the_same_steps(peyton_manning_df,
                                                           compiled_optimizer_module):
    df = peyton_manning_df.iloc[:50].reset_index(drop=True)
    compiled = _fit(df, "cpp", compiled_optimizer_module)
    python = _fit(df, "python", compiled_optimizer_module)
    if len(compiled.opt.loss_trace) != len(python.opt.loss_trace):
        pytest.skip("the twins took different paths on this platform: "
                    f"{len(compiled.opt.loss_trace)} and {len(python.opt.loss_trace)} entries")
    assert compiled.opt.n_iterations == python.opt.nit


# -- the cap, on the Python twin ---------------------------------------------

def _quadratic(curvature):
    return (lambda z: 0.5 * curvature * float(z @ z)), (lambda z: curvature * z)


def test_a_test_met_on_the_last_allowed_step_is_convergence(monkeypatch):
    """One step allowed, and it meets tol_obj: the gradient is 1e-5 at the
    start, above tol_grad, and the step gains 5e-15."""
    monkeypatch.setattr(optimizer, "STAN_MAX_ITERATIONS", 1)
    objective, gradient = _quadratic(1e4)
    result = optimizer.projected_newton(objective, gradient, np.array([1e-9]),
                                        np.array([-np.inf]), np.array([np.inf]))
    assert (result.status, result.message, result.nit) == (0, "converged", 1)


def test_running_out_of_steps_is_the_cap(monkeypatch):
    monkeypatch.setattr(optimizer, "STAN_MAX_ITERATIONS", 1)
    objective, gradient = _quadratic(1.0)
    result = optimizer.projected_newton(objective, gradient, np.array([1.0]),
                                        np.array([-np.inf]), np.array([np.inf]))
    assert (result.status, result.nit) == (-2, 1)
    assert result.message == "reached the iteration cap without converging"
