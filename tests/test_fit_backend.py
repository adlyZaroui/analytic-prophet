"""
Issue #98: `fit` runs the compiled core, and the backend is an argument.

Before this, `fit` ran the readable reference implementation and `fit_cpp`
ran the deliverable -- so a script ported from Prophet, which writes
`model.fit(df)` and nothing else, silently got the slow path. The compiled
core with the hand-derived gradient is the point of the project, and it was
the one call a ported script could not reach without editing it.

What this file pins is the part that is easy to get wrong later: that no
argument is silently ignored. `fit` takes the union of two backends'
arguments, and one belonging to the backend that was not selected **raises**
rather than being dropped -- the rule #52 set for the constructor, for the
same reason. A flag that does nothing is worse than one that is rejected,
because nothing in the result says so.
"""
import numpy as np
import pytest

from analytic_prophet import AnalyticProphet, BACKEND_ARGUMENTS


@pytest.fixture
def df(peyton_manning_df):
    return peyton_manning_df.iloc[:300].reset_index(drop=True)


# -- what the default is --------------------------------------------------

def test_fit_runs_the_compiled_core_by_default(df, compiled_optimizer_module):
    """The headline. A ported Prophet script keeps `model.fit(df)` and gets
    the deliverable rather than the control."""
    model = AnalyticProphet()
    model.fit(df, lib_path=compiled_optimizer_module)

    assert model._fitted_with_cpp is True
    assert np.all(np.isfinite(model.get_parameters()))


def test_the_python_backend_is_reachable_and_is_the_other_one(df):
    model = AnalyticProphet()
    model.fit(df, backend="python")

    assert model._fitted_with_cpp is False
    assert np.all(np.isfinite(model.get_parameters()))


def test_fit_returns_self_so_calls_chain(df, compiled_optimizer_module):
    """[fc] Prophet.fit returns self, which is what makes
    `Prophet().fit(df).predict(future)` work. `fit` returned None here and
    `fit_cpp` returned a leftover `-1`."""
    model = AnalyticProphet()
    returned = model.fit(df, lib_path=compiled_optimizer_module)
    assert returned is model

    chained = (AnalyticProphet()
               .fit(df, lib_path=compiled_optimizer_module))
    forecast = chained.predict(chained.make_future_dataframe(periods=5))
    assert np.all(np.isfinite(forecast["yhat"]))


def test_the_python_backend_uses_the_analytic_gradient_by_default(df, monkeypatch):
    """The old `fit` defaulted to `analytic=False`, so the natural call to the
    reference path got finite differences -- the control rather than the thing
    being controlled for. Asserted by counting gradient calls: the analytic
    path asks for a jacobian, the numeric one leaves scipy to approximate it.
    """
    from analytic_prophet import forecaster

    seen = {}
    real_minimize = forecaster.minimize

    def recording(*args, **kwargs):
        seen["jac"] = kwargs.get("jac")
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(forecaster, "minimize", recording)
    AnalyticProphet().fit(df, backend="python", algorithm="LBFGS")
    assert callable(seen["jac"]), "the python backend did not pass a gradient"

    seen.clear()
    AnalyticProphet().fit(df, backend="python", analytic=False, algorithm="LBFGS")
    assert seen["jac"] is None, "analytic=False still passed a gradient"


# -- the rejection rule ---------------------------------------------------

def test_an_unknown_backend_is_rejected():
    with pytest.raises(ValueError) as raised:
        AnalyticProphet().fit(None, backend="rust")
    assert "'cpp'" in str(raised.value) and "'python'" in str(raised.value)


@pytest.mark.parametrize("argument", sorted(BACKEND_ARGUMENTS["python"]))
def test_a_python_argument_under_the_compiled_backend_raises(df, argument):
    """Every one of them, from the map rather than from a list written twice."""
    with pytest.raises(ValueError) as raised:
        AnalyticProphet().fit(df, **{argument: True})
    message = str(raised.value)
    assert argument in message
    assert "backend='python'" in message      # says which backend wants it
    assert "backend='cpp'" in message         # and which one was asked for


@pytest.mark.parametrize("argument", sorted(BACKEND_ARGUMENTS["cpp"]))
def test_a_compiled_argument_under_the_python_backend_raises(df, argument):
    with pytest.raises(ValueError) as raised:
        AnalyticProphet().fit(df, backend="python", **{argument: True})
    message = str(raised.value)
    assert argument in message
    assert "backend='cpp'" in message


def test_a_falsy_value_is_still_a_value(df):
    """The reason for the sentinel. `analytic=False` is both the old default
    and a thing a caller can mean, so "was it passed?" cannot be answered by
    comparing against a default -- and under the compiled backend it has to
    raise exactly as `analytic=True` does."""
    with pytest.raises(ValueError, match="analytic"):
        AnalyticProphet().fit(df, analytic=False)
    with pytest.raises(ValueError, match="verbose"):
        AnalyticProphet().fit(df, backend="python", verbose=False)


def test_the_shared_arguments_are_accepted_by_both(df, compiled_optimizer_module):
    """`initial_params` and `algorithm` are the two neither backend rejects."""
    initial = {"k": 0.0, "m": 0.0}
    AnalyticProphet().fit(df, initial_params=initial, algorithm="LBFGS",
                          lib_path=compiled_optimizer_module)
    AnalyticProphet().fit(df, backend="python", initial_params=initial,
                          algorithm="LBFGS")


def test_every_non_shared_argument_is_claimed_by_exactly_one_backend():
    """The map is the rejection rule, so an argument appearing in both or in
    neither would make the rule say nothing."""
    cpp, python = set(BACKEND_ARGUMENTS["cpp"]), set(BACKEND_ARGUMENTS["python"])
    assert cpp and python
    assert cpp.isdisjoint(python)

    import inspect
    taken = set(inspect.signature(AnalyticProphet.fit).parameters) - {
        "self", "df", "backend", "initial_params", "algorithm"}
    assert taken == cpp | python, (
        "fit takes an argument no backend claims, so it can never be rejected")


# -- the old name is gone -------------------------------------------------

def test_fit_cpp_is_gone(df):
    """Removed rather than deprecated: nothing outside this repo depends on
    it, and an alias would keep the confusion the issue was filed about."""
    assert not hasattr(AnalyticProphet(), "fit_cpp")
