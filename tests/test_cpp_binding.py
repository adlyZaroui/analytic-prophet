"""
Issue #1: keep ctypes, or move to pybind11? Resolved as pybind11.

The issue named two concrete problems with the ctypes binding:

1. "the load path is a hardcoded relative string that breaks unless the
   working directory happens to be right" -- fit_cpp() did
   ctypes.CDLL('./liboptimization.so'). The extension is now an ordinary
   importable module, so it is found on sys.path like anything else.

2. "every new function exposed to Python means hand-writing another ctypes
   signature block" -- the old binding carried a 19-entry argtypes list that
   had to be kept in lockstep with the C++ signature by hand, with array
   lengths passed as separate ints and results returned through
   out-parameters. Getting any of that wrong is undefined behaviour, not an
   error: ctypes hands over raw pointers and trusts the declaration.

These tests cover what the move actually buys, since the numerical behaviour
is unchanged and already covered elsewhere (test_cpp_optimizer_convergence.py,
test_fit_cpp_parity.py). What is new is that wrong input is now *rejected*
rather than reinterpreted as a pointer.
"""
import sys

import numpy as np
import pytest

import customProphet
from customProphet import (CustomProphet, CPP_MODULE_NAME, N_CHANGE_POINTS, n_yearly,
                           SIGMA_OBS_PRIOR_SCALE, YEARLY_PERIOD, load_cpp_module)
from conftest import pin_yearly_only

SIGMA_OBS = 1.0
CPP_PARAM_SIZE = 2 + N_CHANGE_POINTS + 2 * n_yearly + 1   # [k, m, delta, beta, zeta]


def _valid_params():
    """A C++-layout vector at Prophet's initialization: everything zero, with
    zeta = log(1.0) = 0 for the trailing sigma_obs slot."""
    return np.zeros(CPP_PARAM_SIZE)


@pytest.fixture
def call_kwargs(prepared_model):
    """A valid optimize() call, for tests that then break one argument."""
    return {
        "params": _valid_params(),
        "t": prepared_model.t,
        "changepoints_t": prepared_model.changepoints_t,
        "t_seasonality": prepared_model.t_seasonality,
        "y_scaled": prepared_model.y_scaled,
        "sigma_obs_prior_scale": SIGMA_OBS_PRIOR_SCALE,
        "sigma_k": prepared_model.sigma_k,
        "sigma_m": prepared_model.sigma_m,
        "sigmas": prepared_model.sigmas,
        "changepoint_prior_scale": prepared_model.changepoint_prior_scale,
        "fourier_orders": [n_yearly],
        "seasonality_periods": [YEARLY_PERIOD],
    }


def test_optimize_returns_a_named_result(cpp_module, call_kwargs):
    """The result carries named attributes. The ctypes binding could only
    return through out-parameters -- a preallocated loss buffer plus its
    length, and two byref'd ints.

    No assertion on the specific status code here: on this (full) dataset the
    run ends at the optimum but with -998, the line search hitting its
    evaluation cap once there is nothing left to gain. Convergence itself is
    asserted in test_cpp_optimizer_convergence.py, against the optimality
    conditions rather than against a status code.
    """
    result = cpp_module.optimize(**call_kwargs)

    assert result.n_iterations > 0
    # loss_trace samples evaluations, not iterations (LBFGSpp has no
    # per-iteration hook), so it is >= n_iterations rather than equal to it
    assert len(result.loss_trace) >= result.n_iterations
    assert result.params.shape == (CPP_PARAM_SIZE,)
    assert np.all(np.isfinite(result.params))
    assert result.loss_trace[-1] < result.loss_trace[0]

    # every code maps to a sentence, so a failure is readable without going to
    # lbfgs.h -- and the repr carries it too
    assert result.status_message and not result.status_message.startswith("liblbfgs error code")
    assert result.status_message in repr(result)


def test_optimize_does_not_mutate_the_callers_array(cpp_module, call_kwargs):
    """The ctypes binding optimized in place, because it was handed a pointer
    into the caller's numpy buffer. The typed binding takes a copy, so the
    input survives and the result is returned."""
    params = _valid_params()
    call_kwargs["params"] = params

    result = cpp_module.optimize(**call_kwargs)

    np.testing.assert_array_equal(params, _valid_params())
    assert not np.allclose(result.params, 0.0)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"changepoint_prior_scale": 0.0}, "changepoint_prior_scale must be positive"),
        ({"sigma_obs_prior_scale": -1.0}, "sigma_obs_prior_scale must be positive"),
        ({"y_scaled": np.zeros(7)}, "same length"),
        ({"params": np.zeros(3)}, "too short"),
    ],
)
def test_invalid_input_raises_instead_of_corrupting_memory(cpp_module, call_kwargs, override, message):
    """Each of these was undefined behaviour under ctypes: a mismatched length
    or a zero denominator was simply passed through to C++ as-is."""
    call_kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        cpp_module.optimize(**call_kwargs)


def test_wrong_argument_type_raises_type_error(cpp_module, call_kwargs):
    """ctypes would have tried to reinterpret this as a double pointer."""
    call_kwargs["t"] = "not an array"

    with pytest.raises(TypeError):
        cpp_module.optimize(**call_kwargs)


def test_optimize_requires_its_arguments(cpp_module, call_kwargs):
    """A missing argument is an error at the boundary. The ctypes call was
    positional with 19 entries, where a dropped argument shifted everything
    after it."""
    del call_kwargs["changepoint_prior_scale"]

    with pytest.raises(TypeError):
        cpp_module.optimize(**call_kwargs)


def test_gradient_entry_point_returns_a_tuple(cpp_module, call_kwargs, prepared_model):
    """Exposing a second function needed no new signature block -- the point of
    the issue's second complaint."""
    del call_kwargs["params"]
    mlp, gradient = cpp_module.minus_log_posterior_and_gradient(
        params=_valid_params(), **call_kwargs)

    assert isinstance(mlp, float)
    assert gradient.shape == (CPP_PARAM_SIZE,)

    # include_l1_prior is a real keyword argument with a default, not a magic int
    point = np.full(CPP_PARAM_SIZE, 0.1)
    mlp_without_l1, _ = cpp_module.minus_log_posterior_and_gradient(
        params=point, include_l1_prior=False, **call_kwargs)
    mlp_with_l1, _ = cpp_module.minus_log_posterior_and_gradient(
        params=point, include_l1_prior=True, **call_kwargs)

    expected_l1 = np.sum(np.abs(np.full(N_CHANGE_POINTS, 0.1))) / prepared_model.changepoint_prior_scale
    assert mlp_with_l1 - mlp_without_l1 == pytest.approx(expected_l1)


def test_module_is_importable_by_name_not_just_by_path(compiled_optimizer_module, monkeypatch):
    """The fix for the issue's first complaint: the extension is found the way
    any module is, rather than through a relative path that depends on the
    process's working directory.

    Simulated here by putting the build directory on sys.path, which is what
    installing the package would do.
    """
    build_dir = str(compiled_optimizer_module.rsplit("/", 1)[0])
    monkeypatch.syspath_prepend(build_dir)
    monkeypatch.delitem(sys.modules, CPP_MODULE_NAME, raising=False)

    module = load_cpp_module()

    assert callable(module.optimize)


def test_missing_extension_raises_a_helpful_import_error(monkeypatch, tmp_path):
    """Neither importable nor built in place -- the error should say how to
    build it, rather than ctypes' bare OSError about a missing file.

    Pointing the loader at a module name that cannot exist makes both lookups
    miss for real -- the import and the glob for a build sitting next to the
    source. Patching just the directory would leave the outcome depending on
    whether the developer happens to have an extension built in place.
    """
    monkeypatch.setattr(customProphet, "CPP_MODULE_NAME", "analytic_prophet_cpp_not_built")

    with pytest.raises(ImportError, match="optimize.cpp"):
        load_cpp_module()


def test_loaded_module_is_cached(compiled_optimizer_module):
    """fit_cpp() calls this on every fit; loading the extension repeatedly
    should not re-execute it."""
    assert load_cpp_module(compiled_optimizer_module) is load_cpp_module(compiled_optimizer_module)


def test_fit_cpp_reports_the_termination_status_in_words(peyton_manning_df, compiled_optimizer_module):
    """The old binding surfaced a bare integer; diagnosing issue #8 meant
    looking -1001 up in lbfgs.h by hand."""
    model = pin_yearly_only(CustomProphet())
    model.fit_cpp(
        peyton_manning_df.iloc[:300].reset_index(drop=True),
        initial_params={
            "k": 0.0, "m": 0.0,
            "delta": np.zeros(N_CHANGE_POINTS),
            "beta": np.zeros(2 * n_yearly),
        },
        lib_path=compiled_optimizer_module,
    )

    # No assertion on the specific code: this run ends at the optimum but with
    # -998, the line search hitting its evaluation cap once there is nothing
    # left to gain. What matters here is that the code arrives with a sentence
    # attached rather than as a bare integer.
    assert model.opt_status_message
    assert not model.opt_status_message.startswith("liblbfgs error code")
