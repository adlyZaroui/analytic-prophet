"""The compiled backend and the conversions it takes.

[fc] prophet/models.py, which is where Prophet keeps its Stan backends. This
is the same seam: where Prophet hands the problem to Stan, this hands it to
a gradient written out by hand and compiled from optimize.cpp.
"""
import glob
import importlib
import importlib.util
import os

import numpy as np

from .layout import DEFAULT_LAYOUT, extract_params


CPP_MODULE_NAME = 'analytic_prophet_cpp'

BUILD_HINT = (
    "Build it from analytic_prophet/optimize.cpp -- see the compile command in "
    "that file's trailing comment, or let tests/conftest.py's "
    "compiled_optimizer_module fixture build it for you."
)

_cpp_module_cache = {}

def load_cpp_module(lib_path=None):
    """Import the compiled pybind11 extension backing `fit()`.

    With lib_path=None this is an ordinary import, so a built or installed
    extension is found on sys.path like any other module; failing that, it
    looks for one built in place next to this file. Passing lib_path loads a
    specific .so, which is how the tests point at one built into a temp dir.

    The ctypes binding this replaced hardcoded a *relative* path
    ('./liboptimization.so'), so it only worked when the process happened to be
    running from the right directory.
    """
    if lib_path is None:
        try:
            return importlib.import_module(CPP_MODULE_NAME)
        except ImportError:
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = sorted(glob.glob(os.path.join(here, CPP_MODULE_NAME + '*.so')))
            if not candidates:
                raise ImportError(
                    f"{CPP_MODULE_NAME} is not importable and no build of it was found "
                    f"in {here}. {BUILD_HINT}"
                )
            lib_path = candidates[0]

    lib_path = os.path.abspath(lib_path)
    if lib_path not in _cpp_module_cache:
        spec = importlib.util.spec_from_file_location(CPP_MODULE_NAME, lib_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"{lib_path} is not loadable as a Python extension module. {BUILD_HINT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _cpp_module_cache[lib_path] = module
    return _cpp_module_cache[lib_path]

def canonical_to_cpp(params, layout=DEFAULT_LAYOUT):
    """(k, m, delta, sigma_obs, beta) -> (k, m, delta, beta, zeta).

    The C++ core carries zeta = log(sigma_obs) as the LAST element, whereas the
    canonical layout keeps sigma_obs between delta and beta, mirroring the order
    of Stan's `parameters` block. Both describe the same model; only the
    packing differs. The log is what keeps sigma_obs positive in the C++, since
    liblbfgs has no box constraints.
    """
    k, m, delta, sigma_obs, beta = extract_params(params, layout)
    return np.concatenate(([k], [m], delta, beta, [np.log(sigma_obs)]))

def cpp_to_canonical(params, layout=DEFAULT_LAYOUT):
    """Inverse of canonical_to_cpp: sigma_obs = exp(zeta), moved into place."""
    params = np.asarray(params, dtype=float)
    n_delta = layout.n_changepoints
    k, m = params[0], params[1]
    delta = params[2:2 + n_delta]
    beta = params[2 + n_delta:-1]
    sigma_obs = np.exp(params[-1])
    return np.concatenate(([k], [m], delta, [sigma_obs], beta))
