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

from .build import (  # noqa: F401 -- re-exported, and the names tests
    # monkeypatch must stay bound in this module
    BUILD_HINT, CPP_MODULE_NAME, CPP_SOURCE, ToolchainMissing,
    build_cpp_extension, cache_dir, find_eigen_include, find_lbfgspp_include)
from .layout import DEFAULT_LAYOUT, extract_params

_cpp_module_cache = {}

def load_cpp_module(lib_path=None):
    """Import the compiled pybind11 extension backing `fit()`.

    With lib_path=None this is an ordinary import, so a built or installed
    extension is found on sys.path like any other module; failing that, it
    looks for one built in place next to this file; failing that, it **builds
    one** and caches it (#108). Passing lib_path loads a specific .so, which
    is how the tests point at one built into a temp dir.

    `ToolchainMissing` comes out of the build when the machine cannot do it,
    naming the one thing that is absent. It subclasses `ImportError`, which is
    what this raised before there was a builder.

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
            # Nothing installed and nothing built in place: build it, once,
            # into a content-keyed cache (#108). Before this, `fit(df)` --
            # the default call, and the one a ported Prophet script makes --
            # raised here on any machine that had not built the extension by
            # hand, which is every fresh clone.
            lib_path = candidates[0] if candidates else build_cpp_extension()

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
