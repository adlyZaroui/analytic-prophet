"""Build the C++ core into the wheel, when a wheel is what is being built.

Issue #97. Until now there was no `setup.py` at all: `optimize.cpp` shipped as
package *data* and was compiled on first use, so `pip install` needed no
toolchain and `fit(df)` did (#108 made that automatic). That is the right
answer for a clone and the wrong one for `pip install analytic-prophet`, where
the whole promise is that a user needs nothing.

So a wheel carries the compiled extension, and this is what compiles it.

**The extension is optional on purpose.** If the compiler, Eigen or LBFGSpp
are missing, the build logs why and produces a package without it rather than
failing. That keeps the old contract intact for the two cases where it still
matters: installing the sdist on a platform with no wheel, and installing from
a clone. In both, `analytic_prophet.build` compiles the core on first use
exactly as before. A wheel that quietly lacks its extension is not a silent
failure either -- the first `fit` builds it and says so.

The flags are `analytic_prophet.build.COMPILE_FLAGS`, read from that module
rather than repeated here, because a wheel compiled differently from the
on-demand path would make every published timing describe a binary nobody has.
"""
import importlib.util
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent


def _build_module():
    """`analytic_prophet/build.py`, loaded as a file rather than imported.

    `import analytic_prophet.build` would run the package's `__init__`, which
    imports the forecaster, which imports numpy -- and numpy is a *runtime*
    dependency, absent from the isolated environment pip builds wheels in.
    The module itself imports nothing but the standard library, which is what
    makes loading it directly safe.
    """
    spec = importlib.util.spec_from_file_location(
        "_analytic_prophet_build", HERE / "analytic_prophet" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_build = _build_module()
COMPILE_FLAGS = _build.COMPILE_FLAGS
CPP_MODULE_NAME = _build.CPP_MODULE_NAME
find_eigen_include = _build.find_eigen_include
find_lbfgspp_include = _build.find_lbfgspp_include


class OptionalBuildExt(build_ext):
    """Compile the core if the toolchain is here, and carry on if it is not."""

    def run(self):
        try:
            super().run()
        except Exception as error:                      # noqa: BLE001
            self._carry_on(error)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as error:                      # noqa: BLE001
            self._carry_on(error)

    @staticmethod
    def _carry_on(error):
        print(f"\nanalytic-prophet: the C++ core was not compiled into this "
              f"build ({type(error).__name__}: {error}).\n"
              f"The package still works: analytic_prophet.build compiles it on "
              f"first use, which needs a C++17 compiler, Eigen and LBFGSpp. "
              f"fit(df, backend='python') needs none of them.\n", file=sys.stderr)


def extensions():
    """The one extension, or none when its headers are not to be found.

    cibuildwheel sets both variables in `before-all`; a developer building a
    wheel by hand gets the same search `analytic_prophet.build` uses.
    """
    eigen = find_eigen_include()
    lbfgspp = find_lbfgspp_include()
    if eigen is None or lbfgspp is None:
        missing = ", ".join(name for name, found in
                            (("Eigen", eigen), ("LBFGSpp", lbfgspp)) if found is None)
        print(f"analytic-prophet: {missing} not found; building without the "
              f"compiled core. Set EIGEN_INCLUDE_DIR / LBFGSPP_INCLUDE_DIR to "
              f"include it.", file=sys.stderr)
        return []

    try:
        import pybind11
    except ImportError:
        print("analytic-prophet: pybind11 is not installed; building without "
              "the compiled core.", file=sys.stderr)
        return []

    # `-shared` and `-fPIC` are setuptools' to add, and `-o` is its own; what
    # is shared with the on-demand path is the language level and the
    # optimisation, which are what change the binary.
    flags = [flag for flag in COMPILE_FLAGS if flag not in ("-shared", "-fPIC")]
    link = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []

    return [Extension(
        f"analytic_prophet.{CPP_MODULE_NAME}",
        sources=["analytic_prophet/optimize.cpp"],
        include_dirs=[eigen, lbfgspp, pybind11.get_include()],
        extra_compile_args=flags,
        extra_link_args=link,
        language="c++",
    )]


setup(ext_modules=extensions(), cmdclass={"build_ext": OptionalBuildExt})
