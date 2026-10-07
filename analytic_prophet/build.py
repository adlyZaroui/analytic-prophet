"""Building the C++ core: on demand, once, and from one place.

Three things wanted this and none of them had it (#108).

`fit(df)` runs the compiled core, which is the point of the project -- and
until now nothing built it. `load_cpp_module` imported an extension or raised,
so the default call failed on a fresh clone and the README's claim that the
core was "built on demand" was true only of the test suite and the benchmarks.

Those two had a builder each, and the package that needed one had none. The
copies had drifted: `tests/conftest.py` compiled at `-O2` and
`benchmark/_common.py` at `-O3`, so the performance numbers came from a
different binary than the parity tests verified. There is one builder now, it
compiles at `-O3`, and all three callers reach it -- which also means every
test run and every CI job exercises the code path a user's first `fit` takes.

**The build is cached by content.** The key is a digest of the source, the
compile command and the interpreter's `EXT_SUFFIX`, so editing `optimize.cpp`
or switching Python rebuilds, and nothing else does. That matters for
development as much as for speed: a stale binary that silently survives a
source edit would make every measurement a lie.
"""
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

logger = logging.getLogger("analytic_prophet")

CPP_MODULE_NAME = "analytic_prophet_cpp"

PACKAGE_DIR = Path(__file__).resolve().parent
CPP_SOURCE = PACKAGE_DIR / "optimize.cpp"

# [fc] nothing. The flags are the compile command in optimize.cpp's trailing
# comment, which was the documented way to build it by hand.
#
# -O3 rather than -O2 because that is what the published timings were measured
# with. Neither enables -ffast-math, so the arithmetic is unchanged: -O3 buys
# inlining and vectorisation that are not allowed to reassociate floating
# point, which is why the parity tests hold to 1.5e-8 either way.
#
# -g0 is for the wheels and does nothing here (#135). This path invokes the
# compiler directly, so no debug info was ever requested; `setup.py` goes
# through setuptools, which *prepends* the interpreter's own `OPT` -- on a
# manylinux image `-DNDEBUG -g -fwrapv -O3 -Wall` -- and `extra_compile_args`
# are appended, so nothing cancelled the `-g`. The published 0.1.0 Linux
# wheels carry 8 DWARF sections and a 34.8 MB shared object against macOS's
# 0.6 MB, because macOS leaves DWARF in a separate `.dSYM` rather than in the
# binary. With -g0 the Linux object is 0.54 MB. Stripping as well reaches
# 0.44 MB, which is not worth a post-processing step over not emitting it.
COMPILE_FLAGS = ("-std=c++17", "-shared", "-fPIC", "-O3", "-g0")

BUILD_TIMEOUT_SECONDS = 300

BUILD_HINT = (
    "Build it from analytic_prophet/optimize.cpp -- see the compile command in "
    "that file's trailing comment -- or use fit(df, backend='python'), which "
    "needs no compiler."
)


class ToolchainMissing(ImportError):
    """The C++ core cannot be built here, and why.

    An `ImportError` subclass so that code already written around
    `load_cpp_module` -- which raised `ImportError` before there was a builder
    -- keeps catching it. The message names the one thing that is missing
    rather than the list of things that might be, because the caller's next
    action depends on which (#108).
    """


def _logging_is_listening():
    """Whether an INFO record from this logger would reach anyone.

    `logging.lastResort` only emits WARNING and above, so a library that logs
    at INFO into an application that never configured logging is talking to
    itself.
    """
    if not logger.isEnabledFor(logging.INFO):
        return False
    current = logger
    while current is not None:
        if current.handlers:
            return True
        current = current.parent if current.propagate else None
    return False


def _announce(message, *args):
    """Say it through logging, and through stderr when nothing is listening.

    A first build blocks for ten seconds or so. An application that configures
    logging hears about it there, and one that does not would otherwise meet
    an unexplained pause on its first `fit` -- which is half of what #108 was
    about. Saying it twice to someone who is listening would be worse, so this
    checks.
    """
    logger.info(message, *args)
    if not _logging_is_listening():
        print("analytic-prophet: " + (message % args if args else message),
              file=sys.stderr)


def _compiler():
    return shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")


def _brew_prefix(formula):
    """`brew --prefix <formula>`, or None off macOS and without brew."""
    try:
        prefix = subprocess.run(["brew", "--prefix", formula], capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return prefix or None


def find_eigen_include():
    """The directory holding `Eigen/Dense`, or None."""
    candidates = [os.environ.get("EIGEN_INCLUDE_DIR")]
    prefix = _brew_prefix("eigen")
    if prefix:
        candidates.append(str(Path(prefix, "include", "eigen3")))
    candidates += [
        "/opt/homebrew/opt/eigen/include/eigen3",
        "/usr/local/opt/eigen/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "Eigen" / "Dense").exists():
            return candidate
    return None


def find_lbfgspp_include():
    """The directory holding `LBFGSB.h`, or None.

    NOTE: LBFGSpp ships `LBFGS.h`, which shadows liblbfgs's `lbfgs.h` on a
    case-insensitive filesystem. That stopped mattering when #23 removed the
    liblbfgs dependency, but it is why the two must not share an include path.
    """
    candidates = [os.environ.get("LBFGSPP_INCLUDE_DIR")]
    prefix = _brew_prefix("lbfgspp")
    if prefix:
        candidates.append(str(Path(prefix, "include")))
    candidates += ["/opt/homebrew/include", "/usr/local/include", "/usr/include"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "LBFGSB.h").exists():
            return candidate
    return None


def cache_root():
    """The one directory this project caches anything under.

    `ANALYTIC_PROPHET_CACHE` wins, then `XDG_CACHE_HOME`, then `~/.cache` --
    **the same rule on every platform**, which is the point of #111. The
    variable was read in two places with two different defaults: here, and in
    `evaluation/corpora.py` for the M4 corpus. Setting it moved both and
    leaving it unset moved them apart, because this one took the platform
    cache directory on macOS and that one did not. A reader could not say
    where either landed without reading both modules.

    `~/.cache` rather than `~/Library/Caches` on macOS, because that is what
    the corpus loader already used: picking the other way round would have
    orphaned every M4 download on disk to tidy up a build directory that
    rebuilds itself in twelve seconds.

    Never beside the package: an installed one may be read-only, and writing
    into site-packages is not this library's business.
    """
    override = os.environ.get("ANALYTIC_PROPHET_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".cache") \
        / "analytic-prophet"


def cache_dir():
    """Where a built extension is kept between runs: `<root>/build`.

    A subdirectory rather than the root itself, so the compiled artefact and
    the downloaded corpus cannot be mistaken for each other or for a stray
    file somebody left there (#111).
    """
    return cache_root() / "build"


def _build_command(compiler, eigen, lbfgspp, pybind11_include, out_path):
    command = [compiler, *COMPILE_FLAGS, "-o", str(out_path), str(CPP_SOURCE),
               f"-I{eigen}", f"-I{pybind11_include}",
               f"-I{sysconfig.get_paths()['include']}", f"-I{lbfgspp}"]
    if sys.platform == "darwin":
        # Extension modules resolve CPython's symbols from the host
        # interpreter at load time rather than linking libpython.
        command += ["-undefined", "dynamic_lookup"]
    return command


def _digest(command):
    """A key for this source compiled this way by this interpreter.

    The command carries the compiler, the flags and every include path, and
    EXT_SUFFIX carries the Python version and the platform. Hashing the source
    text is what makes an edit to optimize.cpp rebuild rather than quietly
    reuse the binary it no longer describes.
    """
    digest = hashlib.sha256()
    digest.update(CPP_SOURCE.read_bytes())
    digest.update("\0".join(command).encode())
    digest.update((sysconfig.get_config_var("EXT_SUFFIX") or ".so").encode())
    return digest.hexdigest()[:16]


def build_cpp_extension(dest=None, force=False):
    """Compile `optimize.cpp` and return the path to the extension.

    With `dest` the build goes there and is reused if already present, which
    is what the tests and benchmarks want: a directory they control. Without
    it the build goes to a content-keyed subdirectory of `cache_dir()` and is
    reused across processes, which is what `fit(df)` wants.

    Raises `ToolchainMissing`, naming the one thing that is absent.
    """
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

    if dest is not None:
        out_dir = Path(dest)
        out_path = out_dir / f"{CPP_MODULE_NAME}{suffix}"
        if out_path.exists() and not force:
            return str(out_path)

    compiler = _compiler()
    if compiler is None:
        raise ToolchainMissing(
            "no C++ compiler found (looked for c++, g++, clang++), so the C++ "
            f"core cannot be built. {BUILD_HINT}")

    try:
        import pybind11
    except ImportError:
        raise ToolchainMissing(
            "pybind11 is not installed, so the C++ core cannot be built. "
            f"`pip install pybind11`. {BUILD_HINT}") from None

    eigen = find_eigen_include()
    if eigen is None:
        raise ToolchainMissing(
            "Eigen headers not found (set EIGEN_INCLUDE_DIR, or "
            "`brew install eigen` / `apt install libeigen3-dev`), so the C++ "
            f"core cannot be built. {BUILD_HINT}")

    lbfgspp = find_lbfgspp_include()
    if lbfgspp is None:
        raise ToolchainMissing(
            "LBFGSpp headers not found (set LBFGSPP_INCLUDE_DIR, or "
            "`brew install lbfgspp`), so the C++ core cannot be built. "
            f"{BUILD_HINT}")

    if dest is None:
        probe = _build_command(compiler, eigen, lbfgspp, pybind11.get_include(), "")
        out_dir = cache_dir() / _digest(probe)
        out_path = out_dir / f"{CPP_MODULE_NAME}{suffix}"
        if out_path.exists() and not force:
            return str(out_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Built under a private name and moved into place, so that two processes
    # racing here cannot have one import the other's half-written file.
    # os.replace is atomic within a directory.
    staging = out_dir / f".{CPP_MODULE_NAME}.{os.getpid()}{suffix}"

    _announce("building the C++ core from %s (first use here; takes a few "
              "seconds, cached afterwards)", CPP_SOURCE.name)
    started = time.perf_counter()
    result = subprocess.run(
        _build_command(compiler, eigen, lbfgspp, pybind11.get_include(), staging),
        capture_output=True, text=True, timeout=BUILD_TIMEOUT_SECONDS)
    if result.returncode != 0:
        staging.unlink(missing_ok=True)
        raise ToolchainMissing(
            f"the C++ core failed to compile:\n{result.stderr[-800:]}")

    os.replace(staging, out_path)
    _announce("built the C++ core in %.1fs -> %s",
              time.perf_counter() - started, out_path)
    return str(out_path)
