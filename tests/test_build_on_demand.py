"""
Issue #108: `fit(df)` builds the C++ core if it has to.

`fit(df)` runs the compiled core, which is the point of the project, and
until now nothing built it -- `load_cpp_module` imported an extension or
raised. So the default call, the one a script ported from Prophet makes,
failed on every machine that had not compiled it by hand, which is every
fresh clone. The README said the core was "built on demand" and that was true
of the test suite and the benchmarks and never of a user's `fit`.

There were two builders before this, in `tests/conftest.py` and
`benchmark/_common.py`, and none in the package that needed one. They had
drifted to different optimisation levels, so the published timings came from
a different binary than the parity tests verified. There is one now, and
these tests are about it.

What is deliberately *not* here: a test that compiles from scratch to measure
the cold path. Every test in this suite that touches the compiled core
already builds it through this code, once per session, via the
`compiled_optimizer_module` fixture -- so the build is exercised constantly
and what is left to check is the parts around it.
"""
import logging
import os
import sys
import sysconfig
from pathlib import Path

import pytest

from analytic_prophet import build, load_cpp_module, models


# -- the cache ------------------------------------------------------------

def test_the_cache_is_keyed_by_the_source(tmp_path, monkeypatch):
    """A stale binary surviving an edit to optimize.cpp would make every
    measurement after it a lie, so the key is a digest of the source."""
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path))
    command = ["c++", "-O3", "-o", "out.so", str(build.CPP_SOURCE)]

    before = build._digest(command)
    assert before == build._digest(command), "the digest is not stable"

    # the same compile of different source must land elsewhere
    source = tmp_path / "optimize.cpp"
    source.write_text(build.CPP_SOURCE.read_text() + "\n// a change\n")
    monkeypatch.setattr(build, "CPP_SOURCE", source)
    assert build._digest(command) != before


def test_the_cache_is_keyed_by_the_compile_command(tmp_path, monkeypatch):
    """Changing a flag or an include path has to rebuild too -- the binary
    that comes out is not the same one."""
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path))
    at_O3 = build._digest(["c++", "-O3", "-o", "x", str(build.CPP_SOURCE)])
    at_O0 = build._digest(["c++", "-O0", "-o", "x", str(build.CPP_SOURCE)])
    assert at_O3 != at_O0


def test_the_cache_location_is_overridable(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path / "somewhere"))
    assert build.cache_dir() == tmp_path / "somewhere"


def test_the_default_cache_is_not_inside_the_package(monkeypatch):
    """An installed package may be read-only, and writing into site-packages
    is not this library's business."""
    monkeypatch.delenv("ANALYTIC_PROPHET_CACHE", raising=False)
    assert build.PACKAGE_DIR not in build.cache_dir().parents
    assert build.cache_dir() != build.PACKAGE_DIR


def test_a_second_build_reuses_the_first(tmp_path):
    """The build happens once, not per process -- #108's third acceptance
    criterion. Checked by mtime: a rebuild would replace the file."""
    first = build.build_cpp_extension(dest=tmp_path)
    stamp = os.stat(first).st_mtime_ns

    second = build.build_cpp_extension(dest=tmp_path)
    assert second == first
    assert os.stat(second).st_mtime_ns == stamp, "it rebuilt instead of reusing"


def test_the_cached_path_is_the_one_a_user_takes(tmp_path, monkeypatch):
    """Every other test here passes an explicit `dest`, which is what the
    suite and the benchmarks do -- so the branch a *user* takes, with no
    destination at all, would otherwise be the one thing never run.

    ANALYTIC_PROPHET_CACHE points it at a directory this test owns, so the
    real code path runs without writing to the developer's cache.
    """
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path))
    assert build.cache_dir() == tmp_path

    built = Path(build.build_cpp_extension())
    assert built.exists()
    assert tmp_path in built.parents, "it did not land in the cache"
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    assert built.name == build.CPP_MODULE_NAME + suffix

    # content-keyed: the digest is a directory under the cache, not the file
    assert built.parent.parent == tmp_path

    # and the second call is a lookup rather than a compile
    stamp = os.stat(built).st_mtime_ns
    again = build.build_cpp_extension()
    assert Path(again) == built
    assert os.stat(again).st_mtime_ns == stamp


def test_force_rebuilds(tmp_path):
    first = build.build_cpp_extension(dest=tmp_path)
    stamp = os.stat(first).st_mtime_ns
    again = build.build_cpp_extension(dest=tmp_path, force=True)
    assert os.stat(again).st_mtime_ns != stamp


def test_nothing_is_left_behind_by_a_failed_build(tmp_path, monkeypatch):
    """The staging file is private and must not survive a failure, or the
    next run would find a half-written extension where it expects a good one.

    The failure is real rather than stubbed: `false` is a compiler that
    compiles nothing and exits 1. Patching `subprocess.run` instead would
    also intercept the `brew --prefix` lookups the finders make, which is how
    the first version of this test broke.
    """
    import shutil as _shutil
    false = _shutil.which("false")
    if false is None:
        pytest.skip("no `false` to stand in for a failing compiler")

    monkeypatch.setattr(build, "_compiler", lambda: false)
    with pytest.raises(build.ToolchainMissing, match="failed to compile"):
        build.build_cpp_extension(dest=tmp_path)

    assert list(tmp_path.iterdir()) == [], f"left behind: {list(tmp_path.iterdir())}"


# -- what it says when it cannot ------------------------------------------

@pytest.mark.parametrize("missing,patch,expected", [
    ("compiler", lambda m: m.setattr(build, "_compiler", lambda: None),
     "no C++ compiler"),
    ("eigen", lambda m: m.setattr(build, "find_eigen_include", lambda: None),
     "Eigen headers not found"),
    ("lbfgspp", lambda m: m.setattr(build, "find_lbfgspp_include", lambda: None),
     "LBFGSpp headers not found"),
])
def test_it_names_the_one_thing_that_is_missing(tmp_path, monkeypatch,
                                                missing, patch, expected):
    """"Could not build" is not actionable; which of four things is absent
    decides what the caller does next."""
    patch(monkeypatch)
    with pytest.raises(build.ToolchainMissing) as raised:
        build.build_cpp_extension(dest=tmp_path)

    message = str(raised.value)
    assert expected in message
    assert "backend='python'" in message, "it should say what works without a toolchain"
    assert isinstance(raised.value, ImportError), "callers catch ImportError"


# -- the announcement -----------------------------------------------------

def test_a_build_says_so_when_nothing_is_listening(tmp_path, monkeypatch, capsys):
    """A first build blocks for ten seconds or so, and an application that
    never configured logging would otherwise meet an unexplained pause."""
    monkeypatch.setattr(build, "_logging_is_listening", lambda: False)
    build.build_cpp_extension(dest=tmp_path, force=True)
    assert "building the C++ core" in capsys.readouterr().err


def test_a_build_stays_quiet_when_logging_is_listening(tmp_path, monkeypatch, capsys):
    """And does not say it twice to someone who is."""
    monkeypatch.setattr(build, "_logging_is_listening", lambda: True)
    build.build_cpp_extension(dest=tmp_path, force=True)
    assert capsys.readouterr().err == ""


def test_logging_is_listening_only_when_a_handler_would_emit():
    logger = logging.getLogger("analytic_prophet")
    had = list(logger.handlers)
    level = logger.level
    try:
        logger.handlers = []
        logger.setLevel(logging.INFO)
        # no handler anywhere on the chain that is ours; the root's, if the
        # test runner installed one, is what this is allowed to find
        logger.handlers = [logging.NullHandler()]
        assert build._logging_is_listening() is True

        logger.setLevel(logging.WARNING)
        assert build._logging_is_listening() is False
    finally:
        logger.handlers = had
        logger.setLevel(level)


# -- the loader -----------------------------------------------------------

def test_load_cpp_module_builds_when_there_is_nothing_to_import(monkeypatch, tmp_path):
    """The whole point: `fit(df)` on a machine with a toolchain and no build.

    The import and the look-beside-the-package are both made to miss, as in
    test_cpp_binding, and the build is pointed at a directory this test owns.
    """
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path))
    built = {}

    real_build = build.build_cpp_extension

    def recording(dest=None, force=False):
        built["called"] = True
        return real_build(dest=tmp_path, force=force)

    monkeypatch.setattr(models, "build_cpp_extension", recording)
    monkeypatch.setattr(models, "CPP_MODULE_NAME", "analytic_prophet_cpp_absent")
    monkeypatch.setattr(models.glob, "glob", lambda pattern: [])

    with pytest.raises(ImportError):
        # it builds, then fails to find the init symbol for the patched name,
        # which is as far as this can go without a second real extension
        load_cpp_module()
    assert built.get("called"), "the loader did not try to build"


# -- the benchmarks still start -------------------------------------------

def test_the_benchmark_plumbing_imports_as_a_script_does():
    """`benchmark/_common.py` is imported with sys.path[0] = benchmark/, not
    the repo root, because the benchmarks are standalone scripts rather than
    part of `pytest tests/`.

    Under pytest the repo root is on the path anyway, so an import that only
    works there passes the whole suite and fails the moment someone runs
    `python benchmark/benchmark_fit_time.py`. That is exactly what happened
    when #108 added a package import to that file above the line that puts
    the repo root on the path.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", "import _common; _common.IMPLEMENTATIONS"],
        cwd=Path(__file__).parent.parent / "benchmark",
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-1500:]
