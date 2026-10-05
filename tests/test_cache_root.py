"""
Issue #111: one cache, one rule, both halves of it.

`ANALYTIC_PROPHET_CACHE` was read in two places with two different defaults --
`analytic_prophet/build.py` for the compiled core and `evaluation/corpora.py`
for the M4 corpus. Setting it moved both; leaving it unset moved them apart,
because the first took the platform cache directory on macOS and the second
did not. Nothing collided, because one used an `m4/` subdirectory and the
other a content digest, but a reader could not say where either landed
without reading both modules.

That is the kind of thing that is fine until it is not: a test redirecting
the variable to a `tmp_path` redirects both, and one that did so while
touching M4 would have got an empty corpus rather than the cached one. Nothing
did, which was luck.

These tests are the fence. They fail if the two ever derive their location
differently again.
"""
import os
import sys
from pathlib import Path

import pytest

from analytic_prophet import build


@pytest.fixture
def corpora_module():
    """`evaluation/corpora.py`, which pytest puts on the path."""
    import corpora

    return corpora


def test_both_caches_hang_off_the_one_root(tmp_path, monkeypatch, corpora_module):
    """The whole issue in one assertion."""
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path))

    root = build.cache_root()
    assert root == tmp_path
    assert build.cache_dir().parent == root, "the build cache left the root"
    assert corpora_module.m4_cache().parent == root, "the corpus cache left the root"


def test_the_two_do_not_collide(tmp_path, monkeypatch, corpora_module):
    """Separate subdirectories, so neither can be mistaken for the other."""
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path))
    assert build.cache_dir() != corpora_module.m4_cache()


def test_the_default_root_is_the_same_rule_on_every_platform(monkeypatch):
    """The bug was a platform-dependent default on one side only. There is no
    branch on `sys.platform` left to disagree about."""
    monkeypatch.delenv("ANALYTIC_PROPHET_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    assert build.cache_root() == Path.home() / ".cache" / "analytic-prophet"
    assert "darwin" not in Path(build.__file__).read_text().split(
        "def cache_root")[1].split("def cache_dir")[0], (
        "cache_root branches on the platform again")


def test_xdg_cache_home_is_honoured(tmp_path, monkeypatch):
    monkeypatch.delenv("ANALYTIC_PROPHET_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert build.cache_root() == tmp_path / "analytic-prophet"


def test_the_explicit_variable_beats_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path / "explicit"))
    assert build.cache_root() == tmp_path / "explicit"


def test_redirecting_the_variable_moves_the_corpus_too(tmp_path, monkeypatch,
                                                       corpora_module):
    """`M4_CACHE` was a module-level constant bound at import, so a test that
    set the variable afterwards kept using the real cache and would have said
    nothing about it. A function reads the environment when asked."""
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path / "first"))
    first = corpora_module.m4_cache()
    monkeypatch.setenv("ANALYTIC_PROPHET_CACHE", str(tmp_path / "second"))
    assert corpora_module.m4_cache() != first


def test_the_readme_says_where_things_land():
    """A cache nobody can find is a cache nobody can clear."""
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    one_line = " ".join(readme.split())
    assert "ANALYTIC_PROPHET_CACHE" in one_line
    assert "~/.cache/analytic-prophet" in one_line, (
        "the README does not say where the cache defaults to")
