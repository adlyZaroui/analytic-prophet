"""
Issue #180: one command regenerates every number and figure in the paper.

  * **one command** -- `make reproduce` under paper/ runs every experiment
    script and the evaluation tiers the paper quotes, then the numbers and the
    PDF; checked from `make -n`, so a new script cannot be left out of it;
  * **nothing orphaned** -- every committed result and figure is written by a
    script that command runs, and every figure the paper includes exists;
  * **the environment pinned** -- paper/requirements.txt, and every result's
    record of the environment it ran in agrees with it;
  * **the build commit printed** -- written by `make` at each build, and quoted
    by the reproducibility section.

That every number in the generated include matches its source result is
tests/test_paper_numbers.py's job, and it has been since #167.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"
HELPERS = {"quote_numbers", "stan_runner"}   # imported by the others; write no result

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="needs make")


@pytest.fixture(scope="module")
def recipe():
    """What `make reproduce` would run, in order, without running it."""
    out = subprocess.run(["make", "-n", "reproduce"], cwd=PAPER, capture_output=True,
                         text=True, check=True).stdout
    return out.splitlines()


def _scripts():
    return sorted(path.stem for path in EXPERIMENTS.glob("*.py")
                  if path.stem not in HELPERS and "def main(" in path.read_text())


def test_make_reproduce_runs_every_experiment(recipe):
    ran = [m.group(1) for line in recipe for m in [re.search(r"experiments/(\w+)\.py", line)] if m]
    assert set(_scripts()) <= set(ran), f"not run by make reproduce: {set(_scripts()) - set(ran)}"
    assert any("experiments/kkt.py --detail" in line for line in recipe)
    # the numbers come after every experiment, and the PDF after the numbers
    last_experiment = max(i for i, line in enumerate(recipe)
                          if re.search(r"experiments/(?!quote_numbers)\w+\.py", line))
    numbers = max(i for i, line in enumerate(recipe) if "quote_numbers.py" in line)
    assert last_experiment < numbers < max(i for i, line in enumerate(recipe) if "latexmk" in line)


def test_make_reproduce_runs_the_evaluation_tiers_the_paper_quotes(recipe):
    quoted = set(re.findall(r'_rows\("tier(\d)_', (EXPERIMENTS / "quote_numbers.py").read_text()))
    assert quoted, "quote_numbers.py no longer reads an evaluation tier"
    line = next(line for line in recipe if "evaluation/run.py" in line)
    run = set(re.search(r"--tiers ([\d ]+)", line).group(1).split())
    assert quoted <= run, f"the paper quotes tiers {quoted}; make reproduce runs {run}"


def test_every_result_and_figure_is_written_by_a_script_make_runs():
    sources = "\n".join((EXPERIMENTS / f"{name}.py").read_text() for name in _scripts())
    for path in [*(PAPER / "results").glob("*.csv"), *(PAPER / "figures").glob("*.pdf")]:
        assert path.name in sources, f"no script under experiments/ writes {path.name}"


def test_every_figure_the_paper_includes_exists():
    tex = "\n".join(p.read_text() for p in [PAPER / "main.tex", *(PAPER / "sections").glob("*.tex")])
    included = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    assert included
    for name in included:
        assert (PAPER / f"{name}.pdf").exists(), name


def _pins():
    text = (PAPER / "requirements.txt").read_text()
    pins = dict(re.findall(r"^([\w-]+)==(\S+)$", text, flags=re.M))
    python = re.search(r"^#\s+python (\S+)$", text, flags=re.M).group(1)
    return pins, python


def test_every_result_was_produced_in_the_pinned_environment():
    """A result made with another numpy is a different measurement; this is
    where a rerun in a drifted environment would show."""
    pins, python = _pins()
    assert {"prophet", "cmdstanpy", "numpy", "pandas", "scipy"} <= set(pins)
    metas = [*(PAPER / "results").glob("*.meta.json"),
             *(REPO / "evaluation" / "results").glob("*.meta.json")]
    assert len(metas) >= 10
    for path in metas:
        meta = json.loads(path.read_text())
        assert meta["python"] == python, path.name
        for package, version in meta["versions"].items():
            if version is not None and package in pins:
                assert version == pins[package], f"{path.name}: {package} {version}"


def test_cmdstan_is_pinned_where_it_is_enforced():
    text = (PAPER / "requirements.txt").read_text()
    runner = (EXPERIMENTS / "stan_runner.py").read_text()
    version = re.search(r'^CMDSTAN_VERSION = "([^"]+)"$', runner, flags=re.M).group(1)
    assert f"CmdStan {version}" in text


def test_the_reproducibility_section_states_what_is_pinned():
    text = (PAPER / "sections" / "reproducibility.tex").read_text()
    for macro in ("BuildCommit", "CensusDigest", "EnvPython", "EnvProphet", "EnvCmdStan",
                  "EnvCmdstanpy", "EnvNumpy", "EnvPandas", "EnvScipy"):
        assert f"\\{macro}" in text, macro
    assert "make reproduce" in text
    tex = "\n".join(p.read_text() for p in [PAPER / "main.tex", *(PAPER / "sections").glob("*.tex")])
    assert not re.search(r"\\pending\{\d+\}", tex), "a result is still pending"


def test_make_writes_the_commit_the_pdf_is_built_from(tmp_path):
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("needs a git checkout")
    build = PAPER / "generated" / "build.tex"
    saved = build.read_text() if build.exists() else None
    try:
        subprocess.run(["make", "build"], cwd=PAPER, capture_output=True, check=True)
        written = re.search(r"\\newcommand\{\\BuildCommit\}\{([^}]*)\}", build.read_text()).group(1)
    finally:
        if saved is None:
            build.unlink(missing_ok=True)
        else:
            build.write_text(saved)
    head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert re.fullmatch(re.escape(head) + r"(\+modified)?", written), written


def test_the_build_commit_is_never_committed_and_never_missing():
    """Committed, it would name the commit before its own; missing, LaTeX
    would stop. It is ignored, and main.tex falls back without it."""
    ignored = subprocess.run(["git", "check-ignore", "-q", "paper/generated/build.tex"],
                             cwd=REPO).returncode == 0
    assert ignored
    main = (PAPER / "main.tex").read_text()
    assert r"\InputIfFileExists{generated/build.tex}{}{\newcommand{\BuildCommit}{unrecorded}}" in main
