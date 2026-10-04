"""
Issue #103: the workflows, checked rather than trusted.

A CI configuration is code that runs nowhere else, so nothing would notice it
rotting. The failure mode the issue is about is specific and quiet: a runner
without Eigen and LBFGSpp reports a **green** suite that silently skipped the
compiled path, which is worse than a red one because it looks like evidence.
Dropping `--require-cpp` from a job, or adding a job that forgets it, would
bring that back and no test would fail. Hence this file.

It also pins the two things that have to agree with something else in the
repo: the matrix floor against `requires-python`, and the set of
`--require-*` options against the ones conftest.py actually registers.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import conftest

REPO = Path(__file__).parent.parent
WORKFLOWS = REPO / ".github" / "workflows"


def _load(name):
    """A workflow as a dict, with YAML 1.1's `on` -> True quirk undone."""
    with open(WORKFLOWS / name) as handle:
        loaded = yaml.safe_load(handle)
    if True in loaded:                      # `on:` is a YAML 1.1 boolean
        loaded["on"] = loaded.pop(True)
    return loaded


# a line that *invokes* pytest, as against one that merely names pytest.out
INVOKES_PYTEST = re.compile(r"^\s*pytest\s", re.MULTILINE)


def _all_shell(name):
    """Every `run:` block in a workflow, as one string."""
    return " ".join(step.get("run", "")
                    for job in _load(name)["jobs"].values()
                    for step in job["steps"])


def _pytest_steps(workflow):
    """(job name, the shell of every step that invokes pytest)."""
    for job_name, job in workflow["jobs"].items():
        for step in job["steps"]:
            run = step.get("run", "")
            if INVOKES_PYTEST.search(run):
                yield job_name, run


def test_there_is_a_workflow_at_all():
    """The whole of #103 in one assertion: a suite that nothing ran."""
    assert WORKFLOWS.is_dir()
    assert (WORKFLOWS / "tests.yml").exists()
    assert (WORKFLOWS / "evaluation.yml").exists()


def test_the_suite_runs_on_pushes_and_pull_requests():
    triggers = _load("tests.yml")["on"]
    assert "pull_request" in triggers
    assert "push" in triggers


def test_every_job_that_runs_pytest_requires_the_cpp_toolchain():
    """The point of the issue.

    Each job installs Eigen and LBFGSpp on purpose, so a toolchain skip there
    is a broken runner rather than a contributor without a compiler --
    `--require-cpp` is what turns it into a failure. A job without it would
    report green for a run that never built the C++ core.
    """
    steps = list(_pytest_steps(_load("tests.yml")))
    assert steps, "no job in tests.yml runs pytest"
    for job_name, run in steps:
        assert "--require-cpp" in run, f"job {job_name!r} runs pytest without --require-cpp"


def test_one_job_installs_prophet_and_requires_it():
    """Prophet is the single largest cost in a run, so only one job pays it --
    and that job has to assert the comparisons actually ran."""
    jobs = _load("tests.yml")["jobs"]
    requiring = [name for name, job in jobs.items()
                 if any("--require-prophet" in step.get("run", "")
                        for step in job["steps"])]
    assert len(requiring) == 1, f"expected exactly one prophet job, found {requiring}"

    job = jobs[requiring[0]]
    installs = " ".join(step.get("run", "") for step in job["steps"])
    # [dev] is [test,compare] and compare is prophet; requiring it without
    # installing it would fail every run of that job.
    assert "[dev]" in installs


def test_the_toolchain_is_installed_wherever_it_is_required():
    for job_name, job in _load("tests.yml")["jobs"].items():
        shell = " ".join(step.get("run", "") for step in job["steps"])
        if "--require-cpp" not in shell:   # nothing to install for
            continue
        assert "libeigen3-dev" in shell, f"job {job_name!r} requires Eigen without installing it"
        assert "LBFGSpp" in shell, f"job {job_name!r} requires LBFGSpp without fetching it"
        assert any("LBFGSPP_INCLUDE_DIR" in str(step.get("env", {}))
                   for step in job["steps"]), \
            f"job {job_name!r} fetches LBFGSpp without telling conftest where it is"


def test_the_matrix_floor_is_the_version_pyproject_claims(pyproject_text):
    """`requires-python = ">=3.9"` had only ever been run on one version.

    If the floor is raised or lowered, the matrix has to move with it --
    otherwise the claim goes back to being unevidenced.
    """
    floor = pyproject_text.split('requires-python = ">=')[1].split('"')[0]
    versions = _load("tests.yml")["jobs"]["suite"]["strategy"]["matrix"]["python"]
    as_tuples = sorted(tuple(int(part) for part in v.split(".")) for v in versions)
    assert as_tuples[0] == tuple(int(part) for part in floor.split("."))
    assert len(as_tuples) == len(set(as_tuples))


def test_the_evaluation_tiers_are_not_in_the_push_workflow():
    """Tier 2 alone is ~11 minutes and needs the network. The issue is
    explicit that it belongs in its own workflow."""
    assert "evaluation/run.py" not in _all_shell("tests.yml")


def test_the_evaluation_workflow_is_manual_and_keeps_its_output():
    """Manual or scheduled, never per-push -- and it uploads the regenerated
    results rather than committing them, since refreshing the committed
    numbers is a decision someone makes after reading the diff."""
    workflow = _load("evaluation.yml")
    assert "workflow_dispatch" in workflow["on"]
    assert "push" not in workflow["on"]
    assert "pull_request" not in workflow["on"]

    steps = workflow["jobs"]["evaluate"]["steps"]
    shell = " ".join(step.get("run", "") for step in steps)
    assert "evaluation/run.py" in shell       # tier 0 gates it by exit code
    assert "evaluation/report.py" in shell
    assert any("upload-artifact" in str(step.get("uses", "")) for step in steps)
    assert "git commit" not in shell and "git push" not in shell


@pytest.mark.parametrize("name", ["tests.yml", "evaluation.yml"])
def test_no_workflow_interpolates_into_a_shell(name):
    """`${{ }}` is expanded before bash sees the script, so anything
    attacker-controllable spliced in there is a command-injection seam. Values
    reach the shell through `env:` instead.

    The one input here is a dispatch field, which needs write access to set --
    low risk, and still not a habit worth having in a public repository.
    """
    for job_name, job in _load(name)["jobs"].items():
        for step in job["steps"]:
            assert "${{" not in step.get("run", ""), \
                f"{name}: step {step.get('name', '?')!r} in {job_name!r} interpolates into its shell"


# ---------------------------------------------------------------------------
# the mechanism the workflows lean on


class _Config:
    """Just enough of pytest's config to answer getoption."""

    def __init__(self, **flags):
        self._flags = flags

    def getoption(self, name):
        return self._flags.get(name, False)


def test_an_environment_gap_skips_by_default():
    """A contributor without a compiler should get a skip, not a red suite:
    it is an environment gap, not a code defect."""
    with pytest.raises(pytest.skip.Exception) as raised:
        conftest.environment_gap(_Config(), "cpp", "no compiler")
    assert conftest.ENVIRONMENT_PREFIX in str(raised.value)


@pytest.mark.parametrize("requirement", sorted(conftest.REQUIREMENTS))
def test_the_flag_turns_the_gap_into_a_failure(requirement):
    """And where the environment was built on purpose, the same gap is a
    failure -- otherwise CI reports green for a run that skipped the half
    that matters."""
    option = conftest.REQUIREMENTS[requirement][0]
    with pytest.raises(pytest.fail.Exception) as raised:
        conftest.environment_gap(_Config(**{option: True}), requirement, "missing")
    assert option in str(raised.value)
    assert "missing" in str(raised.value)


def test_the_options_the_workflows_pass_are_the_ones_conftest_registers():
    """The two sides of #103's requirement, pinned together: a typo in the
    workflow would otherwise be silently ignored by pytest's argument
    parser... which is not true, but a *renamed* option here would be, and
    this is what catches it."""
    registered = {option for option, _ in conftest.REQUIREMENTS.values()}
    passed = set()
    for _, run in _pytest_steps(_load("tests.yml")):
        passed.update(word for word in run.split() if word.startswith("--require-"))
    assert passed <= registered, f"workflow passes unregistered options: {passed - registered}"
    assert passed == registered, f"registered but never exercised in CI: {registered - passed}"


def test_every_environment_skip_in_conftest_goes_through_the_gap():
    """A new `pytest.skip` for a missing dependency would not be reported as
    an environment gap and `--require-*` would not catch it, so the next
    silently-green CI run would look exactly like this issue.

    The test fixtures' own skips are fine -- this is about conftest, where the
    shared dependencies are found.
    """
    source = (Path(conftest.__file__)).read_text()
    offenders = [line.strip() for line in source.splitlines()
                 if "pytest.skip(" in line
                 # the one the helper itself makes, which is the point of it
                 and "ENVIRONMENT_PREFIX" not in line]
    assert offenders == [], (
        "these skips bypass environment_gap, so --require-* cannot see them: "
        f"{offenders}")


def test_the_skip_report_names_every_reason(tmp_path):
    """The run has to say what it skipped, grouped, or "green" cannot be read.

    Checked by running a real pytest, because the hook that prints it is
    pytest's to call -- but on a test file written here rather than one of the
    repo's own. The first version pointed at tests/test_packaging.py and
    asserted its skips, which it has on this machine because the editable
    install is broken on macOS and does *not* have on Linux, where the install
    works: the test failed in all seven CI jobs for the thing it was supposed
    to be indifferent to (#103).

    conftest.py comes in as a plugin because the generated file lives outside
    tests/, so it is loaded once rather than twice.
    """
    generated = tmp_path / "test_generated_skip.py"
    generated.write_text("import pytest\n\n\n"
                         "def test_skips_for_a_reason_of_its_own():\n"
                         "    pytest.skip('a reason of its own')\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(generated), "-q",
         "-p", "conftest", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
        env=dict(os.environ, PYTHONPATH=str(REPO / "tests")))
    assert "skipped, by reason" in result.stdout, result.stdout[-2000:]
    assert "a reason of its own" in result.stdout, result.stdout[-2000:]


@pytest.fixture(scope="module")
def pyproject_text():
    return (REPO / "pyproject.toml").read_text()
