"""
Issue #105: the things the first visitor sees.

None of them is interesting alone. Together they are what makes a repository
look like one somebody maintains: bounded dependencies, an install command
that works or says when it does not, a contributing guide describing the
workflow this project actually follows, data that says where it came from,
and no code sitting there doing nothing with an explanation attached.

Two of the issue's seven items were already closed by #103 and are not
retested here: `requires-python` is now exercised rather than implied,
because CI runs 3.9 through 3.14, and the stale "722 tests" counts are gone
from the README, which stopped quoting a number that goes stale every time
somebody adds a test.
"""
import ast
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def pyproject():
    with open(REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


# -- bounds ---------------------------------------------------------------

@pytest.mark.parametrize("package", ["numpy", "pandas", "scipy"])
def test_every_runtime_dependency_carries_a_floor(pyproject, package):
    """Unbounded, every measured claim here was relative to whatever happened
    to be installed beside it."""
    declared = {d.split(">=")[0].split("==")[0]: d
                for d in pyproject["project"]["dependencies"]}
    assert package in declared, f"{package} is not declared"
    assert ">=" in declared[package], f"{package} has no lower bound"


def test_the_comparison_dependency_is_bounded(pyproject):
    """The one that matters: every claim is relative to Prophet, so a Prophet
    release could move the baselines with nothing looking wrong."""
    compare = pyproject["project"]["optional-dependencies"]["compare"]
    assert any(">=" in entry for entry in compare), (
        "prophet is unbounded, which is what makes the evidence base drift")


def test_the_recorded_prophet_version_is_the_one_the_results_used():
    """pyproject names a known-good Prophet; the committed results record what
    actually produced them. If those two ever disagree, one of them is lying
    about the evidence base.
    """
    import json

    text = (REPO / "pyproject.toml").read_text()
    assert "1.4.0" in text, "pyproject no longer records the known-good Prophet"

    results = REPO / "evaluation" / "results"
    recorded = set()
    for meta in results.glob("*.meta.json"):
        with open(meta) as handle:
            version = json.load(handle).get("versions", {}).get("prophet")
        if version:
            recorded.add(version)
    assert recorded == {"1.4.0"}, (
        f"the committed results were produced with {sorted(recorded)}, and "
        f"pyproject records 1.4.0")


# -- the things a visitor reads -------------------------------------------

@pytest.mark.parametrize("path", [
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/defect.md",
    ".github/ISSUE_TEMPLATE/parity.md",
    "tests/data/README.md",
])
def test_the_community_file_exists_and_says_something(path):
    target = REPO / path
    assert target.exists(), f"{path} is missing"
    assert len(target.read_text().split()) > 40, f"{path} is a stub"


def test_contributing_describes_the_workflow_actually_used():
    """The specific parts a contributor cannot infer: branch from main rather
    than from another branch, and run the benchmarks as well as the suite --
    both of which this project learned by getting them wrong."""
    text = " ".join((REPO / "CONTRIBUTING.md").read_text().split()).lower()
    assert "benchmark" in text, "it does not say to run the benchmarks"
    assert "do not merge your own" in text
    assert "from `main`" in text or "from main" in text


def test_the_data_says_where_it_came_from():
    """Every claim here rests on this series, so its provenance belongs beside
    it rather than in somebody's memory."""
    text = (REPO / "tests" / "data" / "README.md").read_text()
    assert "peyton_manning.csv" in text
    assert "MIT" in text, "the licence of the vendored series is not stated"
    assert "M4" in text, "the fetched corpus has no terms recorded"


def test_the_quickstart_carries_the_install_caveat():
    """The warning that `pip install -e .` succeeds and then does not import
    on macOS was two hundred lines below the command it is about."""
    readme = (REPO / "README.md").read_text()
    install = readme.index("pip install -e")
    caveat = readme.index("can succeed without working")
    assert 0 < caveat - install < 500, (
        "the caveat is no longer beside the command it is about")


# -- no code that does nothing --------------------------------------------

def _dead_locals(path):
    tree = ast.parse(Path(path).read_text())
    found = []
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        assigned = {target.id: target.lineno
                    for node in ast.walk(function) if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        used = {node.id for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        found += [(function.name, name, line)
                  for name, line in assigned.items()
                  if name not in used and not name.startswith("_")]
    return found


@pytest.mark.parametrize("module", sorted(
    p.name for p in (REPO / "analytic_prophet").glob("*.py")))
def test_no_local_is_assigned_and_never_read(module):
    """Four of these had accumulated, and the problem was never the lines.

    Three carried `[fc]` comments explaining Prophet's Poisson rate and the
    Laplace guard, left behind when `_sample_trends` was extracted and
    recomputed both -- so there were two explanations beside one copy of the
    code and no way to tell which was authoritative. The fourth was hiding a
    discarded Jacobian in the objective.
    """
    dead = _dead_locals(REPO / "analytic_prophet" / module)
    assert dead == [], f"assigned and never read: {dead}"


def test_the_objective_does_not_compute_a_jacobian_it_discards(monkeypatch,
                                                               peyton_manning_df):
    """The fourth dead local, and the only one that cost anything.

    `_minus_log_posterior` called `logistic_trend_and_jacobian` and dropped
    the Jacobian -- 52% of that call, on every iteration of every logistic
    fit. Counted rather than timed, because a timing test on a 30% difference
    is a flaky test.
    """
    from analytic_prophet import AnalyticProphet, forecaster

    calls = []
    real = forecaster.logistic_trend_and_jacobian
    monkeypatch.setattr(forecaster, "logistic_trend_and_jacobian",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    frame = peyton_manning_df.iloc[:200].reset_index(drop=True)
    frame = frame.assign(cap=frame["y"].max() + 2.0)
    model = AnalyticProphet(growth="logistic")
    model.preprocess(frame)
    model._minus_log_posterior(
        forecaster.from_dict_to_array(model.calculate_initial_params(), model.layout))

    assert calls == [], "the objective still computes a Jacobian it throws away"


def test_the_trend_only_path_is_bit_identical(peyton_manning_df):
    """Splitting the computation must not move the arithmetic: the objective
    is sensitive to the last bits, which is why the linear branch has its own
    expression in the first place."""
    import numpy as np

    from analytic_prophet.trend import logistic_trend, logistic_trend_and_jacobian

    rng = np.random.default_rng(0)
    T, S = 300, 12
    t = np.linspace(0, 1, T)
    changepoints = np.linspace(0, 0.8, S)
    A = (t[:, None] >= changepoints[None, :]).astype(float)
    cap = np.full(T, 2.0)
    delta = rng.normal(0, 0.05, S)

    alone = logistic_trend(0.3, 0.1, delta, t, cap, A, changepoints)
    both, _jacobian = logistic_trend_and_jacobian(0.3, 0.1, delta, t, cap, A,
                                                  changepoints)
    assert np.array_equal(alone, both)
