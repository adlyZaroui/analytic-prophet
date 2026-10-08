"""
Issue #164: the evaluation corpus, frozen before the results exist.

Every accuracy claim rested on 36 series drawn with a seed, and a seeded
sample invites exactly one question -- what else was tried. The answer should
not depend on trusting anybody, so the membership is written down first.

**It is a census, not a sample.** Every M4 Weekly and Daily series the protocol
can measure is in it, 3008 of them, and there is no sampling step to
second-guess. The eligibility test is the protocol itself: a series qualifies
when `generate_cutoffs` yields at least one rolling-origin split. That is not
the same question as a length floor and the difference is large -- 4137 Daily
series clear 120 observations and only 2714 of those produce a cutoff, because
a floor counts points where the protocol needs calendar span.

What these tests pin: that the manifest is internally consistent, that editing
it is detectable, that the rule in the file is the rule in the code, that the
series measured are the series frozen, and that a parallel run and a serial one
produce the same numbers -- without which "rerunning produces an unchanged
file" would stop being true the moment the work was distributed.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
MANIFEST = REPO / "evaluation" / "corpus" / "m4_census_v1.json"
TIER2 = REPO / "evaluation" / "results" / "tier2_accuracy.csv"


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


# -- the manifest itself ---------------------------------------------------

def test_the_manifest_is_internally_consistent(manifest):
    assert manifest["total"] == sum(len(v) for v in manifest["series"].values())
    for frequency, identifiers in manifest["series"].items():
        assert manifest["counts"][frequency] == len(identifiers), frequency
        assert len(set(identifiers)) == len(identifiers), f"{frequency} repeats a series"
        assert identifiers == sorted(identifiers), f"{frequency} is not sorted"


def test_the_manifest_declares_a_census_rather_than_a_sample(manifest):
    """The property the whole issue turns on. If a future manifest samples, it
    has to say so here and this test has to be changed deliberately."""
    assert "census" in manifest["rule"].lower()
    assert "nothing is sampled" in manifest["rule"].lower()


def test_editing_the_manifest_is_detectable():
    """A pre-registered corpus that can be edited quietly is not
    pre-registered. The digest is over the membership, canonicalised."""
    import corpora

    good = json.loads(MANIFEST.read_text())
    assert corpora.manifest_digest(good["series"]) == good["digest"]

    tampered = json.loads(MANIFEST.read_text())
    tampered["series"]["Weekly"] = tampered["series"]["Weekly"][:-1]
    path = Path(pytest.ensuretemp("corpus") if hasattr(pytest, "ensuretemp")
                else REPO / "evaluation" / "corpus") / "_tampered.json"
    try:
        path.write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match="edited since it was frozen"):
            corpora.load_manifest(path)
    finally:
        path.unlink(missing_ok=True)


def test_the_digest_depends_on_membership_not_on_order():
    """Otherwise a parallel generator could produce a different digest for the
    same corpus, and the fingerprint would be about scheduling."""
    import corpora

    forward = {"Weekly": ["W1", "W2"], "Daily": ["D1"]}
    shuffled = {"Daily": ["D1"], "Weekly": ["W2", "W1"]}

    assert corpora.manifest_digest(forward) == corpora.manifest_digest(shuffled)
    assert corpora.manifest_digest(forward) != corpora.manifest_digest(
        {"Weekly": ["W1"], "Daily": ["D1"]})


def test_the_protocol_in_the_file_is_the_protocol_in_the_code(manifest):
    """The manifest records the horizons and the floor it was built under. If
    the code's change and the file's do not, the corpus silently describes a
    different experiment from the one that runs."""
    import corpora

    assert manifest["min_length"] == corpora.CENSUS_MIN_LENGTH
    assert manifest["horizons"] == {k: list(v)
                                    for k, v in corpora.CENSUS_HORIZONS.items()}
    assert set(manifest["series"]) == set(corpora.CENSUS_HORIZONS)


def test_the_frozen_corpus_reaches_the_target(manifest):
    """3000 was the objective. Stated as a floor, since the census can only
    grow as frequencies are added on the way to M4 entire."""
    assert manifest["total"] >= 3000, (
        f"the census is {manifest['total']} series, below the 3000 target")


def test_the_generator_can_rederive_what_is_frozen(request):
    """The rule and its output must not drift apart. Runs the verifier, which
    rebuilds the membership from the M4 files and compares.

    Behind `--verify-corpus` because it calls `generate_cutoffs` on every row
    of every frequency and takes about three minutes -- too much for every
    push, and the wrong cadence anyway: the corpus changes when somebody
    freezes it, not when somebody edits the optimizer. The evaluation workflow
    passes the flag.
    """
    if not request.config.getoption("--verify-corpus"):
        pytest.skip("re-deriving the corpus takes ~3 minutes; pass "
                    "--verify-corpus, or run evaluation/freeze_corpus.py")
    pytest.importorskip("prophet")
    import corpora

    for frequency in corpora.CENSUS_HORIZONS:
        if not corpora.m4_available(frequency, download=False):
            pytest.skip(f"M4 {frequency} is not cached")

    completed = subprocess.run(
        [sys.executable, str(REPO / "evaluation" / "freeze_corpus.py")],
        capture_output=True, text=True, cwd=REPO,
        env={**__import__("os").environ,
             "PYTHONPATH": f"{REPO}:{REPO / 'evaluation'}:{REPO / 'benchmark'}"})
    assert completed.returncode == 0, (
        f"the frozen corpus no longer reproduces:\n{completed.stdout}\n{completed.stderr}")


# -- the results are the frozen corpus ------------------------------------

def test_every_measured_series_was_frozen(manifest):
    """The tie between the pre-registration and the numbers. A series in the
    results that is not in the manifest means something outside the frozen
    corpus was measured, which is the thing the manifest exists to rule out."""
    import csv

    if not TIER2.exists():
        pytest.skip("Tier 2 has not been run")
    measured = {row["series"] for row in csv.DictReader(open(TIER2))
                if row["series"] not in ("paired", "corpus")}
    if not measured:
        pytest.skip("no per-series rows")

    frozen = {f"m4_{frequency.lower()}_{identifier}"
              for frequency, identifiers in manifest["series"].items()
              for identifier in identifiers}
    outside = measured - frozen
    assert not outside, (
        f"{len(outside)} measured series are not in the frozen corpus, "
        f"e.g. {sorted(outside)[:5]}")


# -- what makes a distributed run reproducible -----------------------------

def _fake(series, metric, implementation, value):
    from harness import Measurement
    return Measurement(2, series, "Weekly", implementation, metric, value)


def test_paired_differences_do_not_depend_on_arrival_order():
    """The property parallelism threatens.

    Under a process pool the completion order is whatever the scheduler did.
    `_differences` sorts by series name before computing so the file is the
    same whatever that order was -- a median would not notice, but the promise
    that an unchanged rerun produces an unchanged diff would.
    """
    import random
    from tiers import tier2

    rows = []
    for index in range(12):
        rows.append(_fake(f"m4_weekly_W{index}", "mae", "analytic_prophet", index))
        rows.append(_fake(f"m4_weekly_W{index}", "mae", "prophet", index * 2))

    forward = tier2._differences(rows, "mae")
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)

    assert tier2._differences(shuffled, "mae") == forward
    assert forward == sorted(forward, key=lambda _: 0), "order must be series order"
    # and the series order is the sorted one, not the insertion one
    assert tier2._differences(rows, "mae") == [
        -index for index in sorted(range(12), key=lambda i: f"m4_weekly_W{i}")]


def test_the_paired_summary_ignores_the_corpus_bookkeeping_rows():
    """`corpus` and `paired` rows share the CSV with per-series ones. Folding
    either into the differences would quietly corrupt every summary."""
    from tiers import tier2

    rows = [_fake("m4_weekly_W1", "mae", "analytic_prophet", 1.0),
            _fake("m4_weekly_W1", "mae", "prophet", 2.0),
            _fake("corpus", "mae", "both", 999.0),
            _fake("paired", "mae", "difference", 888.0)]

    assert tier2._differences(rows, "mae") == [-1.0]


def test_a_checkpoint_round_trips(tmp_path):
    """A census is hours of fitting and M4 entire is days, so each series is
    durable the moment it finishes."""
    from tiers import tier2

    rows = [_fake("m4_weekly_W1", "mae", "analytic_prophet", 1.5),
            _fake("m4_weekly_W1", "rmse", "prophet", 2.5)]

    tier2._save_checkpoint(tmp_path, "m4_weekly_W1", rows)
    assert tier2._checkpoint_path(tmp_path, "m4_weekly_W1").exists()
    assert tier2._load_checkpoint(tmp_path, "m4_weekly_W1") == rows


def test_the_report_keeps_the_strata_apart():
    """#164 added a per-stratum summary under the same `paired` series name as
    the pooled one. The report keyed only on the metric, so without filtering
    on configuration the two sets collide and whichever the CSV sorted last
    becomes the pooled table."""
    sys.path.insert(0, str(REPO / "evaluation"))
    import report

    rows = [{"series": "paired", "configuration": "all", "metric": "mae_median",
             "value": 1.0},
            {"series": "paired", "configuration": "Weekly", "metric": "mae_median",
             "value": 2.0},
            {"series": "paired", "configuration": "Daily", "metric": "mae_median",
             "value": 3.0}]

    assert report._paired_rows(rows)["mae_median"] == 1.0
    assert report._paired_rows(rows, "Weekly")["mae_median"] == 2.0
    assert report._paired_rows(rows, "Daily")["mae_median"] == 3.0
