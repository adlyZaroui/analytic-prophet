"""Freeze the evaluation corpus, or verify that the frozen one is reproducible.

Issue #164. Every accuracy claim rested on 36 series drawn with a seed, and a
seeded sample invites exactly one question: what else was tried. This writes
the membership down before any result exists, so the question has an answer
that does not depend on trusting anybody.

**The rule is a census, not a sample.** Every M4 Weekly and Daily series that
the protocol can measure is in, which is 3008 of them. There is no sampling
step to second-guess, no `n_series`, and the seed plays no part in choosing --
it seeds the fits' own randomness and nothing else.

A series qualifies when `generate_cutoffs` yields at least one rolling-origin
split under the frozen horizons. That test is the protocol itself rather than
a proxy for it, which matters: a 120-observation floor admits 4137 Daily series
and only 2714 of those produce a cutoff, because a floor counts points where
the protocol needs calendar span. Deciding it at freeze time is what makes the
listed count the measured count.

    python evaluation/freeze_corpus.py            # verify, exit 1 on drift
    python evaluation/freeze_corpus.py --write    # (re)write the manifest

Verifying re-derives the membership from the M4 files and compares it with what
is committed, so the rule and its output cannot drift apart silently. Freezing
is a deliberate act and a reviewable diff.
"""
import argparse
import json
import sys
from datetime import date, timezone, datetime

import corpora
import harness


def build():
    """The manifest, derived from the M4 files under the frozen protocol."""
    series = {frequency: corpora.census_eligible(frequency)
              for frequency in sorted(corpora.CENSUS_HORIZONS)}
    return {
        "protocol": "m4-weekly-daily-census-v1",
        "rule": (
            "Every M4 series of the listed frequencies that has at least "
            f"{corpora.CENSUS_MIN_LENGTH} observations, a parseable "
            "StartingDate, and yields at least one rolling-origin cutoff under "
            "the horizons below. A census: nothing is sampled and no seed takes "
            "part in choosing. The seed in the results is for the fits."),
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "min_length": corpora.CENSUS_MIN_LENGTH,
        "horizons": {frequency: list(values)
                     for frequency, values in corpora.CENSUS_HORIZONS.items()},
        "fit_seed": harness.SEED,
        "counts": {frequency: len(identifiers)
                   for frequency, identifiers in series.items()},
        "total": sum(len(identifiers) for identifiers in series.values()),
        "digest": corpora.manifest_digest(series),
        "series": series,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="write the manifest instead of verifying it")
    args = parser.parse_args(argv)

    for frequency in corpora.CENSUS_HORIZONS:
        if not corpora.m4_available(frequency, download=True):
            print(f"M4 {frequency} is not available and could not be fetched.",
                  file=sys.stderr)
            return 1

    built = build()
    corpora.CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if args.write:
        corpora.MANIFEST.write_text(json.dumps(built, indent=2) + "\n")
        print(f"wrote {corpora.MANIFEST.relative_to(harness.REPO)}")
        for frequency, count in sorted(built["counts"].items()):
            print(f"  {frequency:8} {count:5} series")
        print(f"  {'total':8} {built['total']:5} series")
        print(f"  digest   {built['digest']}")
        return 0

    if not corpora.MANIFEST.exists():
        print("no manifest; run with --write", file=sys.stderr)
        return 1

    frozen = corpora.load_manifest()
    if frozen["series"] != built["series"]:
        print("the frozen corpus is no longer what the rule produces.",
              file=sys.stderr)
        for frequency in sorted(set(frozen["series"]) | set(built["series"])):
            was = set(frozen["series"].get(frequency, ()))
            now = set(built["series"].get(frequency, ()))
            if was != now:
                print(f"  {frequency}: {len(now - was)} added, "
                      f"{len(was - now)} removed", file=sys.stderr)
        return 1

    print(f"corpus reproduces: {frozen['total']} series, "
          f"digest {frozen['digest'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
