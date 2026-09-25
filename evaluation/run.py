"""Run the evaluation suite (#79).

    python evaluation/run.py                  # every registered tier
    python evaluation/run.py --tiers 0 3      # a subset
    python evaluation/run.py --list           # what is registered

Tier 0 is a gate rather than a measurement: if the two implementations are not
fitting the same model, every number the other tiers produce is about something
nobody asked. A failing gate stops the run.

Tiers register themselves in TIERS below as they land. The runner deliberately
does nothing clever with an empty registry -- until a tier exists, this reports
that there is nothing to run.
"""
import argparse
import importlib
import importlib.util
import sys
import time

import harness

# A tier's callable takes no arguments and returns an iterable of
# harness.Measurement. The registry itself lives in harness -- see the note
# there for why it cannot live in this file.
TIERS = harness.TIERS


def _load_tiers():
    """Import whichever tier modules exist. They are added over separate PRs,
    so a missing one is the normal state.

    A tier that exists but cannot be imported is *not* normal, and is raised
    rather than skipped: swallowing it reports "no tiers registered" for what is
    actually a broken import, which is a long way to debug from.
    """
    for number in range(4):
        name = f"tiers.tier{number}"
        if importlib.util.find_spec(name) is None:
            continue
        importlib.import_module(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", type=int, nargs="+", default=None,
                        help="which tiers to run (default: all registered)")
    parser.add_argument("--list", action="store_true", help="list registered tiers")
    parser.add_argument("--results", default=str(harness.RESULTS),
                        help="where to write results")
    args = parser.parse_args(argv)

    _load_tiers()

    if args.list or not TIERS:
        if not TIERS:
            print("no tiers registered yet -- each lands as its own PR (#79)")
            return 0
        for number in sorted(TIERS):
            name, _, gate = TIERS[number]
            print(f"  tier {number}  {name}{'  [gate]' if gate else ''}")
        return 0

    if not harness.prophet_available():
        print(harness.PROPHET_INSTALL_HINT, file=sys.stderr)
        return 1

    selected = sorted(TIERS if args.tiers is None else
                      {n: TIERS[n] for n in args.tiers if n in TIERS})
    for number in selected:
        name, run, gate = TIERS[number]
        print(f"tier {number}: {name}")
        started = time.perf_counter()
        measurements = list(run())
        elapsed = time.perf_counter() - started
        csv_path, _ = harness.write(f"tier{number}_{name}", measurements,
                                    {"tier": number, "seconds": round(elapsed, 3)},
                                    results_dir=args.results)
        print(f"  {len(measurements)} measurements in {elapsed:.1f}s -> {csv_path.name}")

        if gate and any(m.metric == "gate_failed" and m.value for m in measurements):
            print("  gate failed: the two implementations are not fitting the "
                  "same model, so the remaining tiers would measure nothing. "
                  "Stopping.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
