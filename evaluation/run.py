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
from analytic_prophet.build import cache_root

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


def _options(run, args, number):
    """The arguments this tier accepts, out of the ones the caller gave.

    Tiers are registered as plain callables with whatever signature suits them
    -- Tier 2 takes workers and a checkpoint directory since #164, the others
    take neither -- so the runner offers rather than imposes.

    **Checkpoints are keyed by commit**, which is the part that matters. A
    census of 3008 series is hours of fitting and M4 entire is days, so
    resuming has to be the default or an interrupted run is a wasted one. But
    a checkpoint directory shared across commits would quietly serve results
    fitted by code that has since changed, which is worse than losing them.
    Keying on the commit means editing anything starts a fresh run by itself.
    """
    import inspect

    accepted = inspect.signature(run).parameters
    options = {}
    if "workers" in accepted and args.workers:
        options["workers"] = args.workers
    if "checkpoint" in accepted and not args.no_resume:
        commit = (harness.run_metadata().get("commit") or "uncommitted")[:12]
        options["checkpoint"] = (cache_root() / "checkpoints"
                                 / f"tier{number}" / commit)
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", type=int, nargs="+", default=None,
                        help="which tiers to run (default: all registered)")
    parser.add_argument("--list", action="store_true", help="list registered tiers")
    parser.add_argument("--results", default=str(harness.RESULTS),
                        help="where to write results")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers for tiers that support it "
                             "(default: one per core, less one)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore checkpoints and refit every series")
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
        measurements = list(run(**_options(run, args, number)))
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
