"""How much of the speed difference is the cost of calling CmdStan (#178).

The README attributes the speed advantage at small T to Prophet's fixed cost of
launching a CmdStan subprocess. Tier 3 records only whole fits, so that
attribution had no measurement behind it. This supplies one.

At each length it times, on Peyton Manning's first T rows in Prophet's default
configuration:

    prophet_fit           Prophet().fit, end to end
    prophet_optimize      the CmdStanModel.optimize call inside it
    one_iteration_call    the same call with iter=1 -- launching the process,
                          writing the data, one iteration, reading the result:
                          the part of a fit that does not depend on how many
                          iterations the optimizer needs
    ours_fit              AnalyticProphet().fit on the compiled path

Each is the median of REPEATS runs after one warm-up. Wall-clock times on one
machine are not portable and the metadata records which machine; what the paper
quotes are ratios, which travel better.

    python paper/experiments/overhead.py

Writes `paper/results/overhead.csv` and its `.meta.json`.
"""
import csv
import json
import logging
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kkt                                          # noqa: E402,F401  (sets up paths)
import corpora                                      # noqa: E402
import harness                                      # noqa: E402

RESULTS = PAPER / "results" / "overhead.csv"
LENGTHS = (50, 100, 300, 1000, 2905)
REPEATS = 15
NEWTON_BELOW = 100
FIELDS = ["observations", "prophet_algorithm", "repeats", "prophet_fit",
          "prophet_optimize", "one_iteration_call", "ours_fit"]


def measure(size, lib_path, repeats=REPEATS):
    """Medians of the four timings at one length."""
    from cmdstanpy import CmdStanModel
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    df = corpora.peyton_manning(size)
    original = CmdStanModel.optimize
    inside = {}

    def timed(self, **kwargs):
        started = time.perf_counter()
        result = original(self, **kwargs)
        inside["seconds"] = time.perf_counter() - started
        return result

    samples = {"prophet_fit": [], "prophet_optimize": [], "one_iteration_call": [],
               "ours_fit": []}
    CmdStanModel.optimize = timed
    try:
        for attempt in range(repeats + 1):        # the first is a warm-up
            started = time.perf_counter()
            Prophet(**harness.PROPHET_KWARGS).fit(df)
            fit, optimize = time.perf_counter() - started, inside["seconds"]
            Prophet(**harness.PROPHET_KWARGS).fit(df, iter=1)
            one_iteration = inside["seconds"]
            started = time.perf_counter()
            AnalyticProphet(**harness.PROPHET_KWARGS).fit(df, lib_path=lib_path)
            ours = time.perf_counter() - started
            if attempt:
                samples["prophet_fit"].append(fit)
                samples["prophet_optimize"].append(optimize)
                samples["one_iteration_call"].append(one_iteration)
                samples["ours_fit"].append(ours)
    finally:
        CmdStanModel.optimize = original
    row = {"observations": size,
           "prophet_algorithm": "Newton" if size < NEWTON_BELOW else "LBFGS",
           "repeats": repeats}
    row.update({key: statistics.median(values) for key, values in samples.items()})
    return row


def read(path=RESULTS):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in FIELDS[3:]:
            row[key] = float(row[key])
        row["observations"] = int(row["observations"])
    return rows


def main():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    warnings.simplefilter("ignore")
    lib_path = harness.build_extension(tempfile.mkdtemp())
    rows = [measure(size, lib_path) for size in LENGTHS]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(value) if isinstance(value, float) else value)
                             for key, value in row.items()})
    with open(RESULTS.with_suffix(".meta.json"), "w") as handle:
        json.dump(harness.run_metadata({"lengths": list(LENGTHS), "repeats": REPEATS}),
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    for row in rows:
        print(f"T={row['observations']:5d} {row['prophet_algorithm']:6s} "
              f"prophet {row['prophet_fit']:.3f}s  one-iteration call "
              f"{row['one_iteration_call']:.3f}s ({row['one_iteration_call'] / row['prophet_fit']:.0%})"
              f"  ours {row['ours_fit']:.3f}s")
    print(f"wrote {RESULTS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
