"""Fitting wall-clock time: original Prophet vs this project's two paths.

The project's premise is that replacing Stan's autodiff with a closed-form
gradient should not cost speed, so a slower fit is a defect rather than a
tradeoff. This measures that.

    python benchmark/benchmark_fit_time.py
    python benchmark/benchmark_fit_time.py --sizes 300 1000 --repeats 5

What is timed is a full `fit(df)` call on each side, so dataframe setup,
scaling and changepoint placement are inside the measurement for all
implementations. Compiling the C++ extension is not -- it is a build step, not
part of fitting, and is done once before timing starts.

The first fit of each implementation is discarded. It pays one-off costs
(imports, cmdstan model load, first-touch page faults) that say nothing about
steady-state fitting speed.
"""
import argparse
import statistics
import sys
import tempfile
import time
import traceback

import _common as common


def time_one(name, df, lib_path, repeats):
    """Returns (best, median, n_ok) in seconds, or None if unavailable."""
    try:
        common.run_one(name, df, lib_path)          # warmup, discarded
    except Exception as exc:
        print(f"    {name}: unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            common.run_one(name, df, lib_path)
        except Exception:
            traceback.print_exc()
            return None
        samples.append(time.perf_counter() - start)
    return min(samples), statistics.median(samples), len(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(common.DEFAULT_SIZES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip", nargs="*", default=[], help="implementation names to skip")
    args = parser.parse_args()

    have_prophet = common.prophet_available()
    if not have_prophet:
        print(common.PROPHET_INSTALL_HINT, file=sys.stderr)
        print(file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        lib_path = common.build_cpp_extension(tmp)

        names = [n for n in common.IMPLEMENTATIONS if n not in args.skip]
        if not have_prophet:
            names = [n for n in names if n != "prophet"]

        print(f"fitting time, best of {args.repeats} (seconds)")
        print()
        header = f"{'T':>6}  " + "  ".join(f"{n:>19}" for n in names)
        print(header)
        print("-" * len(header))

        results = {}
        for size in args.sizes:
            df = common.load_data(size)
            row = [f"{size:>6}  "]
            for name in names:
                measured = time_one(name, df, lib_path, args.repeats)
                results[(size, name)] = measured
                row.append(f"{measured[0]:>19.3f}" if measured else f"{'n/a':>19}")
            print("  ".join(row))

        if have_prophet:
            print()
            print("ratio to prophet (>1 means slower than the original)")
            print()
            others = [n for n in names if n != "prophet"]
            header = f"{'T':>6}  " + "  ".join(f"{n:>19}" for n in others)
            print(header)
            print("-" * len(header))
            for size in args.sizes:
                base = results.get((size, "prophet"))
                row = [f"{size:>6}  "]
                for name in others:
                    got = results.get((size, name))
                    row.append(f"{got[0] / base[0]:>18.2f}x" if (got and base) else f"{'n/a':>19}")
                print("  ".join(row))
            print()
            print("Note: below T=100 Prophet uses Newton rather than L-BFGS, so those rows")
            print("compare different algorithms -- see issue #25.")


if __name__ == "__main__":
    main()
