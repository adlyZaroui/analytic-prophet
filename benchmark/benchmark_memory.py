"""Peak memory during fitting: original Prophet vs this project's two paths.

The expectation is the reverse of the timing benchmark: a closed-form gradient
should use *less* memory than autodiff, which has to retain a tape of the
forward pass to reverse over. If this project is not cheaper here, the analytic
approach is not paying for itself.

    python benchmark/benchmark_memory.py
    python benchmark/benchmark_memory.py --sizes 300 2905

Method. Each fit runs in a fresh subprocess and reports its own peak RSS, for
two reasons:

  - Peak RSS is a high-water mark that never comes back down within a process,
    so several fits in one process would each inherit the largest previous
    peak and report nonsense.
  - Prophet does the real work in a cmdstan subprocess. Measuring only this
    process would attribute almost none of its memory to it. The child sums
    RUSAGE_SELF with RUSAGE_CHILDREN, which covers cmdstan.

Each implementation is also measured with the fit skipped, giving the cost of
interpreter plus imports. The difference is what fitting itself added, which is
the number worth comparing; the absolute peak is reported alongside because a
library that is expensive to merely import is still expensive to deploy.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import _common as common

CHILD = """
import json, sys
sys.path.insert(0, {bench!r})
import _common as common

name, size, do_fit, lib_path = {name!r}, {size!r}, {do_fit!r}, {lib_path!r}
baseline_error = None
try:
    if name == "prophet":
        import prophet          # noqa: F401
    else:
        import customProphet    # noqa: F401
    if do_fit:
        common.run_one(name, common.load_data(size), lib_path)
except Exception as exc:
    baseline_error = f"{{type(exc).__name__}}: {{exc}}"

print("RESULT " + json.dumps({{"peak": common.peak_rss_bytes(), "error": baseline_error}}))
"""


def measure(name, size, do_fit, lib_path, bench_dir):
    code = CHILD.format(bench=str(bench_dir), name=name, size=size,
                        do_fit=do_fit, lib_path=lib_path)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=1800)
    for line in result.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT "):])
            if payload["error"]:
                return None, payload["error"]
            return payload["peak"], None
    return None, (result.stderr.strip().splitlines() or ["no output"])[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 2905])
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    have_prophet = common.prophet_available()
    if not have_prophet:
        print(common.PROPHET_INSTALL_HINT, file=sys.stderr)
        print(file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        lib_path = common.build_cpp_extension(tmp)
        names = [n for n in common.IMPLEMENTATIONS if n not in args.skip]
        if not have_prophet:
            names = [n for n in names if n != "prophet"]

        print("peak resident memory, one fresh process per measurement")
        print()
        print(f"{'T':>6}  {'implementation':<20} {'import only':>14} {'with fit':>14} {'fit added':>14}")
        print("-" * 74)

        for size in args.sizes:
            for name in names:
                baseline, err = measure(name, size, False, lib_path, bench_dir)
                peak, err2 = measure(name, size, True, lib_path, bench_dir)
                if err or err2:
                    print(f"{size:>6}  {name:<20} {'unavailable: ' + (err or err2)[:40]}")
                    continue
                print(f"{size:>6}  {name:<20} {common.human_bytes(baseline):>14} "
                      f"{common.human_bytes(peak):>14} {common.human_bytes(peak - baseline):>14}")
            print()

        if not have_prophet:
            print("Without prophet installed these are absolute numbers only -- the")
            print("comparison the benchmark exists for needs the original to measure against.")


if __name__ == "__main__":
    main()
