"""Plumbing shared by every tier of the evaluation suite (#79).

Three things live here and nowhere else, because four tiers landing on four
branches would otherwise each invent their own:

  * **how evaluation/ reaches the rest of the repo.** Tiers run as scripts, not
    under pytest, so they do not get pyproject.toml's `pythonpath`. Everything
    a tier needs from `benchmark/` is re-exported below rather than imported
    from there directly, so the day those helpers move there is one import to
    change instead of four.
  * **the results contract.** A tier emits `Measurement` rows and nothing else.
    They are written as tidy CSV -- one row per measured number -- which keeps
    the schema stable while tiers measure entirely different things, and keeps a
    rerun to a reviewable diff rather than a number someone has to trust.
  * **run metadata.** Seeds, versions, commit, machine. A result without them
    is not reproducible, and the differences between two runs are usually in
    here rather than in the code.
"""
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVALUATION = REPO / "evaluation"
RESULTS = EVALUATION / "results"

# The package and the benchmark helpers, for a tier run as `python evaluation/...`
for path in (REPO, REPO / "benchmark"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _common as _benchmark                         # noqa: E402
import _prophet_bridge as _bridge                    # noqa: E402

# Re-exported so tiers import from one place. `_common` and `_prophet_bridge`
# are the benchmark package's own privates; reaching through this module is
# what keeps that an implementation detail.
build_extension = _benchmark.build_cpp_extension
prophet_available = _benchmark.prophet_available
peak_rss_bytes = _benchmark.peak_rss_bytes
peak_rss_split = _benchmark.peak_rss_split
child_peak_rss_is_reliable = _benchmark.child_peak_rss_is_reliable
human_bytes = _benchmark.human_bytes
PROPHET_KWARGS = _benchmark.PROPHET_KWARGS
PROPHET_INSTALL_HINT = _benchmark.PROPHET_INSTALL_HINT
capture_stan_model = _bridge.capture_stan_model
stan_log_prob = _bridge.stan_log_prob
validate_bridge = _bridge.validate_bridge

# One seed for the whole suite, recorded in the metadata. Tiers derive their own
# from it rather than each picking a number.
SEED = 20260925

# number -> (name, callable, is_gate). The registry lives here rather than in
# run.py because run.py is executed as a script: `python evaluation/run.py`
# makes it `__main__`, and a tier doing `import run` then gets a *second*
# instance of the same file with its own empty registry. Registering into it
# populated the copy nobody read, and the runner reported no tiers at all.
# harness is only ever imported, never run, so there is exactly one of it.
TIERS = {}


def register(number, name, run, gate=False):
    """Add a tier. Called at import time by each evaluation/tiers/tierN.py."""
    TIERS[number] = (name, run, gate)


@dataclass(frozen=True)
class Measurement:
    """One measured number, with enough context to be read on its own.

    Tidy rather than wide: tiers measure different things, and a column per
    metric would mean a schema change every time one is added. `horizon` is
    None except for the per-horizon accuracy metrics of Tier 2.
    """
    tier: int
    series: str
    configuration: str
    implementation: str
    metric: str
    value: float
    unit: str = ""
    horizon: int = None

    def row(self):
        record = asdict(self)
        record["horizon"] = "" if self.horizon is None else self.horizon
        return record


FIELDS = ["tier", "series", "configuration", "implementation", "metric",
          "value", "unit", "horizon"]


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _version(module_name):
    try:
        from importlib.metadata import version
        return version(module_name)
    except Exception:
        return None


def run_metadata(extra=None):
    """What a result has to carry to be reproducible.

    The differences between two runs of the same code are usually in here --
    a numpy release, a different machine, a cmdstan rebuild -- so a results
    file without it records a number nobody can place.
    """
    metadata = {
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "seed": SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "versions": {name: _version(name) for name in
                     ("numpy", "pandas", "scipy", "prophet", "cmdstanpy",
                      "analytic-prophet")},
    }
    metadata.update(extra or {})
    return metadata


def write(tier_name, measurements, metadata=None, results_dir=RESULTS):
    """Write one tier's measurements, and return the paths written.

    Sorted, so that rerunning an unchanged tier produces an unchanged file and
    a diff shows only what actually moved.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted((m.row() for m in measurements),
                  key=lambda r: (r["tier"], r["series"], r["configuration"],
                                 r["implementation"], r["metric"], str(r["horizon"])))
    csv_path = results_dir / f"{tier_name}.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = results_dir / f"{tier_name}.meta.json"
    with open(json_path, "w") as handle:
        json.dump(run_metadata(metadata), handle, indent=1, sort_keys=True)
        handle.write("\n")
    return csv_path, json_path


def read(tier_name, results_dir=RESULTS):
    """A tier's measurements back, for the report."""
    with open(Path(results_dir) / f"{tier_name}.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["tier"] = int(row["tier"])
        row["value"] = float(row["value"])
        row["horizon"] = int(row["horizon"]) if row["horizon"] else None
    return rows
