"""What M4's calendar can and cannot support (#178).

Prophet is built around holidays and day-of-week effects, and the paper's
limitations say M4 cannot test them. This records why, from M4's own info file
rather than from memory: per frequency, how many series M4 has, what metadata
it carries (a domain category, no location), and how the census's starting
dates are distributed -- including how many share a single timestamp, and how
many the census loader reads as starting after the year it ran.

The last is a property of this project's loader, not of M4: `StartingDate` has
two-digit years, and pandas resolves them with a window that moves with the
current year. Both implementations see the same dates, so the comparison is
unaffected, but the dates themselves are not evidence of anything.

    python paper/experiments/m4_calendar.py     # needs the M4 cache

Writes `paper/results/m4_calendar.csv`.
"""
import csv
import sys
import warnings
from datetime import date
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
for path in (REPO, REPO / "evaluation", REPO / "benchmark"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import corpora                                      # noqa: E402

RESULTS = PAPER / "results" / "m4_calendar.csv"
FIELDS = ["frequency", "m4_series", "census_series", "metadata_columns",
          "distinct_starts", "modal_start", "modal_count", "starts_after_run_year",
          "run_year"]


def collect():
    import pandas as pd

    info = pd.read_csv(corpora.m4_cache() / "M4-info.csv")
    manifest = corpora.load_manifest()
    run_year = date.today().year
    rows = []
    for frequency, group in info.groupby("SP"):
        identifiers = manifest["series"].get(frequency, [])
        starts = info.set_index("M4id").loc[identifiers, "StartingDate"] if identifiers else pd.Series([], dtype=str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(starts, dayfirst=True, errors="coerce")
        counts = starts.value_counts()
        rows.append({
            "frequency": frequency,
            "m4_series": len(group),
            "census_series": len(identifiers),
            "metadata_columns": "+".join(info.columns),
            "distinct_starts": int(starts.nunique()),
            "modal_start": counts.index[0] if len(counts) else "",
            "modal_count": int(counts.iloc[0]) if len(counts) else 0,
            "starts_after_run_year": int((parsed.dt.year > run_year).sum()),
            "run_year": run_year,
        })
    return sorted(rows, key=lambda row: row["frequency"])


def main():
    rows = collect()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {RESULTS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
