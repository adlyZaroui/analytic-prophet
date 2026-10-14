"""Recompute Tier 2's paired summary from its committed per-series results (#200).

    python evaluation/resummarise.py

The census is not refitted. Its per-series rows are what was measured; the
paired summary -- medians, counts, the sign test -- is arithmetic on them, and
`tiers.tier2.resummarise` rebuilds it with the code a full run uses. Run
`python evaluation/report.py` afterwards to regenerate the report.
"""
import importlib

import harness

tier2 = importlib.import_module("tiers.tier2")

if __name__ == "__main__":
    path = tier2.resummarise()
    print(f"rebuilt the paired summary in {path.relative_to(harness.REPO)}")
