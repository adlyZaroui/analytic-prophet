"""The series the tiers measure on.

A loader returns `(name, DataFrame)` pairs with Prophet's `ds`/`y` columns, so
a tier iterates corpora without knowing where they came from.

**M4 is fetched, not vendored.** The daily file alone is 96 MB, and a corpus in
the repository is a corpus nobody can change their mind about. It is downloaded
once into a user cache and read from there afterwards; a run without network and
without a cache skips the tier that needs it rather than failing.

One limitation, recorded here rather than discovered later: **M4 series are
anonymised and carry only a start date and a frequency.** Holidays and
day-of-week effects -- the features Prophet is specifically built for -- barely
exercise on them. M4 buys breadth and neutrality, not coverage of the model, so
the calendar-real series are reported separately rather than pooled in.
"""
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from harness import REPO

PEYTON_MANNING = REPO / "tests" / "data" / "peyton_manning.csv"

M4_BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
M4_CACHE = Path(os.environ.get("ANALYTIC_PROPHET_CACHE",
                               Path.home() / ".cache" / "analytic-prophet")) / "m4"

# frequency -> (file stem, pandas offset, periods per year). Weekly and Daily
# only: they are the M4 subsets whose spacing a Prophet-shaped model has
# anything to say about, and the monthly file is 48000 series of 12-ish points.
M4_FREQUENCIES = {
    "Weekly": ("Weekly", "W", 52),
    "Daily": ("Daily", "D", 365),
}


def peyton_manning(n_rows=None):
    """The log daily page views for Peyton Manning's Wikipedia entry.

    Prophet's own documentation series, and until Tier 2 lands the *only* one
    this project has ever measured on -- every number in the README rests on
    it, which is the single biggest limitation the evaluation suite exists to
    remove.
    """
    df = pd.read_csv(PEYTON_MANNING)
    return df if n_rows is None else df.iloc[:n_rows].reset_index(drop=True)


def local():
    """Every series vendored in the repo, as (name, frame) pairs."""
    yield "peyton_manning", peyton_manning()


def _download(name, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{M4_BASE}/{name}"
    print(f"    fetching {url}")
    urllib.request.urlretrieve(url, destination)
    return destination


def m4_available(frequency="Weekly", download=True):
    """Whether the files for a frequency are on disk, fetching them if allowed.

    Returns False rather than raising when there is no network and no cache:
    a tier that cannot get its corpus should skip, not fail the whole run.
    """
    stem = M4_FREQUENCIES[frequency][0]
    wanted = [("M4-info.csv", M4_CACHE / "M4-info.csv"),
              (f"Train/{stem}-train.csv", M4_CACHE / f"{stem}-train.csv")]
    for remote, local in wanted:
        if local.exists():
            continue
        if not download:
            return False
        try:
            _download(remote, local)
        except (urllib.error.URLError, OSError) as error:
            print(f"    M4 unavailable ({type(error).__name__}), and nothing cached")
            return False
    return True


def m4(frequency="Weekly", n_series=20, seed=0, min_length=120, download=True):
    """A sample of M4 series for one frequency, as (name, frame) pairs.

    `min_length` keeps out series too short for a rolling-origin evaluation to
    have anything to roll over. The sample is drawn with a fixed seed so a rerun
    measures the same series.
    """
    if not m4_available(frequency, download=download):
        return
    stem, offset, _ = M4_FREQUENCIES[frequency]
    info = pd.read_csv(M4_CACHE / "M4-info.csv").set_index("M4id")
    frame = pd.read_csv(M4_CACHE / f"{stem}-train.csv")

    lengths = frame.iloc[:, 1:].notna().sum(axis=1).to_numpy()
    eligible = np.flatnonzero(lengths >= min_length)
    if eligible.size == 0:
        return
    chosen = np.random.default_rng(seed).permutation(eligible)[:n_series]

    for position in sorted(chosen):
        row = frame.iloc[position]
        identifier = row.iloc[0]
        values = pd.to_numeric(row.iloc[1:], errors="coerce").dropna().to_numpy(dtype=float)
        if identifier not in info.index:
            continue
        start = pd.to_datetime(info.loc[identifier, "StartingDate"],
                               dayfirst=True, errors="coerce")
        if pd.isna(start):
            continue
        dates = pd.date_range(start=start, periods=len(values), freq=offset)
        yield f"m4_{frequency.lower()}_{identifier}", pd.DataFrame(
            {"ds": dates, "y": values})
