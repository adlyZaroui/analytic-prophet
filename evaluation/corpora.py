"""The series the tiers measure on.

The M4 competition dataset is published by the M Open Forecasting Center for
research use, and is downloaded from the M4-methods repository rather than
vendored: it is large, only this suite needs it, and the tests that use it
skip cleanly when it is absent. `tests/data/README.md` records where both
corpora come from and under what terms (#105).

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

from analytic_prophet.build import cache_root
from harness import REPO

PEYTON_MANNING = REPO / "tests" / "data" / "peyton_manning.csv"

M4_BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
def m4_cache():
    """Where the M4 files are kept: `<cache root>/m4`.

    A function rather than the module-level constant this was, so that
    redirecting `ANALYTIC_PROPHET_CACHE` moves it -- as a constant it was
    bound at import and a test that set the variable afterwards silently kept
    using the real cache. The root is the package's, so the corpus and the
    compiled core land under one directory by one rule (#111).
    """
    return cache_root() / "m4"

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
    wanted = [("M4-info.csv", m4_cache() / "M4-info.csv"),
              (f"Train/{stem}-train.csv", m4_cache() / f"{stem}-train.csv")]
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
    info = pd.read_csv(m4_cache() / "M4-info.csv").set_index("M4id")
    frame = pd.read_csv(m4_cache() / f"{stem}-train.csv")

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


# -- the frozen corpus (#164) ----------------------------------------------

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MANIFEST = CORPUS_DIR / "m4_census_v1.json"

# The protocol the manifest was frozen against. Changing any of these changes
# which series qualify, so a manifest built under different values is a
# different corpus and the digest will say so.
CENSUS_MIN_LENGTH = 120
CENSUS_HORIZONS = {
    "Weekly": ("52 W", "104 W", "52 W"),
    "Daily": ("90 D", "730 D", "180 D"),
}


def _m4_rows(frequency):
    """`(identifier, dates, values)` for every row of one M4 frequency file.

    The two filters the loader has always applied are here rather than at the
    call site, because the manifest has to list series that actually load:
    a length floor, and a `StartingDate` pandas can parse.
    """
    stem, offset, _ = M4_FREQUENCIES[frequency]
    info = pd.read_csv(m4_cache() / "M4-info.csv").set_index("M4id")
    frame = pd.read_csv(m4_cache() / f"{stem}-train.csv")
    lengths = frame.iloc[:, 1:].notna().sum(axis=1).to_numpy()

    for position in range(len(frame)):
        if lengths[position] < CENSUS_MIN_LENGTH:
            continue
        identifier = frame.iloc[position, 0]
        if identifier not in info.index:
            continue
        start = pd.to_datetime(info.loc[identifier, "StartingDate"],
                               dayfirst=True, errors="coerce")
        if pd.isna(start):
            continue
        values = pd.to_numeric(frame.iloc[position, 1:],
                               errors="coerce").dropna().to_numpy(dtype=float)
        yield identifier, pd.date_range(start=start, periods=len(values),
                                        freq=offset), values


def census_eligible(frequency):
    """Every identifier of one frequency that the protocol can measure, sorted.

    **The eligibility test is the protocol itself**, not a proxy for it: a
    series qualifies when `prophet.diagnostics.generate_cutoffs` yields at
    least one rolling-origin split for it. A length floor is not the same
    question and gets it badly wrong -- 4137 Daily series clear 120
    observations and only 2714 of those produce a cutoff, because the floor
    counts points where the protocol needs calendar span.

    Deciding it here rather than at run time is what lets the corpus be a
    census: the manifest can list exactly the series that will return results,
    so "3008 series" is the number measured rather than the number attempted.
    """
    from prophet.diagnostics import generate_cutoffs

    horizon, initial, period = (pd.Timedelta(text)
                                for text in CENSUS_HORIZONS[frequency])
    eligible = []
    for identifier, dates, values in _m4_rows(frequency):
        frame = pd.DataFrame({"ds": dates, "y": values})
        try:
            cutoffs = generate_cutoffs(frame, horizon, initial, period)
        except ValueError:
            continue                      # less span than the horizon needs
        if cutoffs:
            eligible.append(identifier)
    return sorted(eligible)


def manifest_digest(series_by_frequency):
    """A SHA-256 over the selected identifiers, as the manifest's fingerprint.

    So that an edited list is detectable rather than merely discouraged. The
    input is canonicalised -- frequencies sorted, identifiers sorted within
    each -- so the digest depends on the membership and not on the order a
    particular run happened to produce.
    """
    import hashlib

    canonical = "\n".join(
        f"{frequency}:{identifier}"
        for frequency in sorted(series_by_frequency)
        for identifier in sorted(series_by_frequency[frequency]))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_manifest(path=None):
    """The frozen corpus, as a dict. Raises if it has been edited."""
    import json

    manifest = json.loads(Path(path or MANIFEST).read_text())
    recomputed = manifest_digest(manifest["series"])
    if recomputed != manifest["digest"]:
        raise ValueError(
            f"{Path(path or MANIFEST).name} has been edited since it was frozen: "
            f"its digest says {manifest['digest'][:12]}, its contents hash to "
            f"{recomputed[:12]}. The corpus is pre-registered; regenerate it "
            "with `python evaluation/freeze_corpus.py` and say in the commit "
            "why the membership changed.")
    return manifest


def m4_census(frequency, manifest=None, download=True):
    """The manifest's series for one frequency, as `(name, frame)` pairs.

    Yields in the manifest's order, which is sorted, so a run is reproducible
    and a parallel run can be reassembled into the same file.
    """
    if not m4_available(frequency, download=download):
        return
    manifest = manifest or load_manifest()
    wanted = set(manifest["series"].get(frequency, ()))
    if not wanted:
        return
    for identifier, dates, values in _m4_rows(frequency):
        if identifier in wanted:
            yield (f"m4_{frequency.lower()}_{identifier}",
                   pd.DataFrame({"ds": dates, "y": values}))
