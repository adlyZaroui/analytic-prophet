"""The series the tiers measure on.

A loader returns `(name, DataFrame)` pairs with Prophet's `ds`/`y` columns, so
a tier iterates corpora without knowing where they came from. Only the series
already in the repo is here; the M4 fetcher lands with Tier 2, which is what
needs breadth (#79).
"""
import pandas as pd

from harness import REPO

PEYTON_MANNING = REPO / "tests" / "data" / "peyton_manning.csv"


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
