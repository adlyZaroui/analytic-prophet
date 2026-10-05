# Data

## `peyton_manning.csv`

The daily log page-views of the English Wikipedia article for **Peyton Manning**, from
2007-12-10 to 2016-01-20 — 2905 rows, columns `ds` and `y`.

It is **Prophet's own example series**, the one its quickstart and documentation use,
vendored here from the Prophet repository (`examples/example_wp_log_peyton_manning.csv`).
Prophet is MIT licensed, and the file carries that licence with it.

It is here rather than fetched because almost every claim in this repository rests on it:
the posterior comparison, the convergence tolerances, the agreement benchmark, and most of
the test suite. A measurement whose input can change under it is not reproducible, and a
test suite that needs the network is not a test suite.

The series is log-transformed page views, which is why it is roughly 8–12 rather than in
the hundreds of thousands. Prophet's documentation uses it to demonstrate holiday effects
(Manning's playoff and Super Bowl appearances are the spikes) and a trend with changepoints.

## The M4 corpus

Not vendored. `evaluation/corpora.py` downloads the Weekly and Daily subsets on demand from
the [M4-methods](https://github.com/Mcompetitions/M4-methods) repository and caches them
under `~/.cache/analytic-prophet/m4` (or `$ANALYTIC_PROPHET_CACHE`).

The M4 competition dataset is published by the M Open Forecasting Center for research use.
It is fetched rather than committed because it is large, because only the evaluation suite
needs it, and because the tests that use it skip cleanly when it is absent — which is what
keeps `pytest` working with no network.

Tier 2's committed results name the 36 series they used, so the selection is reproducible
even when the corpus is not at hand.
