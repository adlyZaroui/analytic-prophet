# Evaluation suite

What `benchmark/` is not. Those three scripts are fast enough to run against a
change; this is the claim-level study from
[#79](https://github.com/adlyZaroui/analytic-prophet/issues/79), and it is
expected to take minutes to hours.

```bash
python evaluation/run.py            # every registered tier
python evaluation/run.py --tiers 0 3
python evaluation/run.py --list
python evaluation/run.py --tiers 2 --workers 10     # the census, in parallel
```

## The corpus is pre-registered

Tier 2 measures a **census**, not a sample: every M4 Weekly and Daily series the
protocol can measure, 3008 of them, listed in
[`corpus/m4_census_v1.json`](corpus/m4_census_v1.json) with a digest of its own
membership. There is no `n_series` and no sampling seed, because there is
nothing to sample — which is the point. A seeded sample of 36 invited one
question, what else was tried, and the only answer that does not rest on trust
is a membership fixed before the results exist
([#164](https://github.com/adlyZaroui/analytic-prophet/issues/164)).

A series qualifies when `prophet.diagnostics.generate_cutoffs` yields at least
one rolling-origin split under the frozen horizons. That is the protocol itself
rather than a proxy for it, and the difference is not small: 4137 Daily series
clear the 120-observation floor and only **2714** of those produce a cutoff,
because a floor counts points where the protocol needs calendar span. Deciding
it at freeze time is what makes the listed count the measured count.

```bash
python evaluation/freeze_corpus.py            # verify; exit 1 if it drifted
python evaluation/freeze_corpus.py --write    # re-freeze, as a reviewable diff
pytest tests/test_corpus_census.py --verify-corpus
```

Verifying re-derives the membership from the M4 files and compares it with what
is committed, so the rule and its output cannot drift apart silently. It takes
about three minutes, so `tests.yml` does not run it and `evaluation.yml` does —
before the tiers rather than after, since a run against a manifest that no
longer matches its own rule produces numbers about an unknown set of series.

**Running it.** The work is one series per worker and the series do not
interact, so it parallelises almost linearly. **The full census took 5.6 hours
on eight workers** — about 60 seconds of fitting per series, held steady across
the run. Results are checkpointed per series under the cache, keyed by commit,
so an interrupted run resumes and editing the code starts a fresh one by
itself. `--no-resume` ignores them.

That figure replaces a projection of 45 minutes, which was wrong by a factor of
seven and is worth recording because of how. It was timed on the first series
the loader yields, and the head of the M4 Daily file is short series with
**1.8** rolling-origin cutoffs each where the stratum averages **15.2**. Cost
here scales with cutoffs, not observations, so a few dozen series off the top
of a file are the cheapest corpus there is. Time a random sample.

**On the way to M4 entire.** At 60 seconds a series, 100,000 series is about
1,600 worker-hours — eight days on eight workers, or a day on sixty-four. The
machinery reaches it unchanged; what it needs is the other four frequencies
added to `CENSUS_HORIZONS` with their own protocols, after which the manifest is
re-frozen and the count grows. The per-series cost is dominated by the cutoff
count, so the other lever is the protocol itself: fewer, wider rolling windows
would cut it directly, and is a decision to make deliberately rather than to
discover.

## Why it exists

Every measured claim in the README rests on **one series**. Peyton Manning is
behind the +2.34 nat posterior margin, the 0.10–0.68× fit times and the
0.17–0.59% prediction agreement. That was enough to find real bugs and is not
enough to make a general claim.

Settling whether the better posterior means better *forecasts* was the suite's
centre of gravity, and Tier 2 settled it: it does, held out, on 36 M4 series.

The explanation that used to sit beside that question — that Prophet fits the
training data better while we pay less in the Laplace prior — turned out to be
backwards in both halves, and the metric supporting it was a threshold artifact
([#95](https://github.com/adlyZaroui/analytic-prophet/issues/95)). Sparsity is
now reported as Σ|δ| and as the number of *exact* zeros, neither of which needs
a cutoff to be argued about.

## The tiers

| tier | question | status |
|---|---|---|
| 0 | are we fitting the same model? | **done** |
| 1 | who recovers the true parameters? | **done** |
| 2 | **does the better MAP point forecast better?** | **done** |
| 3 | what does it cost to fit and predict? | **done** |
| — | the report | **done** |

Tier 0 is a **gate**, not a measurement. If the two implementations are not
fitting the same specification, every number the others produce is about
something nobody asked; a failing gate stops the run.

The tiers do not call each other — only this scaffolding. What couples them is
the deliverable: one command, a shared results format, and a report that reads
all four.

## The contract a tier follows

A tier returns `harness.Measurement` rows and nothing else:

```python
Measurement(tier, series, configuration, implementation, metric, value, unit, horizon)
```

Tidy rather than wide, because tiers measure entirely different things and a
column per metric would mean a schema change for each one. Results are written
to `results/<tier>.csv`, **sorted**, so that rerunning an unchanged tier
produces an unchanged file and a rerun is a reviewable diff rather than a number
someone has to trust. `results/<tier>.meta.json` carries the seed, the commit,
library versions and the machine — the differences between two runs of the same
code usually live there.

## What is here now

| | |
|---|---|
| `harness.py` | the results contract, run metadata, and the one place `evaluation/` reaches `benchmark/` |
| `metrics.py` | the metrics more than one tier needs |
| `corpora.py` | series loaders — the vendored series, and M4 fetched on demand |
| `synthetic.py` | generators that run the model forward, with the parameters that made each series |
| `run.py` | the runner, tier registry and the gate |
| `report.py` | tables and figures, from the committed results and nothing else |
| `tiers/` | one module per tier |
| `results/` | the committed CSVs, their metadata, the report and its figures |

## The report

```bash
python evaluation/run.py        # produce results/
python evaluation/report.py     # produce results/report.md + figures/
```

[**results/report.md**](results/report.md) is generated from the committed CSVs
and nothing else — it refits nothing, so regenerating it on another machine
reproduces the same file and two reports diff to show what moved. A tier that
has not been run is named in the report with the command that would produce it,
rather than silently missing.

## The results

**They live in [`results/report.md`](results/report.md)**, generated from the
committed CSVs by `report.py`. They are deliberately not repeated here: two
hand-maintained copies of the same table drift the moment a tier is re-run, and
the generated one is the copy that cannot.

The headlines, with the full treatment one link away:

- **Tier 0** — the gate passes. Design matrices agree to 7.3e-12, changepoints
  and prior scales exactly, and our `lp__` is ahead at every size.
- **Tier 1** — weighted by curvature we recover the truth better on 14 of 18
  synthetic series (p = 0.0034), where raw distance is a coin flip at 7 of 18.
- **Tier 2** — we forecast better held out (MAE p = 0.0063), with higher
  coverage (p = 0.0025) at indistinguishable interval width. **Both**
  implementations under-cover badly, which is larger than anything separating
  them.
- **Tier 3** — fitting is faster and the fit's memory is about a third of
  Prophet's. Prediction is measured on **four** paths, because
  `predict(vectorized=True)` is the default on both sides and is an
  approximation rather than a faster form of the exact sampler (#93). We are
  faster on both diagonals: 1.8× approximate, 2.6× exact.

---

## The two metrics worth explaining

### `metrics.quadratic_change`

The metric that makes a parameter-space distance mean something. Raw distance
between two fits is dominated by directions the data does not identify — `k`
against `delta`, which trade off almost freely — so two fits can be far apart
and be the same function. Weighted by the curvature,

```
f(other) − f(theta)  ≈  g·d + ½ d'Hd + (the Laplace term, exactly)
```

the distance *is* the loss gap. It turns out to be near-exact rather than merely
second order: with a Gaussian likelihood, Gaussian priors and a model linear in
its parameters the objective is quadratic in these coordinates, and agreement is
~1e-10 relative even where the gap is thousands of nats.

### `metrics.curvature_spectrum`

The flat directions as a number. On 300 days of Peyton Manning, linear growth
spans 8.1e5 down to 5.4e-2 — a condition number of 1.5e7. Under **flat growth
the trend block is exactly singular**: the likelihood never sees `k` or `delta`,
and the Laplace prior holding `delta` contributes no curvature at all. That is
the README's "flat growth removes those directions" measured rather than
described, and it is why flat growth ties with Prophet exactly while linear
growth does not.
