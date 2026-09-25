# Evaluation suite

What `benchmark/` is not. Those three scripts are fast enough to run against a
change; this is the claim-level study from
[#79](https://github.com/adlyZaroui/analytic-prophet/issues/79), and it is
expected to take minutes to hours.

```bash
python evaluation/run.py            # every registered tier
python evaluation/run.py --tiers 0 3
python evaluation/run.py --list
```

## Why it exists

Every measured claim in the README rests on **one series**. Peyton Manning is
behind the +2.34 nat posterior margin, the 0.10–0.68× fit times and the
0.17–0.59% prediction agreement. That was enough to find real bugs and is not
enough to make a general claim.

The README also states outright that better forecasts are **not** claimed, and
the one adjacent measurement points the other way: Prophet fits the training
data marginally better while we score better on the posterior, because we find a
sparser trend. Settling that is the suite's centre of gravity.

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
- **Tier 2** — we forecast better held out (MAE p = 0.0063) with narrower
  intervals at indistinguishable coverage. **Both** implementations under-cover
  badly, which is larger than anything separating them.
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
