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
| 2 | **does the better MAP point forecast better?** | pending |
| 3 | what does it cost to fit and predict? | pending |

Tier 0 is a **gate**, not a measurement. If the two implementations are not
fitting the same specification, every number the others produce is about
something nobody asked; a failing gate stops the run.

Each tier lands as its own PR against `main` — they do not depend on each
other, only on this scaffolding.

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
| `corpora.py` | series loaders; M4 arrives with Tier 2 |
| `run.py` | the runner, tier registry and the gate |

## Tier 0 — the gate

```
tier  metric                       T=300      T=1000     T=2905
0     design_matrix_max_abs_diff   7.3e-12    7.3e-12    7.3e-12
0     prior_scales_max_abs_diff    0          0          0
0     changepoints_max_abs_diff    0          0          0
0     lp__  prophet                813.351    2852.768   8004.798
0     lp__  analytic_prophet       815.337    2855.528   8005.159
0     lp___difference             +1.986     +2.760     +0.361
```

Four checks, in the order of how fundamental they are: the design matrix
element for element, the per-column prior scales, the changepoints, and only
then the posterior. A difference in the first three is a *modelling*
difference, and reporting only the posterior would let one hide inside the
other. The 7.3e-12 is the Fourier basis being evaluated in a different order —
arithmetic, not model.

`beta` is passed straight into Stan's density rather than projected into its
basis, which is only legitimate because the design matrices are identical. That
is what the first check establishes; a projection step would otherwise absorb a
real difference and report agreement.

## Tier 1 — parameter recovery

18 series generated from the model (T ∈ {100, 300, 1000} × noise ∈ {0.05, 0.2} ×
3 seeds), both implementations fitted, both compared to the parameters that made
them.

| | ours wins |
|---|---|
| raw distance `‖θ̂ − θ*‖` | 7 / 18 |
| **identified error** `½(θ̂−θ*)ᵀH(θ*)(θ̂−θ*)` | **14 / 18** |

Median paired difference −0.265 nats, Wilcoxon p = 0.0034.

**The two disagree, and that is the finding.** Raw distance is a coin flip
because it is dominated by the directions the data does not identify — `k`
against `delta`, where being far away costs nothing and means nothing. Weight by
the curvature and a real difference appears.

The ruler is the curvature **at the truth**, not at either fit. An earlier
version expanded around the fitted point, which carries the gradient term and
therefore reports a loss gap: whichever implementation reached the lower
objective is then further from the truth *by construction*. It produced a clean
0/18 that was a restatement of the posterior comparison and nothing else.
`tests/test_tier1_recovery.py` pins the distinction.

A second observation with no obvious reading yet: on these series the truth has
6–11 active changepoints, we find 0–9 and Prophet finds 2–20. Neither is
calibrated — we are consistently too sparse and Prophet consistently too dense.

The identified error is in **nats** and the curvature grows with the sample, so
it compares two implementations on one series and does not pool across series of
different lengths. The tier pairs per series for that reason.

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
