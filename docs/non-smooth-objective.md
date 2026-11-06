# The non-smooth objective

Why Prophet's optimizer stops short of its own posterior mode, what this package does
instead, and where each part is measured. This page is the argument in brief. The paper
makes it in full, with every number (`paper/`, built with `make`). The
[evaluation report](../evaluation/results/report.md) has the forecasts. This page's
earlier, longer version, and the history it carried, are in the
[changelog](../CHANGELOG.md#what-the-non-smooth-objective-page-used-to-say).

[← back to the README](../README.md)

---

### The kink

Prophet's trend changes slope at its changepoints, by rates `δⱼ`, and a Laplace prior on
the rates puts `Σ|δⱼ|/τ` in the objective it optimizes, with `τ` the
`changepoint_prior_scale`. That term is not differentiable where a rate is zero, and zero
is where the prior pulls most rates: the optimum sits on the kinks. On the full Peyton
Manning series, 15 of its 25 rates are exactly zero. Across the 3,008 series of the
census, the median is 11.

### Why a smooth optimizer stops short there

L-BFGS models the objective from its gradients, and Newton's method from its curvature,
and both models assume the objective is smooth. Across `δⱼ = 0` the gradient jumps by
`2/τ` and the curvature is undefined, so the model is wrong exactly where the optimum
is. Rates near zero overshoot it and change sign, and the steps shrink. The run ends
short of the optimum, with no rate exactly zero: by Stan's stopping tests, which measure
progress rather than optimality, or by a line search that can make no more progress.

What was measured, each in the paper under the section named:

- **Prophet's L-BFGS stops short**, by Prophet's own density, at every series length
  measured. The tolerances are not the cause: with all five of Stan's stopping tests at
  machine precision the gap remains ("The optimum, under Prophet's own density").
- **Its point fails the first-order optimality conditions** on every one of the 3,008
  census series, even with each rate read in the way most favourable to it
  ("First-order optimality").
- **The kink is what stops it.** Smoothing the prior inside Prophet's own Stan program,
  with nothing else changed, brings the same L-BFGS 8 to 1,592 times closer to the
  optimum ("Smoothing the prior inside Stan").
- **A second-order method stops short the same way.** This package's own Newton method,
  run on the objective as written, oscillates across the kink and stops 1.2 to 7.1 nats
  short; on the split below, the same code reaches the optimum ("One second-order
  method, two parameterisations").
- **Prophet's own Newton method comes closer, not all the way.** On 200 census series it
  stops a median of 0.0039 nats short, against 5.1 for its L-BFGS, and takes a median of
  12 times as long ("Discussion").

### The fix: split each rate

```
δ = δ⁺ − δ⁻,    δ⁺, δ⁻ ≥ 0
```

The penalty becomes `Σ(δ⁺ⱼ + δ⁻ⱼ)/τ`, which is linear, and the non-smoothness becomes a
bound, which a bound-constrained solver handles natively. This package runs L-BFGS-B,
and a projected Newton method where Prophet's rule runs Newton (below). At an optimum at
most one of each pair is non-zero, so the two problems have the same minimisers
("Exactness"), and a rate can land exactly on zero.

### What it does and does not claim

It claims a better optimum of the model Prophet specifies, by Prophet's own posterior
density, at every series length measured. The forecasts move a little. On 3,008 held-out
M4 series the RMSE is lower on 1,929 of them, by a median of 0.40%: a real improvement,
and usually a small one. The intervals are wider, by a median of 0.56% on 2,153 of the
3,008 series. Both implementations cover far less than the nominal 80%: mean coverage is
0.345 for this package and 0.341 for Prophet. The
[evaluation report](../evaluation/results/report.md#tier-2--does-the-better-map-point-forecast-better)
has every comparison.

It does not claim:

- **A mechanism for the size of the margin.** The margin moves with the series length
  and with the width of the model, and not monotonically in either; only its direction
  holds everywhere it was measured. No mechanism for the *size* of the margin is claimed
  here.
- **That flat growth isolates the kink.** With `growth="flat"` the likelihood never sees
  the trend, and the two implementations land on the same point to Stan's printed
  precision. That shows the same model is being fitted. It does not separate the kink
  from the trend's ill-conditioning, because flat growth removes both ("A negative
  control: removing the trend").

### Prophet's rule for short series, and what it costs

Below 100 observations Prophet runs Stan's Newton method rather than L-BFGS, and retries
with Newton when L-BFGS exits abnormally at any length. This package follows the rule,
with a projected Newton method on the split problem, so a script ported from Prophet runs
the optimizer Prophet's documentation describes. Its Newton method and its L-BFGS-B land
on the same optimum at every length tested (`tests/test_short_series.py`). The rule costs
time, not accuracy: each Newton step pays `2n` gradient evaluations for its Hessian.

### What upstream has said

Two Prophet issues come close to this, and neither reports the kink.

- [prophet#1422](https://github.com/facebook/prophet/issues/1422): a user took the
  cutoff that decides which changepoints a plot marks for the model's choice of
  breakpoints. A contributor corrected it: the cutoff is "an absolute threshold applied
  only for visualization and doesn't affect the model or its predictions", and it occurs
  nowhere but `prophet/plot.py`. The thread says nothing about non-differentiability. It
  shows why such a cutoff exists at all: Prophet's rates never reach zero, so a line has
  to be drawn somewhere, and once drawn it gets read as part of the model.
- [prophet#1280](https://github.com/facebook/prophet/issues/1280): a float32 round-trip
  of `changepoint_prior_scale` changed forecast accuracy. The reporter has "not been able
  to reproduce" the headline case. A maintainer offered "One possibility", that "the
  optimizer was running into issues with one of the changepoint prior scales", and
  suggested the Newton optimizer as "a bit more robust". The verdict was that the
  variation "seems inline with expectation". That is a
  hypothesis offered in a support thread, not a finding, and this page does not upgrade it
  into one.

### Where each part is measured

| question | the paper's section | script |
|---|---|---|
| Does Prophet stop short, by its own density? | The optimum, under Prophet's own density | [`margin.py`](../paper/experiments/margin.py) |
| Is it the stopping tolerances? | The optimum, under Prophet's own density | [`tolerance.py`](../paper/experiments/tolerance.py) |
| Does its point satisfy the optimality conditions? | First-order optimality | [`kkt.py`](../paper/experiments/kkt.py), [`rounding.py`](../paper/experiments/rounding.py) |
| Is it the kink, or L-BFGS? | Smoothing the prior inside Stan | [`smoothing.py`](../paper/experiments/smoothing.py) |
| Does a second-order method stop short too? | One second-order method, two parameterisations | [`newton.py`](../paper/experiments/newton.py) |
| Is the same model being fitted? | A negative control: removing the trend | [`flat.py`](../paper/experiments/flat.py) |
| Do the forecasts move? | Consequences, and Appendix A | [`evaluation/`](../evaluation/), Tier 2 |

---
