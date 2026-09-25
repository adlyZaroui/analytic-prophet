# The non-smooth objective

The project's central technical argument: Prophet's posterior is not
differentiable exactly where its optimum sits, that costs it real accuracy, and
a reformulation removes the problem rather than working around it.

[← back to the README](../README.md)

---

The Laplace prior on `delta` puts `Σ|δⱼ|/τ` in the objective. That makes the posterior
**non-differentiable at `δ = 0`** — and the optimum sits exactly on those kinks, because
the prior is what drives most changepoint rates to zero. L-BFGS assumes a smooth
objective. This one is not.

This caused three separate, independently-discovered failures here:

1. **liblbfgs** (the original C++ backend) terminated after ~2 iterations with
   `LBFGSERR_ROUNDING_ERROR`. More-Thuente narrows an interval of uncertainty until the
   strong Wolfe conditions hold; across a kink that interval collapses instead. Sweeping
   `tau` made the mechanism unambiguous — iterations-before-death tracked the L1
   strength exactly: `1/tau = 20` died at 2 iterations, `1/tau = 1e-6` (an effectively
   smooth objective) ran 3528.
2. **scipy's L-BFGS-B** on the natural parameterization stalled 17.8% above the optimum
   while reporting `success=True`. Not a tolerance problem: with `ftol=0, gtol=0` it
   still stopped at the same point.
3. **Stan itself**, as shipped in Prophet, stops short too — see below.

### The fix: a smooth reformulation

Split `delta` into non-negative parts:

```
delta = delta_pos − delta_neg,    delta_pos, delta_neg ≥ 0
```

Then `|δ|` becomes `delta_pos + delta_neg`: linear, smooth, with the non-smoothness
moved into simple box constraints that a bound-constrained solver handles natively. At
an optimum at most one of each pair is non-zero, so the two problems have the same
solution. Both fit paths now use this.

### Prophet's optimizer stops short on this objective

Scoring both solutions under **Stan's own log density**, fitted on the same model
specification (Prophet's own `t_change`, so `delta` is comparable):

| | Stan `lp__` (higher is better) |
|---|---|
| Prophet's optimum | 7794.9104 |
| this implementation | **7797.2523** |
| difference | **+2.34 nats** |

Prophet does not reach that value even with its stopping rule effectively removed. Run
with `tol_rel_obj=1` (one machine epsilon), `tol_obj=1e-20`, `tol_grad=1e-20`,
`tol_param=1e-20`, it runs 1252 iterations, reaches **7796.96** — still 0.29 nats short
— and then terminates with:

> `Optimization terminated with error: Line search failed to achieve a sufficient decrease, no more progress can be made`

Its own iteration log shows the cause: repeated `LS failed, Hessian reset` entries, and
`||grad||` sitting between 55 and 90 throughout, never approaching a gradient tolerance.
That is the same non-smoothness, defeating Stan's line search the way it defeated
liblbfgs's and scipy's.

**It is not specific to L-BFGS.** Stan also ships a Newton optimizer, which Prophet uses
on series under 100 observations, and it is genuinely the better of the two *for Prophet*
— it beats Prophet's own L-BFGS at every size measured. It still lands below this
implementation's optimum, at every size:

| T | Prophet, Newton | Prophet, L-BFGS | this implementation |
|---|---|---|---|
| 20 | 74.40912 | 73.93879 | **74.89546** |
| 50 | 139.93934 | 139.55784 | **139.95119** |
| 99 | 246.80793 | 246.38023 | **246.82546** |
| 150 | 380.44485 | 379.47301 | **380.56898** |

That matters for what can be concluded. A first-order method stopping short is
consistent with "L-BFGS is a poor fit for this objective" — which would be a statement
about the algorithm. A *second-order* method stopping short in the same place is not:
Newton uses curvature, and curvature is exactly what a kink does not have. Two
different algorithms, both defeated at the same point, is the behaviour the Laplace
prior's non-differentiability predicts.

The third piece is [flat growth](#flat-growth-is-an-exact-tie-and-that-is-the-point),
where the trend's flat directions are removed and the disagreement vanishes entirely —
identical `lp__` to Stan's printed precision. Taken together: two of Prophet's
algorithms stop short where the objective has a kink, and neither does when the kink
stops mattering.

**Using `lp__` as the yardstick is what made this legible.** Parameters cannot be
compared directly across the two implementations (rotated Fourier basis, different
changepoints), but `lp__` is a single scalar, reparameterization-invariant, and is the
quantity the model is actually defined by. Both implementations can be scored on it.

### What this does and does not claim

It claims: **this implementation finds a better optimum of the model Prophet
specifies**, by Prophet's own objective.

For a long time it did **not** claim better forecasts, because that is a different
question and the one adjacent measurement pointed the other way:

| | training SSE | active changepoints (`\|δ\| > 1e-6`) |
|---|---|---|
| Prophet | **788.646** | 17 |
| this implementation | 789.679 | 8 |

Prophet fits the *training data* marginally better; this implementation scores better on
the posterior because it pays less in the Laplace prior, i.e. it finds a sparser trend.
Whether a better MAP point generalizes was left open, pending held-out evaluation.

**That evaluation has now been done, and it reverses.** On 36 M4 series, rolling-origin,
using Prophet's own cutoffs and its own scorer, this implementation is more accurate —
MAE better on 26 of 36 (p = 0.0063), RMSE on 25 of 36 (p = 0.0183) — and with *narrower*
intervals at statistically indistinguishable coverage. The mechanism is visible in the
table above: the sparser trend, which loses in-sample, wins out of sample. It fits fewer
active changepoints than Prophet on **all 36** series.
→ [Tier 2](../evaluation/results/report.md#tier-2--does-the-better-map-point-forecast-better)

Two things keep that from being the last word. The corpus is M4, whose series are
anonymised and therefore barely exercise holidays or day-of-week effects — the features
Prophet is built for. And **both** implementations badly under-cover: the nominal 80%
interval contains about 35% of the points, for Prophet as much as for us, which is larger
than anything separating them.

### Prophet's rule for short series, and what it costs

Prophet does not always run L-BFGS. [fc] `CmdStanPyBackend.fit`: `'Newton' if T < 100
else 'LBFGS'`, one retry with Newton when the first attempt raises — at any length,
controlled by `newton_fallback` — and an explicit `algorithm=` overriding the choice.
Both fit paths here follow that rule, including the override
([#25](https://github.com/adlyZaroui/analytic-prophet/issues/25)).

It was implemented **against** the measurement rather than because of it. #25 was filed
on the premise that running a different algorithm on short series meant we could not
expect matching parameters there, and proposed measuring before deriving a Hessian. The
measurement inverted the premise: this implementation's L-BFGS already beats Prophet's
Newton at every size, including below 100
([table above](#prophets-optimizer-stops-short-on-this-objective)). Adopting Newton
there could only match that, not improve on it.

The rule is implemented anyway, because the contract is *the same fit under the same
data*. Keeping a better-scoring algorithm because it scores better is exactly the silent
divergence the rest of this file is spent ruling out, and it would mean a script ported
from Prophet quietly running a different optimizer than Prophet's own documentation
describes.

**What it costs: nothing.** Scored under Stan's density on Prophet's changepoints, so
only the optimizer differs:

| T | rule picks | Prophet, Newton | ours, L-BFGS | ours, Newton (C++) | ours, Newton (Python) |
|---|---|---|---|---|---|
| 20 | Newton | 74.40912 | 74.89546 | 74.89546 | 74.89546 |
| 30 | Newton | 91.61102 | 91.61988 | 91.61988 | 91.61988 |
| 50 | Newton | 139.93934 | 139.95119 | 139.95119 | 139.95119 |
| 75 | Newton | 185.66277 | 185.66397 | 185.66397 | 185.66397 |
| 99 | Newton | 246.80793 | 246.82546 | 246.82546 | 246.82546 |
| 150 | L-BFGS | 380.44485 | 380.56898 | 380.56898 | 380.56898 |

All four of our columns agree to five decimals at every size, so a default fit under the
rule lands exactly where L-BFGS would — still ahead of Prophet at every size. The cost is
time, not accuracy: Newton converges in 36–85 iterations but pays `2n` gradient
evaluations per iteration for its Hessian, which makes `fit_cpp` about 6× slower at these
sizes — 0.016s against 0.003s at T=50, where Prophet takes 0.171s.

That agreement is a property of the damping schedule, not of the objective, and the
first version of this Newton did not have it: with the Levenberg parameter keyed to
whether a step was *accepted*, it landed 1.39 nats short at T=20 and 0.05 at T=30, and
crawled — at T=50, a median of 14 backtracks per step, an accepted step length of 6e-5,
and 658 iterations. Keying it to whether the step had to be **backtracked** instead — a
step the line search had to shorten is a step the quadratic model was trusted too far on,
so the damping rises even though the step succeeded — cut that to 44 iterations and
removed both shortfalls. `tests/test_short_series.py` asserts the agreement, so it stays
measured.

**Newton needed the split reformulation too**, and that is the part worth keeping. The
first version ran on the natural parameterization — `(k, m, delta, beta, zeta)`, no
bounds, the Laplace prior entering through its subgradient, which is what Stan's Newton
does. It does not converge. It oscillates across the kink at `delta = 0` making about
**1e-5** progress per step, and after Prophet's entire 10,000-iteration budget it is
still **66 nats** short. Moved onto the
[split reformulation](#the-fix-a-smooth-reformulation), where the L1 term is linear and
its curvature is exactly zero rather than undefined, the same code converges in under a
hundred iterations.

That is the [central claim](#prophets-optimizer-stops-short-on-this-objective) measured
from the other side. A second-order method, with an exact gradient, given Prophet's whole
iteration budget, cannot reach the optimum of this objective as written; remove the
single kink and it arrives in under a hundred steps. The obstacle is the
non-differentiability, not the optimizer.

The Hessian is not derived in closed form, which is what #25 expected the blocking work
to be. It is central differences of the **analytic** gradient — `2n` gradient
evaluations, accurate to about 1e-8. Differencing an exact gradient is a different thing
from differencing an objective, and at these sizes it is cheap.

**The fallback has something to catch on one path.** `fit_cpp` still converges at every
size down to ten observations, so its retry has never fired outside a test. `fit()` does
fail in one configuration — yearly forced at order 10 on 328 days from an all-zero start,
which is `test_fit_cpp_parity`'s matched-initialization setup and not one the auto rule
would select. There the retry converges and lands on the same objective to within 1e-9:
it corrects the reported status without moving the answer.

One boundary worth naming: running out of iterations is **not** a failure. CmdStan
reports an exhausted `iter` budget as a warning and cmdstanpy raises only on a non-zero
exit code, so Prophet keeps that fit. Treating it as a failure here made a widened-prior
fit measurably worse by retrying a run that had not actually failed, which is why
`SCIPY_LINE_SEARCH_FAILURE` and `CPP_SOLVER_RAISED` name one status each instead of the
two paths sharing a `status != 0` test.

### Flat growth is an exact tie, and that is the point

Every comparison above ends with our posterior a few nats ahead, attributed throughout
to Prophet stopping short on the trend's flat directions — `k` against `delta`, which
trade off almost freely.

Flat growth removes those directions. `k` and `delta` remain parameters and keep their
priors, but the likelihood never sees them, so nothing pulls against the shrinkage and
both are pinned at zero. What is left is well conditioned.

The result is an **exact tie**, at every size measured:

| T | Prophet `lp__` | ours | difference |
|---|---|---|---|
| 300 | 720.336510 | 720.336510 | 0 |
| 1000 | 2698.194700 | 2698.194700 | 0 |
| 2905 | 7494.870800 | 7494.870800 | 0 |

Not "ours is no worse" — identical to Stan's full printed precision, with `beta` agreeing
to ~1e-7 and `m` to 1e-5. This is the strongest evidence the project has that the margin
under linear growth is optimizer behaviour on a flat objective, not a difference in what
is being fitted. Take the flat directions away and both implementations land on the same
point.

### One claim that was withdrawn

An earlier version of this file recorded a **111%** forecast disagreement on a 328-day
slice, explained as the model being under-identified below Prophet's own 730-day
threshold for yearly seasonality. The explanation was right about the mechanism and
wrong about the cause.

Prophet's rule *disables* yearly under 730 days. The benchmarks were forcing it on for
both sides, because yearly was the only component this implementation could fit.
Comparing against a configuration Prophet would never choose is not a measurement of
disagreement. With the selection rule implemented, both sides fit weekly-only at that
length and agree to **0.709%**.

What survives is smaller and is about the trend rather than the seasonality — see
**Overall fit** under [Known differences](deviations.md#known-differences-from-prophet). The general
lesson is recorded because it recurred: on this project, a large disagreement has so far
always been a difference in what was being compared, not a defect in the gradient.

---
