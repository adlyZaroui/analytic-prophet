# analytic-prophet

A reimplementation of [Facebook Prophet](https://github.com/facebook/prophet)'s fitting
engine that replaces Stan's automatic differentiation with a hand-derived, closed-form
gradient.

**Status: early development.** The model side is feature-complete against Prophet's:
linear, logistic and flat growth; seasonality selected from the history by Prophet's own
rule, with per-component Fourier order, prior scale, mode and condition; holidays, country
holidays and extra regressors; additive and multiplicative modes throughout.

The constructor takes Prophet's arguments, so
`AnalyticProphet(seasonality_mode='multiplicative', changepoint_prior_scale=0.01)` works as
it would there. Arguments for what is *not* implemented — `mcmc_samples`,
`stan_backend`, `scaling='minmax'` — are accepted and then **rejected**, so a ported
script fails where it is wrong rather than at the first `AttributeError`.

It is still not a drop-in replacement: there is no MCMC or plotting, three names have no
counterpart to match, and a refit means something here that it cannot mean in Prophet,
which refuses one. See [Not implemented](#not-implemented),
[Where this deviates on purpose](#where-this-deviates-on-purpose) and
[Known differences](#known-differences-from-prophet).

---

## Why

Prophet was published by Facebook's Core Data Science team in 2017 and is still widely
deployed; its successor NeuralProphet (2021) is a different model and out of scope here.
What this project reimplements is the 2017 model's fitting engine, not the model.

Prophet fits its model by maximum a posteriori estimation, running L-BFGS on a log
posterior whose gradient Stan obtains by reverse-mode automatic differentiation. The
model is small and entirely explicit, so that gradient can be written down in closed
form instead. Two things should follow, and both are measured rather than assumed:

- **Speed should not regress.** An analytic gradient does strictly less work than taping
  a forward pass and reversing over it.
- **Memory should improve.** Reverse-mode autodiff retains a tape; a closed-form
  gradient does not.

Both now hold. See [Benchmarks](#benchmarks).

---

## The model

MAP estimation, no MCMC. With `S` changepoints at `s₁…s_S`, indicator
`aⱼ(t) = 1[t ≥ sⱼ]` and `K` regressor columns `X`, the trend takes one of three forms
([stan] `trend_indicator`):

```
linear     g(t) = (k + a(t)ᵀδ)·t + (m + a(t)ᵀγ),          γⱼ = −sⱼδⱼ
logistic   g(t) = cap · σ((k + a(t)ᵀδ)·(t − (m + a(t)ᵀγ)))
flat       g(t) = m
```

Under logistic growth `γ` is not `−sⱼδⱼ` but a recursion that keeps the curve continuous
where the rate changes, which is what makes its derivative the hardest one here — see the
[development history](#development-history).

Each of the `K` regressor columns is marked additive or multiplicative — Stan's `s_a`
and `s_m` — giving

```
yhat = trend · (1 + X_sm·β) + X_sa·β
```

which is the trend plus the additive columns, scaled by what the multiplicative ones
contribute. With every column additive the multiplier is 1 and this is `trend + X·β`.

### Objective

```
−log p(θ | y) = T·log σ_obs + Σrᵢ²/(2σ_obs²)          likelihood
              + k²/(2σ_k²) + m²/(2σ_m²)               priors on k, m
              + Σ|δⱼ|/τ                                Laplace prior on δ
              + Σβₗ²/(2σ_β²)                           prior on β
              + σ_obs²/(2σ_ε²)                         half-normal prior on σ_obs
```

### Parameters and constants

Checked against
[`prophet.stan`](https://github.com/facebook/prophet/blob/main/python/stan/prophet.stan)
and
[`forecaster.py`](https://github.com/facebook/prophet/blob/main/python/prophet/forecaster.py).
Three categories, kept deliberately distinct — conflating a prior's *scale* with a
fitted *value* is an easy and consequential mistake:

**FITTED** — free parameters, estimated by L-BFGS (Stan's `parameters` block, length `2 + S + 1 + K`):

| parameter | type | meaning | prior |
|---|---|---|---|
| `k` | real | base trend growth rate | `normal(0, 5)` |
| `m` | real | trend offset | `normal(0, 5)` |
| `delta` | vector[S] | rate adjustment per changepoint | `double_exponential(0, tau)` |
| `sigma_obs` | real, > 0 | observation noise | `normal(0, 0.5)`, half |
| `beta` | vector[K] | seasonality coefficients | `normal(0, sigmas)` |

**SCALE** — fixed constants that are the scale of a prior, not the value of anything.
Prophet estimates none of them; there are no hierarchical priors in the model.

| name | value | applies to | user-configurable in Prophet? |
|---|---|---|---|
| `sigma_k` | 5 | `k` | no — hardcoded in `prophet.stan` |
| `sigma_m` | 5 | `m` | no — hardcoded |
| `sigma_obs` prior scale | 0.5 | `sigma_obs` | no — hardcoded |
| `tau` | 0.05 | `delta` | yes (`changepoint_prior_scale`) |
| `sigmas` | 10.0 | `beta` | yes (`seasonality_prior_scale`) |

`sigmas` is a `vector[K]` in Stan, one scale per regressor column, and is built that way
here: each registered seasonality carries a `prior_scale`, repeated across its block of
the design matrix. A component that sets none inherits the model-wide `10.0`, so a
default model gets a uniform vector and the term reduces to the scalar form.

**FIXED** — structural constants: `n_changepoints = 25`, `changepoint_range = 0.8`,
`y` scaled by `max|y|`, `t` scaled to `[0, 1]`.

**HOLIDAYS** — a holidays frame adds an indicator column per occurrence, plus one per
day of any `lower_window`/`upper_window` around it, carrying `holidays_prior_scale`
(default `10.0`). They join the seasonal columns in the same design matrix, so `K` grows
and nothing else changes: Stan writes `beta ~ normal(0, sigmas)` over every regressor
column alike, and the objective and gradient have had that shape since `sigmas` became
per-column.

```python
model.add_holidays(pd.DataFrame({"holiday": "superbowl", "ds": [...],
                                 "lower_window": -1, "upper_window": 1}))
model.add_country_holidays("US")
```

`add_country_holidays` is not shorthand for writing the dates out. A frame contains the
occurrences it lists and no others, so a forecast past the end of it silently loses
them; a country is resolved against whichever years a frame covers, at fit and predict
alike, so the holidays keep coming. It needs the [`holidays`](https://pypi.org/project/holidays/)
package — the same one Prophet uses, imported lazily, so nothing else in the model
requires it.

**GROWTH** — `model.growth = 'logistic'` fits a saturating trend toward a capacity the
caller supplies per row, in a `cap` column on every frame passed to `fit` and `predict`:

```
[stan]  cap .* inv_logit((k + A·delta) .* (t - (m + A·gamma)))
```

`gamma` keeps the curve continuous where the rate changes, and unlike the linear case it
is defined by a recursion — `gamma[i]` depends on every earlier one — so its Jacobian is
accumulated forward alongside it.

`model.growth = 'flat'` fits the constant `m`. `k` and `delta` stay parameters and keep
their priors, but the likelihood never sees them, so both go to zero.

**REGRESSORS** — `add_regressor(name, prior_scale=None, standardize='auto', mode=None)`
names a column that the frames passed to `fit` and `predict` must both carry. Its
`prior_scale` defaults to `holidays_prior_scale` — surprising, and matched deliberately.
`standardize='auto'` standardizes unless the column is binary, with the mean and spread
fitted on the history and reapplied unchanged at predict time.

`regressor_predictor=True` fits a second model on the regressor itself and uses it to
supply the future values, so `predict` needs only `ds`. It fills rows past the end of the
history only — rows inside it come back from the fit's own values, and rows the caller
supplied are left alone. Its uncertainty is propagated: each draw of the interval uses a
different draw from the nested model, so a forecast that has to guess the regressor is
less confident than one that is told it.

**SELECTED** — `K` is not a constant. Which seasonal components a model fits is decided
from the history, exactly as Prophet does it:

| component | period | Fourier order | fitted when |
|---|---|---|---|
| yearly | 365.25 | 10 | history spans ≥ 730 days |
| weekly | 7 | 3 | spans ≥ 2 weeks **and** observations closer than 7 days apart |
| daily | 1 | 4 | spans ≥ 2 days **and** observations closer than 1 day apart |

`yearly_seasonality`, `weekly_seasonality` and `daily_seasonality` override the rule:
`True` forces the default order, `False` leaves the component out, an integer sets the
order directly. Forcing yearly on under 730 days of history warns, as Prophet's does.

`add_seasonality(name, period, fourier_order, prior_scale=None, mode=None,
condition_name=None)` registers a component of your own, and returns the model so calls
chain. A component registered under a built-in name replaces that built-in rather than
colliding with it, so `add_seasonality('weekly', 7, 10)` is how you ask for a
higher-resolution weekly term than the default order 3. `prior_scale` falls back to the
model's `seasonality_prior_scale`. `condition_name` names a boolean
column, required on the frames passed to both `fit` and `predict`, whose False rows have
that component's features zeroed — the columns stay, so `beta` keeps its width and only
the excluded rows inform it. `mode` is `'additive'` or `'multiplicative'`: an
additive component adds to the trend, a multiplicative one scales it, so its effect
grows with the level of the series.

**Initialization** — Prophet overrides Stan's random initialization with deterministic
values, so Stan's `init_r · N(0,1)` default is never reached: `k`, `m` from
`linear_growth_init` (the line through the first and last points of the scaled series),
`delta` and `beta` zero, `sigma_obs` 1.0.

---

## Verification: the objective is Stan's objective

The strongest check in the project. `cmdstanpy` exposes `CmdStanModel.log_prob`, which
evaluates **Stan's own log density** at arbitrary parameter values. Scoring our
closed-form posterior against it at six points — both optima and four random draws
spanning 1e5 to 1e6 in magnitude — the two sum to zero every time, to the eight
significant figures Stan reports:

```
point 0: ours=  194997.205423  stan= -194997.210000  sum=-4.6e-03
point 1: ours=  282630.300634  stan= -282630.300000  sum=+6.3e-04
point 2: ours=  783639.264098  stan= -783639.260000  sum=+4.1e-03
point 3: ours= 1769879.589294  stan=-1769879.600000  sum=-1.1e-02
```

Relative agreement ~1e-8, which is Stan's output precision. Not equal up to a dropped
constant — **identical**. The hand-derived posterior reproduces what Stan's autodiff
computes.

Comparing raw parameters does *not* work, and this trips people up:

- Prophet builds its Fourier features from days since the **1970 epoch**; this
  implementation uses days since the **series start**. Same function, rotated
  `(cos, sin)` basis. `beta` coefficients differ, including in sign. Transferring the
  seasonality vector between the two bases by least squares leaves a residual of 1e-14,
  confirming they span the same space.
- Changepoints sit at different locations (see [Known differences](#known-differences-from-prophet)),
  so `delta` is not comparable element-wise either.

The gradient is separately checked against finite differences, and the C++ and Python
gradients against each other to ~1e-16 relative.

---

## The non-smooth objective, and why it matters

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

It does **not** claim better forecasts. That is a separate, untested question. The one
adjacent measurement currently available points the other way:

| | training SSE | active changepoints (`\|δ\| > 1e-6`) |
|---|---|---|
| Prophet | **788.646** | 17 |
| this implementation | 789.679 | 8 |

Prophet fits the *training data* marginally better; this implementation scores better on
the posterior because it pays less in the Laplace prior, i.e. it finds a sparser trend.
Whether a better MAP point generalizes better is an empirical question requiring
held-out evaluation, which has not been done. The planned prediction benchmark is where
that gets settled.

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
**Overall fit** under [Known differences](#known-differences-from-prophet). The general
lesson is recorded because it recurred: on this project, a large disagreement has so far
always been a difference in what was being compared, not a defect in the gradient.

---

## Benchmarks

Against `prophet` 1.4.0, both sides on their **own defaults** — additive, linear growth,
MAP, seasonality chosen by the rule above. Every number here is re-measured whenever the
fit changes, and the fit is checked bit-for-bit against the previous commit on every
change that should not have moved it. (Prophet's seasonality used to be pinned to
yearly-only, because that was the only component this implementation could fit;
comparing its 26-column design matrix against a 20-column one would have called a
modelling gap "performance".) Peyton Manning series, Apple Silicon. `T` is series
length; `fit_cpp` is the compiled path, `fit` the Python reference.

**Fitting time** (best of 3, seconds):

| T | prophet | `fit` | `fit_cpp` |
|---|---|---|---|
| 300 | 0.048 | 0.115 (2.40×) | **0.018 (0.40×)** |
| 1000 | 0.158 | 0.551 (3.49×) | **0.105 (0.68×)** |
| 2905 | 0.581 | 3.222 (5.55×) | **0.397 (0.68×)** |

**Forecast intervals** — mean band width over the horizon, Peyton Manning at T = 1000:

| | prophet | `fit_cpp` | ratio |
|---|---|---|---|
| trend, 30 days | 0.00505 | 0.00731 | 1.45 |
| trend, 90 | 0.04811 | 0.06043 | 1.26 |
| trend, 365 | 0.50358 | 0.59661 | 1.18 |
| **yhat, 90** | **1.05234** | **1.05518** | **1.003** |

`yhat`'s interval is dominated by the observation noise: fitted `sigma_obs` is 0.41 in
series units, and an 80% interval on `normal(0, 0.41)` is 1.05, which is nearly all of
it. The trend contributes about 5%, which is why its 1.2× shows up as 1.003× there.

The remaining 1.2× is not the sampler: the Laplace scale it draws from is `mean|δ|`, the
band is linear in it, and ours runs 1.171× Prophet's on this series because the two
optimizers land on different rate adjustments — sparser here (6 active against 15) but
larger in mean magnitude.

Two details of this are worth stating, because each looks wrong until you know why.

**The history has a zero-width band.** New changepoints are drawn on `(1, T]` — strictly
past the end of the history — so the sampled trend over the history is the same in every
draw. A forecast therefore reports no trend uncertainty at all over the period it was
fitted on. That is Prophet's behaviour too, verified rather than assumed, and it follows
from what the interval means: it is the uncertainty in *where the trend goes next*, not
in where it has been. (It is zero to about 1e-15 rather than exactly, since each draw
appends its changepoints and `A·δ` then sums a different number of zero terms.)

**The Laplace scale carries `+ 1e-8`.** [fc] `lambda_ = np.mean(np.abs(deltas)) + 1e-8`.
A series straight enough that the fit leaves every changepoint inactive gives `mean|δ|`
of exactly zero, and `Laplace(0, 0)` is degenerate — every draw returns the same trend
and the band is *identically* zero rather than merely narrow. That is a forecast claiming
certainty it does not have, and the epsilon is what prevents it. It is not a rounding
guard: it is the difference between a narrow interval and no interval.

**Peak memory added by fitting** (T = 2905):

| | added |
|---|---|
| prophet | 9.3 MiB |
| `fit` | **2.6 MiB** |
| `fit_cpp` | **4.1 MiB** |

Prophet runs the optimization in a `cmdstan` subprocess, so its memory is measured via
`RUSAGE_CHILDREN`; see `benchmark/README.md` for the methodology, including why each
measurement runs in a fresh process.

`fit` is a readable reference implementation, not a performance target. `fit_cpp` is the
deliverable.

Run them yourself:

```bash
pip install prophet                      # needed for the comparison
python benchmark/benchmark_fit_time.py
python benchmark/benchmark_memory.py
```

---

## Two fit paths

| | `fit()` | `fit_cpp()` |
|---|---|---|
| optimizer | scipy L-BFGS-B | LBFGSpp L-BFGS-B (C++) |
| Newton | `projected_newton` in `analytic_prophet/forecaster.py` | `newton` in `optimize.cpp` |
| gradient | closed form (or finite differences with `analytic=False`) | closed form |
| role | readable reference | the deliverable |

Both paths pick between the two by [Prophet's rule](#prophets-rule-for-short-series-and-what-it-costs):
Newton below `T = 100`, L-BFGS at or above, one Newton retry when L-BFGS fails.
`algorithm='Newton'` or `algorithm='LBFGS'` overrides it, as it does in `Prophet.fit`.
The two Newtons are deliberate twins — same split space, same Levenberg damping, same
stopping tests — so the paths stay comparable under the rule the way they are under
L-BFGS.

`analytic=False` selects scipy's **finite-difference** approximation, not automatic
differentiation — scipy has no AD. It exists as a control: same optimizer, same problem,
gradient obtained the expensive way. It is 22–40× slower, which is what the analytic
gradient buys.

The two agree to **1.5e-8** relative on the full series.

### What Stan's tolerance costs on the Python path

Prophet sets no tolerances — `optimize(algorithm='LBFGS', iter=int(1e4))` — so CmdStan's
defaults apply, and [#21](https://github.com/adlyZaroui/analytic-prophet/issues/21) put
them in the C++ core. It deliberately left `fit()` out, with three overrides. Two of
them are gone ([#24](https://github.com/adlyZaroui/analytic-prophet/issues/24)):

| setting | Stan | `fit()` |
|---|---|---|
| `maxiter` | `iter = 1e4` | same |
| `gtol` | `tol_grad = 1e-8` | same — the override was a no-op |
| `maxfun` | no such cap | `10 × maxiter`, so scipy's default of 15000 cannot end the run first |
| `ftol` | `tol_rel_obj × eps = 2.22e-12` | **`1e-16`** |

`gtol` was disabled because scipy tests the inf-norm of the *projected* gradient where
Stan tests the 2-norm of the full one, and on the split problem the projected norm is
far the smaller. Whatever that was worth when it was written, it is now unmeasurable:
enabling Stan's value leaves the run **bit-identical** — same iteration count, same
fitted vector — at T = 30 through 2905. Prophet's changepoint placement
([#15](https://github.com/adlyZaroui/analytic-prophet/issues/15)) is the likely reason;
index-spaced changepoints land on observations, and the problem is better conditioned
for it.

`ftol` is the one that stays, and the reason is not that Stan's number is wrong but that
**scipy's iterate sequence has plateaus**. On the full series it takes a step with a
relative decrease of **6.8e-16** — four orders below Stan's threshold — while still
**4.80 nats** from the optimum, and then goes on descending for another 3800 iterations.
37% of its steps are below the threshold. Stan's test is a one-step test, so it fires on
the first of them.

Three explanations were checked and ruled out:

- **Not scipy's own stopping rule.** Evaluating Stan's test by hand on the trajectory
  fires at the same step.
- **Not the line search.** The C++ core allows 60 tries where scipy's default is 20;
  `maxls` of 20, 60 and 100 give the identical trajectory.
- **Not the parameterization.** Moving `sigma_obs` to `zeta = log(sigma_obs)`, which is
  what the C++ core optimizes, fixes T = 300 and T = 2905 and breaks T = 1000 instead
  (2.58 nats short). The C++ core runs the *same* split reformulation under Stan's real
  tolerance and converges, so what differs between the two is the iterate sequence, not
  the problem and not the test.

What settles it is the comparison #24 was waiting on — what Prophet itself reaches:

| T | Prophet `lp__` | `fit()` as shipped | `fit()` under Stan's `ftol` |
|---|---|---|---|
| 300 | 813.35084 | **815.33726** | 815.11242 |
| 1000 | 2852.76760 | **2855.52800** | 2855.52800 |
| 2905 | 8004.79800 | **8005.15920** | **8000.37110** |

Matching Stan's number would put the reference path **4.43 nats below the model it
reproduces**. Tightening instead costs iterations and nothing else. `fit_cpp`, the
deliverable, uses Stan's values unchanged.

`tests/test_convergence_tolerances.py` pins every number above, including the ones that
would reopen the question: if Stan's `gtol` ever stops being a no-op, or if Stan's `ftol`
ever stops scoring below Prophet, the tests fail rather than the reasoning quietly going
stale.

---

## Where this deviates on purpose

Three deviations are decisions rather than outstanding gaps. Each was measured before
being settled.

### Three names stay different

Names follow Prophet's. `tests/test_prophet_naming.py` enumerates the mapping and fails
if one moves on either side — including if an exception stops being an exception, since a
silent convergence is as much a surprise as a silent divergence. It checks values too: a
matching name on a different quantity would be worse than no match.

Only these have no counterpart to match:

| here | why |
|---|---|
| `fit_cpp` | Prophet has nothing like it. It is the point of the project. |
| `sigma_k`, `sigma_m` | `[stan]` writes these as literals in `k ~ normal(0, 5)` rather than naming them in the data block, and Prophet does not expose them. |
| `T` | `[stan] T`. Prophet reads the count off `history.shape[0]` rather than keeping an attribute. |

Everything else that used to differ now matches, including the five that were argued for
in an earlier revision of this section — `changepoint_prior_scale` (was `tau`),
`changepoints_t` (was `t_change`), `t` (was `t_scaled`), `params` (was `opt_params`, and
now a dict of arrays shaped as Prophet's) and `make_all_seasonality_features` (was
`seasonality_design_matrix`, and now returns a named frame whose columns are Prophet's
`{component}_delim_{i}`). The case for keeping them was internal consistency; the case
against was that anyone porting a script or reading a traceback beside Prophet's meets
these far more often than the derivations do.

### A model with no seasonality fits `K = 0`, where Prophet fits `K = 1`

Stan declares `K` as `int<lower=1>`, so Prophet cannot hand it an empty design matrix:
[fc] `make_all_seasonality_features` appends a `zeros` column with prior scale `1.0` —
*"Dummy to prevent empty X"*. This implementation allows `K = 0` and fits one parameter
fewer.

That is only safe if the padding column does nothing, which is now measured rather than
argued. It is identically zero **and** carries `s_a = s_m = 0`, so it enters neither
`X_sa` nor `X_sm` and cannot touch the likelihood at any `β`. Its only term is
`normal(0, 1)`, costing `β²/2` — verified at `β = 0.5, 1.0, 2.0` against Stan's own
`log_prob`, giving 0.125, 0.500 and 2.000 exactly. At the `β = 0` where Prophet's fit
puts it, the cost is zero.

And the fits agree. Trend-plus-noise under flat growth, at T = 300, 400 and 1000:
identical `lp__` to Stan's full printed precision and identical forecasts. Under linear
growth ours is ahead by 1.3–3.8 nats, which is the usual margin on the trend's flat
directions — the same thing [flat growth](#flat-growth-is-an-exact-tie-and-that-is-the-point)
shows from the other side.

### The Fourier basis is evaluated in a different order

This implementation folds the frequency into one constant, `(2π/period) · t·(i+1)`; Prophet
computes `2π·t` and scales it by `(i+1)/period`. The two are the same function in exact
arithmetic. In floating point they differ by up to **7e-12**, because `t` is days since
the 1970 epoch, so the angles reach ~15000 radians where one ULP is ~1e-12 in `sin`/`cos`.

Matching Prophet's order was tried, and it does make the feature matrices bit-identical
at every period tested. It was reverted for two measured reasons.

*It buys no accuracy.* Against exact rational arithmetic — a 50-digit π and
`fractions.Fraction`, 27,000 samples over the `t` range of the Peyton Manning series —
neither order is systematically closer to the true angle:

| period | order | this implementation | Prophet | closer |
|---|---|---|---|---|
| 7 | 1 | 0.336 ulp | 0.740 ulp | this |
| 7 | 3 | 0.377 | 0.539 | this |
| 7 | 10 | 0.415 | 0.337 | Prophet |
| 30.5 | 1 | 0.707 | 0.346 | Prophet |
| 30.5 | 3 | 0.553 | 0.391 | Prophet |
| 30.5 | 10 | 0.743 | 0.817 | this |
| 365.25 | 1 | 0.297 | 0.419 | this |
| 365.25 | 3 | 0.412 | 0.348 | Prophet |
| 365.25 | 10 | 0.395 | 0.540 | this |
| **mean** | | **0.470** | **0.498** | — |

Five configurations to four, means well inside one ULP of each other. The difference is
noise, not an advantage either way.

*It perturbs every fit.* A 7e-12 change in the design matrix moves where L-BFGS stops,
because the stopping rule is relative objective progress on an objective with flat
directions. Measured: parameters move by up to 1.7e-3 and `yhat` by 4e-5 relative —
immaterial against the 1% the project is held to, but not nothing, and paid on every fit
for a cosmetic match.

So the basis agrees with Prophet's to **1e-10** rather than exactly, which is eight
orders of magnitude below anything the model resolves, and the fits stay where they are.
`tests/test_conditional_seasonalities.py` asserts the tolerance and explains it at the
point of the check.

### Refitting is allowed, and a refit means something specific

[fc] `Prophet.fit` refuses a second call:

```python
if self.history is not None:
    raise Exception('Prophet object can only be fit once. '
                    'Instantiate a new object.')
```

So there is no original behaviour to copy — there is a *refusal*, and a
reimplementation that allows the call owes a contract in its place. The one implemented
is the one the refusal implies
([#41](https://github.com/adlyZaroui/analytic-prophet/issues/41)):

> **A refit is equivalent to a fresh instance carrying the same user configuration, fit
> on the new data.**

What the caller set survives. What the previous history produced does not.
`FIT_DERIVED_ATTRIBUTES` in `analytic_prophet/forecaster.py` is the list that makes it true, and
`_reset_fit_state` applies it at the top of both `fit()` and `fit_cpp()`.

Keeping refits is the more useful behaviour and costs nothing measurable — the
[posterior and prediction agreement](#benchmarks) are unchanged — but it is a divergence,
so it is a decision on the record rather than something that fell out.

**It was not free when it was written.** Two bugs of exactly this shape were live, and
neither raised:

- A model fit on twenty rows kept `n_changepoints` capped at 15 — [fc]
  `set_changepoints` caps at `floor(T · changepoint_range) − 1` and *overwrites* the
  attribute, which Prophet can do because it never fits twice — and then fitted 15 rather
  than 25 changepoints on every later history, however long. A different model, silently.
- A model fit across one date range kept that range's `train_holiday_names`. [fc]
  `construct_holiday_dataframe` pins the training holiday set so predict produces the
  same columns as fit; on a refit the stale set was forced onto the new history as
  all-zero columns, while holidays the new history actually had were filtered out.

`n_changepoints` is the interesting case, because it is neither purely configuration nor
purely fit-derived: it is configuration that `set_changepoints` overwrites. Undoing the
cap only while the capped value is still standing is what separates the two — a caller
who assigned `model.n_changepoints` between fits keeps theirs, and one who did not gets
the count they configured back.

`changepoints` is the other one worth naming. It is in the reset list, which looks wrong
for something the user can supply — until you see that restoring the *constructed* value
hands back the given list where one was given and clears the generated dates where one
was not. One rule, both cases.

**What keeps the list honest** is `tests/test_refit_contract.py`, not the list itself.
It fits a fresh instance and a refit on the same data across eight configurations, two
history pairs and both paths, and compares **every** attribute rather than the ones known
to have been wrong — so a stateful feature that forgets to reset shows up as a failing
test rather than as a wrong number. That is the point: #41 was filed because the trap is
structural, and an enumeration alone would have gone stale the same way.

Two attributes are excluded, neither for a refit-related reason: `rng`, unseeded by
design since the uncertainty sampling draws from it (two *fresh* instances differ in it
too), and `opt`, the solver's own result object, which comes from pybind11 or from scipy
depending on the path and defines equality in neither — its contents are compared
field by field instead, and also reach the sweep through `params`, `_params_vector` and
`loss_over_iterations`.

---

## Known differences from Prophet

Tracked, deliberate, and not yet closed:

- **One convergence tolerance on the Python path**
  ([#24](https://github.com/adlyZaroui/analytic-prophet/issues/24)). `fit()` runs under
  Stan's iteration cap and Stan's `tol_grad`; its relative-objective tolerance is
  **tighter** than Stan's `2.22e-12`, at `1e-16`. Taking Stan's number would put this
  path 4.43 nats *below* Prophet on the full series — see
  [What Stan's tolerance costs on the Python path](#what-stans-tolerance-costs-on-the-python-path).
  `fit_cpp`, the deliverable, uses Stan's values unchanged.
- **Overall fit**: on series past two years, predictions differ from Prophet's by
  0.17–0.59% of the series scale (history plus a 30-day horizon, measured at T = 730,
  800, 1000, 1500, 2000, 2500, 2905). On shorter series the trend decomposition is
  looser and the gap reaches 1.21% at T = 500. What remains is **only** the optimizer:
  the changepoints are now identical to Prophet's ([#15](https://github.com/adlyZaroui/analytic-prophet/issues/15)),
  the design matrix agrees to 1e-10, and our posterior is the better one at every size
  measured, T = 100 through 2905. See
  [#30](https://github.com/adlyZaroui/analytic-prophet/issues/30).

## Not implemented

The model itself is complete against Prophet's — [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16)'s
fourteen tasks are done. What is missing is around it:

- **MCMC sampling** — MAP only. `mcmc_samples > 0` is rejected rather than ignored.
- **Plotting** — no `plot` or `plot_components`.
- **`scaling='minmax'`** — only `absmax`.

Each of these is accepted as an argument and then rejected, where Prophet has an argument
for it, rather than being absent.

---

## Layout

```
analytic_prophet/
    __init__.py      re-exports the package's surface
    forecaster.py    the model, mirroring prophet/forecaster.py
    optimize.cpp     the compiled core, this project's answer to prophet/models.py
tests/               626 tests, plus the Peyton Manning series under data/
benchmark/           against the original: agreement, fit time, memory
```

`forecaster.py` takes its name from Prophet's own, where `prophet/forecaster.py` holds
the `Prophet` class. The C++ source sits inside the package rather than beside it because
it *is* the implementation, not a build input to it — where Prophet hands the problem to
Stan, this hands it to a gradient written out by hand.

Importing the package needs the repo root on `sys.path`; `tests/conftest.py` and
`benchmark/_common.py` each put it there, so neither an install nor a `PYTHONPATH` is
required to run either.

---

## Building and testing

Requires a C++17 compiler and two header-only libraries:

```bash
brew install eigen lbfgspp          # or equivalent
pip install -r requirements-dev.txt
pytest tests/                        # 626 tests
```

Nothing is linked: the extension needs Eigen and LBFGSpp headers only. The test suite
compiles `analytic_prophet/optimize.cpp` into a temporary directory on the fly, which is
why no binary is checked in. Tests that need the toolchain **skip** rather than fail when
it is absent.

`prophet` itself is deliberately not a dependency — every comparison against the original
needs it, and it pulls `cmdstanpy` plus a compiled Stan model. The agreement tests skip
without it and the benchmarks print an install hint, so `pip install prophet` is only
needed to run those. `holidays` is required for `add_country_holidays` and imported
lazily, so nothing else needs it.

---

## Development history

Every item below was found and fixed with a regression test. Listed because the failure
modes are instructive, and because several were silent — producing plausible wrong
numbers rather than errors.

| | issue | what was wrong |
|---|---|---|
| Correctness | [#2](https://github.com/adlyZaroui/analytic-prophet/issues/2) | Uncertainty intervals de-normalized asymmetrically: `y_absmax` was applied to the intercept term only, leaving the slope-driven part in normalized units |
| | [#4](https://github.com/adlyZaroui/analytic-prophet/issues/4), [#18](https://github.com/adlyZaroui/analytic-prophet/issues/18) | `sigma_obs` was a fixed, unseeded random draw rather than an estimated parameter. Now fitted, in both paths, with the `T·log σ` normalization the likelihood requires |
| | [#12](https://github.com/adlyZaroui/analytic-prophet/issues/12) | Full audit of every constant against Prophet. All prior scales and structural constants were already correct; initialization was not — `fit` started `k = m = 0` instead of `linear_growth_init`, and `fit_cpp` drew from an unseeded `2.0·N(0,1)`, making it non-reproducible |
| | | Changepoint indicator used `>` where Stan uses `>=`. Aligned — and verified numerically inert, since `aⱼ(t)·δⱼ·(t − sⱼ)` vanishes at `t = sⱼ` either way |
| Optimizer | [#8](https://github.com/adlyZaroui/analytic-prophet/issues/8) | liblbfgs died after 2 iterations with `LBFGSERR_ROUNDING_ERROR`. Root cause: the non-smooth Laplace prior |
| | [#21](https://github.com/adlyZaroui/analytic-prophet/issues/21) | No reachable stopping criterion — `past = 0` disabled the objective-change test and the gradient threshold was unattainable, so runs ended on line-search exhaustion ~33k iterations past convergence. Stan's criteria implemented instead |
| | [#23](https://github.com/adlyZaroui/analytic-prophet/issues/23) | OWL-QN replaced by the split reformulation with L-BFGS-B: 5.4× faster and a better optimum |
| | [#28](https://github.com/adlyZaroui/analytic-prophet/issues/28) | The changepoint and Fourier matrices, constant for a whole fit, were rebuilt on every objective evaluation — 57% of each. Built once: `fit_cpp` reached parity with Prophet |
| | [#13](https://github.com/adlyZaroui/analytic-prophet/issues/13) | `fit_cpp` returned NaN while reporting `LBFGS_SUCCESS`. Resolved by the solver change |
| | [#24](https://github.com/adlyZaroui/analytic-prophet/issues/24) | `fit()`'s loss trace recorded the *canonical* objective while scipy minimized the *split* one. Equal at the optimum, so the final value was right and nothing caught it — but the recorded trajectory rose on a run that descends, and was not comparable with `fit_cpp`'s |
| Refitting | [#41](https://github.com/adlyZaroui/analytic-prophet/issues/41) | A model fit once on twenty rows kept `n_changepoints` capped at 15 and fitted 15 rather than 25 on every later history; a model fit across one date range kept that range's holiday names and forced them onto the next as all-zero columns. Both silently fitted a different model. `Prophet.fit` raises on a second call, so neither can arise there |
| Modelling | [#36](https://github.com/adlyZaroui/analytic-prophet/issues/36) | Fourier basis measured days from the series start, not the 1970 epoch, and emitted all `cos` then all `sin` rather than interleaving. A pure reparameterization — but until it was fixed, `beta` could not be compared with Prophet's at all |
| | [#3](https://github.com/adlyZaroui/analytic-prophet/issues/3) | The C++ carried `params.segment(2, 25)` and `fourier_components(..., 10)` as literals, so `S` and `K` could not vary |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 2 | The seasonal component was described in four places that had to agree and nothing checked that they did. A `ParameterLayout` now derives every offset from `(S, K)`, and `K` comes from a registry |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) tasks 3–4 | Yearly was registered unconditionally. Prophet selects components from the span and spacing of the history — so the two were never fitting the same model unless the series happened to suit yearly-only |
| | [#63](https://github.com/adlyZaroui/analytic-prophet/issues/63), [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 14a | `yhat`'s interval was the trend's band shifted, so it carried no observation noise — the term that dominates it. Measured at 0.056× Prophet's, an 18-fold overstatement of precision, and in the dangerous direction: too narrow looks reasonable in a way too wide does not. `predict` now draws `yhat` rather than deriving it, which is also what lets a regressor predictor's own uncertainty enter |
| | [#15](https://github.com/adlyZaroui/analytic-prophet/issues/15) | Changepoints were spaced uniformly in scaled *time*; Prophet spaces them over evenly-spaced *row indices* and takes the dates there, caps the count on short series, and accepts an explicit list. All three ported together, since porting one leaves the other two wrong. They are now bit-identical to Prophet's at every size — and `fit()` stopped terminating ABNORMAL at T = 30, because a changepoint on an observation has data at it where one in a gap does not |
| | [#58](https://github.com/adlyZaroui/analytic-prophet/issues/58), [#35](https://github.com/adlyZaroui/analytic-prophet/issues/35) | The trend interval drew its new changepoints across the whole frame at a per-point rate, so most landed *inside* the fitted history and every draw rewrote the past before extrapolating from it — a band ~170× Prophet's, wider than the data. Placed as Prophet's Poisson process on `(1, T]` instead, which brings it to 1.2×, the residual being our own `mean\|delta\|`. `predict` also stopped writing a `t_scaled` column into the caller's frame |
| | [#52](https://github.com/adlyZaroui/analytic-prophet/issues/52) | The constructor took no arguments, so every setting was an attribute assigned afterwards and a ported Prophet script had to be rewritten line by line. Wiring `interval_width` and `uncertainty_samples` into it exposed three defects in the method they feed: quantiles hardcoded at 95% where Prophet's default is 80%, one draw taken from the global numpy generator so seeding a model did nothing, and `compute_trend` called without the growth mode — a linear band around a logistic fit |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 14d | The `K = 0` divergence above was argued rather than measured, which is the one thing every other agreement claim here is not. Measured: the padding column costs exactly `β²/2` and nothing else, and a trend-only fit under flat growth reproduces Prophet's `lp__` exactly |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 14 | A model with every seasonality disabled could not fit: the C++ length guard read `params_size < 2 + S + 2`, assuming at least one seasonality column. `K = 0` is trend plus noise, and a model. Found through a nested regressor model on 400 days with weekly turned off, which puts yearly below its threshold too |
| | [#33](https://github.com/adlyZaroui/analytic-prophet/issues/33), [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 10 | `add_regressor` accepted a `pd.Series` and discarded it, so a caller's regressor was simply absent with no error. The signature could not have worked either: a series carries the history's values and no way to produce the future ones `predict` needs |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 9 | Country holidays, via the `holidays` package rather than by importing anything from Prophet. Frames and name sets checked against Prophet's own helpers for five countries, since the underlying data moves between releases and asserting specific dates would test the package rather than this code |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 13 | Flat growth, and with it the first exact agreement with Prophet — identical `lp__` at every size, which is what confirms the margin elsewhere is optimizer behaviour rather than a modelling difference |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 12 | Logistic growth. The hardest derivative in the project so far: `gamma` is a recursion, so `d(gamma)/d(k, m, delta)` is accumulated forward rather than written down, and the trend's chain rule carries that S×(2+S) Jacobian through `inv_logit` |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 11 | Multiplicative mode. The first change to the gradient since `sigma_obs` became free: every trend block picks up the `(1 + X_sm·β)` factor and `β`'s picks up the trend. The all-additive case keeps its own branch — not for speed, but because `y − g − s` and `y − (g·1 + s)` differ in the last bits, which was enough to move the scipy path to a point 2.96 nats worse |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 8 | Holidays. The objective and gradient needed no change — a holiday column is a column of `X` — so the work was alignment: columns sorted by name rather than frame order, all-zero columns kept for occurrences outside a frame, and the fit's holiday set reconciled with predict's |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 7 | Conditional seasonalities: a named boolean column zeroes a component's features where it is False. The feature matrix reproduces Prophet's to 1e-10 — see [Where this deviates on purpose](#where-this-deviates-on-purpose) for why not exactly |
| | [#43](https://github.com/adlyZaroui/analytic-prophet/issues/43) | Eigen answers a size mismatch with an assertion, which calls `abort()`: the interpreter died with no traceback and, in a test run, no failing test name. Four such mismatches aborted and eleven more returned plausible wrong numbers. Every dimension and scale is now checked in one place, through both entry points |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 6 | The registry built by tasks 2–5 had no public entry point: fitting anything other than what the auto rule selects meant assigning to `model.seasonalities` directly. `add_seasonality` added, with Prophet's validation checked branch for branch |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 5 | `beta`'s prior used one scalar for every column, where Stan has `vector[K] sigmas`. Verified per-column against Stan's own `sigmas` and `log_prob`, and the shrinkage checked against the ridge algebra rather than merely being nonzero |
| | [#34](https://github.com/adlyZaroui/analytic-prophet/issues/34) | `from_dict_to_array` overwrote `beta` with zeros regardless of what was passed. Invisible only because every caller happened to pass zeros |
| Infrastructure | [#1](https://github.com/adlyZaroui/analytic-prophet/issues/1), [#11](https://github.com/adlyZaroui/analytic-prophet/pull/11) | ctypes → pybind11. The old binding hardcoded a relative library path and carried a 19-entry `argtypes` list kept in sync by hand; mismatches were undefined behaviour rather than errors |
| | [#5](https://github.com/adlyZaroui/analytic-prophet/issues/5) | Parity test between the two fit paths from matched initial conditions |
| | [#26](https://github.com/adlyZaroui/analytic-prophet/pull/26) | Benchmark harness against the original |

One methodological note worth recording: the convexity argument in
[#5](https://github.com/adlyZaroui/analytic-prophet/issues/5) — "the objective is convex,
so a mismatch means a bug" — **stopped being true** once `sigma_obs` became a free
parameter. Along that axis the objective carries `T·log σ_obs`, which is concave, and a
concrete violation of the midpoint inequality is now in the test suite. From perturbed
starting points, 2 of 5 seeds land on local optima 1206 and 912 nats worse than
Prophet's deterministic initialization reaches. That initialization is doing real work.

---

## Licence

See [LICENSE](LICENSE).
