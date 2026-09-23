# analytic-prophet

A reimplementation of [Facebook Prophet](https://github.com/facebook/prophet)'s fitting
engine that replaces Stan's automatic differentiation with a hand-derived, closed-form
gradient.

**Status: early development.** The model is additive with linear growth, and selects
yearly, weekly and daily seasonality from the history by Prophet's own rule. It is not a
drop-in replacement for Prophet yet — see [Not implemented](#not-implemented).

---

## Why

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

Additive, linear growth, MAP estimation, no MCMC. With `S` changepoints at `s₁…s_S`,
indicator `aⱼ(t) = 1[t ≥ sⱼ]`, offset correction `γⱼ = −sⱼδⱼ`, and `K = 2N` Fourier
features:

```
g(t) = (k + a(t)ᵀδ)·t + (m + a(t)ᵀγ)        trend
s(t) = X(t)·β                                seasonality
rᵢ   = yᵢ − g(tᵢ) − s(tᵢ)                    residual
```

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
model's `seasonality_prior_scale`. `mode='multiplicative'` and `condition_name` are
validated exactly as Prophet validates them and then **refused**, since this
implementation cannot fit either yet — see [Not implemented](#not-implemented).

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
MAP, seasonality chosen by the rule above. (Prophet's seasonality used to be pinned to
yearly-only, because that was the only component this implementation could fit;
comparing its 26-column design matrix against a 20-column one would have called a
modelling gap "performance".) Peyton Manning series, Apple Silicon. `T` is series
length; `fit_cpp` is the compiled path, `fit` the Python reference.

**Fitting time** (best of 3, seconds):

| T | prophet | `fit` | `fit_cpp` |
|---|---|---|---|
| 300 | 0.043 | 0.222 (5.13×) | **0.030 (0.69×)** |
| 1000 | 0.157 | 0.414 (2.63×) | **0.130 (0.83×)** |
| 2905 | 0.576 | 1.504 (2.61×) | **0.348 (0.60×)** |

**Peak memory added by fitting** (T = 2905):

| | added |
|---|---|
| prophet | 9.5 MiB |
| `fit` | **2.6 MiB** |
| `fit_cpp` | **3.5 MiB** |

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
| gradient | closed form (or finite differences with `analytic=False`) | closed form |
| role | readable reference | the deliverable |

`analytic=False` selects scipy's **finite-difference** approximation, not automatic
differentiation — scipy has no AD. It exists as a control: same optimizer, same problem,
gradient obtained the expensive way. It is 22–40× slower, which is what the analytic
gradient buys.

The two agree to **1.5e-8** relative on the full series.

---

## Known differences from Prophet

Tracked, deliberate, and not yet closed:

- **Changepoint placement** ([#15](https://github.com/adlyZaroui/analytic-prophet/issues/15)).
  Prophet spaces changepoints over uniformly-spaced *row indices* of the first 80% of
  history; this implementation spaces them uniformly in *scaled time*. Identical for
  regular daily data, divergent otherwise. Measured contribution to the overall
  disagreement: about a fifth of it.
- **Algorithm selection for short series**
  ([#25](https://github.com/adlyZaroui/analytic-prophet/issues/25)). Prophet uses
  **Newton** when `T < 100` and falls back to Newton when L-BFGS raises. This
  implementation is L-BFGS-only and has no fallback. Related: on short series the Python
  path takes 20–40× more iterations than on a series six times longer, and at `T = 100`
  L-BFGS-B terminates with `ABNORMAL_TERMINATION_IN_LNSRCH`.
- **The Python path's convergence tolerances**
  ([#24](https://github.com/adlyZaroui/analytic-prophet/issues/24)) deviate from Stan's,
  because Stan's values make scipy stall on the split reformulation.
- **Refitting is allowed** ([#41](https://github.com/adlyZaroui/analytic-prophet/issues/41)).
  `Prophet.fit` refuses a second call; this implementation accepts one. Neither the
  divergence nor the contract is currently written down, and it has already produced one
  bug — a component the previous history supported surviving into a history that cannot
  identify it. Fixed for seasonality, but every remaining task in
  [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) adds fit-time state
  facing the same question.
- **Overall fit**: on series past two years, predictions differ from Prophet's by
  0.21–0.54% of the series scale (history plus a 30-day horizon, measured at T = 730,
  800, 1000, 1500, 2000, 2500, 2905). On shorter series the trend decomposition is
  looser and the gap reaches 1.22% at T = 500 — not changepoint placement (refitting on
  Prophet's own changepoints moves that figure to 1.21%), but Prophet stopping in a
  flatter region than we do. Our posterior is the better one at every size measured,
  T = 100 through 2905. See
  [#30](https://github.com/adlyZaroui/analytic-prophet/issues/30).

## Not implemented

- Conditional seasonalities (`condition_name`) — `add_seasonality` accepts the argument
  only to reject it
- Holidays and extra regressors
- Multiplicative seasonality (`trend · (1 + X·β)`)
- Logistic and flat growth — linear only
- MCMC sampling — MAP only

Tracked in [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16).

---

## Building and testing

Requires a C++17 compiler and two header-only libraries:

```bash
brew install eigen lbfgspp          # or equivalent
pip install -r requirements-dev.txt
pytest tests/                        # 242 tests
```

Nothing is linked: the extension needs Eigen and LBFGSpp headers only. The test suite
compiles `legacy/optimize.cpp` into a temporary directory on the fly, which is why no
binary is checked in. Tests that need the toolchain **skip** rather than fail when it is
absent.

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
| Modelling | [#36](https://github.com/adlyZaroui/analytic-prophet/issues/36) | Fourier basis measured days from the series start, not the 1970 epoch, and emitted all `cos` then all `sin` rather than interleaving. A pure reparameterization — but until it was fixed, `beta` could not be compared with Prophet's at all |
| | [#3](https://github.com/adlyZaroui/analytic-prophet/issues/3) | The C++ carried `params.segment(2, 25)` and `fourier_components(..., 10)` as literals, so `S` and `K` could not vary |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) task 2 | The seasonal component was described in four places that had to agree and nothing checked that they did. A `ParameterLayout` now derives every offset from `(S, K)`, and `K` comes from a registry |
| | [#16](https://github.com/adlyZaroui/analytic-prophet/issues/16) tasks 3–4 | Yearly was registered unconditionally. Prophet selects components from the span and spacing of the history — so the two were never fitting the same model unless the series happened to suit yearly-only |
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
