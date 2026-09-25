# The model

What is being fitted, term for term, and the evidence that it is the same thing
Prophet fits. Sources are marked `[stan]` for `python/stan/prophet.stan` and
`[fc]` for `python/prophet/forecaster.py` in facebook/prophet.

[← back to the README](../README.md)

---

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
[development history](../CHANGELOG.md).

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
- Changepoints sit at different locations (see [Known differences](deviations.md#known-differences-from-prophet)),
  so `delta` is not comparable element-wise either.

The gradient is separately checked against finite differences, and the C++ and Python
gradients against each other to ~1e-16 relative.

---
