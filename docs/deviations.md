# Deviations from Prophet

Where this implementation differs from the original, split into decisions that
were measured and settled, and gaps that are still open.

[← back to the README](../README.md)

---

## Deliberate deviations

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
directions — the same thing [flat growth](non-smooth-objective.md#flat-growth-is-an-exact-tie-and-that-is-the-point)
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
[posterior and prediction agreement](../evaluation/results/report.md) are unchanged — but it is a divergence,
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

### The default uncertainty sampler is an approximation, in both implementations

Worth stating plainly because it is easy to compare the wrong pair of numbers.
`Prophet.predict(vectorized=True)` is Prophet's default, and it is **not** a faster form
of `vectorized=False` — it is a different computation. Prophet's own two paths disagree by
about **1.4%** on the interval bounds, with `yhat` identical between them.

This implementation had only the exact sampler until
[#93](https://github.com/adlyZaroui/analytic-prophet/issues/93), which is why every
prediction-time number before it compared an exact computation against an approximate one
and reported this project as 2.4× slower. It now offers both, under Prophet's argument
name and Prophet's default.

What the approximation changes, [fc] `_sample_uncertainty`:

- the Poisson process over the horizon becomes an independent coin at each future
  timestep, so at most one changepoint lands per step and the counts agree only as the
  step shrinks;
- the trend is integrated discretely, by a double cumulative sum with a trapezoidal
  correction, rather than evaluated from the piecewise-linear definition;
- rows inside the history get exactly zero, which both samplers do anyway ([#58](https://github.com/adlyZaroui/analytic-prophet/issues/58)).

It costs `O(n_samples × future rows)` against `O(n_samples × T × S)`, and the horizon is
usually a few percent of `T`, which is the entire speed difference.

**Under logistic growth the exact sampler runs regardless.** Prophet's approximation needs
a separate derivation there — a forward-fill and a per-column `gamma` recursion — which
this does not have yet. `model.predicted_vectorized` records which sampler actually ran,
so the fallback is visible rather than silent.

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

---

## What Stan's tolerance costs on the Python path

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
