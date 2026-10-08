# analytic-prophet

[![tests](https://github.com/adlyZaroui/analytic-prophet/actions/workflows/tests.yml/badge.svg)](https://github.com/adlyZaroui/analytic-prophet/actions/workflows/tests.yml)

A reimplementation of [Facebook Prophet](https://github.com/facebook/prophet)'s fitting
engine that replaces Stan with a hand-derived, closed-form gradient and a small C++ core.

It fits the same model, and it fits it better: our optimum is ahead of Prophet's by its
own objective at every size measured, and on held-out M4 series our forecasts are more
accurate. Fitting is faster and uses less memory, which is what the analytic gradient was
for.

**Status: early development.** The model is feature-complete against Prophet's, but there
is no MCMC and no plotting, so this is not yet a drop-in replacement. See
[what this is not](#what-this-is-not).

## Install

```bash
pip install analytic-prophet
```

**On Linux and macOS, Python 3.9–3.14, that is the whole of it.** The wheels carry the
compiled core, so nothing is built at install time and nothing is built at your first
`fit` either: no compiler, no Eigen, no LBFGSpp. Those are the platforms and versions CI
runs the suite on.

**Anywhere else — Windows, or any `pip install` that falls back to the source
distribution — there is no wheel**, and the C++ core is compiled on demand instead: the
first `fit(df)` takes about ten seconds, says so while it does it, and caches the result
for every later run. That path needs a C++17 compiler and the two header-only libraries.
`fit(df, backend="python")` needs neither and works everywhere.

Windows has no wheel because nothing here has ever been tested there, and shipping a
binary for an untested platform is a claim with nothing behind it.

## Quickstart

```python
import pandas as pd
from analytic_prophet import AnalyticProphet

df = pd.read_csv(                                      # columns: ds, y
    "https://raw.githubusercontent.com/adlyZaroui/analytic-prophet"
    "/main/tests/data/peyton_manning.csv")

model = AnalyticProphet(seasonality_mode="multiplicative")
model.fit(df)                                          # the compiled core, built on first use
# model.fit(df, backend="python")                      # the readable reference path, no compiler

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
```

---

## Three series, held out

![three held-out forecasts: the largest advantage, the median, and one Prophet wins](https://raw.githubusercontent.com/adlyZaroui/analytic-prophet/main/evaluation/results/figures/showcase.png)

**What this shows, and what it does not.** Three M4 series, forecast past a cutoff
neither model saw. Each coloured line is continuous through the cutoff: to its left the
model's fit to data it was shown, to its right its forecast. The actual values over the
horizon are drawn in black, and the bands are the nominal 80% intervals.

They are **chosen by rule, not by eye.** Of Tier 2's 36 series, the ranking is taken over
the **11 where a Prophet-shaped model fits at all** — both implementations within 10%
sMAPE held out — because a panel where both miss badly shows the difficulty of the series
rather than the difference between two optimizers. Within those: the series where our
cross-validated RMSE beats Prophet's by the most, the one at the median of that ranking,
and the one where Prophet beats us by the most.

**The top panel is the mechanism; the bottom two are the typical case.** Prophet's
optimizer stops short on the non-differentiable objective, and that costs most where the
trend is doing the work — a regime change, as in the top panel, where the L1 kink is
load-bearing. Elsewhere both implementations fit nearly the same model and the two lines
sit on top of each other. Across all 36 series the median RMSE advantage is **0.45%**, and
past two years of history predictions differ by **0.17–0.59%** of the series scale. A
reader who runs this on their own data should expect the bottom two panels, not the top
one.

**Neither implementation's intervals are well calibrated.** On this corpus they contain
about a third of the held-out points they claim four fifths of — mean coverage **0.356**
for ours and **0.341** for Prophet's. That is a property of the model on long horizons, it
is shared, and it is larger than anything separating the two.

Regenerate it with `python evaluation/showcase.py`; the output is byte-identical because
both sides are seeded.

---

## What it does

**No Stan.** Prophet ships a compiled Stan model and reaches it through `cmdstanpy`,
which spawns a subprocess for every fit. This carries neither. The model is Python, the
arithmetic is a small C++ extension compiled on demand from one source file, and nothing
is linked beyond two header-only libraries.

**No Stan toolchain to deploy.** `pip install` needs no cmdstan, no model compilation and
no subprocess at fit time, on any path.

**From a wheel**, the core is already compiled and nothing is built, ever. **From the
source distribution or a clone** — Windows, or any platform without a wheel — it is built
on demand: the first `fit(df)` compiles `optimize.cpp`, takes about ten seconds, says so
rather than appearing to hang, and caches the result for every later run
([#108](https://github.com/adlyZaroui/analytic-prophet/issues/108)). The cache is keyed by
a digest of the source and the compile command, so editing `optimize.cpp` rebuilds and
nothing else does.

Everything this project caches hangs off **one root**, by the same rule on every
platform: `$ANALYTIC_PROPHET_CACHE`, else `$XDG_CACHE_HOME/analytic-prophet`, else
`~/.cache/analytic-prophet`. The compiled core goes in `build/` and the M4 corpus the
evaluation suite downloads goes in `m4/`, so one variable moves both and deleting the
root is how you start over ([#111](https://github.com/adlyZaroui/analytic-prophet/issues/111)).

Without a compiler the build raises, naming the one thing that is missing — and
`fit(df, backend="python")` needs no compiler at all. The tests that need the toolchain
*skip* rather than fail when it is absent.

**An analytic gradient instead of automatic differentiation.** This is the point of the
project. Reverse-mode autodiff tapes a forward pass and reverses over it; the model is
small and entirely explicit, so the gradient can be written down instead. Both
consequences are measured rather than assumed — fitting is **1.4–8.2× faster** than
Prophet, and which end you get depends on the series: **8.2× at T = 50, falling to 1.4×
by T = 1000 and 1.5× at T = 2905**, because Prophet pays a fixed cmdstan subprocess cost
that matters most when there is least to do. The advantage is a trough rather than a
decline — `T = 1000` is the narrowest point measured, not the longest series. The fit's peak memory is **about two fifths** of Prophet's at
T = 2905, and the decomposition says where that gap is: fitting `memory = fixed + slope·T`
leaves the two fixed costs within 0.3 MiB of each other while the per-observation costs
differ **about 3.6×**. Both numbers are weaker than this README claimed before
[#130](https://github.com/adlyZaroui/analytic-prophet/issues/130), and for a reason worth
knowing: deferring scipy dropped what is resident before the fit from 130 MiB to 73, and
peak RSS is a high-water mark, so fit growth that used to hide under the import's own
transient peak is now visible. The earlier, larger ratio was flattering by accident.
That is the shape a retained tape
predicts, a tape being O(T) in the
operations it records — but peak RSS cannot tell a tape from any other allocation that
grows with T, so the decomposition rules out fixed overhead rather than proving the
mechanism ([#125](https://github.com/adlyZaroui/analytic-prophet/issues/125)). Importing
the library costs **+1.6 MiB** over a numpy+pandas interpreter against Prophet's **+40.2**;
until [#130](https://github.com/adlyZaroui/analytic-prophet/issues/130) that comparison ran
the other way, because scipy was imported whatever backend you asked for and is 57 of the
59 MiB it used to add. It serves only the Python reference backend, so it is now imported
when that runs and not before. Predicting is
faster on both of the paths described below — **1.8×** on the approximate one and **2.6×**
on the exact one.
→ [cost](https://github.com/adlyZaroui/analytic-prophet/blob/main/evaluation/results/report.md#tier-3--what-it-costs)

**Prophet's non-differentiable objective, handled.** The Laplace prior on the changepoint
rates puts `Σ|δ|/τ` in the posterior, which is not differentiable at `δ = 0` — exactly
where the optimum sits, because that prior is what drives most rates to zero. Prophet's
own optimizer stops short there, and so did three others before the objective was
reformulated: **liblbfgs**, which died after two iterations with
`LBFGSERR_ROUNDING_ERROR`; **scipy's L-BFGS-B** on the natural parameterization, which
stalled 17.8% above the optimum while reporting success; and **Stan's own Newton**, which
Prophet uses below 100 observations and which lands short at every size measured. That
last one is the informative one — a second-order method defeated in the same place is
not a statement about L-BFGS, because curvature is exactly what a kink does not have. Splitting `δ` into non-negative parts makes the problem smooth with simple
bounds, and the same solution.
→ [the argument and the evidence](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/non-smooth-objective.md)

**A better optimum, by Prophet's own objective.** Scored under Stan's `log_prob` on
identical changepoints, so only the optimizer differs:

| T | Prophet `lp__` | this implementation |
|---|---|---|
| 300 | 813.351 | **815.337** |
| 1000 | 2852.768 | **2855.528** |
| 2905 | 8004.798 | **8005.159** |

→ [the correctness gate](https://github.com/adlyZaroui/analytic-prophet/blob/main/evaluation/results/report.md#tier-0--are-the-two-fitting-the-same-model)

**Better forecasts, held out.** 36 M4 series, rolling-origin evaluation on cutoffs from
Prophet's own `generate_cutoffs` and scored by its own `performance_metrics`, so neither
the splits nor the definitions are ours:

| | median difference | p |
|---|---|---|
| MAE | −1.914 | 0.0063 |
| RMSE | −2.967 | 0.0183 |
| MAPE | −0.0007 | 0.0013 |
| coverage | **+0.0026** | 0.0025 |
| interval width | +1.350 | 0.470 |

More accurate points, and **higher** coverage at statistically indistinguishable width —
negative is better for the error rows, positive for coverage.
→ [forecast accuracy](https://github.com/adlyZaroui/analytic-prophet/blob/main/evaluation/results/report.md#tier-2--does-the-better-map-point-forecast-better)

**Two uncertainty samplers, and Prophet's default is the approximate one.** This is
worth knowing before comparing any interval or any prediction time.
`Prophet.predict(vectorized=True)` is its default, and it is **not** a faster form of
`vectorized=False` — it is a different computation. Prophet's own two paths disagree by
about **1.4%** on the interval bounds. This implementation offers both, under the same
argument and the same default:

```python
forecast = model.predict(future)                      # approximate, as Prophet defaults to
forecast = model.predict(future, vectorized=False)    # exact, and slower
```

`yhat` is identical either way — only the interval is sampled. The approximation replaces
the Poisson process over the horizon with one coin per timestep and integrates the trend
by a double cumulative sum; the exact sampler places changepoints in continuous time and
evaluates the piecewise-linear trend from its definition. Under logistic growth the exact
sampler runs regardless, and `model.predicted_vectorized` records which one did.
→ [all four paths timed](https://github.com/adlyZaroui/analytic-prophet/blob/main/evaluation/results/report.md#tier-3--what-it-costs)

**Feature-complete against Prophet's model.** Linear, logistic and flat growth;
seasonality selected from the history by Prophet's own rule, with per-component Fourier
order, prior scale, mode and condition; holidays, country holidays and extra regressors;
additive and multiplicative modes throughout.

**A Prophet-compatible API.** The constructor takes Prophet's arguments, and the names
match — `changepoint_prior_scale`, `changepoints_t`, `params`, `make_all_seasonality_features`.
What is *not* implemented is **rejected rather than silently ignored**, so a ported script
fails where it is actually wrong instead of at the first `AttributeError`.

**Save and load**, following Prophet's own API:

```python
from analytic_prophet.serialize import model_to_json, model_from_json

with open("model.json", "w") as handle:
    handle.write(model_to_json(model))
```

A round-tripped model predicts bit-identically, and carries no handle to the compiled
extension — so a model fitted on one machine loads on one that has never built it.

**Refitting is allowed**, where `Prophet.fit` refuses a second call. A refit is
equivalent to a fresh instance carrying the same user configuration, fit on the new data.
That is a divergence, so it is a stated contract rather than an accident.
→ [deviations](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/deviations.md#refitting-is-allowed-and-a-refit-means-something-specific)

**Two backends that agree to 1.5e-8.** `fit(df)` runs the compiled core, which is the
deliverable, so a script ported from Prophet keeps its fit call and gets it.
`fit(df, backend="python")` runs the readable pure-Python reference. Both solve the same
reformulated problem and follow Prophet's algorithm rule — Newton below 100 observations,
L-BFGS at or above, one Newton retry when L-BFGS fails.

Arguments belonging to the backend you did not select are **rejected rather than
ignored**, so `fit(df, analytic=False)` says that `analytic` is the Python backend's
rather than quietly running the compiled one.

**Verified against Stan's own density.** The objective is checked to *be* Prophet's, not
to resemble it: `CmdStanModel.log_prob` evaluated at our parameters must differ from ours
by a constant, and it does to 1e-12.
→ [verification](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/model.md#verification-the-objective-is-stans-objective)

**A reproducible evaluation suite.** Four tiers — a correctness gate, parameter recovery
on synthetic data with known truth, held-out forecast accuracy, and cost — with committed
results and a generated report. One command regenerates everything.
→ [evaluation/](https://github.com/adlyZaroui/analytic-prophet/tree/main/evaluation/)

---

## What this is not

- **No MCMC.** MAP estimation only; `mcmc_samples > 0` is rejected rather than ignored.
- **No plotting.** No `plot` or `plot_components`.
- **Not a drop-in, and here is how far off.** Of Prophet's 40 public methods, 17 are the
  same, 8 are module-level functions here rather than methods, 15 are absent on purpose
  and **none is a gap** — [#114](https://github.com/adlyZaroui/analytic-prophet/issues/114)
  closed the last four. Enumerated member by member, with the attributes a fit sets, in
  [how far from a drop-in](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/deviations.md#how-far-from-a-drop-in-enumerated).
- **The held-out evidence is 36 series.** M4 has 100,000. Thirty-six weekly and daily
  series, rolling-origin, both sides on the same splits and the same scorer, is a
  defensible first pass and it is what every accuracy claim here rests on — but it is a
  small sample for a forecasting result, and the showcase figure ranks within the
  **11** of them where a Prophet-shaped model fits at all. The direction has survived
  multiplicity adjustment on five metrics; the magnitude should be read as "measured on
  this corpus" rather than as a property of the method.
- **The intervals are not well calibrated — in either implementation.** On the M4 corpus
  the nominal 80% interval contains about a third of the points it claims four fifths of
  — mean coverage **0.341** for Prophet and **0.356** for this implementation. That is a property of the model on long
  horizons and volatile series, it is shared, and it is larger than anything separating
  the two. Nothing above should be read without it.

---

## Documentation

| | |
|---|---|
| [The model](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/model.md) | what is fitted, term for term, and the proof that it is Stan's objective |
| [The non-smooth objective](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/non-smooth-objective.md) | the central argument: where Prophet's optimizer stops short, and why |
| [Deviations from Prophet](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/deviations.md) | deliberate divergences, and the gaps still open |
| [Evaluation report](https://github.com/adlyZaroui/analytic-prophet/blob/main/evaluation/results/report.md) | every measured number, generated from committed results |
| [Benchmarks](https://github.com/adlyZaroui/analytic-prophet/tree/main/benchmark/) | the fast micro-benchmarks, for running against a change |
| [Evaluation suite](https://github.com/adlyZaroui/analytic-prophet/tree/main/evaluation/) | the claim-level study and its methodology |
| [Changelog](https://github.com/adlyZaroui/analytic-prophet/blob/main/CHANGELOG.md) | what was wrong, and how it was found |
| [Contributing](https://github.com/adlyZaroui/analytic-prophet/blob/main/CONTRIBUTING.md) | how this project works: building from source, the loop, and what a change carries |

---

## Layout

```
analytic_prophet/
    __init__.py       re-exports the package's surface
    forecaster.py     Prophet's forecaster.py — the model
    constants.py      the numbers the model is defined by
    layout.py         where each parameter sits in the flat vector
    seasonality.py    Fourier basis, registry, selection rule
    make_holidays.py  Prophet's make_holidays.py, plus the design columns
    trend.py          the three growth modes and their derivatives
    optimizer.py      projected Newton, and the stopping tolerances
    models.py         Prophet's models.py — the compiled backend's loader
    build.py          compiling optimize.cpp on demand, and caching it
    serialize.py      Prophet's serialize.py — save and load
    optimize.cpp      that backend
.github/workflows/    CI: the suite on every push, the tiers on request
docs/                 the model, the argument, the deviations
tests/                the suite, plus the Peyton Manning series under data/
benchmark/            fast micro-benchmarks, for running against a change
evaluation/           the claim-level study, and its generated report
```

`forecaster.py`, `make_holidays.py`, `models.py` and `serialize.py` take Prophet's own
names. **The rest have no Prophet file to correspond to, which is the point:** `layout.py`,
`optimizer.py` and the derivatives in `trend.py` are what Stan supplies there, and
`constants.py` holds numbers that live in `prophet.stan` rather than in any Python file.
Writing them down is what this project is, so they get files you can open.

Two of the rest are not that, and saying so costs nothing: `seasonality.py` is code
Prophet has as well, inside its own `forecaster.py`, and `build.py` has no counterpart
because Prophet ships its Stan model already compiled.

The C++ source sits *inside* the package rather than beside it because it is the
implementation, not a build input to it — where Prophet hands the problem to Stan, this
hands it to a gradient written out by hand.

One rule the layout imposes, for anyone adding a test: **patch a name where it is looked
up, not where it is defined.** `analytic_prophet/__init__.py` says why.

---

## Building from source

The wheels cover Linux and macOS on Python 3.9–3.14, so most people need none of this, and
`fit(df, backend="python")` needs no compiler anywhere. Working on the project, or
building the core yourself, needs a C++17 compiler and two header-only libraries.

→ [CONTRIBUTING.md](https://github.com/adlyZaroui/analytic-prophet/blob/main/CONTRIBUTING.md)
has the toolchain, the editable-install caveat on macOS that reports success and then does
not import, how to run the suite and the benchmarks, and what CI makes strict.

---

## Licence

See [LICENSE](https://github.com/adlyZaroui/analytic-prophet/blob/main/LICENSE).
