# analytic-prophet

A reimplementation of [Facebook Prophet](https://github.com/facebook/prophet)'s fitting
engine that replaces Stan with a hand-derived, closed-form gradient and a small C++ core.

It fits the same model, and it fits it better: our optimum is ahead of Prophet's by its
own objective at every size measured, and on held-out M4 series our forecasts are more
accurate. Fitting is faster and uses less memory, which is what the analytic gradient was
for.

**Status: early development.** The model is feature-complete against Prophet's, but there
is no MCMC and no plotting, so this is not yet a drop-in replacement. See
[what this is not](#what-this-is-not).

```bash
git clone https://github.com/adlyZaroui/analytic-prophet
cd analytic-prophet
brew install eigen lbfgspp          # or equivalent; header-only, nothing is linked
pip install -e '.[dev]'
```

```python
import pandas as pd
from analytic_prophet import AnalyticProphet

df = pd.read_csv("tests/data/peyton_manning.csv")     # columns: ds, y

model = AnalyticProphet(seasonality_mode="multiplicative")
model.fit_cpp(df)                                      # or model.fit(df) for the reference path

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
```

---

## What it does

**No Stan.** Prophet ships a compiled Stan model and reaches it through `cmdstanpy`,
which spawns a subprocess for every fit. This carries neither. The model is Python, the
arithmetic is a small C++ extension compiled on demand from one source file, and nothing
is linked beyond two header-only libraries.

**No Stan toolchain to deploy.** `pip install` needs no cmdstan, no model compilation, no
subprocess at fit time. The C++ core is built on demand, and the tests that need a
compiler *skip* rather than fail when there is not one.

**An analytic gradient instead of automatic differentiation.** This is the point of the
project. Reverse-mode autodiff tapes a forward pass and reverses over it; the model is
small and entirely explicit, so the gradient can be written down instead. Both
consequences are measured rather than assumed — fitting is **1.5–10× faster** than
Prophet and the fit's peak memory is **about a third** of Prophet's at T = 2905, with the
gap widening as the series grows, which is what a retained tape predicts. Predicting is
faster on both of the paths described below — **1.8×** on the approximate one and **2.6×**
on the exact one.
→ [cost](evaluation/results/report.md#tier-3--what-it-costs)

**Prophet's non-differentiable objective, handled.** The Laplace prior on the changepoint
rates puts `Σ|δ|/τ` in the posterior, which is not differentiable at `δ = 0` — exactly
where the optimum sits, because that prior is what drives most rates to zero. Prophet's
own optimizer stops short there, and so did three others until the objective was
reformulated. Splitting `δ` into non-negative parts makes the problem smooth with simple
bounds, and the same solution.
→ [the argument and the evidence](docs/non-smooth-objective.md)

**A better optimum, by Prophet's own objective.** Scored under Stan's `log_prob` on
identical changepoints, so only the optimizer differs:

| T | Prophet `lp__` | this implementation |
|---|---|---|
| 300 | 813.351 | **815.337** |
| 1000 | 2852.768 | **2855.528** |
| 2905 | 8004.798 | **8005.159** |

→ [the correctness gate](evaluation/results/report.md#tier-0--are-the-two-fitting-the-same-model)

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
→ [forecast accuracy](evaluation/results/report.md#tier-2--does-the-better-map-point-forecast-better)

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
→ [all four paths timed](evaluation/results/report.md#tier-3--what-it-costs)

**Feature-complete against Prophet's model.** Linear, logistic and flat growth;
seasonality selected from the history by Prophet's own rule, with per-component Fourier
order, prior scale, mode and condition; holidays, country holidays and extra regressors;
additive and multiplicative modes throughout.

**A Prophet-compatible API.** The constructor takes Prophet's arguments, and the names
match — `changepoint_prior_scale`, `changepoints_t`, `params`, `make_all_seasonality_features`.
What is *not* implemented is **rejected rather than silently ignored**, so a ported script
fails where it is actually wrong instead of at the first `AttributeError`.

**Save and load**, `[fc]` Prophet's own API:

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
→ [deviations](docs/deviations.md#refitting-is-allowed-and-a-refit-means-something-specific)

**Two fit paths that agree to 1.5e-8.** `fit()` is a readable pure-Python reference;
`fit_cpp()` is the compiled core and the deliverable. Both solve the same reformulated
problem and follow Prophet's algorithm rule — Newton below 100 observations, L-BFGS at or
above, one Newton retry when L-BFGS fails.

**Verified against Stan's own density.** The objective is checked to *be* Prophet's, not
to resemble it: `CmdStanModel.log_prob` evaluated at our parameters must differ from ours
by a constant, and it does to 1e-12.
→ [verification](docs/model.md#verification-the-objective-is-stans-objective)

**A reproducible evaluation suite.** Four tiers — a correctness gate, parameter recovery
on synthetic data with known truth, held-out forecast accuracy, and cost — with committed
results and a generated report. One command regenerates everything.
→ [evaluation/](evaluation/)

---

## What this is not

- **No MCMC.** MAP estimation only; `mcmc_samples > 0` is rejected rather than ignored.
- **No plotting.** No `plot` or `plot_components`.
- **Not a drop-in.** Three names have no counterpart to match, and a refit means something
  here that it cannot mean in Prophet. See [deviations](docs/deviations.md).
- **The intervals are not well calibrated — in either implementation.** On the M4 corpus
  the nominal 80% interval contains about **35%** of the points, for Prophet (0.342) as
  much as for this implementation (0.353). That is a property of the model on long
  horizons and volatile series, it is shared, and it is larger than anything separating
  the two. Nothing above should be read without it.

---

## Documentation

| | |
|---|---|
| [The model](docs/model.md) | what is fitted, term for term, and the proof that it is Stan's objective |
| [The non-smooth objective](docs/non-smooth-objective.md) | the central argument: where Prophet's optimizer stops short, and why |
| [Deviations from Prophet](docs/deviations.md) | deliberate divergences, and the gaps still open |
| [Evaluation report](evaluation/results/report.md) | every measured number, generated from committed results |
| [Benchmarks](benchmark/) | the fast micro-benchmarks, for running against a change |
| [Evaluation suite](evaluation/) | the claim-level study and its methodology |
| [Changelog](CHANGELOG.md) | what was wrong, and how it was found |

---

## Layout

```
analytic_prophet/
    __init__.py       re-exports the package's surface
    forecaster.py     the model
    constants.py      the numbers the model is defined by
    layout.py         where each parameter sits in the flat vector
    seasonality.py    Fourier basis, registry, selection rule
    make_holidays.py  [fc] prophet/make_holidays.py, plus the design columns
    trend.py          the three growth modes and their derivatives
    optimizer.py      projected Newton, and the stopping tolerances
    models.py         [fc] prophet/models.py — the compiled backend's loader
    serialize.py      [fc] prophet/serialize.py — save and load
    optimize.cpp      that backend
docs/                 the model, the argument, the deviations
tests/                722 tests, plus the Peyton Manning series under data/
benchmark/            fast micro-benchmarks, for running against a change
evaluation/           the claim-level study, and its generated report
```

`forecaster.py`, `models.py` and `make_holidays.py` take Prophet's own names.
**The other four have no Prophet counterpart, which is the point:** Stan supplies the
parameter layout, the derivatives and the optimizer there. Writing them down is what this
project is, so they get files you can open.

The C++ source sits *inside* the package rather than beside it because it is the
implementation, not a build input to it — where Prophet hands the problem to Stan, this
hands it to a gradient written out by hand.

One rule the layout imposes, for anyone adding a test: **patch a name where it is looked
up, not where it is defined.** `analytic_prophet/__init__.py` says why.

---

## Building and testing

Requires a C++17 compiler and two header-only libraries:

```bash
brew install eigen lbfgspp          # or equivalent
pip install -e '.[dev]'
pytest                               # 722 tests
```

`pytest` alone is enough — `pyproject.toml` puts the repo root and `benchmark/` on
`pythonpath` along with `evaluation/`, so a fresh clone runs the suite with **no install and no `PYTHONPATH`**.
`pip install -e .` is for importing the package from elsewhere; nothing in the repo
depends on it.

> **`pip install -e .` on macOS with Python 3.13+ can install successfully and still not
> import.** setuptools writes the editable `.pth` with macOS's `UF_HIDDEN` flag set, and
> Python 3.13 hardened `site.addpackage` to **skip hidden `.pth` files**. The install
> reports success, `pip show` is happy, the metadata resolves — and `import
> analytic_prophet` raises `ModuleNotFoundError` from any directory but the repo root.
> `chflags nohidden .venv/lib/python3.*/site-packages/__editable__*` clears it, though
> something re-applies the flag here, so the fix does not stick.
> `tests/test_packaging.py::test_an_editable_install_actually_imports` is what catches
> this: it skips when the package is not installed and fails with the diagnosis when it
> is installed and broken.

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

## Licence

See [LICENSE](LICENSE).
