# analytic-prophet

[![tests](https://github.com/adlyZaroui/analytic-prophet/actions/workflows/tests.yml/badge.svg)](https://github.com/adlyZaroui/analytic-prophet/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/analytic-prophet)](https://pypi.org/project/analytic-prophet/)
[![Python versions](https://img.shields.io/pypi/pyversions/analytic-prophet)](https://pypi.org/project/analytic-prophet/)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue)](https://github.com/adlyZaroui/analytic-prophet/blob/main/LICENSE)

**Stan-free Prophet forecasting model using an analytical gradient instead of AutoDiff: faster and lighter on memory.**

A reimplementation of [Prophet](https://github.com/facebook/prophet)'s fitting engine
without Stan: the same model and the same API, with the log posterior and its gradient
written out by hand and compiled into a small C++ core. `pip install` needs no Stan
toolchain, a fit never starts a CmdStan subprocess, and fits are faster and lighter on
memory at every series length measured.

The forecasts are slightly better too. Prophet's changepoint prior makes the objective it
maximizes non-differentiable exactly where the optimum sits, and the smooth optimizers it
uses stop short of it; this package solves an exact reformulation with no such kink, so by
Prophet's own posterior density its fits score higher at every series length measured. On
3,008 held-out M4 series its RMSE is lower on 1,929 of them, by a median of 0.40%: a real
improvement, and usually a small one.

**Status: early development.** The model is feature-complete against Prophet's, but there
is no MCMC and no plotting. See [coming from Prophet](#coming-from-prophet).

## Install

```bash
pip install analytic-prophet
```

**On Linux and macOS, Python 3.9–3.14, that is the whole of it.** The wheels carry the
compiled core, so nothing is built at install time or at your first `fit`: no compiler, no
Eigen, no LBFGSpp. Those are the platforms and versions CI runs the suite on.

**Anywhere else — Windows, or any `pip install` that falls back to the source
distribution — there is no wheel**, and the C++ core is compiled on demand: the first
`fit(df)` compiles `optimize.cpp`, takes about ten seconds, says so while it does it, and
caches the result under `~/.cache/analytic-prophet` (set `ANALYTIC_PROPHET_CACHE` to move
it). That path needs a C++17 compiler and two header-only libraries.
`fit(df, backend="python")` needs neither and works everywhere. Windows has no wheel
because the suite has never run there. Building from source and working on the project are
in [CONTRIBUTING.md](https://github.com/adlyZaroui/analytic-prophet/blob/main/CONTRIBUTING.md).

## Quickstart

```python
import pandas as pd
from analytic_prophet import AnalyticProphet

df = pd.read_csv(                                      # columns: ds, y
    "https://raw.githubusercontent.com/adlyZaroui/analytic-prophet"
    "/main/tests/data/peyton_manning.csv")

model = AnalyticProphet(seasonality_mode="multiplicative")
model.fit(df)                                          # the compiled core
# model.fit(df, backend="python")                      # the readable reference backend

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
```

## Coming from Prophet

**The same model and the same API.** Linear, logistic and flat growth; seasonality chosen
from the history by Prophet's own rule, with per-component Fourier order, prior scale, mode
and condition; holidays, country holidays and extra regressors; additive and
multiplicative modes throughout. The constructor takes Prophet's arguments, and the names
match: `changepoint_prior_scale`, `changepoints_t`, `params`,
`make_all_seasonality_features`.

**What is not implemented is rejected rather than ignored**, so a ported script fails
where it is actually wrong. `mcmc_samples > 0` raises: this is MAP estimation only. There is
no plotting: no `plot` and no `plot_components`.

**Not a drop-in, and here is how far off.** Of Prophet's 40 public methods, 17 are the
same, 8 are module-level functions here rather than methods, 15 are absent on purpose and
none is a missing method. Two behavioural differences are still open and tracked, under
[known differences](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/deviations.md#known-differences-from-prophet);
the surface is enumerated member by member in
[how far from a drop-in](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/deviations.md#how-far-from-a-drop-in-enumerated).

**Refitting is allowed**, where `Prophet.fit` refuses a second call. A refit is equivalent
to a fresh instance carrying the same user configuration, fitted on the new data
([the contract](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/deviations.md#refitting-is-allowed-and-a-refit-means-something-specific)).

**Save and load** follow Prophet's own API, and a round-tripped model predicts
bit-identically:

```python
from analytic_prophet.serialize import model_to_json, model_from_json

with open("model.json", "w") as handle:
    handle.write(model_to_json(model))
```

**Two uncertainty samplers, and Prophet's default is the approximate one.**
`Prophet.predict(vectorized=True)` is its default and is not a faster form of
`vectorized=False` but a different computation: Prophet's own two paths disagree by about
1.4% on the interval bounds. This package offers both, under the same argument and the
same default. `yhat` is identical either way; only the interval is sampled.

```python
forecast = model.predict(future)                      # approximate, as Prophet defaults to
forecast = model.predict(future, vectorized=False)    # exact, and slower
```

**Why a separate package, and not a pull request to Prophet.** This replaces how the model
is fitted — without Stan, and to its optimum — and Prophet's maintainers wrote in 2023 that
"at this moment we do not plan to make any more changes to the underlying Stan model"
([facebook/prophet in 2023 and beyond](https://medium.com/@cuongduong_35162/facebook-prophet-in-2023-and-beyond-c5086151c138)).

## 1. A faster, lighter Prophet

Prophet compiles its model with Stan and reaches it through CmdStan, a subprocess for every
fit, with automatic differentiation for the gradient. Here the model's log posterior and its
gradient are written out by hand, in a small C++ core, so a fit needs no Stan toolchain, no
model compilation and no subprocess. `fit(df)` runs that core; `fit(df, backend="python")`
runs a pure-Python reference implementation of the same arithmetic, and the two agree to
about 1e-8.

**It computes Prophet's density, not an imitation of it.** Stan's `log_prob` for Prophet's
own program, evaluated at this package's parameters, equals minus its objective to about
1e-13 relative at every one of 21 points measured, and the two gradients agree to about 1e-9
relative (`paper/experiments/agreement.py`).
→ [verification](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/model.md#verification-the-objective-is-stans-objective)

**What it costs**, against Prophet on the same series (Peyton Manning, default
configuration; T is the number of observations), as Prophet's figure over this
package's — above 1, this package is ahead:

![fit time, prediction time and the memory a fit adds, as Prophet's over this package's, against series length](https://raw.githubusercontent.com/adlyZaroui/analytic-prophet/main/evaluation/results/figures/cost_ratios.png)

The fit gains most where Prophet runs Stan's Newton method, below T = 100, and least from
T = 1000 up; the exact sampler gains more than the approximate one at every length; and the
memory gain grows with the series. Importing the library costs +1.6 MiB over numpy and
pandas, against +40.2 for Prophet.

**Very little of the speed is the gradient.** From T = 100 up, where Prophet runs L-BFGS,
a CmdStan call capped at a single iteration takes 24–54% of Prophet's whole fit, and without
it Prophet's fit takes 0.9–1.1 times as long as this one: the advantage there is the cost
of calling CmdStan. Below 100 observations Prophet runs Stan's Newton method, and at
T = 50 that, not the subprocess, is most of its time (`paper/experiments/overhead.py`).

**Where the memory gap is.** Fitting memory = fixed + slope·T leaves the two fixed costs
within 0.3 MiB of each other while the per-observation costs differ about 3.6×. That is the
shape a retained tape predicts: autodiff records the operations of a pass, so its cost
grows with T and is nothing at T = 0. But peak RSS cannot tell a tape from any other
allocation that grows with T, so this rules out fixed overhead rather than proving the
mechanism.
→ [every cost](https://github.com/adlyZaroui/analytic-prophet/blob/main/evaluation/results/report.md#tier-3--what-it-costs)

## 2. Slightly better forecasts

Reaching the optimum Prophet's optimizer misses (section 3) changes the forecasts a
little, mostly for the better.

On a frozen census of 3,008 held-out M4 series — every weekly and daily series the
evaluation protocol can measure, fixed before anything was measured — this package's RMSE
is lower on 1,929 of them (64%), by a median of **0.40%**:

| | median difference | lower on |
|---|---|---|
| RMSE | −0.40% | 1929/3008 |
| MAE | −0.52% | 1894/3008 |
| MAPE | −0.43% | 1807/3008 |
| sMAPE | −0.48% | 1838/3008 |
| interval width | +0.56% | — |

Each row is the median of the per-series differences, ours / Prophet's − 1, so negative is
better for the errors; the intervals are wider. Each is tested by the sign test
(`scipy.stats.binomtest`) and Holm-adjusted across the comparisons. Cutoffs come from
Prophet's own `generate_cutoffs` and scores from its own `performance_metrics`, so neither
the splits nor the definitions are ours.
→ [the full comparison](https://github.com/adlyZaroui/analytic-prophet/blob/main/evaluation/results/report.md#tier-2--does-the-better-map-point-forecast-better)

![three held-out forecasts: the largest advantage, the median, and one Prophet wins](https://raw.githubusercontent.com/adlyZaroui/analytic-prophet/main/evaluation/results/figures/showcase.png)

Three M4 series, forecast past a cutoff neither model saw. Each coloured line is continuous
through the cutoff: to its left the model's fit to data it was shown, to its right its
forecast. The actual values over the horizon are drawn in black, and the bands are the
nominal 80% intervals. Each horizontal axis counts time from the series' first observation
rather than showing dates: M4's recorded starting dates are partly placeholders, and their
two-digit years put some weekly series in the future.

They are **chosen by rule, not by eye.** The ranking is taken over the **658 where a
Prophet-shaped model fits at all** — both implementations within 10% sMAPE held out —
because a panel where both miss badly shows the difficulty of the series rather than the
difference between two optimizers. Within those: the series where our cross-validated RMSE
beats Prophet's by the most, the one at the median of that ranking, and the one where
Prophet beats us by the most. On the first our RMSE is 54.6% lower; at the median the two
lines sit on top of each other; on the last ours is 78.7% higher. Across all 3,008 series
the median RMSE advantage is **0.40%**. A reader who runs this on their own data should
expect the middle panel.

## 3. Why: the optimum Prophet misses

The Laplace prior puts Σ|δⱼ|/τ in Prophet's objective, where δⱼ are the trend's changepoint
rates: not differentiable where a rate is zero, which is where most of them belong. Stan's
L-BFGS and Newton methods, which Prophet uses, are built for smooth objectives; they stop
short of the optimum there, with no rate exactly zero. This package splits each rate into
two non-negative parts, which turns the penalty linear and leaves simple bounds: the same
optimum, and no kink.

![Prophet's forecast a year ahead jumps as changepoint_prior_scale moves near its default; the exact optimum's moves in a straight line](https://raw.githubusercontent.com/adlyZaroui/analytic-prophet/main/paper/figures/tau_path_readme.png)

*Refitted as `changepoint_prior_scale` moves a little around its default, Prophet's
forecast a year past the data jumps between neighbouring settings; the exact optimum's
moves in a straight line.*

The evidence, each piece reproducible and written up in the paper:

- **Prophet's solution fails the first-order optimality conditions on every one of 3,008
  M4 series**, even when each changepoint rate is read in the way most favourable to it
  ([in the paper](https://github.com/adlyZaroui/analytic-prophet/blob/main/paper/sections/evidence.tex)).
- **From 150 random starts, with the noise scale σ held fixed so that the optimum is
  unique, Stan's L-BFGS with Prophet's settings reaches it from none**
  ([in the paper](https://github.com/adlyZaroui/analytic-prophet/blob/main/paper/sections/evidence.tex)).
- **Smoothing the prior inside Prophet's own Stan program, with nothing else changed,
  brings the same L-BFGS 8 to 1,592 times closer to the optimum**: the kink is what stops
  it ([in the paper](https://github.com/adlyZaroui/analytic-prophet/blob/main/paper/sections/evidence.tex)).

→ [the paper](#paper--citing), and the argument in prose:
[the non-smooth objective](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/non-smooth-objective.md)

## Limitations

- **Two of M4's six frequencies.** The census is 3,008 series of M4's 100,000: every weekly
  and daily series the protocol can measure. Monthly, quarterly, yearly and hourly series
  are absent, and the showcase ranks within the **658** of them where a Prophet-shaped
  model fits at all.
- **The advantage is small and not uniform.** On the worst series for this package its
  RMSE is 325% above Prophet's, and on its best 54.6% below.
- **Wider intervals.** Ours are wider by a median 0.56%, on 2,153 of the 3,008 series.
- **Neither implementation's intervals are calibrated.** The nominal 80% interval contains
  about a third of the held-out points it claims four fifths of: mean coverage **0.345**
  for ours and **0.341** for Prophet's. That is a property of the model on long horizons,
  it is shared, and it is larger than anything separating the two.
- **Significance is uninformative at this size.** Every comparison survives Holm
  adjustment, which across 3,008 paired series says only that the direction is consistent:
  read the medians and the win counts.

## Documentation

| | |
|---|---|
| [The model](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/model.md) | what is fitted, term for term, and the proof that it is Stan's objective |
| [The non-smooth objective](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/non-smooth-objective.md) | the central argument: where Prophet's optimizer stops short, and why |
| [Deviations from Prophet](https://github.com/adlyZaroui/analytic-prophet/blob/main/docs/deviations.md) | deliberate divergences, and the behavioural differences still open |
| [Evaluation report](https://github.com/adlyZaroui/analytic-prophet/blob/main/evaluation/results/report.md) | every measured number, generated from committed results |
| [Evaluation suite](https://github.com/adlyZaroui/analytic-prophet/tree/main/evaluation/) | the claim-level study and its methodology |
| [Benchmarks](https://github.com/adlyZaroui/analytic-prophet/tree/main/benchmark/) | the fast micro-benchmarks, for running against a change |
| [Changelog](https://github.com/adlyZaroui/analytic-prophet/blob/main/CHANGELOG.md) | what was wrong, and how it was found |
| [Contributing](https://github.com/adlyZaroui/analytic-prophet/blob/main/CONTRIBUTING.md) | building from source, the layout, the loop, and what a change carries |

## Paper & citing

The paper's subject is the fix, not the package's speed: it exhibits the problem, measures
its consequences for forecasts and the consequences of fixing it. The argument and every
experiment behind it are written up as a preprint, *Prophet's MAP
Estimate Is Not the Posterior Mode: A Non-Differentiable Changepoint Prior, Smooth
Optimizers, and an Exact Reformulation*, in
[`paper/`](https://github.com/adlyZaroui/analytic-prophet/tree/main/paper/). Every number
in it is generated from committed results, and `make reproduce` there reruns them all. A
link to the arXiv version will go here once it is posted.

To cite the software, use
[`CITATION.cff`](https://github.com/adlyZaroui/analytic-prophet/blob/main/CITATION.cff),
which is what GitHub's "Cite this repository" reads.

## Licence

MIT — see [LICENSE](https://github.com/adlyZaroui/analytic-prophet/blob/main/LICENSE).

Parts of this project are derived from [Prophet](https://github.com/facebook/prophet),
which is also MIT-licensed, copyright Facebook, Inc. and its affiliates; its licence is
[LICENSE-PROPHET](https://github.com/adlyZaroui/analytic-prophet/blob/main/LICENSE-PROPHET).
That covers the Python ported from Prophet's (marked `[fc]` where it follows it), the
copy of Prophet's Stan program and its modified versions in `paper/stan/`, and the
Peyton Manning series in `tests/data/`. The wheels carry both licences.
