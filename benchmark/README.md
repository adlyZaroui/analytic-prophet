# Benchmarks

The premise of this project is a Prophet that differs from the original in one
respect only: the gradient comes from a closed-form expression rather than
Stan's automatic differentiation. Two consequences should follow, and these
benchmarks exist to check them rather than assume them.

- **Fitting time must not regress.** Replacing autodiff with an analytic
  gradient removes work; there is no mechanism by which it should make fitting
  slower. A slower fit is a defect, not a tradeoff.
- **Memory should improve.** Reverse-mode autodiff retains a tape of the
  forward pass to reverse over; a closed-form gradient does not. If this
  project is not cheaper here, the analytic approach is not paying for itself.

## Running them

```bash
python benchmark/benchmark_fit_time.py                       # all sizes, best of 3
python benchmark/benchmark_fit_time.py --sizes 300 1000 --repeats 5
python benchmark/benchmark_memory.py --sizes 300 2905
```

Both run without `prophet` installed, reporting this project's paths alone and
saying so. For the comparison they exist for:

```bash
pip install prophet     # pulls cmdstanpy; needs a cmdstan toolchain
```

Neither script is collected by `pytest tests/` — they live outside `tests/` and
are not named `test_*`. They are slow and measure the machine, not correctness.

## What is compared

| name | what it runs |
|---|---|
| `prophet` | the original, via cmdstanpy (Stan's L-BFGS, autodiff gradient) |
| `fit(analytic=True)` | scipy L-BFGS-B over the split reformulation, closed-form gradient |
| `fit(numeric grad)` | same, but scipy approximates the gradient by finite differences |
| `fit_cpp` | the compiled core: liblbfgs OWL-QN, closed-form gradient |

`fit(numeric grad)` is not a serious candidate — it is included as the control
that shows what the analytic gradient buys, since it is the same optimizer on
the same problem with the gradient obtained the expensive way.

### Prophet is configured to match, not left on defaults

This matters more than it looks. Prophet's defaults add weekly seasonality
(Fourier order 3) and, on sub-daily data, daily seasonality. Benchmarking
against those would compare a 26-column design matrix against our 20-column one
and report the difference as performance. `PROPHET_KWARGS` in `_common.py` pins
Prophet to the model this project actually implements: additive, linear growth,
yearly seasonality only at order 10, MAP rather than MCMC, with matching
changepoint count, range and prior scales.

If this project grows weekly seasonality or holidays (issue #16), that
configuration must grow with it or the comparison quietly stops being fair.

## Method notes

**Timing.** A full `fit(df)` call is timed on each side, so dataframe setup,
scaling and changepoint placement are inside the measurement for everyone.
Compiling the C++ extension is not — that is a build step, done once before
timing starts. The first fit of each implementation is discarded, since it pays
one-off costs (imports, cmdstan model load, first-touch page faults) that say
nothing about steady-state speed. Best-of-N is reported rather than the mean:
the minimum is the least contaminated by scheduling noise.

**Memory.** Each measurement runs in a fresh subprocess, for two reasons. Peak
RSS is a high-water mark that never falls within a process, so several fits in
one process would each inherit the largest previous peak. And Prophet does the
real work in a cmdstan subprocess, so a measurement that ignored children would
attribute almost none of its memory to it — the child sums `RUSAGE_SELF` with
`RUSAGE_CHILDREN`. Each implementation is measured twice, once with the fit
skipped, so the reported "fit added" column is the cost of fitting rather than
of importing pandas. `ru_maxrss` is bytes on macOS and kilobytes on Linux;
`peak_rss_bytes()` normalizes that.

## Agreement benchmark

`benchmark_agreement.py` answers the question the other two cannot: how *close*
the fits are. It is also the project's acceptance criterion (issue #30), enforced
by `tests/test_prophet_agreement.py`.

Comparing **learned parameters** was the obvious design and does not work.
`beta` lives in a rotated Fourier basis — Prophet measures days from the 1970
epoch, this implementation from the series start — so the coefficients differ,
including in sign, while describing the same function. `delta` is indexed
against different changepoints. A test comparing them would measure the
parameterization, not the model.

Two criteria replace it:

1. **Posterior** — our optimum scored under **Stan's own log density**, via
   `CmdStanModel.log_prob`. One scalar, invariant to both reparameterizations,
   and the quantity the model is defined by. Both implementations are fitted on
   the same changepoints here, so this isolates optimizer quality.
2. **Predictions** — `yhat` over history plus a horizon, as a fraction of the
   series scale, each implementation in its *default* configuration.

The tolerance was set from measurement, not chosen: across every identified
slice of the Peyton Manning series (T = 730 … 2905) the observed `yhat`
disagreement is 0.320%–0.419%, so the bound is **1%**.

### The identifiability caveat

Prediction agreement is only asserted on series with at least **730 days** of
history, which is Prophet's own threshold for yearly seasonality being
identifiable. Below it, trend and seasonality trade off almost freely: on a
328-day slice the two implementations agree to 2.6% in-sample and then diverge
to **111%** over a 30-day forecast, with fitted `k` differing eightfold — while
*our* posterior is the better one. That is the model being under-determined,
not either implementation being wrong.

## An observation already worth recording

Short series behave badly in the Python path, which is the regime where Prophet
switches algorithms:

| T | iterations | time | scipy message |
|---|---|---|---|
| 50 | 2056 | 0.355s | CONVERGENCE |
| 100 | 3919 | 0.931s | **ABNORMAL** (line-search failure) |
| 300 | 91 | 0.046s | CONVERGENCE |

Prophet uses Newton below T=100 and falls back to Newton when L-BFGS raises; we
do neither. Tracked in issue #25.
