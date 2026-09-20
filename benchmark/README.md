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

## Planned: agreement benchmark

A third benchmark belongs here and is not written yet: how *close* the fits are,
which is the question the other two cannot answer. Two levels, and they can
disagree:

1. **Learned parameters** — `k`, `m`, `delta`, `beta`, `sigma_obs` against
   Prophet's, from the same data and the same seeds. The strict test.
2. **Predictions** — `yhat` and the trend/seasonality components over a forecast
   horizon. The one that matters in practice: parameters can differ in a flat
   direction of the posterior while predictions agree closely.

The tolerance for both is an open question, deliberately. It should be set from
measurement rather than picked in advance.

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
