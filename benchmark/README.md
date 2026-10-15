# Benchmarks

Three scripts that are fast enough to run against a change. For the claim-level
study — parameter recovery, held-out accuracy, cost across sizes, and the
generated report — see [`evaluation/`](../evaluation/), which takes minutes to
hours rather than seconds.

The division is deliberate: these answer "did that change break something", and
`evaluation/` answers "what can this project claim".

The premise of this project is Prophet's model fitted without Stan: its
gradient written out by hand rather than taken by automatic differentiation,
in-process rather than through a CmdStan subprocess, and its changepoint prior
solved through an exact reformulation that Prophet's optimizer does not use
(#23). Two consequences should follow, and these benchmarks exist to check them
rather than assume them.

- **Fitting time must not regress.** Writing the gradient out and starting no
  subprocess remove work; nothing here should make fitting slower. A slower fit
  is a defect, not a tradeoff.
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
| `prophet` | the original, via cmdstanpy: Stan's L-BFGS, or its Newton method below 100 observations; autodiff gradient |
| `python` | `fit(backend="python")`: scipy's L-BFGS-B over the split reformulation, or a projected Newton method below 100 observations; closed-form gradient |
| `python(numeric grad)` | same, but the gradient is approximated by finite differences |
| `compiled` | `fit(df)`: the compiled core, LBFGSpp's L-BFGS-B over the split reformulation, or a projected Newton method below 100 observations; closed-form gradient |

`python(numeric grad)` is not a serious candidate — it is included as the control
that shows what the analytic gradient buys, since it is the same optimizer on
the same problem with the gradient obtained the expensive way.

### Prophet runs on its own defaults, restated

`PROPHET_KWARGS` in `_common.py` restates Prophet's defaults rather than relying
on them, so that a change in Prophet's defaults shows up as a benchmark change
instead of silently moving the baseline. Both sides choose their seasonal
components from the history by Prophet's own rule (#16 task 3), so the
comparison is against Prophet as a user gets it.

It used to pin both sides to yearly seasonality at order 10, when that was the
only component this project could fit: against Prophet's defaults, a 26-column
design matrix would have been compared with a 20-column one and the difference
reported as performance.

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
attribute almost none of its memory to it — `peak_rss_bytes()` **sums**
`RUSAGE_SELF` with `RUSAGE_CHILDREN`. It took the maximum until #125, which is
not the same thing and here was not even close: the parent is 75–120 MiB of
interpreter, pandas and prophet against a cmdstan child peaking at 4–10 MiB, so
the maximum returned the parent every time and never counted the child at all.
The sum is an upper bound on simultaneous residency, since the two high-water
marks need not coincide, and the maximum a lower one. Each implementation is
measured twice, once with the fit skipped, so the reported "fit added" column is
the cost of fitting rather than of importing pandas. `ru_maxrss` is bytes on
macOS and kilobytes on Linux; `peak_rss_bytes()` normalizes that.

## Agreement benchmark

`benchmark_agreement.py` answers the question the other two cannot: how *close*
the fits are. It is also the project's acceptance criterion (issue #30), enforced
by `tests/test_prophet_agreement.py`.

Comparing **learned parameters** was the obvious design, and it measures the
wrong thing. The parameters are comparable: the Fourier features use Prophet's
1970 epoch (`seasonality.FOURIER_EPOCH`), and since #15 the changepoints are
Prophet's own. But they differ wherever Prophet stops short of the optimum,
which is everywhere it has been measured, and by how much depends on how flat
the objective is in each direction. A tolerance on them would measure that
flatness, not whether either fit is right.

Two criteria replace it:

1. **Posterior** — our optimum scored under **Stan's own log density**, via
   `CmdStanModel.log_prob`. One scalar, invariant to both reparameterizations,
   and the quantity the model is defined by. Both implementations are fitted on
   the same changepoints here, so this isolates optimizer quality.
2. **Predictions** — `yhat` over history plus a horizon, as a fraction of the
   series scale, each implementation in its *default* configuration.

The tolerance was set from measurement, not chosen: across every slice of the
Peyton Manning series past two years (T = 730 … 2905) the observed `yhat`
disagreement is 0.207%–0.537%, so the bound is **1%**.

### The identifiability caveat, and how it closed

This used to read: prediction agreement is asserted only on series with at
least **730 days** of history, because below it a 328-day slice diverged to
**111%** over a 30-day forecast with fitted `k` differing eightfold.

That was an artifact of the comparison, not of the model. Prophet's rule
disables yearly seasonality under 730 days; these benchmarks were forcing it on
*both* sides, because yearly was the only component this implementation could
fit. With `set_auto_seasonalities` implemented (#16 task 3) both sides fit
weekly-only at that length and agree to **0.709%**.

A milder version survives, and it is about the trend rather than the
seasonality. On short series the changepoint/rate decomposition is loose and
Prophet stops in a flatter region than we do, costing up to **1.216%** at
T = 500 against 0.207–0.537% past two years. It is not changepoint placement:
refitting on Prophet's own changepoints moves T = 500 from 1.216% to 1.207%.
Our posterior is the better one at every size measured, T = 100 through 2905,
so this is the same "Prophet stops short" story the posterior criterion exists
to detect — not either implementation being wrong.

## What the short-series regime costs now

This section used to record the Python path behaving badly below T = 100 — 2056
iterations at T = 50, and `ABNORMAL_TERMINATION_IN_LNSRCH` at T = 100. Stan's
convergence criteria (#21) and Prophet's changepoint placement (#15) both
removed that, and #25 then put Prophet's algorithm rule in: Newton below 100,
L-BFGS at or above, one Newton retry when L-BFGS fails.

What is left is a time cost, not an accuracy one. Newton reaches the same
optimum as L-BFGS at every size measured, but pays `2n` gradient evaluations per
iteration for its finite-difference Hessian:

| T | rule picks | iterations | `python` | `compiled` |
|---|---|---|---|---|
| 50 | Newton | 37 | 0.224s | 0.020s |
| 100 | L-BFGS | 157 | 0.028s | 0.009s |
| 300 | L-BFGS | 515 | 0.105s | 0.022s |

The compiled path at T = 50 is 0.12x Prophet's 0.172s, so the Hessian is
affordable where it matters. The Python path is the readable reference rather
than the deliverable, and Newton is where that shows most: at these sizes it is
~16x its own L-BFGS. `analytic=False` still selects finite differences under
Newton as it does under L-BFGS, which is why the numeric-gradient column jumps to
63x at T = 50 -- differencing the objective inside a method that then
differences the gradient.

The same column jumps further at T = 100, to about 30s. Since #229 the Python
path works in log(sigma_obs), and there finite-difference L-BFGS-B reaches the
analytic optimum and then fails its line search on differencing noise. Prophet's
rule retries with Newton, which lands on the same point; in sigma_obs the run had
stopped on `ftol` in half a second. At T = 2905 it now runs to its evaluation
limit instead of stopping on `ftol`, about 15s against 9.6s, and ends 0.67 nats
closer to the optimum. The analytic gradient is not affected: on the Python path,
#229 cut the iterations at four of these five lengths.
