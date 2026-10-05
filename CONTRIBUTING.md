# Contributing

This project has a specific way of working. It is not written down anywhere else, so a
contributor has no way to infer it — which is what this file is for.

## The loop

Every change in this repository has gone through the same sequence:

1. **Read the issue.** Issues here carry the measurement and the reasoning, not just the
   request. Several have been closed by arguing they were ill-posed; one was closed
   because the concept it asked about ("the right number of changepoints") turned out not
   to be well defined. Reading it is the first step, and disagreeing with it is allowed.
2. **Branch.** One branch per issue, from `main`. Never stack a branch on another branch:
   it has been done twice here, and both times the work merged into a sibling branch
   instead of `main` and needed a second PR to rescue it.
3. **Fix it.**
4. **Run the whole suite** — `pytest` — and **run the benchmarks**. The suite does not run
   the benchmark scripts, so an import error in `benchmark/` passes every test and fails
   the moment somebody runs one. That has happened.
5. **Open a pull request** describing what was measured, not only what was changed.
6. **Do not merge your own PR.**

## What a change is expected to carry

- **A test that fails before it and passes after.** Where the change is a claim about
  behaviour, the test asserts the behaviour; where it is a claim about a number, the test
  recomputes the number from committed data rather than restating it.
- **Mutation-checking for anything whose job is to catch drift.** A guard that cannot be
  shown to fail is a guard nobody has tested. Break it deliberately, watch it fail, put it
  back.
- **The reason, in the code.** Comments here say *why*, and `[fc]` marks a decision copied
  from Prophet's `forecaster.py` — which is most of them, because the point of the project
  is to reproduce that model exactly. If you diverge, say so and say why; `docs/deviations.md`
  is where deliberate divergences are recorded, and there is a test that the list stays true.
- **Honest reporting.** If a benchmark got slower, the PR says so. If a number moved, every
  document quoting it moves with it.

## Running things

```bash
pytest                              # the whole suite, no install needed
pytest --require-cpp                # fail rather than skip if the C++ core cannot build
python benchmark/benchmark_fit_time.py
python evaluation/run.py            # the claim-level study; minutes to hours
```

The C++ core builds itself on first use and caches the result. `pytest` needs no install:
`pyproject.toml` puts the repo root, `benchmark/` and `evaluation/` on `pythonpath`.

Tests that need a toolchain **skip** rather than fail when it is absent, so a green run on
a machine without a compiler is green for less than it looks. CI passes `--require-cpp` and
`--require-prophet` to turn those skips into failures, and prints what was skipped and why.

## Measurements

Numbers in this repository are reproducible or they are not claims. The evaluation suite
seeds both implementations — including Prophet's intervals, which come from numpy's global
generator — writes its results to `evaluation/results/`, and commits them, so re-running is
a reviewable diff rather than an act of faith. The committed results were produced with the
Prophet version pinned in `pyproject.toml`; changing it means regenerating them.

If you change something that moves a published number, regenerate the affected tier and
update every document that quotes it. There are tests that recompute the README's figures
from the committed results, so a stale number fails the suite rather than sitting there.

## Scope

- **No MCMC**, and no plotting of the model. Both are stated non-goals, not oversights.
- **Prophet's names**, everywhere there is a counterpart. `docs/deviations.md` lists the two
  that have none and why, and `tests/test_prophet_naming.py` fails if that stops being true.
- **Reject rather than ignore.** An argument this implementation cannot honour raises; it
  does not quietly do nothing. That rule is why the constructor refuses `mcmc_samples` and
  why `fit` refuses a backend's arguments under the other backend.
