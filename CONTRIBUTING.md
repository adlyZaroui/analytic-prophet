# Contributing

This project has a specific way of working. It is not written down anywhere else, so a
contributor has no way to infer it — which is what this file is for.

## Building from source

Requires a C++17 compiler and two header-only libraries. Nothing is linked: the extension
needs the Eigen and LBFGSpp headers only, and the suite compiles
`analytic_prophet/optimize.cpp` into a temporary directory on the fly, which is why no
binary is checked in.

```bash
git clone https://github.com/adlyZaroui/analytic-prophet
cd analytic-prophet
brew install eigen lbfgspp          # or equivalent; header-only, nothing is linked
pip install -e '.[dev]'
pytest                              # the whole suite, about three minutes
```

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

The install is low-stakes because nothing in the repository depends on it — `pip install
-e .` is for importing the package from *elsewhere*, and the suite runs from a fresh clone
without it. `.[test]` is the same as `.[dev]` without `prophet`, which only the
comparisons need. The test count is deliberately not written down: CI reports it, and a
number in prose goes stale between the commit that adds tests and the one that remembers
to update it.

`prophet` itself is deliberately not a dependency — every comparison against the original
needs it, and it pulls `cmdstanpy` plus a compiled Stan model. The agreement tests skip
without it and the benchmarks print an install hint, so `pip install prophet` is only
needed to run those. `holidays` is required for `add_country_holidays` and imported
lazily, so nothing else needs it.

## Where things are cached

Everything this project caches hangs off **one root**, by the same rule on every
platform: `$ANALYTIC_PROPHET_CACHE`, else `$XDG_CACHE_HOME/analytic-prophet`, else
`~/.cache/analytic-prophet`. The compiled core goes in `build/` and the M4 corpus the
evaluation suite downloads goes in `m4/`, so one variable moves both and deleting the root
is how you start over ([#111](https://github.com/adlyZaroui/analytic-prophet/issues/111)).

From the source distribution or a clone, the first `fit(df)` compiles `optimize.cpp`, says
so rather than appearing to hang, and caches the result
([#108](https://github.com/adlyZaroui/analytic-prophet/issues/108)). The cache is keyed by a
digest of the source and the compile command, so editing `optimize.cpp` rebuilds and
nothing else does. Without a compiler the build raises, naming the one thing that is
missing, and the tests that need the toolchain skip rather than fail.

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
paper/                the preprint, its experiments and their committed results
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

The C++ core builds itself on first use and caches the result. `pytest` needs no install
and no `PYTHONPATH`: `pyproject.toml` puts the repo root, `benchmark/` and `evaluation/` on
`pythonpath`, so a fresh clone runs the suite as it stands.

Tests that need a toolchain **skip** rather than fail when it is absent, so a green run on
a machine without a compiler is green for less than it looks. That is why CI makes them
strict rather than leaving it to the caller — see below.

## Continuous integration

Two workflows, under [`.github/workflows/`](.github/workflows):

- **`tests.yml`**, on every push and pull request: the suite across Python 3.9–3.14,
  which is what gives `requires-python = ">=3.9"` any basis — before it, the suite had
  only ever run on one version. One further job installs `prophet` and runs the
  comparisons against the original; it is the only one that pays for cmdstan.
- **`evaluation.yml`**, manual or monthly: `evaluation/run.py` and a regenerated report,
  uploaded as an artifact rather than committed. Tier 0 is a gate and fails the job.
  These tiers are deliberately not per-push — Tier 2 alone is about eleven minutes and
  needs the network.

Every CI job passes `--require-cpp` and `--require-prophet`, so a skip there would mean
CI reported green for a run that never built the C++ core. Every run also prints what it
skipped, grouped by reason, into the job summary — "green" has to be readable.

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
