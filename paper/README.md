# Prophet does not solve the problem it specifies

A preprint for **arXiv stat.CO**: a numerical-software case study with a
mechanism, controls and a fix.

> Prophet specifies a posterior whose optimum sits on a non-differentiable set,
> delegates optimization to solvers designed for smooth objectives, and
> consequently does not solve the problem it specifies. This is diagnosable, has
> user-visible consequences reported in the issue tracker for years, is removable
> by an exact reformulation, and the consequences on the optimum and predictive
> performance are discussed here.

The forecast-accuracy results corroborate that the gap matters a little in
practice; they are not the thesis.

## Building

Needs a TeX Live with `latexmk`, `pdflatex` and BibTeX — MacTeX provides all of
them — and the repository's Python environment.

```bash
cd paper
make            # regenerate the quoted numbers, then build main.pdf
make check      # fail while any result is still marked pending
make arxiv      # source tarball for arXiv; refused while anything is pending
```

`make` builds from the committed results, in seconds. **`make reproduce`
regenerates the results themselves first**: every script under `experiments/`,
in the order the Makefile lists them, and the evaluation tiers the paper quotes
(0 and 2), then the numbers and the PDF. It takes hours, almost all of them the
two passes over the census (`experiments/kkt.py` and Tier 2). It needs CmdStan
2.37.0 (below), and fetches the M4 files on first use (`evaluation/corpora.py`).
Run in the pinned environment on the platform the results record (each
`.meta.json` names it), it regenerates what is committed, timings aside. On
another platform Stan's L-BFGS can stop at slightly different points (#186), and
the tests that refit a committed result allow for that.

```bash
pip install -r paper/requirements.txt && pip install -e .
cd paper && make reproduce
```

`requirements.txt` is the environment every committed result was produced in.
Each result's `.meta.json` records the versions it ran with, and
`tests/test_paper_reproducibility.py` holds every record to the pins, so a
result regenerated under a drifted numpy shows up as a failing test rather than
as a quietly different number.

The PDF prints the commit it was built from. `make` writes it to
`generated/build.tex` at every build, marked `+modified` when the working tree
differs from that commit; the file is not committed, because a file cannot hold
the hash of the commit that contains it.

## Where the numbers come from

**None is typed into a `.tex` file.** `experiments/quote_numbers.py` reads the
committed evaluation results and writes `generated/numbers.tex`, a set of
`\newcommand` macros that the prose uses. Regenerate an experiment and the
sentences that quote it follow; leave a result uncomputable and the document
fails to compile rather than printing a stale value. This is the rule
`evaluation/report.py` already follows, and the one whose absence turned the
suite red when the 3,008-series census replaced the 36-series sample.

The paper's own experiments, as opposed to the evaluation suite's, commit their
output to `results/` and are read from there: the first-order certificate (#170)
is `experiments/kkt.py`, writing `results/kkt_census.csv` and
`results/kkt_peyton_manning.csv`. It is resumable and parallel, and took 37 minutes
on eight workers. The margin against series length (#173) is
`experiments/margin.py`, writing `results/margin.csv` and
`figures/margin_vs_length.pdf` in about three minutes; `--draw` redraws the figure
from the committed CSV without refitting. The natural-vs-split Newton control (#174) is
`experiments/newton.py`, writing `results/newton.csv` and `results/newton_traces.csv` in
about two minutes. The limitations (#178) add two: `experiments/overhead.py` times the
cost of calling CmdStan against whole fits (`results/overhead.csv`, under a minute), and
`experiments/m4_calendar.py` records what M4's info file holds and how the census's
starting dates are distributed (`results/m4_calendar.csv`; needs the M4 cache). The
smoothing control (#168) is `experiments/smoothing.py`: Prophet's Stan program with the
prior smoothed, fitted by Stan's L-BFGS across 17 values of ε at three lengths, writing
`results/smoothing.csv` and `figures/smoothing.pdf` in about two minutes (needs CmdStan
2.37.0, see below). The prophet#842 reproducer (#172) is `experiments/issue842.py`: the
thread's ten-point reprex on current Prophet, this implementation and the smoothed
program, plus the variants the thread lists, writing `results/issue842.csv` and
`results/issue842_variants.csv` in under half a minute. The τ solution path (#171) is
`experiments/tau_path.py`: λ = 1/τ swept for Prophet and for the exact path, σ fixed and
free, with a zoom, a bisection of each path's largest step and prophet#1280's change,
writing `results/tau_path.csv`, `tau_refine.csv`, `tau_pairs.csv` and
`figures/tau_path.pdf` in about seven minutes on eight workers. The multi-start control
(#169) is `experiments/multistart.py`: Prophet's L-BFGS from 50 seeded random starts at
three lengths, σ fixed and free, against this implementation's certified optimum, writing
`results/multistart.csv` in about five minutes.

Four more make the results the paper used to state without a script regenerable (#175).
`experiments/flat.py` is the flat-growth negative control (`results/flat.csv`, seconds).
`experiments/tolerance.py` reruns Prophet's L-BFGS with Stan's five stopping tolerances at
machine precision, and through `Prophet.fit` with the same arguments
(`results/tolerance.csv`, a minute). `experiments/strength.py` sweeps the prior's strength
for SciPy's unbounded L-BFGS-B against an exact, certified optimum
(`results/strength.csv`, two minutes). It replaces a sweep on liblbfgs, the original
backend, which cannot be rerun and which the paper does not quote.
`experiments/rounding.py` measures what CmdStan's eight printed digits do to Prophet's
residual on 200 random census series (`results/rounding.csv`, four minutes on four
workers; needs the M4 cache). No number is typed into a section unless
`tests/test_paper_numbers.py` lists it as a constant, such as Prophet's defaults or the
lengths measured.

The fix restated inside Stan (#202) is `experiments/inside_stan.py`: Hoff's Hadamard
parametrisation and the split declared `<lower=0>`, each fitted by Stan's own L-BFGS
with Prophet's settings, from Prophet's start and #169's seeded starts, at three lengths
with σ fixed and free, writing `results/inside_stan.csv` in about two minutes (needs
CmdStan 2.37.0, below).

## Results not yet measured

A `\pending{#issue}{…}` marker renders in red where the paper states something
it has not yet measured, naming the issue that will. `make check` lists them and
`make arxiv` refuses to package a draft that still has any. The experiments are
tracked under the [`paper`](https://github.com/adlyZaroui/analytic-prophet/labels/paper)
label. None remain: the last, #180, was this reproducibility machinery, and
`tests/test_paper_reproducibility.py` now fails if a marker comes back.

## Layout

```
main.tex              preamble, title, abstract
sections/             one file per section
references.bib        bibliography
experiments/          the scripts behind every number and figure
stan/                 Prophet's Stan program and the modified copies (#183)
generated/            their output — macros and tables the .tex includes, and the
                      build commit `make` writes (not committed)
requirements.txt      the environment every committed result was produced in
figures/              generated figures
```

## Modified copies of Prophet's Stan program

Four experiments change Prophet's program rather than call it: the
smooth-surrogate control (#168), the fixed-σ halves of the multi-start (#169)
and the τ path (#171), and the fix restated inside Stan (#202). Stan cannot hold a
parameter fixed at run time, so each needs a different program. They live in `stan/`, beside Prophet's own
`prophet.stan` copied verbatim, so that each diff is its documentation:

| program | the change |
|---|---|
| `prophet_fixed_sigma.stan` | `sigma_obs` moved from `parameters` to `data` |
| `prophet_smooth.stan` | `\|δ\|` replaced by `√(δ² + ε²) − ε`, with `ε` as data |
| `prophet_fixed_sigma_smooth.stan` | both |
| `prophet_hadamard.stan` | `δ = u ⊙ v`, and `\|δ\|/τ` replaced by `(‖u‖² + ‖v‖²)/(2τ)`: smooth, the same minimisers (#202) |
| `prophet_split.stan` | `δ = δ⁺ − δ⁻`, both `<lower=0>`, the prior linear in them: the paper's fix as Stan has to state it (#202) |
| `prophet_fixed_sigma_hadamard.stan`, `prophet_fixed_sigma_split.stan` | each with `sigma_obs` as data |

`experiments/stan_runner.py` compiles them and fits them the way
`CmdStanPyBackend.fit` does — the data and starting point captured from a real
`Prophet.fit`, Newton below 100 observations and L-BFGS above, 10⁴ iterations,
the Newton fall-back — so the program is the only thing that differs. Executables
are cached under `~/.cache/analytic-prophet/stan/`, keyed by a hash of the
source.

They need **CmdStan 2.37.0**, the version Prophet bundles, and the runner refuses
any other: a different version changes the math library, and then #168 no longer
varies only the smoothness. The CmdStan inside the `prophet` wheel runs Prophet's
model but cannot compile another, so it has to be installed:

```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.37.0')"
```

`tests/test_paper_stan_variants.py` checks, before any variant is trusted, that
the unmodified program compiled here reproduces Prophet's binary — the same
optimum, `lp__` and gradient — and that each variant changes the density by
exactly what its diff says. Those tests skip without CmdStan.

Every program here keeps Meta's header, which licenses it under "the MIT license
found in the LICENSE file in the root directory of this source tree". In this
repository that licence is [`LICENSE-PROPHET`](../LICENSE-PROPHET); the root
`LICENSE` is this project's own (#199).

Every experiment for the paper lives under `experiments/`. The one exception is
the evaluation corpus (#164), which also serves the general evaluation of this
implementation and stays in `evaluation/`.

## Archiving the software, once the results are final

arXiv assigns the paper a DOI on announcement. The software it measured gets
its own, so that the reproducibility section can cite an archived version
rather than a commit on a host that can change. That is done once, after every
result is final, and from the author's accounts:

1. enable this repository in Zenodo's GitHub integration;
2. publish a tagged GitHub release of the commit the numbers were generated
   from — the commit `make reproduce` ran at, which the PDF prints;
3. Zenodo archives the release and issues a DOI from `CITATION.cff`;
4. cite that DOI in the reproducibility section, and rebuild.

The release has to come after the numbers are final, so that the archived
version is the one measured.

## Register

Prophet's authors specified a model in Stan and delegated optimization; Stan
describes its optimizers by reference to a text on smooth optimization, and warns
about absolute values of parameters in its functions reference rather than on the
distribution Prophet uses. Nobody made an error, and the paper says so. Each
upstream issue is cited for what its thread shows — read in full, and never as a
report of non-differentiability, which none of them is. `citations.md` records, for
every source, the passage each claim rests on (#177).
