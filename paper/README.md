# Prophet does not solve the problem it specifies

A preprint for **arXiv stat.CO**: a numerical-software case study with a
mechanism, controls and a fix.

> Prophet specifies a posterior whose optimum sits on a non-differentiable set,
> delegates optimization to solvers documented for smooth objectives, and
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

## Where the numbers come from

**None is typed into a `.tex` file.** `experiments/quote_numbers.py` reads the
committed evaluation results and writes `generated/numbers.tex`, a set of
`\newcommand` macros that the prose uses. Regenerate an experiment and the
sentences that quote it follow; leave a result uncomputable and the document
fails to compile rather than printing a stale value. This is the rule
`evaluation/report.py` already follows, and the one whose absence turned the
suite red when the 3,008-series census replaced the 36-series sample.

## Results not yet measured

A `\pending{#issue}{…}` marker renders in red where the paper states something
it has not yet measured, naming the issue that will. `make check` lists them and
`make arxiv` refuses to package a draft that still has any. The experiments are
tracked under the [`paper`](https://github.com/adlyZaroui/analytic-prophet/labels/paper)
label.

## Layout

```
main.tex              preamble, title, abstract
sections/             one file per section
references.bib        bibliography
experiments/          the scripts behind every number and figure
generated/            their output — macros and tables the .tex includes
figures/              generated figures
```

Every experiment for the paper lives under `experiments/`. The one exception is
the evaluation corpus (#164), which also serves the general evaluation of this
implementation and stays in `evaluation/`.

## Register

Prophet's authors specified a model in Stan and delegated optimization; Stan
documents its optimizers as smooth methods. Nobody made an error, and the paper
says so. Each upstream issue is cited for what its thread shows — read in full,
and never as a report of non-differentiability, which none of them is.
