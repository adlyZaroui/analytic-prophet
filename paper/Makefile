# Build the preprint.  `make` regenerates the quoted numbers and the PDF from the
# committed results; `make reproduce` regenerates the results themselves first.
#
#   make             numbers + pdf, from the committed results (seconds)
#   make reproduce   every experiment and the evaluation tiers the paper quotes,
#                    then numbers + pdf -- hours, almost all of them the census
#   make check       fail while any \pending result remains (the submission gate)
#   make arxiv       a source tarball for arXiv, refused while anything is pending
#   make clean

PYTHON ?= python
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all numbers build pdf reproduce evaluation experiments check arxiv clean

all: numbers pdf

numbers:
	$(PYTHON) experiments/quote_numbers.py

# The commit this PDF is built from, printed in it. Written at every build and
# not committed: a file cannot hold the hash of the commit that contains it.
# "+modified" marks a build from a working tree that differs from that commit.
build:
	@commit=$$(git rev-parse --short=12 HEAD); \
	if [ -n "$$(git status --porcelain --untracked-files=no -- ..)" ]; then \
	  commit="$$commit+modified"; fi; \
	printf '%% Written by make: the commit this PDF was built from.\n\\newcommand{\\BuildCommit}{%s}\n' \
	  "$$commit" > generated/build.tex; \
	echo "building from $$commit"

pdf: numbers build
	$(LATEXMK) main.tex

# One command for everything the paper quotes (#180). Each line writes the
# results and figures named in its script's docstring; tests/test_paper_reproducibility.py
# checks that every script under experiments/ is here and that the evaluation
# tiers are the ones quote_numbers.py reads.
reproduce: evaluation experiments all

evaluation:
	cd .. && $(PYTHON) evaluation/run.py --tiers 0 2

experiments:
	$(PYTHON) experiments/kkt.py
	$(PYTHON) experiments/kkt.py --detail
	$(PYTHON) experiments/agreement.py
	$(PYTHON) experiments/margin.py
	$(PYTHON) experiments/newton.py
	$(PYTHON) experiments/overhead.py
	$(PYTHON) experiments/m4_calendar.py
	$(PYTHON) experiments/smoothing.py
	$(PYTHON) experiments/inside_stan.py
	$(PYTHON) experiments/issue842.py
	$(PYTHON) experiments/tau_path.py
	$(PYTHON) experiments/multistart.py
	$(PYTHON) experiments/flat.py
	$(PYTHON) experiments/tolerance.py
	$(PYTHON) experiments/strength.py
	$(PYTHON) experiments/rounding.py

# A \pending macro marks a result the paper states but has not measured. It
# renders in red so a draft is honest about its gaps; this refuses to call a
# draft finished while any are left.
check:
	@remaining=$$(grep -ho '\\pending{[^}]*}' main.tex sections/*.tex | sort | uniq -c); \
	if [ -n "$$remaining" ]; then \
	  echo "unmeasured results still in the paper:"; echo "$$remaining"; exit 1; \
	else echo "no pending results"; fi

arxiv: check pdf
	rm -f arxiv.tar.gz
	tar czf arxiv.tar.gz main.tex main.bbl references.bib sections generated figures
	@echo "wrote arxiv.tar.gz"

clean:
	latexmk -C main.tex
	rm -f arxiv.tar.gz generated/build.tex
