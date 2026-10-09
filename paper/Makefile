# Build the preprint.  `make` regenerates the quoted numbers and the PDF.
#
#   make           numbers + pdf
#   make check     fail while any \pending result remains (the submission gate)
#   make arxiv     a source tarball for arXiv, refused while anything is pending
#   make clean

PYTHON ?= python
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all numbers pdf check arxiv clean

all: numbers pdf

numbers:
	$(PYTHON) experiments/quote_numbers.py

pdf: numbers
	$(LATEXMK) main.tex

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
	rm -f arxiv.tar.gz
