"""
Issue #177: every citation checked against its source, and kept checked.

`paper/citations.md` records, for each source, the passage the paper's claim
rests on. These tests keep the paper, its bibliography and that record from
drifting apart:

  * every key cited is defined, and every entry defined is cited;
  * every article and book carries a DOI, and the Stan manuals are cited at the
    version of CmdStan that Prophet bundles;
  * every source has its entry in the record;
  * **every quotation in the paper appears verbatim in the record**, so a quote
    cannot enter the text without its source passage being written down;
  * the two claims the check corrected do not come back.
"""
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
BIB = PAPER / "references.bib"
RECORD = PAPER / "citations.md"
SOURCES = [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]


def _tex():
    return "\n".join(path.read_text() for path in SOURCES)


def _entries():
    """{key: (type, body)} from references.bib."""
    text = BIB.read_text()
    return {key: (kind.lower(), body) for kind, key, body in
            re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S)}


def _cited():
    keys = set()
    for group in re.findall(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}", _tex()):
        keys |= {key.strip() for key in group.split(",")}
    return keys


def _normalise(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def test_every_cited_key_is_defined_and_every_entry_is_cited():
    defined, cited = set(_entries()), _cited()
    assert cited <= defined, f"cited but not defined: {sorted(cited - defined)}"
    assert defined <= cited, f"defined but never cited: {sorted(defined - cited)}"


def test_articles_and_books_carry_a_doi():
    for key, (kind, body) in _entries().items():
        if kind != "manual":
            assert re.search(r"doi\s*=\s*\{10\.\d{4,9}/[^}]+\}", body), f"{key} has no DOI"


def test_the_stan_manuals_are_cited_at_the_version_prophet_bundles():
    """The documentation changes between releases; the paper quotes 2.37, the
    CmdStan Prophet ships, and the runner (#183) asserts the same version."""
    spec = importlib.util.spec_from_file_location(
        "stan_runner", PAPER / "experiments" / "stan_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    major_minor = ".".join(runner.CMDSTAN_VERSION.split(".")[:2])
    manuals = {key: body for key, (kind, body) in _entries().items() if kind == "manual"}
    assert manuals
    for key, body in manuals.items():
        assert f"Version {major_minor}" in body, key
        assert f"/docs/{major_minor.replace('.', '_')}/" in body, key


def test_every_source_has_its_entry_in_the_record():
    record = RECORD.read_text()
    for key in _entries():
        assert f"**`{key}`**" in record, f"citations.md has no entry for {key}"


def _quotations():
    """Every ``...'' of three words or more in the paper. Shorter ones are
    scare quotes ("matter", "if"), not citations."""
    found = re.findall(r"``(.*?)''", _tex(), flags=re.S)
    return [quote for quote in found if len(quote.split()) >= 3]


def test_every_quotation_is_recorded_with_its_source():
    """A quotation in the paper must appear, verbatim up to case and line
    breaks, in citations.md -- which is where its source is named."""
    record = _normalise(RECORD.read_text())
    quotes = _quotations()
    assert len(quotes) >= 8
    missing = [quote for quote in quotes if _normalise(quote) not in record]
    assert not missing, f"quoted in the paper but not recorded: {missing}"


@pytest.mark.parametrize("phrase", [
    "documented for smooth objectives",
    "documented as methods for smooth",
    "documents its optimizers as methods for smooth",
])
def test_stan_is_not_said_to_document_its_optimizers_as_smooth_methods(phrase):
    """Stan's optimization chapter refers to Nocedal & Wright and says nothing
    about smoothness itself (citations.md, stan2025reference)."""
    for path in (*SOURCES, PAPER / "README.md"):
        assert phrase not in _normalise(path.read_text()), f"{path.name}: {phrase!r}"


def test_lewis_and_overton_are_not_cited_for_struggling_quasi_newton_methods():
    """They find BFGS converges on nonsmooth functions given a weak Wolfe line
    search; the claim that L-BFGS is ill-suited to an l1 term is Andrew & Gao's."""
    text = _normalise(_tex())
    for match in re.finditer(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{[^}]*lewis2013nonsmooth[^}]*\}", text):
        # the sentence the citation belongs to, from its start to the citation
        sentence = re.split(r"(?<=[.!?])\s", text[:match.start()])[-1]
        assert "struggle" not in sentence and "textbook" not in sentence, sentence
