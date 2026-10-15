"""
Issue #207: the related work a reviewer will raise, each source cited for what
it shows.

Three sources were missing (paper/citations.md has each passage):

  * **Hoff's Hadamard product parametrisation**, with Kolb et al. and Poon &
    Peyre: the same minimisers, smooth -- cited for that, and set beside what
    #202 measured when the reparametrisations ran on Stan's own L-BFGS;
  * **a 2017 reply on the Stan users' list**, reposted on Andrew Gelman's blog:
    the kink and the missing exact zeros were known in Stan's community, so the
    introduction says so;
  * **Prophet's 2023 plan**, verified for #208 and #212 and recorded, not yet
    cited.

`tests/test_paper_citations.py` already holds every quotation to the record and
every entry to a DOI or URL. What is checked here is where each source is cited
and that the sentences around it are true of the committed results.
"""
import csv
import re
import statistics
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
SECTIONS = PAPER / "sections"
RECORD = PAPER / "citations.md"


def _flat(path):
    return re.sub(r"\s+", " ", path.read_text())


def _paragraph(path, title):
    text = _flat(path)
    start = text.index(f"\\paragraph{{{title}}}")
    end = text.find("\\paragraph{", start + 1)
    return text[start:end if end > 0 else None]


def _sentence_citing(text, key):
    match = re.search(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{[^}]*" + key, text)
    assert match, key
    start = max(text.rfind(". ", 0, match.start()), 0)
    end = text.find(". ", match.end())
    return text[start:end if end > 0 else None]


def test_stans_community_is_cited_where_the_paper_disclaims_novelty():
    """'What it does not claim' must say the phenomenon was known in Stan's own
    community, with the thread's words."""
    paragraph = _paragraph(SECTIONS / "introduction.tex", "What it does not claim.")
    assert "\\citet{carpenter2017lasso}" in paragraph
    assert "Stan's own community" in paragraph
    for words in ("the posterior isn't continuously differentiable",
                  "straight-up optimization is unlikely to find estimates of exactly zero"):
        assert words in paragraph, words


def test_the_reparametrisations_are_cited_in_related_work():
    paragraph = _paragraph(SECTIONS / "related.tex",
                           "Smooth reparametrisations, run inside Stan.")
    for key in ("hoff2017lasso", "kolb2026smoothing", "poon2021smooth"):
        assert key in paragraph, key
    # #202's result enters as generated macros, never typed. The split's figures
    # moved to the fix's implementation paragraph (#204), where they are the
    # reason its solver runs outside Stan: tests/test_paper_order.py.
    for macro in ("InsideRuns", "InsideBest", "InsideHadamardStartMin",
                  "InsideHadamardStartMax", "InsideHadamardRandomMedianLong"):
        assert f"\\{macro}" in paragraph, macro
    assert "\\code{paper/experiments/inside_stan.py}" in paragraph


def test_each_reparametrisation_source_is_cited_for_what_it_shows():
    """#177's rule. Poon & Peyre's landscape result is for least squares, after
    one factor is eliminated; Kolb et al.'s remark on zeros is about gradient
    methods; Hoff's stationary point is the reason a start must be chosen."""
    text = _flat(SECTIONS / "related.tex")
    poon = _sentence_citing(text, "poon2021smooth")
    assert "eliminate one factor" in poon
    assert "least-squares" in _paragraph(SECTIONS / "related.tex",
                                         "Smooth reparametrisations, run inside Stan.")
    # their remark on zeros is made of (S)GD, which has no proximal step
    zeros = text.index("the iterates can only asymptotically approach")
    assert "of gradient methods on the product form, which have no proximal step" in \
        text[text.rfind(". ", 0, zeros):zeros]
    assert "stationary point that is not a minimum" in text


def _inside():
    with open(PAPER / "results" / "inside_stan.csv", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["variant"] != "reference"]


def test_the_typed_sentences_about_202_are_true_of_its_result():
    """'neither reaches the optimum', 'No run sets a single rate exactly to
    zero', 'falls further short than Prophet's own L-BFGS from the same starts'
    -- checked against the committed CSV, so a rerun that changed any of them
    fails here."""
    text = _flat(SECTIONS / "related.tex")
    rows = _inside()
    assert {row["variant"] for row in rows} == {"hadamard", "split"}

    assert "neither reaches the optimum" in text
    assert min(float(row["shortfall"]) for row in rows) > 5e-4
    assert "No run sets a single rate exactly to zero" in text
    assert all(int(row["exact_zeros"]) == 0 for row in rows)

    assert ("falls further short than Prophet's own L-BFGS from the same starts, in median, "
            "at every length and setting") in text
    with open(PAPER / "results" / "multistart.csv", newline="") as handle:
        multistart = list(csv.DictReader(handle))
    random_starts = {row["start"] for row in rows if row["start"] != "default"}
    lengths = {row["observations"] for row in rows}
    assert len(lengths) == 3
    for size in lengths:
        for setting in ("fixed", "free"):
            prophet = [float(row["shortfall"]) for row in multistart
                       if row["observations"] == size and row["setting"] == setting
                       and row["implementation"] == "prophet" and row["start"] in random_starts]
            hadamard = [float(row["shortfall"]) for row in rows
                        if row["observations"] == size and row["setting"] == setting
                        and row["variant"] == "hadamard" and row["start"] in random_starts]
            assert prophet and len(prophet) == len(hadamard), (size, setting)
            assert statistics.median(hadamard) > statistics.median(prophet), (size, setting)


def test_what_is_not_new_matches_the_contributions_list():
    """#205 retired 'the attribution of long-standing user reports' from the
    introduction; related work said the same thing in other words."""
    paragraph = _paragraph(SECTIONS / "related.tex", "What is not new.")
    assert "attribution" not in paragraph
    assert "a reading of its long-standing user reports against the mechanism" in paragraph


def test_the_sources_read_from_an_archive_say_so():
    """Two pages refuse automated readers, and their text was read from the
    Internet Archive. The record says which snapshot."""
    record = RECORD.read_text()
    for key, snapshot in (("carpenter2017lasso", "web.archive.org/web/20201125221139/"),
                          ("duong2023prophet", "web.archive.org/web/20230317232722/")):
        entry = record[record.index(f"**`{key}`**"):]
        entry = entry[:entry.find("\n**`", 1) if "\n**`" in entry[1:] else None]
        assert snapshot in entry, key


def test_prophets_2023_plan_is_recorded_verbatim_and_kept_conditional():
    """#208 and #212 will cite the plan. The sentence is recorded in full, and
    wherever the paper or a README quotes it, 'at this moment' stays with it --
    the plan is a statement of intent, not a policy."""
    sentence = ("There have been ideas for further enhancements to the model, but at this "
                "moment we do not plan to make any more changes to the underlying Stan model.")
    assert sentence in re.sub(r"\s+", " ", RECORD.read_text())
    phrase = "any more changes to the underlying Stan model"
    for path in (PAPER / "main.tex", *sorted(SECTIONS.glob("*.tex")),
                 PAPER / "README.md", REPO / "README.md"):
        text = _flat(path)
        for match in re.finditer(re.escape(phrase), text):
            assert "at this moment" in text[max(0, match.start() - 120):match.start()], path.name


def test_the_source_considered_and_not_cited_is_recorded_with_the_reason():
    record = RECORD.read_text()
    assert "Considered and not cited: Ziyin & Wang (ICML 2023)" in record
    assert "ziyin" not in (PAPER / "references.bib").read_text().lower()
