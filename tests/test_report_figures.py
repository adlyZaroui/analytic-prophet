"""
Issue #99: the report's plots, and what they say they measure.

Five figures, every one under-labelled, two actively misleading, and the
measurement the most-quoted table rests on with no plot at all. Nothing in
them plotted the wrong numbers -- the problem was that a reader could not tell
what the numbers were.

The worst was `tier0_parity.png`: a log-log scatter titled "above the line is
better" whose three points sat exactly on the line, because the margin is
0.004%-0.24% of the level. It asserted a claim and displayed its negation.

These tests cannot check that a figure is legible. What they can check is that
every figure has a caption, that the caption names the corpus, the quantity
and which direction is good, and that the figure the suite stopped having --
inference time -- is drawn. The legibility fixes are visible in the committed
PNGs and were judged by looking at them.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
RESULTS = REPO / "evaluation" / "results"
FIGURES = RESULTS / "figures"
REPORT = RESULTS / "report.md"

EXPECTED = (
    "tier0_parity.png",
    "tier1_recovery.png",
    "tier2_coverage.png",
    "tier2_accuracy.png",
    "tier3_cost.png",
)


@pytest.fixture(scope="module")
def report():
    return REPORT.read_text()


@pytest.fixture(scope="module")
def captions(report):
    """{figure file: the paragraph under it}."""
    found = {}
    for match in re.finditer(r"!\[[^\]]*\]\(figures/([^)]+)\)\n\n([^\n]+)\n", report):
        found[match.group(1)] = match.group(2)
    return found


@pytest.mark.parametrize("figure", EXPECTED)
def test_the_figure_is_committed(figure):
    target = FIGURES / figure
    assert target.exists(), f"{figure} is not committed"
    assert target.stat().st_size > 10_000, f"{figure} is suspiciously small"


@pytest.mark.parametrize("figure", EXPECTED)
def test_the_figure_has_a_caption(figure, captions):
    """A figure lifted out of the report should still say what it is."""
    assert figure in captions, f"{figure} is shown with no caption under it"
    assert len(captions[figure].split()) > 25, f"{figure}'s caption is a stub"


@pytest.mark.parametrize("figure", EXPECTED)
def test_the_caption_says_which_direction_is_good(figure, captions):
    """A reader should not need the surrounding paragraph to know."""
    caption = captions[figure].lower()
    assert any(phrase in caption for phrase in
               ("is better", "lower is better", "nearer")), \
        f"{figure}'s caption never says which direction is good"


@pytest.mark.parametrize("figure,wanted", [
    ("tier0_parity.png", ("peyton manning", "log_prob", "changepoints")),
    ("tier1_recovery.png", ("synthetic", "generating", "nats")),
    ("tier2_coverage.png", ("m4", "generate_cutoffs", "80%")),
    ("tier2_accuracy.png", ("per-series", "median", "p-value")),
    ("tier3_cost.png", ("peyton manning", "wall clock", "rusage")),
])
def test_the_caption_names_the_corpus_and_the_quantity(figure, wanted, captions):
    caption = captions[figure].lower()
    for phrase in wanted:
        assert phrase in caption, f"{figure}'s caption does not mention {phrase!r}"


def test_inference_time_is_plotted(report):
    """It was measured as a four-way comparison and plotted nowhere, though it
    is the one place the two implementations differ in kind rather than
    degree."""
    source = (REPO / "evaluation" / "report.py").read_text()
    assert "predict_wall" in source, "the cost figure does not plot inference time"
    assert "compiled(exact)" in source and "prophet(exact)" in source, \
        "the inference panel is not the four-way comparison"

    caption = next(c for f, c in
                   re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)\n\n([^\n]+)\n", report)
                   if f == "tier3_cost.png")
    assert "four paths" in caption.lower() or "all four" in caption.lower()


def test_the_parity_figure_plots_the_difference_not_the_pair():
    """The figure's own axes have to be able to resolve the claim it makes.

    A parity scatter of 813 against 815 cannot, which is why that one showed
    three points on the diagonal under a title saying otherwise.
    """
    source = (REPO / "evaluation" / "report.py").read_text()
    body = source[source.index("def _parity_plot"):source.index("def tier1_section")]
    assert "bar(" in body, "the parity figure is not a difference plot"
    assert "lp__(ours)" in body or "lp__(ours) −" in body or "nats" in body
    assert "xscale=\"log\"" not in body, "a log scale cannot resolve a 0.004% margin"


def test_the_memory_axis_names_the_quantity():
    """"fit added, MiB" meant "peak resident memory attributable to the fit,
    over and above what importing the library cost" -- the distinction is the
    entire measurement and the label hid it."""
    source = (REPO / "evaluation" / "report.py").read_text()
    # the assignment, not the prose: the docstring quotes the old label in
    # order to explain why it is gone, and a plain substring search finds that
    assert 'ylabel="fit added, MiB"' not in source
    assert "peak resident memory added by the fit" in source


def test_every_figure_the_report_shows_exists(report):
    """A broken image in the most-read document in the repository."""
    for name in re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)", report):
        assert (FIGURES / name).exists(), f"the report shows {name}, which is missing"


def test_no_figure_is_committed_that_the_report_never_shows(report):
    """The other direction: a stale PNG nothing references is a figure
    somebody will find and believe."""
    shown = set(re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)", report))
    committed = {p.name for p in FIGURES.glob("*.png")}
    # the README's showcase is deliberately not part of the report
    orphans = committed - shown - {"showcase.png"}
    assert orphans == set(), f"committed but shown nowhere: {sorted(orphans)}"
