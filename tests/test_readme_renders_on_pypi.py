"""
Issue #122: the README is the PyPI page, and PyPI is not GitHub.

`pyproject.toml` ships `README.md` as the long description, so the file is
rendered on https://pypi.org/project/analytic-prophet/ as well as on GitHub.
The two resolve relative links differently: GitHub resolves them against the
repository, and PyPI prepends its own base, turning `docs/model.md` into
`https://pypi.org/project/analytic-prophet/docs/model.md`. A 404.

Nineteen links and the showcase image were relative when the package was
first published, so every `→` pointer in the body was dead for a PyPI reader
-- including the one backing the central claim -- and the most persuasive
thing in the README did not render at all.

Absolute URLs work on both, so there is one README and these tests keep it
that way. They are mechanical on purpose: the next person to add a link will
write a relative one, because that is what works in the editor.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
README = REPO / "README.md"

BLOB = "https://github.com/adlyZaroui/analytic-prophet/blob/main/"
TREE = "https://github.com/adlyZaroui/analytic-prophet/tree/main/"
RAW = "https://raw.githubusercontent.com/adlyZaroui/analytic-prophet/main/"

LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


@pytest.fixture(scope="module")
def readme():
    return README.read_text()


def _slug(heading):
    """GitHub's anchor rule.

    Punctuation is stripped and each space becomes one hyphen -- runs are not
    collapsed, which is why a heading with an em-dash produces a double
    hyphen. Getting this wrong makes a working anchor look broken, which it
    did the first time these were checked.
    """
    text = heading.strip().lstrip("#").strip()
    text = re.sub(r"[`*_\[\]()]", "", text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-").strip("-")


def test_the_readme_is_what_pypi_renders():
    """If this stops being true, everything below is checking the wrong file."""
    from conftest import load_pyproject

    assert load_pyproject()["project"]["readme"] == "README.md"


def test_no_link_is_relative(readme):
    relative = [target for _text, target in LINK.findall(readme)
                if not target.startswith(("http://", "https://", "#", "mailto:"))]
    assert relative == [], (
        f"these resolve against pypi.org for a PyPI reader: {relative}")


def test_no_image_is_relative(readme):
    relative = [target for _alt, target in IMAGE.findall(readme)
                if not target.startswith(("http://", "https://"))]
    assert relative == [], f"these do not render on PyPI: {relative}"


def test_images_come_from_raw_githubusercontent(readme):
    """`blob/` serves an HTML page, not the bytes: an <img> pointing at it
    shows nothing."""
    for _alt, target in IMAGE.findall(readme):
        if "github" in target and "badge" not in target:
            assert target.startswith(RAW), (
                f"{target} is not a raw URL, so it will not render as an image")


def test_every_repo_link_points_at_a_file_that_exists(readme):
    """An absolute URL can be confidently wrong in a way a relative one
    cannot: nothing in the editor complains."""
    missing = []
    for _text, target in LINK.findall(readme):
        for base in (BLOB, TREE):
            if target.startswith(base):
                path = target[len(base):].partition("#")[0]
                if path and not (REPO / path).exists():
                    missing.append(path)
    assert missing == [], f"linked but not in the repository: {missing}"


def test_every_anchor_resolves_to_a_heading(readme):
    """A link to a heading that was renamed is half dead: it loads the page
    and then sits at the top of it."""
    broken = []
    for _text, target in LINK.findall(readme):
        if not target.startswith(BLOB) or "#" not in target:
            continue
        path, _, anchor = target[len(BLOB):].partition("#")
        document = REPO / path
        if not document.exists():
            continue
        headings = {_slug(line) for line in document.read_text().splitlines()
                    if line.startswith("#")}
        if anchor not in headings:
            broken.append(f"{path}#{anchor}")
    assert broken == [], f"anchors with no such heading: {broken}"


# -- what the page says about installing -----------------------------------

def test_the_readme_does_not_say_the_package_is_unpublished(readme):
    """It said "once published" and "Not on PyPI yet" on the PyPI page itself,
    where it read as a package announcing it does not exist."""
    one_line = " ".join(readme.split()).lower()
    for phrase in ("once published", "not on pypi yet", "until then, clone"):
        assert phrase not in one_line, f"the README still says {phrase!r}"


def test_the_install_story_distinguishes_wheel_from_source(readme):
    """The README said both "nothing to build" and "the first fit compiles
    optimize.cpp -- about ten seconds". Both are true, of different users, and
    neither sentence said which.

    "Your first fit pauses for ten seconds" is exactly what someone evaluating
    the package needs a straight answer about, so every claim about building
    has to name the distribution it describes.
    """
    one_line = " ".join(readme.split())
    for claim in ("compiles `optimize.cpp`", "built\non demand", "built on demand"):
        if claim.replace("\n", " ") not in one_line:
            continue
        where = one_line.index(claim.replace("\n", " "))
        context = one_line[max(0, where - 400):where + 200].lower()
        assert "wheel" in context, (
            f"a claim about compiling at fit time ({claim!r}) with no mention "
            f"of which distribution it applies to")
