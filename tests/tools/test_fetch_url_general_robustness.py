"""General-case robustness: invariants that hold for ANY url, not one site.

Every defect pinned here was found by a site-agnostic sweep over 30 URLs that
are NOT in the fixture corpus (news hubs, docs, forums, government, package
registries, non-English press, an RFC text file, a JSON endpoint). Each test
states the invariant, not the page — the URLs in the docstrings are only the
evidence that the invariant was being violated in the wild.
"""
from __future__ import annotations

import pytest

from abstractcore.tools.common_tools import (
    _detect_unrenderable_html,
    _extract_main_content,
    _MIN_REAL_CONTENT_CHARS,
)


# --------------------------------------------------------------------------
# 1. A 2xx that yields no text is never a silent success
# --------------------------------------------------------------------------

def test_small_js_shell_is_reported_not_returned_as_empty_success() -> None:
    """A page that extracts to nothing must fail loudly however small its markup.

    The previous rule required the shell to exceed 20 KB before it counted as
    unrenderable, so small anti-bot shells slipped under it and came back as
    success=True with content="" and no error_class — pubmed ships 5.5 KB and
    reddit 8.4 KB, and both did exactly that.
    """
    shell = "<html><head><title>T</title></head><body><div id=root></div></body></html>"
    assert len(shell) < 20000, "this fixture must sit UNDER the old 20 KB rule"
    detected = _detect_unrenderable_html(shell, "")
    assert detected is not None, "a zero-text HTML response was accepted as content"
    err_class, suggestions = detected
    assert err_class in {"empty_content", "js_required", "bot_challenge"}
    assert suggestions, "an error class must carry actionable suggestions"


def test_a_genuinely_short_page_is_still_real_content() -> None:
    """The zero-text floor must not reject a page that simply says little."""
    page = (
        "<html><head><title>Notice</title></head><body><main>"
        "<h1>Service notice</h1><p>The API will be unavailable on Sunday.</p>"
        "</main></body></html>"
    )
    content = _extract_main_content(page, "https://example.com/n")["content"]
    assert "unavailable on Sunday" in content
    assert _detect_unrenderable_html(page, content) is None, (
        "a short but real page was misclassified as unrenderable"
    )


# --------------------------------------------------------------------------
# 2. A listing page extracts to the LIST, not to one card
# --------------------------------------------------------------------------

def _feed(card_wrapper_classes: list[str] | None = None) -> str:
    """A hub page: six cards, each an <article> with its own headline."""
    cards = []
    for i in range(6):
        inner = (
            f"<article class='card post-{2168000 + i}'>"
            f"<h2><a href='https://example.com/{i}'>Headline number {i}</a></h2>"
            f"<p>Standfirst paragraph for story {i} with enough words to score.</p>"
            f"</article>"
        )
        if card_wrapper_classes is not None:
            inner = f"<div class='{card_wrapper_classes[i % len(card_wrapper_classes)]}'>{inner}</div>"
        cards.append(inner)
    return (
        "<html><head><title>Hub</title></head><body><main class='main'>"
        + "".join(cards)
        + "</main></body></html>"
    )


def test_listing_page_extracts_every_card_not_just_the_best_one() -> None:
    """`article` is in the selector list, so the scorer used to pick ONE card.

    Measured in the wild: arstechnica.com returned 204 chars (one card of ~30),
    stackoverflow.blog 462, nasa.gov/news 794 — 3-5% of the page.
    """
    content = _extract_main_content(_feed(), "https://example.com/")["content"]
    for i in range(6):
        assert f"Headline number {i}" in content, f"card {i} was dropped from the listing"


def test_listing_survives_per_card_wrapper_classes() -> None:
    """Feed cards are often wrapped, and the wrappers differ per card.

    arstechnica gives every <article> its own <div class="lg:order-N">, so the
    articles are not siblings AND the wrapper classes differ — both the
    repeated-shape test and a naive sibling test miss it.
    """
    content = _extract_main_content(
        _feed(["lg:order-1 col-span-2", "lg:order-6 col-span-2", "lg:order-3 col-span-1"]),
        "https://example.com/",
    )["content"]
    found = sum(1 for i in range(6) if f"Headline number {i}" in content)
    assert found == 6, f"only {found}/6 cards survived a wrapped listing"


def test_a_real_article_is_not_widened_into_its_page_shell() -> None:
    """The widening must not fire on an article, which has no repeated siblings."""
    page = (
        "<html><head><title>Story</title></head><body>"
        "<nav>Home About Contact Subscribe</nav>"
        "<main><article><h1>The only story</h1>"
        "<p>" + ("Body prose that is clearly the substance of this page. " * 40) + "</p>"
        "</article></main>"
        "<footer>Copyright notice and site links</footer></body></html>"
    )
    content = _extract_main_content(page, "https://example.com/story")["content"]
    assert "The only story" in content
    assert "Copyright notice" not in content, "widening dragged in the page shell"
    assert "Home About Contact" not in content


# --------------------------------------------------------------------------
# 3. Empty list items are layout, not content
# --------------------------------------------------------------------------

def test_empty_list_items_are_not_emitted_as_bare_bullets() -> None:
    """Icon-only / script-filled <li>s rendered as a bare "-" that says nothing.

    lemonde.fr's front page emitted 31 in a row; arstechnica opened with four.
    """
    page = (
        "<html><head><title>L</title></head><body><main>"
        "<h1>Page</h1>"
        "<ul><li></li><li><span></span></li><li>Real item</li><li>  </li></ul>"
        "<p>" + ("Enough body prose to make this container win selection. " * 20) + "</p>"
        "</main></body></html>"
    )
    content = _extract_main_content(page, "https://example.com/l")["content"]
    assert "- Real item" in content
    bare = [ln for ln in content.splitlines() if ln.strip() == "-"]
    assert not bare, f"{len(bare)} empty list items were emitted as bare bullets"


def test_ordered_list_numbering_skips_dropped_empty_items() -> None:
    """Dropping an empty <li> must not leave a gap in an ordered list."""
    page = (
        "<html><head><title>O</title></head><body><main><h1>H</h1>"
        "<ol><li>First</li><li></li><li>Second</li></ol>"
        "<p>" + ("Body prose for container scoring. " * 25) + "</p>"
        "</main></body></html>"
    )
    content = _extract_main_content(page, "https://example.com/o")["content"]
    assert "1. First" in content
    assert "2. Second" in content, "numbering skipped a value for a dropped item"


# --------------------------------------------------------------------------
# 4. Universal hygiene invariants (hold for any page)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "markup, forbidden",
    [
        ("<!-- internal build note -->", "internal build note"),
        ("<script>var trackingPayload = 1;</script>", "trackingPayload"),
        ("<style>.cls{color:red}</style>", "color:red"),
    ],
)
def test_non_content_nodes_never_reach_the_output(markup: str, forbidden: str) -> None:
    page = (
        "<html><head><title>T</title></head><body><main>"
        f"{markup}<h1>Title</h1><p>" + ("Real prose here. " * 30) + "</p>"
        "</main></body></html>"
    )
    out = _extract_main_content(page, "https://example.com/x")
    assert forbidden not in out["content"]
    assert forbidden not in out["text"]


def test_min_real_content_constant_is_the_shared_floor() -> None:
    """Guards the constant the evidence policy and the retry paths both key on."""
    assert _MIN_REAL_CONTENT_CHARS == 200
