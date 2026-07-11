"""Regression: fetch_url extraction quality against curated gold references.

Deterministic + offline — runs the extraction pipeline over COMMITTED raw-HTML
fixtures (captured 2026-07-11 by adversarial subagents from real pages), scoring
against per-URL fact checklists / junk blacklists curated from the raw bytes
(never from the tool's own output). This pins the maintainer's bar: each URL's
primary `content` must recall >= 90% of the gold facts with ZERO boilerplate
junk. The four fixtures are the exact pages that failed in production
(2026-07-11): techxplore (403 bot-challenge), budgyapp + nextbigfuture
(sidebar/author-box junk), newsletter (consent-overlay + body-level selection).

The live end-to-end variant is env-gated (ABSTRACT_E2E_FETCH_URL=1) because the
network is nondeterministic; CI relies on the fixtures.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fetch_url_fixtures"
sys.path.insert(0, str(FIXTURES))

from harness import GOLD, fixture_html, score, url_keys  # noqa: E402

from abstractcore.tools.common_tools import _extract_main_content, fetch_url


@pytest.mark.parametrize("gold_key", url_keys())
def test_extraction_meets_gold_reference(gold_key: str) -> None:
    """The primary `content` field recalls >=90% of gold facts, zero junk."""
    html = fixture_html(gold_key)
    url = GOLD[gold_key]["url"]
    main = _extract_main_content(html, url, keep_links=True)
    content = str(main.get("content") or "")
    s = score(content, gold_key)

    assert s["recall"] >= 0.90, (
        f"{gold_key}: fact recall {s['recall']:.2f} < 0.90; missed {s['missed']}"
    )
    assert s["junk_ratio"] == 0.0, (
        f"{gold_key}: boilerplate leaked into content: {s['junk_hit']}"
    )
    assert content.strip(), f"{gold_key}: content is empty"


@pytest.mark.parametrize("gold_key", url_keys())
def test_title_extracted(gold_key: str) -> None:
    """`title` is a first-class field carrying the expected page title."""
    html = fixture_html(gold_key)
    url = GOLD[gold_key]["url"]
    main = _extract_main_content(html, url)
    title = str(main.get("title") or "")
    expected = GOLD[gold_key]["title_contains"]
    assert expected.lower() in title.lower(), (
        f"{gold_key}: title {title!r} missing expected {expected!r}"
    )


def test_html_result_exposes_primary_content_keys() -> None:
    """The offline extractor yields the obvious consumer keys (content/title).

    Guards the production root cause: a consumer reading result["content"] must
    get rich text, not an empty string, for a normal HTML page.
    """
    html = fixture_html("budgyapp")
    main = _extract_main_content(html, GOLD["budgyapp"]["url"])
    assert set(main.keys()) >= {"title", "description", "content", "text"}
    assert len(main["content"]) > 1000
    assert "Claude" in main["title"]


@pytest.mark.skipif(
    os.getenv("ABSTRACT_E2E_FETCH_URL") != "1",
    reason="Set ABSTRACT_E2E_FETCH_URL=1 to run the live network variant.",
)
@pytest.mark.parametrize("gold_key", url_keys())
def test_live_fetch_meets_gold(gold_key: str) -> None:
    """Live end-to-end: fetch_url returns success + content passing the gold bar.

    Nondeterministic (bot challenges, A/B walls, content drift) — env-gated.
    """
    url = GOLD[gold_key]["url"]
    result = fetch_url(url=url, timeout=45)
    assert result.get("success") is True, f"{gold_key}: {result.get('error')}"
    content = str(result.get("content") or "")
    s = score(content, gold_key)
    assert s["recall"] >= 0.90, f"{gold_key}: live recall {s['recall']:.2f}; missed {s['missed']}"
