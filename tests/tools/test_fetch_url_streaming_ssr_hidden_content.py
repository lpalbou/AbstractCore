"""Regression: React streaming-SSR pages are extractable, not "empty_content".

Production failure (2026-08-20): fetch_url returned
`No readable content extracted (empty_content)` for a fully server-rendered
Next.js App Router article. React 18/19 streams a completed Suspense boundary
as `<div hidden id="S:1">…the article…</div>` plus an inline script that moves
it into place on the client. The generic hidden-element sweep decomposed that
holder, taking the whole body with it, so a page a browser renders perfectly
looked like a JS-only shell.

Offline + deterministic: builds the exact markup shape React emits.
"""
from __future__ import annotations

from abstractcore.tools.common_tools import (
    _detect_unrenderable_html,
    _extract_clean_text_from_html,
    _extract_main_content,
)

URL = "https://example.com/blog/streaming-post"

PARAGRAPH = (
    "Streaming server rendering sends the shell first and flushes each Suspense "
    "boundary as it resolves, which is why the article text arrives inside a "
    "hidden holder rather than in its final position in the document. "
)


def _streaming_page(*, holder: str) -> str:
    body = "".join(f"<p>{PARAGRAPH}Paragraph {i}.</p>" for i in range(6))
    return f"""<!doctype html>
<html><head><title>Streaming Post | Example</title>
<meta name="description" content="A server-rendered article behind a Suspense boundary."></head>
<body>
  <div id="root"><!--$?--><template id="B:0"></template><div class="skeleton"></div><!--/$--></div>
  {holder.format(body=f'<main><article><h1>Streaming Post</h1>{body}</article></main>')}
  <script>$RC("B:0","S:0")</script>
</body></html>"""


def test_hidden_suspense_holder_content_is_extracted() -> None:
    html = _streaming_page(holder='<div hidden id="S:0">{body}</div>')
    main = _extract_main_content(html, URL, keep_links=True)

    content = str(main.get("content") or "")
    assert "Paragraph 5." in content, f"article body lost; got {content[:200]!r}"
    assert "Streaming server rendering" in content
    assert str(main.get("title") or "").startswith("Streaming Post")


def test_hidden_suspense_holder_is_not_reported_as_empty_content() -> None:
    html = _streaming_page(holder='<div hidden id="S:0">{body}</div>')
    main = _extract_main_content(html, URL)
    assert _detect_unrenderable_html(html, str(main.get("content") or "")) is None


def test_prefixed_and_template_holders_also_recover() -> None:
    """React's `identifierPrefix` and `<template>` holders take the same path."""
    for holder in (
        '<div hidden id="app-S:1">{body}</div>',
        '<template id="S:2">{body}</template>',
    ):
        _, _, text = _extract_clean_text_from_html(_streaming_page(holder=holder), URL)
        assert "Paragraph 5." in text, f"{holder}: body lost; got {text[:200]!r}"


def test_ordinary_hidden_noise_is_still_dropped() -> None:
    """The fix must not re-admit hidden boilerplate on a normal page."""
    body = "".join(f"<p>{PARAGRAPH}Paragraph {i}.</p>" for i in range(6))
    html = f"""<!doctype html>
<html><head><title>Plain Post</title></head><body>
  <div hidden id="mobile-menu"><a href="/pricing">Pricing</a> Buy the annual plan now</div>
  <div aria-hidden="true">Decorative offscreen chrome text</div>
  <main><article><h1>Plain Post</h1>{body}</article></main>
</body></html>"""
    content = str(_extract_main_content(html, URL).get("content") or "")
    assert "Paragraph 5." in content
    assert "Buy the annual plan" not in content
    assert "Decorative offscreen" not in content
