from __future__ import annotations

import os

import pytest


@pytest.mark.e2e
def test_fetch_url_real_html_normalized_text_has_no_markup_or_scripts() -> None:
    if os.environ.get("ABSTRACT_E2E_FETCH_URL") != "1":
        pytest.skip("Set ABSTRACT_E2E_FETCH_URL=1 to run this test.")

    from abstractcore.tools.common_tools import fetch_url

    urls = [
        "https://example.com/",
        "https://httpbin.org/html",
    ]

    for url in urls:
        out = fetch_url(url, timeout=30, include_full_content=False)
        assert out.get("success") is True

        rendered = str(out.get("rendered") or "")
        assert "🌐 HTML Document Analysis" in rendered
        assert "<html" not in rendered.lower()

        normalized = str(out.get("normalized_text") or "")
        assert len(normalized) > 20
        assert "<html" not in normalized.lower()
        assert "<script" not in normalized.lower()

