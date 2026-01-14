from __future__ import annotations

import pytest


@pytest.mark.basic
def test_parse_html_extracts_clean_text_with_bs4() -> None:
    from abstractcore.tools import common_tools

    html = (
        "<html><head><title>R-Type</title>"
        "<meta name=\"description\" content=\"A classic shmup.\" /></head>"
        "<body>"
        "<nav>Home • About • Contact</nav>"
        "<h1>Main</h1><p>Force pod, wave cannon.</p>"
        "<script>console.log('noise')</script>"
        "</body></html>"
    ).encode("utf-8")

    out = common_tools._parse_content_by_type(
        html,
        "text/html; charset=utf-8",
        "https://example.com/",
        extract_links=False,
        include_binary_preview=False,
        include_full_content=False,
    )

    assert "🌐 HTML Document Analysis" in out
    assert "📰 Title: R-Type" in out
    assert "📝 Description: A classic shmup." in out
    assert "Markdown Content Preview" in out
    assert "# Main" in out
    assert "Force pod, wave cannon." in out
    assert "<html" not in out
    assert "Links (first 20)" not in out
