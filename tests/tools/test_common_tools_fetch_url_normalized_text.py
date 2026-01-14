from __future__ import annotations

import pytest


@pytest.mark.basic
def test_normalize_text_for_evidence_html_strips_tags_scripts_and_layout() -> None:
    from abstractcore.tools import common_tools

    html = (
        "<html><head><title>R-Type</title>"
        "<meta name=\"description\" content=\"A classic shmup.\" />"
        "<style>.nav{display:none}</style>"
        "</head>"
        "<body>"
        "<nav>Home • About • Contact</nav>"
        "<article><header><h1>Main</h1></header><p>Force pod, wave cannon.</p></article>"
        "<script>console.log('noise')</script>"
        "<footer>© 2026</footer>"
        "</body></html>"
    )

    out = common_tools._normalize_text_for_evidence(
        raw_text=html,
        content_type_header="text/html; charset=utf-8",
        url="https://example.com/",
    )

    assert "R-Type" in out
    assert "A classic shmup." in out
    assert "Force pod, wave cannon." in out
    assert "<html" not in out.lower()
    assert "console.log" not in out
    assert "Home • About • Contact" not in out


@pytest.mark.basic
def test_normalize_text_for_evidence_detects_html_when_content_type_is_text_plain() -> None:
    from abstractcore.tools import common_tools

    html = "<html><body><p>Hello</p></body></html>"
    out = common_tools._normalize_text_for_evidence(
        raw_text=html,
        content_type_header="text/plain; charset=utf-8",
        url="https://example.com/",
    )

    assert "Hello" in out
    assert "<html" not in out.lower()

