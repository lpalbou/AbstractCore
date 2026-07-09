from __future__ import annotations

import pytest

pytestmark = pytest.mark.basic


class _FakeResponse:
    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: bytes,
        status_code: int = 200,
        reason: str = "OK",
    ):
        self.url = url
        self.headers = headers
        self.status_code = status_code
        self.reason = reason
        self.ok = 200 <= status_code < 400
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def iter_content(self, chunk_size: int = 1):
        for idx in range(0, len(self._body), int(chunk_size)):
            yield self._body[idx : idx + int(chunk_size)]


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls.append(dict(kwargs))
        return self._response


def test_fetch_url_sniffs_pdf_and_extracts_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    if not common_tools._ensure_requests():
        pytest.skip('requests not available; install with: pip install "abstractcore[tools]"')

    pdf = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 24 Tf
100 100 Td
(Hello PDF) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000241 00000 n 
0000000334 00000 n 
trailer
<< /Root 1 0 R /Size 6 >>
startxref
404
%%EOF
"""

    fake = _FakeResponse(
        url="https://example.com/doc.pdf",
        headers={"content-type": "application/octet-stream", "content-length": str(len(pdf))},
        body=pdf,
    )
    monkeypatch.setattr(common_tools.requests, "Session", lambda: _FakeSession(fake))

    out = common_tools.fetch_url("https://example.com/doc.pdf", include_full_content=False)

    assert out.get("success") is True
    assert out.get("content_type") == "application/octet-stream"
    assert "🧭 Detected-As: pdf" in str(out.get("rendered") or "")
    assert "Hello PDF" in str(out.get("rendered") or "")
    assert "Hello PDF" in str(out.get("raw_text") or "")
    assert "Hello PDF" in str(out.get("normalized_text") or "")
    assert str(out.get("pdf_text_backend") or "")
    assert isinstance(out.get("pdf_backend_attempts"), list)
    assert out.get("page_count") == 1


def test_fetch_url_sniffs_xml_feed_and_normalizes_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    if not common_tools._ensure_requests():
        pytest.skip('requests not available; install with: pip install "abstractcore[tools]"')

    feed = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<rss><channel><title>Example Feed</title><description>Latest updates</description>"
        "<item><title>First Post</title><link>https://example.com/1</link><description>Intro item</description></item>"
        "</channel></rss>"
    ).encode("utf-8")

    fake = _FakeResponse(
        url="https://example.com/feed.xml",
        headers={"content-type": "application/octet-stream", "content-length": str(len(feed))},
        body=feed,
    )
    monkeypatch.setattr(common_tools.requests, "Session", lambda: _FakeSession(fake))

    out = common_tools.fetch_url("https://example.com/feed.xml", include_full_content=False)

    assert out.get("success") is True
    assert "🧭 Detected-As: xml" in str(out.get("rendered") or "")
    assert "Example Feed" in str(out.get("rendered") or "")
    assert "First Post" in str(out.get("rendered") or "")
    assert "Example Feed" in str(out.get("normalized_text") or "")
    assert "First Post" in str(out.get("normalized_text") or "")


def test_fetch_url_normalizes_string_tool_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    if not common_tools._ensure_requests():
        pytest.skip('requests not available; install with: pip install "abstractcore[tools]"')

    body = ("start " + ("middle " * 700) + "ENDMARKER").encode("utf-8")
    fake = _FakeResponse(
        url="https://example.com/long.txt",
        headers={"content-type": "text/plain", "content-length": str(len(body))},
        body=body,
    )
    session = _FakeSession(fake)
    monkeypatch.setattr(common_tools.requests, "Session", lambda: session)

    out = common_tools.fetch_url(
        "https://example.com/long.txt",
        timeout="15",
        include_full_content="False",
        keep_links="False",
        include_binary_preview="False",
    )

    assert out.get("success") is True
    assert session.calls
    assert session.calls[0]["timeout"] == 15.0
    rendered = str(out.get("rendered") or "")
    assert "📄 Content Preview:" in rendered
    assert "ENDMARKER" not in rendered


def test_fetch_url_rejects_invalid_string_timeout() -> None:
    import abstractcore.tools.common_tools as common_tools

    out = common_tools.fetch_url("https://example.com/long.txt", timeout="slow")

    assert out.get("success") is False
    assert out.get("error") == "timeout must be a positive number"
