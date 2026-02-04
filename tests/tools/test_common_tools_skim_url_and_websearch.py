from __future__ import annotations

import json

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

    def __enter__(self) -> "_FakeResponse":
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

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(self, *args: object, **kwargs: object) -> _FakeResponse:
        return self._response


def test_skim_url_extracts_title_description_headings_and_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    html = (
        "<html><head><title>R-Type</title>"
        '<meta name="description" content="A classic shmup." />'
        "</head>"
        "<body>"
        "<h1>Main</h1>"
        "<h2>Weapons</h2>"
        "<p>Force pod, wave cannon.</p>"
        "<script>" + ("a" * 10_000) + "</script>"
        "</body></html>"
    ).encode("utf-8")

    fake = _FakeResponse(
        url="http://example.com/page",
        headers={"content-type": "text/html; charset=utf-8", "content-length": str(len(html))},
        body=html,
    )
    monkeypatch.setattr(common_tools.requests, "Session", lambda: _FakeSession(fake))

    out = common_tools.skim_url("http://example.com/page", max_bytes=900, max_preview_chars=600, max_headings=5)

    assert "🌐 URL Skim" in out
    assert "📰 Title: R-Type" in out
    assert "📝 Description: A classic shmup." in out
    assert "🏷️ Headings (H1–H3):" in out
    assert "- H1: Main" in out
    assert "- H2: Weapons" in out
    assert "Force pod, wave cannon." in out
    assert "(partial; limit 900)" in out


def test_skim_websearch_filters_results_by_snippet(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    sample = {
        "engine": "duckduckgo",
        "query": "pets",
        "params": {"num_results": 10},
        "results": [
            {"rank": 1, "title": "Cats", "url": "https://example.com/cats", "snippet": "All about cats"},
            {"rank": 2, "title": "Dogs", "url": "https://example.com/dogs", "snippet": "All about dogs"},
            {"rank": 3, "title": "Cats and Dogs", "url": "https://example.com/both", "snippet": "Cats and dogs together"},
        ],
    }

    monkeypatch.setattr(common_tools, "web_search", lambda *args, **kwargs: json.dumps(sample))

    out_any = common_tools.skim_websearch(query="pets", required_terms=["cats"], num_results=2)
    data_any = json.loads(out_any)
    urls_any = [r["url"] for r in data_any["results"]]
    assert urls_any == ["https://example.com/cats", "https://example.com/both"]

    out_all = common_tools.skim_websearch(query="pets", required_terms="cats,dogs", match="all", num_results=5)
    data_all = json.loads(out_all)
    urls_all = [r["url"] for r in data_all["results"]]
    assert urls_all == ["https://example.com/both"]

