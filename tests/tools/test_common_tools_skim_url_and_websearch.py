from __future__ import annotations

import json
import sys
import types

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
        self.content = body

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

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(self, *args: object, **kwargs: object) -> _FakeResponse:
        return self._response


class _SequenceSession:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.headers: dict[str, str] = {}

    def __enter__(self) -> _SequenceSession:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(self, *args: object, **kwargs: object) -> _FakeResponse:
        assert self._responses, "No fake responses left"
        return self._responses.pop(0)


def test_skim_url_extracts_title_description_headings_and_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.tools.common_tools as common_tools

    if not common_tools._ensure_requests():
        pytest.skip('requests not available; install with: pip install "abstractcore[tools]"')

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

    out = common_tools.skim_url(
        "http://example.com/page", max_bytes=900, max_preview_chars=600, max_headings=5
    )

    assert "🌐 URL Skim" in out
    assert "📰 Title: R-Type" in out
    assert "📝 Description: A classic shmup." in out
    assert "🏷️ Headings (H1–H3):" in out
    assert "- H1: Main" in out
    assert "- H2: Weapons" in out
    assert "Force pod, wave cannon." in out
    assert "(partial; limit 900)" in out


def test_skim_url_parses_rss_when_server_labels_it_octet_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.tools.common_tools as common_tools

    if not common_tools._ensure_requests():
        pytest.skip('requests not available; install with: pip install "abstractcore[tools]"')

    feed = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<rss><channel><title>Example Feed</title><description>Latest updates</description>"
        "<item><title>First Post</title><link>https://example.com/1</link><description>Intro item</description></item>"
        "<item><title>Second Post</title><link>https://example.com/2</link><description>Follow-up item</description></item>"
        "</channel></rss>"
    ).encode("utf-8")

    fake = _FakeResponse(
        url="https://example.com/feed.xml",
        headers={"content-type": "application/octet-stream", "content-length": str(len(feed))},
        body=feed,
    )
    monkeypatch.setattr(common_tools.requests, "Session", lambda: _FakeSession(fake))

    out = common_tools.skim_url("https://example.com/feed.xml")

    assert "Detected-As: xml" in out
    assert "Example Feed" in out
    assert "1. First Post" in out
    assert "https://example.com/1" in out


def test_skim_url_refetches_small_pdf_to_extract_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.tools.common_tools as common_tools

    if not common_tools._ensure_requests():
        pytest.skip('requests not available; install with: pip install "abstractcore[tools]"')

    pdf_prefix = b"%PDF-1.7\n1 0 obj\n<<>>\nstream\npartial"
    full_pdf = pdf_prefix + b"\nendstream\nendobj\nxref\ntrailer\n%%EOF"

    first = _FakeResponse(
        url="https://example.com/report.pdf",
        headers={"content-type": "application/pdf", "content-length": str(len(full_pdf))},
        body=pdf_prefix,
    )
    second = _FakeResponse(
        url="https://example.com/report.pdf",
        headers={"content-type": "application/pdf", "content-length": str(len(full_pdf))},
        body=full_pdf,
    )

    def _fake_route(pdf_bytes: bytes, **kwargs: object) -> dict[str, object]:
        if len(pdf_bytes) < len(full_pdf):
            return {
                "rendered": "",
                "title": "",
                "raw_text": "",
                "warnings": ["PDF extraction failed: Cannot find Root object in pdf"],
            }
        return {
            "title": "Quarterly Results",
            "raw_text": "Revenue grew 12 percent year over year.",
            "warnings": [],
        }

    monkeypatch.setattr(common_tools, "route_pdf_bytes", _fake_route)
    monkeypatch.setattr(common_tools.requests, "Session", lambda: _SequenceSession([first, second]))

    out = common_tools.skim_url("https://example.com/report.pdf", max_bytes=64, max_preview_chars=500)

    assert "Detected-As: pdf" in out
    assert "Quarterly Results" in out
    assert "Revenue grew 12 percent year over year." in out
    assert "Refetched full PDF for preview" in out


def test_skim_websearch_filters_results_by_snippet(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    sample = {
        "engine": "duckduckgo",
        "query": "pets",
        "params": {"num_results": 10},
        "results": [
            {
                "rank": 1,
                "title": "Cats",
                "url": "https://example.com/cats",
                "snippet": "All about cats",
            },
            {
                "rank": 2,
                "title": "Dogs",
                "url": "https://example.com/dogs",
                "snippet": "All about dogs",
            },
            {
                "rank": 3,
                "title": "Cats and Dogs",
                "url": "https://example.com/both",
                "snippet": "Cats and dogs together",
            },
        ],
    }

    monkeypatch.setattr(common_tools, "web_search", lambda *args, **kwargs: json.dumps(sample))

    out_any = common_tools.skim_websearch(query="pets", required_terms=["cats"], num_results=2)
    data_any = json.loads(out_any)
    urls_any = [r["url"] for r in data_any["results"]]
    assert urls_any == ["https://example.com/cats", "https://example.com/both"]

    out_all = common_tools.skim_websearch(
        query="pets", required_terms="cats,dogs", match="all", num_results=5
    )
    data_all = json.loads(out_all)
    urls_all = [r["url"] for r in data_all["results"]]
    assert urls_all == ["https://example.com/both"]


def test_skim_websearch_truncates_long_snippets_to_keep_outputs_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.tools.common_tools as common_tools

    long_snippet = "A" * 2000
    sample = {
        "engine": "duckduckgo",
        "query": "pets",
        "params": {"num_results": 10},
        "results": [
            {
                "rank": 1,
                "title": "Cats",
                "url": "https://example.com/cats",
                "snippet": long_snippet,
            }
        ],
    }

    monkeypatch.setattr(common_tools, "web_search", lambda *args, **kwargs: json.dumps(sample))

    out = common_tools.skim_websearch(query="pets", num_results=1)
    data = json.loads(out)
    snippet = data["results"][0]["snippet"]
    assert "… (truncated)" in snippet
    assert 240 < len(snippet) <= 720


def test_skim_websearch_propagates_upstream_failure_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    sample = {
        "success": False,
        "status_hint": "error",
        "degraded": True,
        "backend_used": "duckduckgo.html",
        "query": "pets",
        "results": [],
        "error": "requests is not installed",
        "warnings": ["Primary backend ddgs.text failed; used duckduckgo.html fallback."],
    }

    monkeypatch.setattr(common_tools, "web_search", lambda *args, **kwargs: json.dumps(sample))

    out = common_tools.skim_websearch(query="pets", num_results=1)
    data = json.loads(out)
    assert data["success"] is False
    assert data["status_hint"] == "error"
    assert data["error"] == "requests is not installed"
    assert data["backend_used"] == "duckduckgo.html"
    assert data["results"] == []


def test_web_search_uses_current_ddgs_query_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    class _FakeDDGS:
        def __enter__(self) -> _FakeDDGS:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def text(self, query: str, **kwargs: object) -> list[dict[str, str]]:
            assert query == "pets"
            assert kwargs["max_results"] == 1
            return [
                {
                    "title": "Cats",
                    "href": "https://example.com/cats",
                    "body": "All about cats",
                }
            ]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=_FakeDDGS))

    out = common_tools.web_search("pets", num_results=1)
    data = json.loads(out)

    assert data["success"] is True
    assert data["source"] == "ddgs.text"
    assert data["results"][0]["url"] == "https://example.com/cats"


def test_web_search_coerces_string_num_results_before_calling_ddgs(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    class _FakeDDGS:
        def __enter__(self) -> _FakeDDGS:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def text(self, query: str, **kwargs: object) -> list[dict[str, str]]:
            assert query == "pets"
            assert kwargs["max_results"] == 5
            assert isinstance(kwargs["max_results"], int)
            return [
                {
                    "title": "Cats",
                    "href": "https://example.com/cats",
                    "body": "All about cats",
                }
            ]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=_FakeDDGS))

    out = common_tools.web_search("pets", num_results="5")
    data = json.loads(out)

    assert data["success"] is True
    assert data["degraded"] is False
    assert data["source"] == "ddgs.text"
    assert data["params"]["num_results"] == 5
    assert data["backend_attempts"] == [{"name": "ddgs.text", "success": True, "module": "ddgs"}]


def test_web_search_rejects_invalid_num_results_instead_of_silent_fallback() -> None:
    import abstractcore.tools.common_tools as common_tools

    out = common_tools.web_search("pets", num_results="abc")
    data = json.loads(out)

    assert data["success"] is False
    assert data["status_hint"] == "error"
    assert data["error"] == "num_results must be a positive integer"
    assert data["results"] == []
    assert data["params"]["num_results"] == "abc"


def test_web_search_uses_legacy_duckduckgo_search_module_when_ddgs_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.tools.common_tools as common_tools

    real_import_module = common_tools.importlib.import_module

    class _LegacyDDGS:
        def __enter__(self) -> _LegacyDDGS:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def text(self, keywords: str, **kwargs: object) -> list[dict[str, str]]:
            assert keywords == "pets"
            assert kwargs["max_results"] == 2
            return [
                {
                    "title": "Cats",
                    "href": "https://example.com/cats",
                    "body": "All about cats",
                }
            ]

    def _fake_import_module(name: str):
        if name == "ddgs":
            raise ImportError("ddgs missing in legacy environment")
        if name == "duckduckgo_search":
            return types.SimpleNamespace(DDGS=_LegacyDDGS)
        return real_import_module(name)

    monkeypatch.setattr(common_tools.importlib, "import_module", _fake_import_module)

    out = common_tools.web_search("pets", num_results=2)
    data = json.loads(out)

    assert data["success"] is True
    assert data["source"] == "ddgs.text"
    assert data["backend_attempts"] == [{"name": "ddgs.text", "success": True, "module": "duckduckgo_search"}]


def test_skim_websearch_rejects_invalid_num_results_instead_of_silent_default() -> None:
    import abstractcore.tools.common_tools as common_tools

    out = common_tools.skim_websearch(query="pets", num_results="abc")
    data = json.loads(out)

    assert data["success"] is False
    assert data["status_hint"] == "error"
    assert data["error"] == "num_results must be a positive integer"
    assert data["results"] == []
    assert data["params"]["num_results"] == "abc"


def test_skim_websearch_surfaces_compact_result_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.tools.common_tools as common_tools

    sample = {
        "success": True,
        "status_hint": "ok",
        "backend_used": "ddgs.text",
        "query": "pets",
        "results": [
            {
                "rank": i,
                "title": f"Cats {i}",
                "url": f"https://example.com/{i}",
                "snippet": "All about cats",
            }
            for i in range(1, 21)
        ],
    }

    monkeypatch.setattr(common_tools, "web_search", lambda *args, **kwargs: json.dumps(sample))

    out = common_tools.skim_websearch(query="pets", num_results=20)
    data = json.loads(out)

    assert data["counts"]["returned"] == 15
    assert "num_results was capped at 15 for compact skim output." in data["warnings"]
    assert "num_results_capped_at_15" in data["limitations"]
