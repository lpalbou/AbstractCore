"""Pins the fetch_url envelope contract on the REAL tool, not on a rebuild.

The offline harness rebuilds the envelope field by field so it can be measured
without a network; this file drives `fetch_url` itself over a fake transport, so
a drift between the two is visible from BOTH sides.

What it pins:
  * single-canonical-payload — `content` is the payload, `raw_text` and
    `normalized_text` are evidence mirrors withheld once they stop being cheap,
    and a `*_withheld` descriptor says exactly what was dropped;
  * `link_dominant` — the one case where fetch_url overrides `keep_links=False`
    (a listing page's links ARE its content) must be visible in the result;
  * amplification — total JSON characters per character of real content.

Offline and deterministic: committed fixtures through a stubbed requests layer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fetch_url_fixtures"
sys.path.insert(0, str(FIXTURES))

from harness import GOLD  # noqa: E402

import abstractcore.tools.common_tools as ct  # noqa: E402

# The fetch is faked; the SSRF check still resolves the host first. Answer that
# from a fake resolver instead of real DNS (network guard finding, 2026-09-24).
pytestmark = pytest.mark.usefixtures("fake_public_dns")

# Amplification bar: what a caller pays, in JSON characters, per character of
# extracted content. Before the payload policy a 29k article cost 222k of JSON.
MAX_ENVELOPE_AMPLIFICATION = 2.0


class _FakeResponse:
    def __init__(self, *, url: str, body: bytes, content_type: str) -> None:
        self.url = url
        self.headers = {"content-type": content_type, "content-length": str(len(body))}
        self.status_code = 200
        self.reason = "OK"
        self.ok = True
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_content(self, chunk_size: int = 1):
        body = self._body
        for start in range(0, len(body), int(chunk_size)):
            yield body[start : start + int(chunk_size)]


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def mount(self, *args: object, **kwargs: object) -> None:
        return None

    def request(self, *args: object, **kwargs: object) -> _FakeResponse:
        return self._response


def _fetch_fixture(monkeypatch: pytest.MonkeyPatch, gold_key: str, **kwargs: object) -> dict:
    """Run the real `fetch_url` against a committed fixture, no network."""
    if not ct._ensure_requests():  # pragma: no cover - environment guard
        pytest.skip('requests not available; install with: pip install "abstractcore[tools]"')

    body = (FIXTURES / GOLD[gold_key]["fixture"]).read_bytes()
    url = GOLD[gold_key]["url"]
    response = _FakeResponse(url=url, body=body, content_type="text/html; charset=utf-8")

    real_requests = ct.requests

    class _Shim:
        exceptions = real_requests.exceptions
        adapters = real_requests.adapters

        @staticmethod
        def Session():  # noqa: N802 - mirrors requests.Session
            return _FakeSession(response)

        def __getattr__(self, name: str) -> object:
            return getattr(real_requests, name)

    monkeypatch.setattr(ct, "requests", _Shim())
    result = ct.fetch_url(url=url, timeout=10, **kwargs)
    assert result.get("success") is True, result.get("error")
    return result


# ---------------------------------------------------------------------------
# single canonical payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gold_key", ["techstartups", "wikipedia_bert", "bbc_tech_hub"])
def test_evidence_mirrors_are_withheld_with_a_receipt(monkeypatch, gold_key):
    result = _fetch_fixture(monkeypatch, gold_key)
    assert len(str(result["content"] or "")) > ct.FETCH_URL_MAX_INLINE_EVIDENCE_CHARS

    for field in ("raw_text", "normalized_text"):
        assert field in result, f"{field} must stay in the envelope for old consumers"
        assert result[field] is None, f"{field} is a duplicate of `content` and must not ship inline"
        descriptor = result[f"{field}_withheld"]
        assert set(descriptor) >= {"chars", "bytes", "sha256", "content_type", "reason"}
        assert descriptor["chars"] > ct.FETCH_URL_MAX_INLINE_EVIDENCE_CHARS
        assert len(descriptor["sha256"]) == 64


@pytest.mark.parametrize("gold_key", ["techstartups", "wikipedia_bert", "bbc_tech_hub", "pydocs_json"])
def test_envelope_amplification_is_bounded(monkeypatch, gold_key):
    result = _fetch_fixture(monkeypatch, gold_key)
    content_chars = len(str(result["content"] or ""))
    total = len(json.dumps(result, ensure_ascii=False, default=str))
    amplification = total / max(1, content_chars)
    assert amplification <= MAX_ENVELOPE_AMPLIFICATION, (
        f"{gold_key}: {amplification:.2f}x amplification "
        f"(content={content_chars}, total_json={total})"
    )


def test_rendered_is_a_header_not_a_second_copy(monkeypatch):
    """`rendered` quotes a preview and points at `content`; it never inlines it."""
    result = _fetch_fixture(monkeypatch, "techstartups")
    rendered = str(result["rendered"] or "")
    content = str(result["content"] or "")
    assert len(rendered) < 0.25 * len(content), f"rendered is {len(rendered)} chars"
    assert "📰 Title:" in rendered, "consumers parse the title out of rendered"
    assert "`content` field" in rendered, "rendered must point at the canonical payload"
    assert "<script" not in rendered.lower()


def test_content_chars_reports_the_payload_size(monkeypatch):
    result = _fetch_fixture(monkeypatch, "techstartups")
    assert result["content_chars"] == len(str(result["content"] or ""))


# ---------------------------------------------------------------------------
# link_dominant: the one place keep_links=False is overridden
# ---------------------------------------------------------------------------


def test_listing_page_reports_that_it_kept_links_anyway(monkeypatch):
    """A hub page's links ARE its content, so they survive `keep_links=False` —
    and the caller who asked for no links has to be able to SEE that."""
    result = _fetch_fixture(monkeypatch, "bbc_tech_hub", keep_links=False)
    assert result["link_dominant"] is True
    assert "](http" in str(result["content"] or ""), "a listing page must keep its URLs"


def test_article_respects_keep_links_false_and_says_so(monkeypatch):
    """The other side: an ordinary article is not link-dominant, so the flag is
    False and `keep_links=False` is honoured exactly as asked."""
    result = _fetch_fixture(monkeypatch, "techstartups", keep_links=False)
    assert result["link_dominant"] is False
    assert "](http" not in str(result["content"] or "")
