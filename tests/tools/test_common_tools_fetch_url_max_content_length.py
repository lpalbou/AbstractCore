from __future__ import annotations

import pytest


class _FakeResponse:
    def __init__(self, *, url: str, headers: dict[str, str], body: bytes, status_code: int = 200, reason: str = "OK"):
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


@pytest.mark.basic
def test_fetch_url_respects_max_content_length_via_content_length_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.common_tools import fetch_url
    import abstractcore.tools.common_tools as common_tools

    monkeypatch.setattr(common_tools, "FETCH_URL_MAX_CONTENT_LENGTH_BYTES", 5000)
    body = b"a" * 10_000

    fake = _FakeResponse(
        url="http://example.com/small",
        headers={
            "content-type": "application/octet-stream",
            "content-length": str(len(body)),
        },
        body=body,
    )
    monkeypatch.setattr(common_tools.requests, "Session", lambda: _FakeSession(fake))

    out = fetch_url(
        "http://example.com/small",
        timeout=10,
        include_full_content=False,
    )
    assert out.get("success") is False
    assert out.get("error") == "Content too large"
    assert out.get("content_length") == len(body)
    assert out.get("max_content_length") == 5000
    rendered = str(out.get("rendered") or "")
    assert "Content too large" in rendered
    assert "max: 5,000" in rendered


@pytest.mark.basic
def test_fetch_url_respects_max_content_length_via_streaming_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.common_tools import fetch_url
    import abstractcore.tools.common_tools as common_tools

    cap = 5000
    monkeypatch.setattr(common_tools, "FETCH_URL_MAX_CONTENT_LENGTH_BYTES", cap)
    body = b"b" * (cap + 123)

    fake = _FakeResponse(
        url="http://example.com/bin",
        headers={
            "content-type": "application/octet-stream",
        },
        body=body,
    )
    monkeypatch.setattr(common_tools.requests, "Session", lambda: _FakeSession(fake))

    out = fetch_url(
        "http://example.com/bin",
        timeout=10,
        include_full_content=False,
    )
    assert out.get("success") is False
    assert out.get("error") == "Content exceeded size limit during download"
    assert out.get("max_content_length") == cap
    assert out.get("downloaded_size") == len(body)
    rendered = str(out.get("rendered") or "")
    assert "Content exceeded size limit during download" in rendered
