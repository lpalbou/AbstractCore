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
def test_fetch_url_clamps_too_small_max_content_length(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.common_tools import fetch_url
    import abstractcore.tools.common_tools as common_tools

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
        max_content_length=5000,
    )
    assert out.get("success") is True
    assert out.get("requested_max_content_length") == 5000
    assert out.get("max_content_length") == 2 * 1024 * 1024
    assert out.get("truncated") is False
    assert out.get("size_bytes") == len(body)
    rendered = str(out.get("rendered") or "")
    assert "Requested max_content_length=5,000" in rendered
    assert "effective cap=2,097,152" in rendered


@pytest.mark.basic
def test_fetch_url_truncates_instead_of_error_when_content_exceeds_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.common_tools import fetch_url
    import abstractcore.tools.common_tools as common_tools

    cap = 2 * 1024 * 1024
    body = b"b" * (cap + 123)

    fake = _FakeResponse(
        url="http://example.com/bin",
        headers={
            "content-type": "application/octet-stream",
            "content-length": str(len(body)),
        },
        body=body,
    )
    monkeypatch.setattr(common_tools.requests, "Session", lambda: _FakeSession(fake))

    out = fetch_url(
        "http://example.com/bin",
        timeout=10,
        include_full_content=False,
        max_content_length=cap,
    )
    assert out.get("success") is True
    assert out.get("max_content_length") == cap
    assert out.get("truncated") is True
    assert out.get("size_bytes") == cap
    assert out.get("content_length") == len(body)
    rendered = str(out.get("rendered") or "")
    assert "Download truncated" in rendered
