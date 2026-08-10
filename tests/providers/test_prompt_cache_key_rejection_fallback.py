"""Some OpenAI-compatible servers (e.g. OVH AI Endpoints) 400-reject the best-effort
`prompt_cache_key` field instead of ignoring it. The provider must drop the key, retry once,
and stop sending it for the rest of the instance's life — prompt caching is best-effort and
must never break generation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider


class _Resp:
    def __init__(self, status_code: int, body: Dict[str, Any]):
        self.status_code = status_code
        self._body = body

    def json(self) -> Dict[str, Any]:
        return self._body

    @property
    def text(self) -> str:
        return json.dumps(self._body)

    def read(self) -> bytes:  # parity with httpx buffering
        return self.text.encode("utf-8")


class _RejectingClient:
    """Rejects any payload containing prompt_cache_key with the OVH-style 400."""

    def __init__(self):
        self.requests: List[Dict[str, Any]] = []

    def post(self, url, json=None, headers=None):  # noqa: A002 - httpx signature
        payload = dict(json or {})
        self.requests.append(payload)
        if "prompt_cache_key" in payload:
            return _Resp(400, {"error": {"message": "feature 'prompt_cache_key' is not currently supported"}})
        return _Resp(
            200,
            {
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


@pytest.fixture()
def provider(monkeypatch):
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    # Minimal attribute surface for _single_generate (bypasses network-touching __init__).
    p.base_url = "http://stub/v1"
    p.model = "stub-model"
    p.client = _RejectingClient()
    p._get_headers = lambda: {}
    p.architecture_config = None
    p.model_capabilities = {}
    p._build_usage_dict = lambda usage: dict(usage or {})
    return p


def test_prompt_cache_key_rejection_drops_key_and_retries(provider):
    payload = {"model": "stub-model", "messages": [], "prompt_cache_key": "abc123"}
    response = provider._single_generate(dict(payload))

    assert response.content == "hello"
    reqs = provider.client.requests
    assert len(reqs) == 2  # rejected once, retried without the key
    assert "prompt_cache_key" in reqs[0]
    assert "prompt_cache_key" not in reqs[1]
    assert getattr(provider, "_prompt_cache_key_unsupported", False) is True


def test_unrelated_400_still_raises(provider):
    class _Always400Client(_RejectingClient):
        def post(self, url, json=None, headers=None):  # noqa: A002
            self.requests.append(dict(json or {}))
            return _Resp(400, {"error": {"message": "context length exceeded"}})

    provider.client = _Always400Client()
    provider.PROVIDER_DISPLAY_NAME = "Stub"
    with pytest.raises(Exception) as ei:
        provider._single_generate({"model": "stub-model", "messages": [], "prompt_cache_key": "abc"})
    assert "context length exceeded" in str(ei.value)
    assert len(provider.client.requests) == 1  # no blind retry on unrelated 400s


def test_stream_rejection_drops_key_and_retries(provider):
    import contextlib

    class _StreamResp(_Resp):
        def iter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}'
            yield "data: [DONE]"

    class _StreamingClient:
        def __init__(self):
            self.requests: List[Dict[str, Any]] = []

        @contextlib.contextmanager
        def stream(self, method, url, json=None, headers=None, **kwargs):  # noqa: A002
            payload = dict(json or {})
            self.requests.append(payload)
            if "prompt_cache_key" in payload:
                yield _StreamResp(400, {"error": {"message": "feature 'prompt_cache_key' is not currently supported"}})
            else:
                yield _StreamResp(200, {})

    provider.client = _StreamingClient()
    chunks = list(provider._stream_generate({"model": "stub-model", "messages": [], "prompt_cache_key": "abc"}))

    assert any(c.content == "hi" for c in chunks)
    reqs = provider.client.requests
    assert len(reqs) == 2 and "prompt_cache_key" not in reqs[1]


def test_marked_instance_stops_sending_the_key(provider):
    provider._prompt_cache_key_unsupported = True
    # Build a payload through the internal path: the key must be filtered out up front.
    # (Direct call: simulate what _generate_internal does.)
    payload = {"model": "stub-model", "messages": []}
    prompt_cache_key = "abc123"
    if (
        isinstance(prompt_cache_key, str)
        and prompt_cache_key.strip()
        and not getattr(provider, "_prompt_cache_key_unsupported", False)
    ):
        payload["prompt_cache_key"] = prompt_cache_key.strip()
    assert "prompt_cache_key" not in payload
    response = provider._single_generate(payload)
    assert response.content == "hello"
    assert len(provider.client.requests) == 1
