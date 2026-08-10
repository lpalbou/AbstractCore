"""Streamed OpenAI-compatible calls must not go dark on usage accounting.

Adversarial find (code seat, 2026-07-12): every streamed call landed with
prompt_tokens=None because (1) the request never asked for the usage chunk
(`stream_options: {"include_usage": true}` is the standard OpenAI mechanism),
and (2) even a server that volunteered usage lost it in the parser — the
usage-bearing final chunk has EMPTY `choices` and was silently skipped, and
content chunks never mapped `usage` into GenerateResponse.

Contract pinned here:
- streamed payloads carry stream_options.include_usage (sync + async builders);
- the parser surfaces usage from BOTH shapes: usage riding the last content
  chunk (LM Studio style) and a final empty-choices usage chunk (OpenAI
  stream_options style);
- strict servers that 400-reject `stream_options` get drop-and-retry with a
  per-instance latch (same best-effort discipline as `prompt_cache_key`);
- non-streamed requests never carry stream_options.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, Dict, List

import pytest

from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider


class _Resp:
    def __init__(self, status_code: int, body: Dict[str, Any], lines: List[str] = None):
        self.status_code = status_code
        self._body = body
        self._lines = lines or []

    def json(self) -> Dict[str, Any]:
        return self._body

    @property
    def text(self) -> str:
        return json.dumps(self._body)

    def read(self) -> bytes:
        return self.text.encode("utf-8")

    def iter_lines(self):
        yield from self._lines


class _StreamingClient:
    """Serves a canned SSE line sequence; optionally rejects stream_options."""

    def __init__(self, lines: List[str], reject_stream_options: bool = False):
        self.lines = lines
        self.reject_stream_options = reject_stream_options
        self.requests: List[Dict[str, Any]] = []

    @contextlib.contextmanager
    def stream(self, method, url, json=None, headers=None, **kwargs):  # noqa: A002 - httpx signature
        payload = dict(json or {})
        self.requests.append(payload)
        if self.reject_stream_options and "stream_options" in payload:
            yield _Resp(400, {"error": {"message": "unknown field 'stream_options'"}})
        else:
            yield _Resp(200, {}, lines=self.lines)


def _provider() -> OpenAICompatibleProvider:
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.base_url = "http://stub/v1"
    p.model = "stub-model"
    p._get_headers = lambda: {}
    p.architecture_config = None
    p.model_capabilities = {}
    return p


# --- parser: usage reaches the consumer in both server shapes ---------------


def test_stream_surfaces_usage_from_final_empty_choices_chunk():
    """OpenAI stream_options shape: last chunk has choices=[] and usage."""
    p = _provider()
    p.client = _StreamingClient([
        'data: {"choices": [{"delta": {"content": "hel"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128}}',
        "data: [DONE]",
    ])

    chunks = list(p._stream_generate({"model": "stub-model", "messages": [], "stream": True}))

    assert "".join(c.content or "" for c in chunks) == "hello"
    usage_chunks = [c for c in chunks if c.usage]
    assert len(usage_chunks) == 1
    usage = usage_chunks[0].usage
    assert usage["prompt_tokens"] == 120 and usage["input_tokens"] == 120
    assert usage["completion_tokens"] == 8 and usage["output_tokens"] == 8
    assert usage["total_tokens"] == 128
    assert usage_chunks[0].content == ""      # accounting chunk carries no text


def test_stream_surfaces_usage_riding_last_content_chunk():
    """LM Studio shape: usage attached to the final content-bearing chunk."""
    p = _provider()
    p.client = _StreamingClient([
        'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}], '
        '"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}',
        "data: [DONE]",
    ])

    chunks = list(p._stream_generate({"model": "stub-model", "messages": [], "stream": True}))

    assert chunks[0].content == "hi"
    assert chunks[0].usage["total_tokens"] == 12


def test_stream_without_usage_stays_none():
    """No fabricated zeros: servers that report nothing yield usage=None."""
    p = _provider()
    p.client = _StreamingClient([
        'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}',
        "data: [DONE]",
    ])

    chunks = list(p._stream_generate({"model": "stub-model", "messages": [], "stream": True}))

    assert all(c.usage is None for c in chunks)


def test_mid_stream_error_event_raises_provider_api_error():
    """Live find (2026-07-13 operator drive): LM Studio evicted the model
    MID-STREAM and sent `data: {"error": {"message": "Model unloaded."}}` —
    the old parser skipped it (no choices, no usage), so the stream ended
    looking like a normal stop and the consumer kept a TRUNCATED answer with
    no signal. Error events must raise ProviderAPIError (the retryable class),
    never end the stream silently."""
    from abstractcore.exceptions import ProviderAPIError

    p = _provider()
    p.client = _StreamingClient([
        'data: {"choices": [{"delta": {"content": "half an ans"}, "finish_reason": null}]}',
        'data: {"error": {"message": "Model unloaded."}}',
        "data: [DONE]",
    ])

    with pytest.raises(ProviderAPIError) as e:
        list(p._stream_generate({"model": "stub-model", "messages": [], "stream": True}))
    assert "Model unloaded" in str(e.value)


def test_mid_stream_error_event_string_shape_also_raises():
    """Some servers send `{"error": "text"}` (string, not object) — same raise."""
    from abstractcore.exceptions import ProviderAPIError

    p = _provider()
    p.client = _StreamingClient([
        'data: {"error": "backend crashed"}',
        "data: [DONE]",
    ])

    with pytest.raises(ProviderAPIError, match="backend crashed"):
        list(p._stream_generate({"model": "stub-model", "messages": [], "stream": True}))


# --- request builder: stream_options only on streamed requests --------------


def test_payload_builder_adds_stream_options_only_when_streaming():
    p = _provider()
    captured: List[Dict[str, Any]] = []

    def _capture_stream(payload):
        captured.append(payload)
        return iter(())

    def _capture_single(payload):
        captured.append(payload)
        from abstractcore.core.types import GenerateResponse
        return GenerateResponse(content="ok", model="stub-model", finish_reason="stop")

    p._stream_generate = _capture_stream
    p._single_generate = _capture_single
    p.tool_handler = type("T", (), {"supports_native": False, "supports_prompted": False})()
    p.temperature = 0.7
    p.execute_tools = False
    p._prepare_generation_kwargs = lambda **kw: dict(kw)
    p._get_provider_max_tokens_param = lambda kw: 100
    p._normalize_system_messages_for_strict_servers = lambda msgs: msgs
    p._apply_model_parameter_constraints = lambda payload: payload
    p._mutate_payload = lambda payload, **kw: payload

    p._generate_internal("hello", stream=True)
    p._generate_internal("hello", stream=False)

    assert captured[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in captured[1]


def test_latched_instance_stops_sending_stream_options():
    p = _provider()
    p._stream_options_unsupported = True
    # Mirror the builder's guard directly (as the prompt_cache_key twin does).
    payload: Dict[str, Any] = {"model": "stub-model", "messages": [], "stream": True}
    if True and not getattr(p, "_stream_options_unsupported", False):
        payload["stream_options"] = {"include_usage": True}
    assert "stream_options" not in payload


# --- strict-server rejection: drop, latch, retry -----------------------------


def test_stream_options_rejection_drops_retries_and_latches():
    p = _provider()
    p.client = _StreamingClient(
        ['data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}', "data: [DONE]"],
        reject_stream_options=True,
    )

    chunks = list(
        p._stream_generate(
            {"model": "stub-model", "messages": [], "stream": True, "stream_options": {"include_usage": True}}
        )
    )

    assert any(c.content == "hi" for c in chunks)
    reqs = p.client.requests
    assert len(reqs) == 2                       # rejected once, retried without
    assert "stream_options" in reqs[0]
    assert "stream_options" not in reqs[1]
    assert getattr(p, "_stream_options_unsupported", False) is True


def test_unrelated_stream_400_still_raises():
    p = _provider()
    p.PROVIDER_DISPLAY_NAME = "Stub"

    class _Always400(_StreamingClient):
        @contextlib.contextmanager
        def stream(self, method, url, json=None, headers=None, **kwargs):  # noqa: A002
            self.requests.append(dict(json or {}))
            yield _Resp(400, {"error": {"message": "context length exceeded"}})

    p.client = _Always400([])
    with pytest.raises(Exception) as ei:
        list(
            p._stream_generate(
                {"model": "stub-model", "messages": [], "stream": True, "stream_options": {"include_usage": True}}
            )
        )
    assert "context length exceeded" in str(ei.value)
    assert len(p.client.requests) == 1          # no blind retry on unrelated 400s


# --- async twin ---------------------------------------------------------------


def test_async_stream_surfaces_usage_and_retries_on_rejection():
    import asyncio

    class _AsyncResp(_Resp):
        async def aread(self) -> bytes:
            return self.read()

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _AsyncStreamingClient:
        def __init__(self, lines, reject_stream_options=False):
            self.lines = lines
            self.reject_stream_options = reject_stream_options
            self.requests: List[Dict[str, Any]] = []

        @contextlib.asynccontextmanager
        async def stream(self, method, url, json=None, headers=None, **kwargs):  # noqa: A002
            payload = dict(json or {})
            self.requests.append(payload)
            if self.reject_stream_options and "stream_options" in payload:
                yield _AsyncResp(400, {"error": {"message": "unknown field 'stream_options'"}})
            else:
                yield _AsyncResp(200, {}, lines=self.lines)

    async def _run():
        p = _provider()
        # async_client is a lazy read-only property over _async_client.
        p._async_client = _AsyncStreamingClient(
            [
                'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}',
                'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}',
                "data: [DONE]",
            ],
            reject_stream_options=True,
        )
        chunks = []
        async for c in p._async_stream_generate(
            {"model": "stub-model", "messages": [], "stream": True, "stream_options": {"include_usage": True}}
        ):
            chunks.append(c)
        return p, chunks

    p, chunks = asyncio.run(_run())

    assert any(c.content == "hi" for c in chunks)
    usage_chunks = [c for c in chunks if c.usage]
    assert len(usage_chunks) == 1 and usage_chunks[0].usage["total_tokens"] == 6
    assert len(p.async_client.requests) == 2    # rejected once, retried without
    assert "stream_options" not in p.async_client.requests[1]
    assert getattr(p, "_stream_options_unsupported", False) is True
