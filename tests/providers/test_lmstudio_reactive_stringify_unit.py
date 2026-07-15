"""Reactive LM Studio tool-argument stringify: latch + single retry, no new side effects.

Why reactive (2026-07-15, maintainer constraint "no new uncontrolled side
effects"): the earlier UNCONDITIONAL stringify in `LMStudioProvider._mutate_payload`
silently mutated the prompt for models whose chat template renders the WHOLE
arguments dict (`arguments | tojson` — Llama-3.2/Qwen3/Granite/Ministral
conventions): their prior args got quoted (`"n": 10` -> `"n": "10"`). The
transform is only needed when LM Studio's minja-class engine PROVABLY cannot
render the standard payload (per-argument `tojson | safe` convention:
Qwen3-Coder/Ornith — HTTP 400 "Unknown StringValue filter: safe" /
"Cannot apply filter 'string' to type: NullValue").

Contract pinned here (mirrors the `stream_options`/`prompt_cache_key`
rejection-latch house pattern):
- first call sends the STANDARD wire;
- a template-render failure latches `_lmstudio_minja_arg_stringify_needed`,
  emits ONE #FALLBACK warning, and retries the SAME request ONCE with
  stringified tool-call history args (sync + async, stream + non-stream);
  the stream lanes catch BOTH failure shapes: a connection-time HTTP 400 and
  LM Studio's live shape — HTTP 200 whose FIRST SSE event carries the error
  (repair only before anything was yielded; mid-generation errors stay loud);
- once latched, later calls stringify proactively (no wasted first attempt,
  no further reactive retries);
- BOUNDED: at most one reactive retry per call; a failing retry raises and the
  outer RetryManager does not multiply it (render-400s raise
  InvalidRequestError, a non-retryable class); streams bypass RetryManager
  entirely (the generator is returned unconsumed), so the in-stream reactive
  retry is the ONLY stream retry;
- NO-OP guard: when the transform would change nothing, the retry cannot help
  a render-400 (different cause) and is skipped;
- LM Studio ONLY: the base OpenAICompatibleProvider hook never repairs;
  vLLM/OpenRouter/Portkey inherit the base hook;
- warning reconciliation: the reactive path owns the retry warning;
  `_raise_for_status`'s template warning fires only on TERMINAL failures.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
from typing import Any, Dict, List, Optional

import pytest

from abstractcore.exceptions import InvalidRequestError
from abstractcore.providers.lmstudio_provider import (
    LMStudioProvider,
    stringify_tool_call_history_argument_values,
)
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.openrouter_provider import OpenRouterProvider
from abstractcore.providers.portkey_provider import PortkeyProvider
from abstractcore.providers.vllm_provider import VLLMProvider

RENDER_400_BODY = {"error": 'Error rendering prompt with jinja template: "Unknown StringValue filter: safe".'}
OK_BODY = {
    "choices": [{"message": {"content": "All good."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}
SSE_LINES = [
    'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}',
    "data: [DONE]",
]


def _history(arguments):
    return [
        {"role": "user", "content": "what are the news today"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "id": "call_1", "function": {"name": "web_search", "arguments": arguments}}
            ],
        },
        {"role": "tool", "content": "1. Example headline.", "tool_call_id": "call_1"},
    ]


def _typed_history():
    return _history(json.dumps({"query": "news", "num_results": 10}))


class _Resp:
    def __init__(self, status_code: int, body: Optional[Dict[str, Any]] = None, lines: Optional[List[str]] = None):
        self.status_code = status_code
        self._body = body or {}
        self._lines = lines or []

    def json(self) -> Dict[str, Any]:
        return self._body

    @property
    def text(self) -> str:
        return json.dumps(self._body)

    def read(self) -> bytes:
        return self.text.encode("utf-8")

    async def aread(self) -> bytes:
        return self.text.encode("utf-8")

    def iter_lines(self):
        yield from self._lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _payload_has_nonstring_history_args(payload: Dict[str, Any]) -> bool:
    """Behavioral LM Studio simulation: the render 400 fires whenever the
    replayed assistant tool-call history carries any NON-STRING argument value
    (the `tojson | safe` branch); all-string values render via `| string`."""
    for message in payload.get("messages") or []:
        if not (isinstance(message, dict) and message.get("role") == "assistant"):
            continue
        for call in message.get("tool_calls") or []:
            container = call.get("function") if isinstance(call.get("function"), dict) else call
            raw = container.get("arguments")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    continue
            if isinstance(raw, dict) and any(not isinstance(v, str) for v in raw.values()):
                return True
    return False


class _MinjaClient:
    """Fake LM Studio: render error on non-string history args; success otherwise.

    `stream_error_shape` selects how the STREAM lane reports the failure:
    - "http400": connection-time HTTP 400 (spec-conservative shape);
    - "event": HTTP 200 whose FIRST SSE event is `data: {"error": ...}` —
      the shape the real LM Studio produces (live-verified 2026-07-15).
    """

    def __init__(
        self,
        fail_always: bool = False,
        error_body: Optional[Dict[str, Any]] = None,
        stream_error_shape: str = "http400",
    ):
        self.requests: List[Dict[str, Any]] = []
        self.fail_always = fail_always
        self.error_body = error_body or RENDER_400_BODY
        self.stream_error_shape = stream_error_shape

    def _should_fail(self, payload: Dict[str, Any]) -> bool:
        return self.fail_always or _payload_has_nonstring_history_args(payload)

    def _stream_response(self, payload: Dict[str, Any]) -> _Resp:
        if not self._should_fail(payload):
            return _Resp(200, {}, lines=SSE_LINES)
        if self.stream_error_shape == "event":
            return _Resp(200, {}, lines=["data: " + json.dumps(self.error_body)])
        return _Resp(400, self.error_body)

    def post(self, url, json=None, headers=None, **kwargs):  # noqa: A002 - httpx signature
        payload = dict(json or {})
        self.requests.append(copy.deepcopy(payload))
        if self._should_fail(payload):
            return _Resp(400, self.error_body)
        return _Resp(200, OK_BODY)

    @contextlib.contextmanager
    def stream(self, method, url, json=None, headers=None, **kwargs):  # noqa: A002
        payload = dict(json or {})
        self.requests.append(copy.deepcopy(payload))
        yield self._stream_response(payload)


class _AsyncMinjaClient(_MinjaClient):
    async def post(self, url, json=None, headers=None, **kwargs):  # noqa: A002
        payload = dict(json or {})
        self.requests.append(copy.deepcopy(payload))
        if self._should_fail(payload):
            return _Resp(400, self.error_body)
        return _Resp(200, OK_BODY)

    def stream(self, method, url, json=None, headers=None, **kwargs):  # noqa: A002
        payload = dict(json or {})
        self.requests.append(copy.deepcopy(payload))
        outer = self

        class _CM:
            async def __aenter__(self_inner):
                return outer._stream_response(payload)

            async def __aexit__(self_inner, *exc):
                return False

        return _CM()


class _CapLogger:
    def __init__(self):
        self.warnings: List[str] = []

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _provider(monkeypatch, client: Optional[_MinjaClient] = None) -> LMStudioProvider:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    p = LMStudioProvider(model="ornith-1.0-35b", base_url="http://127.0.0.1:9/v1")
    p.logger = _CapLogger()
    if client is not None:
        p.client = client
    return p


def _wire_history_args(payload: Dict[str, Any]):
    assistant = [m for m in payload["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    raw = assistant[0]["tool_calls"][0]["function"]["arguments"]
    return json.loads(raw) if isinstance(raw, str) else raw


def _reactive_warnings(p) -> List[str]:
    return [w for w in p.logger.warnings if "#FALLBACK" in w and "stringif" in w.lower()]


def _terminal_warnings(p) -> List[str]:
    return [w for w in p.logger.warnings if "RENDERING" in w]


# ---------------------------------------------------------------------------
# Sync non-stream: reactive retry, latch persistence, single warning
# ---------------------------------------------------------------------------


def test_render_400_triggers_latch_warn_and_single_retry_sync(monkeypatch):
    client = _MinjaClient()
    p = _provider(monkeypatch, client)

    response = p.generate("Summarize.", messages=_typed_history(), max_output_tokens=32)

    # Exactly two requests THROUGH the full generate() path (RetryManager did
    # not multiply the reactive retry into N logical attempts).
    assert len(client.requests) == 2
    assert _wire_history_args(client.requests[0]) == {"query": "news", "num_results": 10}  # standard wire first
    assert _wire_history_args(client.requests[1]) == {"query": "news", "num_results": "10"}  # stringified retry
    assert response.content == "All good."
    assert p._lmstudio_minja_arg_stringify_needed is True
    assert len(_reactive_warnings(p)) == 1  # exactly ONE #FALLBACK warning
    assert not _terminal_warnings(p)  # retry succeeded -> no terminal template warning


def test_latched_instance_stringifies_proactively_no_second_retry(monkeypatch):
    client = _MinjaClient()
    p = _provider(monkeypatch, client)

    p.generate("Summarize.", messages=_typed_history(), max_output_tokens=32)  # latches (2 requests)
    p.generate("Summarize.", messages=_typed_history(), max_output_tokens=32)  # proactive (1 request)

    assert len(client.requests) == 3
    assert _wire_history_args(client.requests[2]) == {"query": "news", "num_results": "10"}
    assert len(_reactive_warnings(p)) == 1  # no re-warn on the proactive lane


def test_retry_also_failing_raises_original_class_and_does_not_loop(monkeypatch):
    client = _MinjaClient(fail_always=True)
    p = _provider(monkeypatch, client)

    with pytest.raises(InvalidRequestError):
        p.generate("Summarize.", messages=_typed_history(), max_output_tokens=32)

    # One reactive retry, then terminal: exactly 2 requests total even through
    # the RetryManager (InvalidRequestError is non-retryable by classification).
    assert len(client.requests) == 2
    # Reconciliation: reactive path warned about the retry; _raise_for_status
    # owns the TERMINAL template warning (fired on the retry's response).
    assert len(_reactive_warnings(p)) == 1
    assert len(_terminal_warnings(p)) == 1


def test_noop_transform_skips_retry_and_keeps_latch_false(monkeypatch):
    # All-string args: the stringify changes nothing, so the render-400 has a
    # different cause and burning a retry could not help.
    client = _MinjaClient(fail_always=True)
    p = _provider(monkeypatch, client)

    with pytest.raises(InvalidRequestError):
        p.generate("Summarize.", messages=_history(json.dumps({"query": "news"})), max_output_tokens=32)

    assert len(client.requests) == 1  # no retry burned
    assert p._lmstudio_minja_arg_stringify_needed is False
    assert not _reactive_warnings(p)
    assert len(_terminal_warnings(p)) == 1  # terminal warning still fires


def test_unrelated_400_never_retries_or_latches(monkeypatch):
    client = _MinjaClient(fail_always=True, error_body={"error": "invalid request: bad parameter"})
    p = _provider(monkeypatch, client)

    with pytest.raises(InvalidRequestError):
        p.generate("Summarize.", messages=_typed_history(), max_output_tokens=32)

    assert len(client.requests) == 1
    assert p._lmstudio_minja_arg_stringify_needed is False
    assert not _reactive_warnings(p)


def test_cannot_apply_filter_error_shape_also_detected(monkeypatch):
    body = {"error": "Cannot apply filter 'string' to type: NullValue"}
    client = _MinjaClient(error_body=body)
    p = _provider(monkeypatch, client)

    response = p.generate("Summarize.", messages=_typed_history(), max_output_tokens=32)

    assert len(client.requests) == 2
    assert response.content == "All good."
    assert p._lmstudio_minja_arg_stringify_needed is True


# ---------------------------------------------------------------------------
# Streaming (sync): 400 surfaces at connection time; retry re-establishes
# ---------------------------------------------------------------------------


def test_sync_stream_render_400_retries_and_reestablishes_stream(monkeypatch):
    client = _MinjaClient()
    p = _provider(monkeypatch, client)

    payload = {"model": p.model, "messages": _typed_history(), "stream": True}
    chunks = list(p._stream_generate(payload))

    assert any(c.content == "hi" for c in chunks)  # stream re-established after retry
    assert len(client.requests) == 2
    assert _wire_history_args(client.requests[0]) == {"query": "news", "num_results": 10}
    assert _wire_history_args(client.requests[1]) == {"query": "news", "num_results": "10"}
    assert p._lmstudio_minja_arg_stringify_needed is True
    assert len(_reactive_warnings(p)) == 1


def test_sync_stream_retry_also_failing_raises_no_loop(monkeypatch):
    client = _MinjaClient(fail_always=True)
    p = _provider(monkeypatch, client)

    with pytest.raises(InvalidRequestError):
        list(p._stream_generate({"model": p.model, "messages": _typed_history(), "stream": True}))

    assert len(client.requests) == 2  # bounded: the recursion guard is the latch


# ---------------------------------------------------------------------------
# Streaming, LIVE LM Studio shape: HTTP 200 whose FIRST SSE event is the error
# ---------------------------------------------------------------------------


def test_sync_stream_error_event_repairs_before_first_yield(monkeypatch):
    client = _MinjaClient(stream_error_shape="event")
    p = _provider(monkeypatch, client)

    chunks = list(p._stream_generate({"model": p.model, "messages": _typed_history(), "stream": True}))

    assert any(c.content == "hi" for c in chunks)
    assert len(client.requests) == 2
    assert _wire_history_args(client.requests[1]) == {"query": "news", "num_results": "10"}
    assert p._lmstudio_minja_arg_stringify_needed is True
    assert len(_reactive_warnings(p)) == 1


def test_async_stream_error_event_repairs_before_first_yield(monkeypatch):
    client = _AsyncMinjaClient(stream_error_shape="event")
    p = _provider(monkeypatch)
    p._async_client = client

    async def _collect():
        return [c async for c in p._async_stream_generate({"model": p.model, "messages": _typed_history(), "stream": True})]

    chunks = asyncio.run(_collect())

    assert any(c.content == "hi" for c in chunks)
    assert len(client.requests) == 2
    assert _wire_history_args(client.requests[1]) == {"query": "news", "num_results": "10"}
    assert p._lmstudio_minja_arg_stringify_needed is True


def test_sync_stream_error_event_retry_also_failing_raises_no_loop(monkeypatch):
    from abstractcore.exceptions import ProviderAPIError

    client = _MinjaClient(fail_always=True, stream_error_shape="event")
    p = _provider(monkeypatch, client)

    with pytest.raises(ProviderAPIError):
        list(p._stream_generate({"model": p.model, "messages": _typed_history(), "stream": True}))

    assert len(client.requests) == 2  # bounded via the latch, exactly one retry


def test_mid_generation_error_event_after_yield_stays_loud_no_retry(monkeypatch):
    """An error AFTER tokens were yielded (e.g. model eviction) is never repaired:
    a silent re-request would replay already-delivered content."""
    from abstractcore.exceptions import ProviderAPIError

    client = _MinjaClient()
    p = _provider(monkeypatch, client)
    lines = [
        'data: {"choices": [{"delta": {"content": "partial"}, "finish_reason": null}]}',
        "data: " + json.dumps(RENDER_400_BODY),
    ]
    client._stream_response = lambda payload: _Resp(200, {}, lines=lines)

    received = []
    with pytest.raises(ProviderAPIError):
        for chunk in p._stream_generate({"model": p.model, "messages": _typed_history(), "stream": True}):
            received.append(chunk)

    assert [c.content for c in received] == ["partial"]
    assert len(client.requests) == 1  # no repair once anything was yielded
    assert p._lmstudio_minja_arg_stringify_needed is False


def test_unrelated_pre_yield_error_event_raises_without_retry(monkeypatch):
    from abstractcore.exceptions import ProviderAPIError

    client = _MinjaClient(
        fail_always=True, stream_error_shape="event", error_body={"error": {"message": "Model unloaded."}}
    )
    p = _provider(monkeypatch, client)

    with pytest.raises(ProviderAPIError, match="Model unloaded"):
        list(p._stream_generate({"model": p.model, "messages": _typed_history(), "stream": True}))

    assert len(client.requests) == 1
    assert p._lmstudio_minja_arg_stringify_needed is False


# ---------------------------------------------------------------------------
# Async parity (non-stream + stream)
# ---------------------------------------------------------------------------


def test_async_single_render_400_retries_once(monkeypatch):
    client = _AsyncMinjaClient()
    p = _provider(monkeypatch)
    p._async_client = client

    payload = {"model": p.model, "messages": _typed_history(), "stream": False}
    response = asyncio.run(p._async_single_generate(payload))

    assert response.content == "All good."
    assert len(client.requests) == 2
    assert _wire_history_args(client.requests[1]) == {"query": "news", "num_results": "10"}
    assert p._lmstudio_minja_arg_stringify_needed is True


def test_async_stream_render_400_retries_once(monkeypatch):
    client = _AsyncMinjaClient()
    p = _provider(monkeypatch)
    p._async_client = client

    async def _collect():
        out = []
        async for chunk in p._async_stream_generate({"model": p.model, "messages": _typed_history(), "stream": True}):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())

    assert any(c.content == "hi" for c in chunks)
    assert len(client.requests) == 2
    assert _wire_history_args(client.requests[1]) == {"query": "news", "num_results": "10"}
    assert p._lmstudio_minja_arg_stringify_needed is True


def test_async_single_retry_also_failing_is_bounded(monkeypatch):
    client = _AsyncMinjaClient(fail_always=True)
    p = _provider(monkeypatch)
    p._async_client = client

    with pytest.raises(InvalidRequestError):
        asyncio.run(p._async_single_generate({"model": p.model, "messages": _typed_history(), "stream": False}))

    assert len(client.requests) == 2


# ---------------------------------------------------------------------------
# Scope: LM Studio only — base + siblings never repair
# ---------------------------------------------------------------------------


def test_base_hook_never_repairs_and_siblings_inherit_it():
    base_hook = OpenAICompatibleProvider._render_400_repaired_payload
    for cls in (VLLMProvider, OpenRouterProvider, PortkeyProvider, OpenAICompatibleProvider):
        assert cls._render_400_repaired_payload is base_hook, f"{cls.__name__} must not repair render-400s"
    assert LMStudioProvider._render_400_repaired_payload is not base_hook

    p = OpenAICompatibleProvider(
        model="test-model", base_url="http://127.0.0.1:9/v1", api_key="x", validate_model=False
    )
    payload = {"model": p.model, "messages": _typed_history(), "stream": False}
    assert p._render_400_repaired_payload(_Resp(400, RENDER_400_BODY), payload) is None


def test_openai_compatible_render_400_fails_without_retry(monkeypatch):
    p = OpenAICompatibleProvider(
        model="test-model", base_url="http://127.0.0.1:9/v1", api_key="x", validate_model=False
    )
    p.logger = _CapLogger()
    client = _MinjaClient(fail_always=True)
    p.client = client

    with pytest.raises(InvalidRequestError):
        p._single_generate({"model": p.model, "messages": _typed_history(), "stream": False})

    assert len(client.requests) == 1  # never mutated, never retried
    assert len(_terminal_warnings(p)) == 1  # _raise_for_status still warns loudly


# ---------------------------------------------------------------------------
# Repair-hook properties: idempotency, payload preservation, latch semantics
# ---------------------------------------------------------------------------


def test_transform_is_idempotent_second_apply_is_identity():
    once = stringify_tool_call_history_argument_values(_typed_history())
    twice = stringify_tool_call_history_argument_values(once)
    assert twice is once  # no change on re-apply -> same object, no double-encoding
    args = json.loads(twice[1]["tool_calls"][0]["function"]["arguments"])
    assert args == {"query": "news", "num_results": "10"}


def test_latched_mutate_payload_reapply_is_stable(monkeypatch):
    p = _provider(monkeypatch)
    p._lmstudio_minja_arg_stringify_needed = True

    payload1 = p._mutate_payload({"model": p.model, "messages": _typed_history(), "stream": False})
    payload2 = p._mutate_payload(dict(payload1))

    assert _wire_history_args(payload1) == {"query": "news", "num_results": "10"}
    assert payload2["messages"] is payload1["messages"]  # idempotent re-apply


def test_repaired_payload_preserves_everything_except_messages(monkeypatch):
    p = _provider(monkeypatch)
    payload = {
        "model": p.model,
        "messages": _typed_history(),
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 128,
        "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {}}}],
        "tool_choice": "auto",
        "seed": 7,
    }
    original = copy.deepcopy(payload)

    repaired = p._render_400_repaired_payload(_Resp(400, RENDER_400_BODY), payload)

    assert repaired is not None and repaired is not payload
    assert payload == original  # caller payload never mutated
    for key in payload:
        if key == "messages":
            continue
        assert repaired[key] == payload[key]  # output-cap/tools/params ride along untouched
    assert _wire_history_args(repaired) == {"query": "news", "num_results": "10"}


def test_hook_returns_none_once_latched_recursion_guard(monkeypatch):
    p = _provider(monkeypatch)
    payload = {"model": p.model, "messages": _typed_history(), "stream": False}

    first = p._render_400_repaired_payload(_Resp(400, RENDER_400_BODY), payload)
    assert first is not None and p._lmstudio_minja_arg_stringify_needed is True

    # Latched: the same render-400 (e.g. on the retried payload in the stream
    # recursion, or any later call) must never mint another retry.
    assert p._render_400_repaired_payload(_Resp(400, RENDER_400_BODY), payload) is None
    assert p._render_400_repaired_payload(_Resp(400, RENDER_400_BODY), first) is None
    assert len(_reactive_warnings(p)) == 1


def test_hook_ignores_non_400_and_missing_messages(monkeypatch):
    p = _provider(monkeypatch)
    assert p._render_400_repaired_payload(_Resp(500, RENDER_400_BODY), {"messages": _typed_history()}) is None
    assert p._render_400_repaired_payload(_Resp(200, {}), {"messages": _typed_history()}) is None
    assert p._render_400_repaired_payload(_Resp(400, RENDER_400_BODY), {"messages": []}) is None
    assert p._render_400_repaired_payload(_Resp(400, RENDER_400_BODY), {}) is None
    assert p._lmstudio_minja_arg_stringify_needed is False
