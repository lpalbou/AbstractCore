"""MLX: a STREAMED call reports the same prompt-cache telemetry as a sync call.

The runtime refuses to stream a call whose streamed record would lack
`metadata["prompt_cache"]` (its stream-parity gate). Until 2026-09-26 the MLX
provider attached that record on the sync lane only, so every gateway chat call
on MLX with a prompt-cache key ran non-streamed. Contract pinned here:

* every MLX lane ends a stream with ONE terminal chunk carrying `finish_reason`,
  `usage` and — with a prompt-cache key — `metadata["prompt_cache"]`, and that
  chunk is the stream's last;
* the record is identical to the sync lane's for the same prompt and key
  (same helper, same inputs);
* on the native APC lanes the counts are only known once generation has run,
  so the record is built after the source generator is exhausted.

Fakes only (no model load); the live twin is
`test_mlx_stream_prompt_cache_parity_live.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.mlx_provider import MLXProvider


class _Tok:
    def encode(self, text: str) -> List[int]:
        return list(range(len(str(text).split())))


def _mlx_lm_provider(words: List[str]) -> MLXProvider:
    """mlx-lm lane (no native processor/runtime) over a fake generator."""
    p = MLXProvider.__new__(MLXProvider)
    p.model = "fake/mlx-lm"
    p.logger = Mock()
    p.llm = object()
    p.tokenizer = _Tok()
    p._mtp_processor = None
    p._native_runtime = None
    p._build_mlx_sampler = lambda *a, **k: None

    def stream_generate_fn(model, tokenizer, prompt, **kwargs):
        for i, w in enumerate(words):
            yield SimpleNamespace(text=w, generation_tokens=i + 1, prompt_tokens=7,
                                  finish_reason="stop" if i == len(words) - 1 else None)

    def generate_fn(model, tokenizer, prompt=None, **kwargs):
        return "".join(words)

    p.stream_generate_fn = stream_generate_fn
    p.generate_fn = generate_fn
    return p


KEY_TELEMETRY = {"mode": "key", "key": "k", "outcome": "hit_extend", "cached_tokens": 4, "fed_tokens": 7}


def test_mlx_lm_stream_ends_with_the_sync_lanes_accounting_and_prompt_cache():
    words = ["Hello", " there", " friend"]
    p = _mlx_lm_provider(words)
    sync = p._single_generate("the prompt", 16, 0.0, 1.0, usage_prompt="the full prompt")
    sync_cache = p._final_prompt_cache_telemetry(dict(KEY_TELEMETRY))

    chunks = list(p._stream_generate("the prompt", 16, 0.0, 1.0,
                                     usage_prompt="the full prompt",
                                     prompt_cache_telemetry=dict(KEY_TELEMETRY)))
    assert "".join(c.content for c in chunks) == sync.content
    terminal = chunks[-1]
    assert terminal.content == ""
    assert terminal.finish_reason == sync.finish_reason == "stop"
    assert terminal.usage == sync.usage
    assert terminal.metadata["prompt_cache"] == sync_cache == KEY_TELEMETRY
    assert all(c.finish_reason is None and c.usage is None for c in chunks[:-1])
    assert all("prompt_cache" not in (c.metadata or {}) for c in chunks[:-1])


def test_mlx_lm_stream_reports_length_like_the_sync_lane():
    p = _mlx_lm_provider(["a", " b", " c"])
    sync = p._single_generate("q", 3, 0.0, 1.0)
    chunks = list(p._stream_generate("q", 3, 0.0, 1.0))
    assert sync.finish_reason == "length"
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].usage == sync.usage
    assert chunks[-1].metadata is None  # no key, no prompt_cache record


def test_failed_stream_has_no_terminal_accounting_chunk():
    p = _mlx_lm_provider(["x"])

    def boom(*a, **k):
        yield SimpleNamespace(text="x", generation_tokens=1, prompt_tokens=1, finish_reason=None)
        raise RuntimeError("metal oom")

    p.stream_generate_fn = boom
    chunks = list(p._stream_generate("q", 8, 0.0, 1.0, prompt_cache_telemetry=dict(KEY_TELEMETRY)))
    assert chunks[-1].finish_reason == "error"
    assert all("prompt_cache" not in (c.metadata or {}) for c in chunks)


def _apc_provider() -> MLXProvider:
    """In-process mlx-vlm APC lane: counts exist only once the generator is exhausted."""
    p = MLXProvider.__new__(MLXProvider)
    p.model = "fake/mlx-vlm"
    p.logger = Mock()
    p.llm = object()
    p.tokenizer = _Tok()
    p._mtp_processor = object()
    p._native_runtime = None
    p._mtp_last_result = None
    p._mtp_last_apc = None
    counters = {"exact_stores": 1, "stores": 0, "memory_skips": 0, "rejects": 0,
                "memory_max_bytes": 1 << 30, "resident_bytes": 1 << 20}

    def result(text: str, n: int) -> SimpleNamespace:
        return SimpleNamespace(text=text, prompt_tokens=40, cached_tokens=32, generation_tokens=n,
                               prompt_tps=1.0, generation_tps=2.0, peak_memory=0.5,
                               finish_reason=None)

    def stream_generate_fn(model, tokenizer, prompt, **kwargs):
        try:
            for i, w in enumerate(["Hi", " you"]):
                r = result(w, i + 1)
                p._mtp_last_result = r
                yield r
        finally:
            # like `_mtp_stream_generate_fn.observed`: APC counters after the end
            p._mtp_last_apc = dict(counters)

    def generate_fn(model, tokenizer, prompt=None, **kwargs):
        p._mtp_last_result = result("Hi you", 2)
        p._mtp_last_apc = dict(counters)
        return "Hi you"

    p.stream_generate_fn = stream_generate_fn
    p.generate_fn = generate_fn
    return p


def test_native_apc_stream_builds_the_record_after_the_generator_is_exhausted():
    p = _apc_provider()
    base = {"mode": "key", "key": "k", "backend": "mlx_vlm_apc"}
    sync = p._single_generate("q", 16, 0.0, 1.0)
    sync_cache = p._final_prompt_cache_telemetry(dict(base))

    p._mtp_last_result = None
    p._mtp_last_apc = None
    chunks = list(p._stream_generate("q", 16, 0.0, 1.0, prompt_cache_telemetry=dict(base)))
    terminal = chunks[-1]
    assert terminal.metadata["prompt_cache"] == sync_cache
    assert terminal.metadata["prompt_cache"]["cached_tokens"] == 32
    assert terminal.metadata["prompt_cache"]["fed_tokens"] == 8
    assert terminal.metadata["prompt_cache"]["outcome"] == "hit_restore"
    assert terminal.metadata["prompt_cache"]["apc"]["exact_stores"] == 1
    assert terminal.usage == sync.usage
    assert terminal.finish_reason == sync.finish_reason
    assert terminal.metadata["performance"] == sync.metadata["performance"]


def test_native_runtime_final_result_carries_the_record():
    p = MLXProvider.__new__(MLXProvider)
    p.model = "fake/native"
    p.logger = Mock()
    p.llm = object()
    p.tokenizer = _Tok()
    p._mtp_processor = object()
    p._native_runtime = object()
    p._mtp_last_apc = None

    def stream_generate_fn(model, tokenizer, prompt, **kwargs):
        for i, (w, fr) in enumerate([("A", None), ("B", "stop")]):
            r = SimpleNamespace(text=w, prompt_tokens=20, cached_tokens=16, generation_tokens=i + 1,
                                finish_reason=fr, metadata={})
            p._mtp_last_result = r
            yield r

    p.stream_generate_fn = stream_generate_fn
    chunks = list(p._stream_generate("q", 16, 0.0, 1.0,
                                     prompt_cache_telemetry={"mode": "key", "key": "k", "backend": "mlx_vlm_apc"}))
    assert [c.finish_reason for c in chunks] == [None, "stop"]
    assert "prompt_cache" not in (chunks[0].metadata or {})
    assert chunks[-1].metadata["prompt_cache"] == {
        "mode": "key", "key": "k", "backend": "mlx_vlm_apc",
        "cached_tokens": 16, "fed_tokens": 4, "outcome": "hit_restore",
    }


def test_tools_wrapper_keeps_the_terminal_chunk_last():
    p = _mlx_lm_provider(["one", " two"])
    p.tool_handler = SimpleNamespace(supports_prompted=False)
    chunks = list(p._stream_generate_with_tools("q", 16, 0.0, 1.0, None, None,
                                                prompt_cache_telemetry=dict(KEY_TELEMETRY)))
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].metadata["prompt_cache"] == KEY_TELEMETRY
    assert "".join(c.content for c in chunks) == "one two"


def test_tool_execution_text_comes_before_the_terminal_chunk():
    """Provider-side tool execution appends result text AFTER the model's stream;
    the terminal (accounting) chunk must still be the last one, and the only one
    with a finish_reason."""
    p = _mlx_lm_provider(["call", " it"])
    p.tool_handler = SimpleNamespace(supports_prompted=True)
    p._handle_prompted_tool_execution = lambda response, tools, **kw: GenerateResponse(
        content=response.content + "\n[tool result]", model=p.model, finish_reason="stop")
    chunks = list(p._stream_generate_with_tools("q", 16, 0.0, 1.0, None, [{"name": "t"}],
                                                prompt_cache_telemetry=dict(KEY_TELEMETRY)))
    assert "".join(c.content for c in chunks) == "call it\n[tool result]"
    assert [c.finish_reason for c in chunks].count("stop") == 1
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].metadata["prompt_cache"] == KEY_TELEMETRY


def test_concurrent_native_requests_never_swap_their_counts():
    """Two interleaved scheduled requests on ONE provider: each terminal chunk
    reports its own request's cached/fed counts (per-request view state)."""
    p = MLXProvider.__new__(MLXProvider)
    p.model = "fake/native"
    p.logger = Mock()
    p.llm = object()
    p.tokenizer = _Tok()
    p._mtp_processor = object()
    p._mtp_last_result = SimpleNamespace(cached_tokens=999, prompt_tokens=999)  # stale, must not leak
    p._mtp_last_apc = None

    class _Handle:
        def __init__(self, results):
            self._it = iter(results)

        def __iter__(self):
            return self._it

        def close(self):
            pass

    class _Runtime:
        def stream(self, request, cancel_event=None, **kw):
            cached = request["cached"]
            return _Handle([
                SimpleNamespace(text="a", prompt_tokens=100, cached_tokens=cached, generation_tokens=1,
                                finish_reason=None, metadata={}, media_records=()),
                SimpleNamespace(text="b", prompt_tokens=100, cached_tokens=cached, generation_tokens=2,
                                finish_reason="stop", metadata={}, media_records=()),
            ])

    p._native_runtime = _Runtime()
    p._native_runtime_request = lambda text, kwargs: {"cached": 10 if text == "one" else 70}
    p._record_native_media = lambda: None
    one, two = p._native_request_view(), p._native_request_view()
    assert one._mtp_last_result is None and two._mtp_last_result is None
    base = {"mode": "key", "backend": "mlx_vlm_apc"}
    s1 = one._stream_generate("one", 8, 0.0, 1.0, prompt_cache_telemetry=dict(base, key="one"))
    s2 = two._stream_generate("two", 8, 0.0, 1.0, prompt_cache_telemetry=dict(base, key="two"))
    out1, out2 = [], []
    for a, b in zip(s1, s2):  # strictly interleaved
        out1.append(a)
        out2.append(b)
    assert out1[-1].metadata["prompt_cache"]["cached_tokens"] == 10
    assert out1[-1].metadata["prompt_cache"]["fed_tokens"] == 90
    assert out2[-1].metadata["prompt_cache"]["cached_tokens"] == 70
    assert out2[-1].metadata["prompt_cache"]["fed_tokens"] == 30
