"""A llama.cpp decode failure mid-request (the context memory is full: "decode: failed to find a memory
slot for batch of size 512", then `RuntimeError("llama_decode returned 1")`) frees the context memory and
fails with the clear ContextWindowExceededError — never an "Error: llama_decode returned 1" answer, never a
context left full for the next request (OVH 2026-10-28: the same failure repeated on every later call)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from abstractcore.exceptions import ContextWindowExceededError
from abstractcore.providers.huggingface_provider import HuggingFaceProvider


class _Ctx:
    def __init__(self) -> None:
        self.cleared = []

    def memory_clear(self, data: bool) -> None:
        self.cleared.append(("memory_clear", data))


class _Llm:
    def __init__(self) -> None:
        self._ctx = _Ctx()
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def n_ctx(self) -> int:
        return 8192


def _provider() -> HuggingFaceProvider:
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    p.llm = _Llm()
    p.model = "unsloth/gemma-4-E4B-it-GGUF"
    p.logger = logging.getLogger("test")
    return p


def test_a_decode_failure_frees_the_memory_and_says_so() -> None:
    p = _provider()
    with pytest.raises(ContextWindowExceededError) as err:
        p._gguf_raise_if_decode_failed(RuntimeError("llama_decode returned 1"))
    assert p.llm.resets == 1 and p.llm._ctx.cleared == [("memory_clear", True)]
    text = str(err.value)
    assert "ran out of context memory" in text and "8,192-token context" in text and "memory freed" in text
    assert err.value.context_length == 8192
    # Anything else is not this failure: nothing freed, nothing raised.
    p2 = _provider()
    p2._gguf_raise_if_decode_failed(RuntimeError("something else"))
    assert p2.llm.resets == 0


def test_a_streamed_decode_failure_is_converted_too() -> None:
    p = _provider()

    def chunks():
        yield {"choices": [{"delta": {"content": "Hel"}}]}
        raise RuntimeError("llama_decode returned 1")

    out = []
    with pytest.raises(ContextWindowExceededError):
        for chunk in p._gguf_context_checked_stream(chunks()):
            out.append(chunk)
    assert len(out) == 1 and p.llm.resets == 1


def test_an_older_llama_cpp_python_is_cleared_with_kv_cache_clear() -> None:
    p = _provider()
    calls = []
    p.llm._ctx = SimpleNamespace(kv_cache_clear=lambda: calls.append("kv_cache_clear"))
    p._gguf_free_context_memory()
    assert calls == ["kv_cache_clear"]
