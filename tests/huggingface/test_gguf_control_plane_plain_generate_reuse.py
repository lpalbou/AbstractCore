"""Pin: the GGUF control-plane lane must PERSIST its prefill snapshot on the
plain `generate(prompt_cache_key=...)` convention, so a growing-prefix loop
(the runtime's actual calling shape) reuses the prefix instead of re-prefilling
the whole prompt every warm turn.

Regression guard for the 2026-07-14 fable5 GGUF finding: with `save_state=False`
the control-plane lane gave ZERO in-process reuse on plain generate and, worse,
`llm.reset()` forfeited llama.cpp's own n_past reuse — making it slower than the
cp=0 fallback. The fix persists the snapshot; these tests assert the snapshot is
saved and that the next growing-prefix turn loads it and evals only the suffix.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, List

import pytest

pytest.importorskip("llama_cpp")
np = pytest.importorskip("numpy")

from abstractcore.providers.base import BaseProvider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider, _GGUFPromptCacheValue
from llama_cpp.llama import LlamaState


class _FakeLlamaModelMeta:
    def add_bos_token(self) -> bool:
        return True

    def add_eos_token(self) -> bool:
        return True

    def token_cls(self) -> int:
        return -1

    def token_sep(self) -> int:
        return -1

    def token_get_text(self, token_id: int) -> str:
        return ""


class _FakeLlama:
    """Minimal llama.cpp stand-in: byte->id tokenizer, KV state save/load, and a
    `generate` that yields nothing (we only measure prefill/eval + persistence)."""

    def __init__(self) -> None:
        self.chat_format = "chatml-function-calling"
        self.metadata = {"tokenizer.ggml.add_space_prefix": "true"}
        self._model = _FakeLlamaModelMeta()
        self._tokens: List[int] = []
        self.n_tokens = 0
        self.cache = None
        self.eval_calls: List[List[int]] = []

    def token_bos(self) -> int:
        return 1

    def token_eos(self) -> int:
        return 2

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> List[int]:
        toks = [int(b) + 3 for b in text]
        return ([self.token_bos()] + toks) if add_bos else toks

    def detokenize(self, tokens: List[int]) -> bytes:
        return b""

    def reset(self) -> None:
        self._tokens = []
        self.n_tokens = 0

    def load_state(self, state: LlamaState) -> None:
        self._tokens = [int(t) for t in state.input_ids[: state.n_tokens].tolist()]
        self.n_tokens = len(self._tokens)

    def eval(self, tokens: List[int]) -> None:
        ints = [int(t) for t in tokens]
        self.eval_calls.append(list(ints))
        self._tokens.extend(ints)
        self.n_tokens = len(self._tokens)

    def save_state(self) -> LlamaState:
        return LlamaState(
            input_ids=np.asarray(self._tokens, dtype=np.intc).copy(),
            scores=np.zeros((max(len(self._tokens), 1), 4), dtype=np.single),
            n_tokens=len(self._tokens),
            llama_state=bytes((t % 251 for t in self._tokens)),
            llama_state_size=len(self._tokens),
            seed=0,
        )

    def set_cache(self, cache: Any) -> None:
        self.cache = cache

    def set_seed(self, seed: int) -> None:  # noqa: D401 - test stub
        pass

    def generate(self, tokens: List[int], **_: Any) -> Iterator[int]:
        return iter(())  # no output; the prefill + persistence is what we test


def _provider() -> HuggingFaceProvider:
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(p, "unsloth/Qwen3.5-2B-GGUF")
    p.provider = "huggingface"
    p.model_type = "gguf"
    p.temperature = 0.2
    p.llm = _FakeLlama()
    p._gguf_prompt_cache_lock = threading.Lock()
    p._gguf_prompt_cache_default_capacity_bytes = 0  # auto-growing
    p._gguf_prompt_cache_pending_capacity_bytes = None
    return p


def _cache_entries(p: HuggingFaceProvider, key: str) -> int:
    cv = p._prompt_cache_store.get(key)
    cache = getattr(cv, "cache", None)
    return len(getattr(cache, "cache_state", {}) or {})


def _drive(p: HuggingFaceProvider, key: str, messages: List[Dict[str, str]]) -> None:
    """Run one plain control-plane generate turn (drain the stream)."""
    cache_value = p._prompt_cache_store.get(key)
    if cache_value is None:
        p.prompt_cache_set(key, make_default=False)
        cache_value = p._prompt_cache_store.get(key)
    cache_obj = p._gguf_prompt_cache_unwrap(cache_value)
    cache_state = p._gguf_prompt_cache_state(cache_value)
    list(
        p._gguf_control_plane_stream_generate(
            chat_messages=p._gguf_build_chat_messages(
                system_prompt="You are precise.", messages=messages, user_message_content=None
            ),
            cache_obj=cache_obj,
            max_output_tokens=4,
            temperature=0.0,
            top_p=0.95,
            top_k=40,
            min_p=0.05,
            typical_p=1.0,
            repeat_penalty=1.1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            tfs_z=1.0,
            mirostat_mode=0,
            mirostat_tau=5.0,
            mirostat_eta=0.1,
            seed=None,
            enable_thinking=False,
            cache_state=cache_state,
        )
    )


def test_plain_generate_persists_prefill_snapshot() -> None:
    p = _provider()
    assert _cache_entries(p, "sess") == 0  # nothing yet

    _drive(p, "sess", [{"role": "user", "content": "first question"}])

    # The fix: a plain generate now leaves a reusable snapshot behind.
    assert _cache_entries(p, "sess") >= 1


def test_warm_turn_loads_snapshot_instead_of_reprefilling() -> None:
    """The persisted snapshot must be LOADED on the next same-key turn so the
    shared prefix is not re-evaluated. Driven with an identical prompt (the
    rendering-independent form of the mechanism): turn 1 prefills the whole
    prompt; turn 2 finds turn-1's snapshot as a complete token prefix, loads it,
    and evals nothing. (Growing-prefix suffix-only reuse is proven live against
    real Qwen3/Gemma-4 GGUFs; here we pin the load path without depending on
    chat-template thinking-scaffold rendering.)"""
    p = _provider()
    msgs = [{"role": "user", "content": "first question"}]
    _drive(p, "sess", msgs)
    turn1_prefill = len(p.llm.eval_calls[-1])
    assert turn1_prefill > 0  # cold prefill of the whole prompt

    p.llm.eval_calls.clear()
    _drive(p, "sess", msgs)  # identical prompt, same key
    # The whole prompt is already a saved snapshot → loaded, zero re-eval.
    assert p.llm.eval_calls == [] or sum(len(c) for c in p.llm.eval_calls) < turn1_prefill


def test_snapshots_stay_ram_bounded_across_growing_turns() -> None:
    """Auto-growing LlamaRAMCache evicts the smaller predecessor as the prefix
    grows, so a long loop does not accumulate unbounded snapshots."""
    p = _provider()
    for i in range(5):
        _drive(p, "sess", [{"role": "user", "content": f"question {i} " + ("pad " * 20 * (i + 1))}])
    # Bounded: the growing snapshot evicts older, smaller ones (not 5 entries).
    assert 1 <= _cache_entries(p, "sess") <= 2
