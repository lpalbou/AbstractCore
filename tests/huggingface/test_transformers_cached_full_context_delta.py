"""HF transformers cached lane: full-context callers get delta discipline.

Live find (three-backend cache confirmation, 2026-07-12): the cached lane
(`_single_generate_transformers_cached`) only consumed `messages` on the FIRST
call of a key; every warm call rendered a delta from `prompt` alone. The
runtime/ReAct shape passes the WHOLE transcript via `messages` with
`prompt=""` every call — so warm calls fed an empty-ish fragment over the
stale cache and the model answered the PREVIOUS question (wrong content), with
`messages` silently ignored.

Contract (parity with the MLX delta lane): `messages is not None` = full
context. Warm calls LCP the newly rendered transcript against the recorded
`state.prompt_tokens`, crop the cache back to the shared prefix
(`DynamicCache.crop`), and feed ONLY the suffix. Divergence with a
crop-refusing cache (hybrids) rebuilds fresh — one cold prefill, never a
stale-context answer. Prompt-only callers keep the append lane unchanged.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.cache_utils import DynamicCache

from abstractcore.providers.base import BaseProvider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider


class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2
    add_bos_token = True

    def encode(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        _ = add_special_tokens
        # Word-level, deterministic, collision-free enough for LCP tests.
        return [3 + (hash(w) % 50_000) for w in str(text or "").split()]

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> Dict[str, Any]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, ids: List[int], *, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join("tok" for _ in (ids or []))


class _RecordingModel:
    """Counts prefill/generate token loads so the delta is observable."""

    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.empty(0, device="cpu"))
        self.fed_lengths: List[int] = []

    def parameters(self):
        yield self._param

    def __call__(self, *, input_ids, attention_mask=None, past_key_values=None, use_cache=True, **kw):
        _ = (attention_mask, use_cache, kw)
        cache = past_key_values if past_key_values is not None else DynamicCache()
        n = int(input_ids.shape[1])
        self.fed_lengths.append(n)
        for layer_idx in range(2):
            k = torch.zeros((1, 1, n, 1), dtype=torch.float32)
            cache.update(k, k.clone(), layer_idx=layer_idx)

        class _Out:
            pass

        out = _Out()
        out.past_key_values = cache
        return out

    def generate(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        past = kwargs.get("past_key_values") or DynamicCache()
        n = int(input_ids.shape[1])
        self.fed_lengths.append(n)
        sequences = torch.cat([input_ids, torch.tensor([[101, 102]])], dim=1)
        for layer_idx in range(2):
            k = torch.zeros((1, 1, n + 2, 1), dtype=torch.float32)
            past.update(k, k.clone(), layer_idx=layer_idx)

        class _Out:
            pass

        out = _Out()
        out.sequences = sequences
        out.past_key_values = past
        return out


def _provider() -> HuggingFaceProvider:
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(p, "__abstractcore_generic_fallback__")
    p.provider = "huggingface"
    p.model_type = "transformers"
    p.temperature = 0.0
    p.tokenizer = _FakeTokenizer()
    p.model_instance = _RecordingModel()
    p.pipeline = object()
    p.device = "cpu"
    p._gguf_prompt_cache_lock = threading.Lock()
    p._gguf_prompt_cache_default_capacity_bytes = 512 << 20
    p._gguf_prompt_cache_pending_capacity_bytes = None
    # Chat template: plain concatenation so token streams extend cleanly.
    p._transformers_build_prompt_fragment = (  # type: ignore[method-assign]
        lambda *, prompt="", messages=None, system_prompt=None, tools=None,
               add_generation_prompt=False, prefilled_modules=None, enable_thinking=None:
        " ".join(
            ([f"sys {system_prompt}"] if system_prompt else [])
            + [f"{m.get('role')} {m.get('content')}" for m in (messages or [])]
            + ([f"user {prompt}"] if prompt else [])
            + (["assistant"] if add_generation_prompt else [])
        )
    )
    return p


def _cached_call(p: HuggingFaceProvider, msgs: List[Dict[str, str]], key: str = "sess"):
    return p._single_generate_transformers_cached(
        prompt="",
        prompt_cache_key=key,
        messages=msgs,
        system_prompt="anchor rules",
        tools=None,
        prefilled_modules=None,
        max_new_tokens=8,
        temperature=0.0,
        top_p=0.9,
    )


def test_full_context_warm_call_feeds_only_the_suffix():
    p = _provider()
    ledger = " ".join(f"fact{i} value{i};" for i in range(120))   # long shared prefix
    t1 = [{"role": "user", "content": ledger}, {"role": "assistant", "content": "a one"}]

    r1 = _cached_call(p, t1)
    assert r1.finish_reason == "stop"
    first_fed = p.model_instance.fed_lengths[-1]

    # Warm: transcript grew by one exchange — only the new tail (plus the
    # crop-then-refeed of the previous generation-prompt tokens) may be fed,
    # never the whole transcript again.
    t2 = t1 + [{"role": "user", "content": "q two"}, {"role": "assistant", "content": "a two"},
               {"role": "user", "content": "q three"}]
    r2 = _cached_call(p, t2)
    assert r2.finish_reason == "stop"
    second_fed = p.model_instance.fed_lengths[-1]

    assert first_fed > 200                                # the prefix dominates
    assert second_fed < 30, (first_fed, second_fed)       # ~the new tail only
    # Usage still reports the full logical prompt.
    assert r2.usage["input_tokens"] > second_fed


def test_full_context_warm_call_sees_the_new_messages():
    """The stale-context bug: pre-fix, messages were ignored on warm calls."""
    p = _provider()
    t1 = [{"role": "user", "content": "alpha"}]
    _cached_call(p, t1)

    t2 = t1 + [{"role": "assistant", "content": "beta"}, {"role": "user", "content": "gamma"}]
    _cached_call(p, t2)

    # The recorded token stream must END with the rendered new tail —
    # impossible if messages were ignored.
    state = p._transformers_prompt_cache_state(p._prompt_cache_store.get("sess"))
    rendered_tail = p.tokenizer.encode("assistant beta user gamma assistant")
    recorded = list(state.prompt_tokens)
    # generation adds 2 fixed tokens (101, 102) after the prompt
    assert recorded[-len(rendered_tail) - 2:-2] == rendered_tail


def test_full_context_divergence_without_crop_rebuilds_fresh():
    p = _provider()
    t1 = [{"role": "user", "content": "left branch words here"}]
    _cached_call(p, t1)

    # Break crop support (hybrid-cache shape).
    state = p._transformers_prompt_cache_state(p._prompt_cache_store.get("sess"))
    state.cache.crop = None  # type: ignore[attr-defined]

    t2 = [{"role": "user", "content": "right branch words now"}]
    r = _cached_call(p, t2)
    assert r.finish_reason == "stop"

    state2 = p._transformers_prompt_cache_state(p._prompt_cache_store.get("sess"))
    rendered = p.tokenizer.encode("sys anchor rules user right branch words now assistant")
    # Fresh rebuild: the recorded stream is exactly the new transcript (+2 gen).
    assert len(state2.prompt_tokens) == len(rendered) + 2 + 1  # +1 BOS


def test_crop_verify_refuses_no_op_crops():
    """Adversarial find: transformers no-ops crop on some hybrid layer classes
    (zamba keeps even attention KV). 'Didn't raise' is not 'was exact' — the
    post-crop length verify must refuse, or warm calls run on wrong context."""
    from abstractcore.providers.huggingface_provider import (
        HuggingFaceProvider,
        _TransformersPromptCacheValue,
    )

    p = _provider()

    class _NoOpCropCache:
        def __init__(self, length):
            self._len = length
        def crop(self, n):
            pass  # silently keeps everything (the zamba shape)
        def get_seq_length(self):
            return self._len

    state = _TransformersPromptCacheValue(cache=_NoOpCropCache(100))
    assert p._transformers_crop_cache(state, 40) is False   # verify caught it

    class _ExactCropCache:
        def __init__(self, length):
            self._len = length
        def crop(self, n):
            self._len = min(self._len, int(n))
        def get_seq_length(self):
            return self._len

    state2 = _TransformersPromptCacheValue(cache=_ExactCropCache(100))
    assert p._transformers_crop_cache(state2, 40) is True
    assert state2.cache.get_seq_length() == 40


def test_linear_layer_crop_warns_once():
    """Hybrids that crop attention exactly but keep linear state are an
    accepted APPROXIMATION — labeled once, never silent."""
    from abstractcore.providers.huggingface_provider import _TransformersPromptCacheValue

    p = _provider()
    warnings_log = []
    p.logger = type("L", (), {
        "warning": lambda self, msg, *a, **k: warnings_log.append(str(msg)),
        "debug": lambda self, *a, **k: None,
    })()

    class _LinearLayer:
        pass
    _LinearLayer.__name__ = "Qwen3_5LinearAttentionLayer"

    class _HybridCache:
        def __init__(self):
            self._len = 100
            self.layers = [_LinearLayer(), object()]
        def crop(self, n):
            self._len = min(self._len, int(n))
        def get_seq_length(self):
            return self._len

    state = _TransformersPromptCacheValue(cache=_HybridCache())
    assert p._transformers_crop_cache(state, 40) is True
    assert len([w for w in warnings_log if "#FALLBACK" in w]) == 1

    state2 = _TransformersPromptCacheValue(cache=_HybridCache())
    p._transformers_crop_cache(state2, 20)
    assert len([w for w in warnings_log if "#FALLBACK" in w]) == 1  # once per provider


def test_prompt_only_callers_keep_the_append_lane():
    p = _provider()
    r = p._single_generate_transformers_cached(
        prompt="turn one words",
        prompt_cache_key="sess",
        messages=None,
        system_prompt=None,
        tools=None,
        prefilled_modules=None,
        max_new_tokens=8,
        temperature=0.0,
        top_p=0.9,
    )
    assert r.finish_reason == "stop"
    fed_first = p.model_instance.fed_lengths[-1]

    r2 = p._single_generate_transformers_cached(
        prompt="turn two words",
        prompt_cache_key="sess",
        messages=None,
        system_prompt=None,
        tools=None,
        prefilled_modules=None,
        max_new_tokens=8,
        temperature=0.0,
        top_p=0.9,
    )
    assert r2.finish_reason == "stop"
    # Append lane: the second fragment is fed as-is (similar size), no rebuild.
    assert p.model_instance.fed_lengths[-1] <= fed_first + 2
