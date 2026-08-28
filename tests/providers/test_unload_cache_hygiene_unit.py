"""Unload hygiene: `unload_model` must drop the instance's session caches.

Session/prompt caches (KV store entries, MLX hybrid snapshots) are only useful
with the weights resident and are the memory hogs — an unload that strands
them frees almost nothing. CPU-only, model-free (repo `__new__` precedent).
"""

from __future__ import annotations

from abstractcore.providers.base import PromptCacheStore
from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.providers.mlx_provider import MLXProvider


def _mlx_provider_with_caches() -> MLXProvider:
    p = MLXProvider.__new__(MLXProvider)
    p.provider = "mlx"
    p.model = "mlx-test-model"
    p.llm = object()
    p.tokenizer = object()
    p._default_prompt_cache_key = "session-1"
    p._prompt_cache_store = PromptCacheStore(max_entries=8)
    p._ensure_hybrid_snapshot_state()
    p._prompt_cache_store.set("session-1", {"state": "a"})
    p._prompt_cache_store.set("session-2", {"state": "b"})
    p._store_hybrid_snapshot("session-1", object(), [1, 2, 3])
    return p


def test_mlx_unload_model_clears_prompt_cache_store_and_hybrid_snapshots() -> None:
    p = _mlx_provider_with_caches()
    assert len(p._prompt_cache_store.keys()) == 2
    assert len(p._hybrid_snapshots) == 1

    p.unload_model("mlx-test-model")

    assert p.llm is None
    assert p.tokenizer is None
    assert p._prompt_cache_store.keys() == []
    assert p._hybrid_snapshots == {}
    assert p._default_prompt_cache_key is None


def test_mlx_unload_model_reports_not_loaded_residency_afterwards() -> None:
    p = _mlx_provider_with_caches()

    p.unload_model("mlx-test-model")
    residency = p.get_model_residency()

    assert residency["provider_residency_verified"] is True
    assert residency["loaded"] is False


def _hf_provider_with_store(model_type: str) -> HuggingFaceProvider:
    p = object.__new__(HuggingFaceProvider)
    p.provider = "huggingface"
    p.model = "hf-test-model"
    p.model_type = model_type
    p.llm = None
    p.tokenizer = None
    p.processor = None
    p.model_instance = object()
    p.pipeline = object()
    p._default_prompt_cache_key = "chat-1"
    p._prompt_cache_store = PromptCacheStore(max_entries=8)
    p._prompt_cache_store.set("chat-1", {"state": "x"})
    return p


def test_huggingface_gguf_unload_model_clears_prompt_cache_store() -> None:
    p = _hf_provider_with_store("gguf")

    p.unload_model("hf-test-model")

    assert p.model_instance is None
    assert p.pipeline is None
    assert p._prompt_cache_store.keys() == []
    assert p._default_prompt_cache_key is None


def test_huggingface_transformers_unload_model_clears_prompt_cache_store() -> None:
    p = _hf_provider_with_store("transformers")

    p.unload_model("hf-test-model")

    assert p._prompt_cache_store.keys() == []
    assert p._default_prompt_cache_key is None
    # The pre-existing snapshot hygiene must stay intact alongside the store clear.
    assert dict(p._transformers_snapshots) == {}
