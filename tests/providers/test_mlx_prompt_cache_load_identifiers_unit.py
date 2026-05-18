from __future__ import annotations

import sys
import types
from pathlib import Path

from abstractcore.providers.base import PromptCacheStore
from abstractcore.providers.mlx_provider import MLXProvider


def test_mlx_prompt_cache_load_accepts_equivalent_resolved_model_id(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("mlx_lm.models.cache")

    def _load_prompt_cache(filename: str, return_metadata: bool = True):
        _ = (filename, return_metadata)
        return ["cache-layer"], {"model": "/resolved/qwen3", "model_resolved_id": "/resolved/qwen3"}

    module.load_prompt_cache = _load_prompt_cache  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", module)

    provider = MLXProvider.__new__(MLXProvider)
    provider.model = "mlx-community/Qwen3.6-27B-4bit"
    provider.provider = "mlx"
    provider._resolved_model_id = "/resolved/qwen3"
    provider._prompt_cache_store = PromptCacheStore()
    provider._default_prompt_cache_key = None

    result = provider.prompt_cache_load(str(tmp_path / "cache.safetensors"), key="k1", make_default=False)

    assert result["key"] == "k1"
    meta = provider._prompt_cache_store.meta("k1") or {}
    assert meta.get("model_resolved_id") == "/resolved/qwen3"
