from __future__ import annotations

import sys
import types
from pathlib import Path

from abstractcore.providers.base import PromptCacheStore
from abstractcore.providers.mlx_provider import MLXProvider


class _FakeCacheLayer:
    def __init__(self, *, size: int, offset: int = 0) -> None:
        self._size = int(size)
        self.offset = int(offset)

    def size(self) -> int:
        return self._size


class _FakeTokenizer:
    bos_token = "<bos>"


class _FakeToolHandler:
    supports_prompted = False


def test_mlx_prompt_cache_token_count_uses_max_hybrid_layer_offset() -> None:
    provider = MLXProvider.__new__(MLXProvider)
    cache = [
        _FakeCacheLayer(size=1024, offset=3837),
        _FakeCacheLayer(size=3837, offset=3837),
        _FakeCacheLayer(size=1024, offset=3837),
    ]

    assert provider._prompt_cache_backend_token_count(cache) == 3837


def test_mlx_gemma4_prompt_fragment_uses_turn_template_and_optional_bos() -> None:
    provider = MLXProvider.__new__(MLXProvider)
    provider.model = "mlx-community/gemma-4-26b-a4b-4bit"
    provider.architecture_config = {"message_format": "gemma_turn"}
    provider.tokenizer = _FakeTokenizer()
    provider.tool_handler = _FakeToolHandler()

    initial = provider._build_prompt_fragment(
        messages=[{"role": "user", "content": "FILEBOX"}],
        add_generation_prompt=True,
    )
    suffix = provider._build_prompt_fragment(
        prompt="QUESTION",
        add_generation_prompt=True,
        include_bos=False,
    )

    assert initial == "<bos><|turn>user\nFILEBOX<turn|>\n<|turn>model\n"
    assert suffix == "<|turn>user\nQUESTION<turn|>\n<|turn>model\n"


def test_mlx_gemma4_postprocess_truncates_turn_delimiter() -> None:
    provider = MLXProvider.__new__(MLXProvider)
    provider.architecture_config = {
        "message_format": "gemma_turn",
        "assistant_suffix": "<turn|>\n",
    }
    provider.model_capabilities = {}

    cleaned, reasoning = provider._postprocess_generated_text("answer<turn|>\n<|turn>user\nnext")

    assert cleaned == "answer"
    assert reasoning is None


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
