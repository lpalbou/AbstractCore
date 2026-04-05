from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.cache_utils import DynamicCache

from abstractcore.providers.base import BaseProvider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider, _TransformersPromptCacheValue


class _FakeToolHandler:
    supports_prompted = True

    def format_tools_prompt(self, tools: List[Any], *, include_tool_list: bool = True) -> str:
        names: List[str] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function") if isinstance(tool.get("function"), dict) else None
            name = ""
            if func:
                name = str(func.get("name") or "").strip()
            if not name:
                name = str(tool.get("name") or "").strip()
            if name:
                names.append(name)
        lines: List[str] = []
        if include_tool_list:
            lines.append("## Tools (session)")
        for name in names:
            lines.append(f"- {name}")
        return "\n".join(lines)


class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2
    add_bos_token = True

    def encode(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        _ = add_special_tokens
        return [3 + (ord(c) % 251) for c in str(text or "")]

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> Dict[str, Any]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, ids: List[int], *, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return "x" * len(ids or [])


class _FakeOutput:
    def __init__(self, cache: Any) -> None:
        self.past_key_values = cache


class _FakeGenerateOutput:
    def __init__(self, sequences: Any, cache: Any) -> None:
        self.sequences = sequences
        self.past_key_values = cache


class _FakeModel:
    def __init__(self, *, n_layers: int = 2, device: str = "cpu") -> None:
        self._n_layers = int(n_layers)
        self._param = torch.nn.Parameter(torch.empty(0, device=device))

    def parameters(self):
        yield self._param

    def __call__(
        self,
        *,
        input_ids: Any,
        attention_mask: Any = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> _FakeOutput:
        _ = (attention_mask, use_cache, kwargs)
        cache = past_key_values
        if cache is None:
            cache = DynamicCache()
        delta_len = int(getattr(input_ids, "shape", [1, 0])[1])
        for layer_idx in range(self._n_layers):
            key_states = torch.zeros((1, 1, delta_len, 1), device=input_ids.device, dtype=torch.float32)
            value_states = torch.zeros((1, 1, delta_len, 1), device=input_ids.device, dtype=torch.float32)
            cache.update(key_states, value_states, layer_idx=layer_idx)
        return _FakeOutput(cache)

    def generate(self, **kwargs: Any) -> _FakeGenerateOutput:
        input_ids = kwargs.get("input_ids")
        past = kwargs.get("past_key_values")
        delta_len = int(getattr(input_ids, "shape", [1, 0])[1])
        # Append a fixed 2-token completion.
        sequences = torch.cat([input_ids, torch.tensor([[101, 102]], device=input_ids.device)], dim=1)
        if past is None:
            past = DynamicCache()
        for layer_idx in range(self._n_layers):
            key_states = torch.zeros((1, 1, delta_len + 2, 1), device=input_ids.device, dtype=torch.float32)
            value_states = torch.zeros((1, 1, delta_len + 2, 1), device=input_ids.device, dtype=torch.float32)
            past.update(key_states, value_states, layer_idx=layer_idx)
        return _FakeGenerateOutput(sequences, past)


def _new_provider() -> HuggingFaceProvider:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(provider, "__abstractcore_generic_fallback__")
    provider.provider = "huggingface"
    provider.model_type = "transformers"
    provider.temperature = 0.2
    provider.tool_handler = _FakeToolHandler()
    provider.tokenizer = _FakeTokenizer()
    provider.model_instance = _FakeModel(device="cpu")
    provider.pipeline = object()
    provider.device = "cpu"
    provider._gguf_prompt_cache_lock = threading.Lock()
    provider._gguf_prompt_cache_default_capacity_bytes = 512 << 20
    provider._gguf_prompt_cache_pending_capacity_bytes = None
    return provider


def test_transformers_prompt_cache_capabilities_report_local_control_plane() -> None:
    provider = _new_provider()

    caps = provider.get_prompt_cache_capabilities()

    assert caps.supported is True
    assert caps.mode == "local_control_plane"
    assert caps.supports_prepare_modules is True
    assert provider.prompt_cache_supports_kv_source_of_truth() is True


def test_transformers_prompt_cache_prepare_modules_fork_update_and_save_load(tmp_path) -> None:
    provider = _new_provider()

    prepared = provider.prompt_cache_prepare_modules(
        namespace="tenant:model",
        modules=[
            {"module_id": "system", "system_prompt": "You are helpful."},
            {"module_id": "tools", "tools": [{"type": "function", "function": {"name": "shell"}}]},
        ],
        make_default=False,
    )
    assert prepared["supported"] is True
    prefix_key = prepared["final_cache_key"]

    prefix_state = provider._prompt_cache_store.get(prefix_key)
    assert isinstance(prefix_state, _TransformersPromptCacheValue)
    assert prefix_state.prompt_tokens

    assert provider.prompt_cache_fork(prefix_key, "sess", make_default=False) is True

    assert provider.prompt_cache_update(
        "sess",
        messages=[{"role": "user", "content": "hi"}],
        add_generation_prompt=False,
    ) is True

    session_state = provider._prompt_cache_store.get("sess")
    assert isinstance(session_state, _TransformersPromptCacheValue)
    assert session_state.prompt_tokens[: len(prefix_state.prompt_tokens)] == prefix_state.prompt_tokens
    assert len(session_state.prompt_tokens) > len(prefix_state.prompt_tokens)

    filename = tmp_path / "cache.safetensors"
    saved = provider.prompt_cache_save("sess", str(filename))
    assert saved["supported"] is True

    provider.prompt_cache_clear()
    loaded = provider.prompt_cache_load(str(filename), make_default=False)
    assert loaded["supported"] is True

    loaded_state = provider._prompt_cache_store.get(loaded["key"])
    assert isinstance(loaded_state, _TransformersPromptCacheValue)
    assert loaded_state.prompt_tokens

