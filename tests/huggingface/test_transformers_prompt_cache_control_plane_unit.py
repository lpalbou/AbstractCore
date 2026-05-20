from __future__ import annotations

import threading
from types import SimpleNamespace
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


def test_transformers_zero_temperature_pipeline_uses_greedy_decoding() -> None:
    provider = _new_provider()
    captured: Dict[str, Any] = {}

    class _Pipeline:
        def __call__(self, text: str, **kwargs: Any) -> List[Dict[str, str]]:
            captured["text"] = text
            captured.update(kwargs)
            return [{"generated_text": "ok"}]

    provider.pipeline = _Pipeline()

    response = provider._single_generate_transformers("hello", 8, 0.0, 0.95, seed=None)

    assert response.finish_reason == "stop"
    assert response.content == "ok"
    assert captured["do_sample"] is False
    assert "temperature" not in captured
    assert "top_p" not in captured


def test_transformers_pipeline_forwards_top_k_when_sampling() -> None:
    provider = _new_provider()
    captured: Dict[str, Any] = {}

    class _Pipeline:
        def __call__(self, text: str, **kwargs: Any) -> List[Dict[str, str]]:
            captured["text"] = text
            captured.update(kwargs)
            return [{"generated_text": "ok"}]

    provider.pipeline = _Pipeline()

    response = provider._single_generate_transformers("hello", 8, 0.5, 0.8, top_k=20, seed=None)

    assert response.finish_reason == "stop"
    assert response.content == "ok"
    assert captured["do_sample"] is True
    assert captured["temperature"] == 0.5
    assert captured["top_p"] == 0.8
    assert captured["top_k"] == 20


def test_transformers_loaded_generation_config_defaults_apply_when_omitted() -> None:
    provider = _new_provider()
    provider.model_instance.generation_config = SimpleNamespace(
        temperature=0.33,
        top_p=0.44,
        top_k=17,
    )

    provider._apply_loaded_generation_config_defaults()

    assert provider.temperature == 0.33
    assert provider.top_p == 0.44
    assert provider.top_k == 17


def test_transformers_loaded_generation_config_respects_explicit_constructor_values() -> None:
    provider = _new_provider()
    provider._explicit_generation_params = frozenset({"temperature", "top_p", "top_k"})
    provider.temperature = 0.2
    provider.top_p = 0.5
    provider.top_k = 10
    provider.model_instance.generation_config = SimpleNamespace(
        temperature=0.33,
        top_p=0.44,
        top_k=17,
    )

    provider._apply_loaded_generation_config_defaults()

    assert provider.temperature == 0.2
    assert provider.top_p == 0.5
    assert provider.top_k == 10


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


def test_transformers_dynamic_cache_load_preserves_configured_layer_types(tmp_path) -> None:
    config_mod = pytest.importorskip("transformers.models.mistral.configuration_mistral")

    provider = _new_provider()
    config = config_mod.MistralConfig(num_hidden_layers=2, sliding_window=8)
    provider.model = "mistral-test"
    provider.model_instance.config = config

    cache = DynamicCache(config=config)
    for idx in range(2):
        keys = torch.full((1, 1, 4, 1), float(idx), dtype=torch.float32)
        values = torch.full((1, 1, 4, 1), float(idx + 1), dtype=torch.float32)
        cache.update(keys, values, layer_idx=idx)

    provider._prompt_cache_store.set(
        "sliding",
        _TransformersPromptCacheValue(cache=cache, prompt_tokens=(1, 2, 3, 4)),
        meta={"backend": "transformers"},
    )

    filename = tmp_path / "sliding.safetensors"
    saved = provider.prompt_cache_save("sliding", str(filename))
    assert saved["meta"]["cache_schema"] == "dynamic-cache-layers/v1"
    assert "cache_layer_sequence_lengths" in saved["meta"]["cache_json_attrs"]

    provider.prompt_cache_clear()
    loaded = provider.prompt_cache_load(str(filename), key="loaded-sliding", make_default=False)
    loaded_state = provider._prompt_cache_store.get(loaded["key"])

    assert isinstance(loaded_state, _TransformersPromptCacheValue)
    assert [layer.__class__.__name__ for layer in loaded_state.cache.layers] == [
        "DynamicSlidingWindowLayer",
        "DynamicSlidingWindowLayer",
    ]
    assert loaded_state.cache.get_seq_length() == 4
    assert [layer.cumulative_length for layer in loaded_state.cache.layers] == [4, 4]
    assert torch.equal(loaded_state.cache.layers[1].keys, torch.full((1, 1, 4, 1), 1.0))
    assert torch.equal(loaded_state.cache.layers[1].values, torch.full((1, 1, 4, 1), 2.0))


def test_transformers_prompt_cache_update_uses_unified_thinking_control() -> None:
    provider = _new_provider()
    provider.model = "Qwen/Qwen3.5-4B"
    provider.architecture = "qwen3_5"
    captured: List[Any] = []
    original = provider._transformers_build_prompt_fragment

    def _capture_fragment(**kwargs: Any) -> str:
        captured.append(kwargs.get("enable_thinking"))
        return original(**kwargs)

    provider._transformers_build_prompt_fragment = _capture_fragment  # type: ignore[method-assign]
    assert provider.prompt_cache_set("thinking", make_default=False) is True
    assert provider.prompt_cache_update(
        "thinking",
        prompt="hi",
        add_generation_prompt=True,
        thinking="off",
    ) is True

    assert False in captured


def test_transformers_prompt_cache_save_load_preserves_qwen35_hybrid_cache(tmp_path) -> None:
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    config_mod = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")

    config = config_mod.Qwen3_5TextConfig()
    cache = modeling.Qwen3_5DynamicCache(config)
    for idx, layer_type in enumerate(cache.layer_types):
        if layer_type == "full_attention":
            cache.key_cache[idx] = torch.full((1, 1, 3, 1), float(idx), dtype=torch.float32)
            cache.value_cache[idx] = torch.full((1, 1, 3, 1), float(idx + 1), dtype=torch.float32)
        else:
            cache.conv_states[idx] = torch.full((1, 2, 4), float(idx + 2), dtype=torch.float32)
            cache.recurrent_states[idx] = torch.full((1, 2, 3), float(idx + 3), dtype=torch.float32)

    provider = _new_provider()
    provider.model = "Qwen/Qwen3.5-4B"
    provider.model_instance.config = config
    provider._prompt_cache_store.set(
        "qwen",
        _TransformersPromptCacheValue(cache=cache, prompt_tokens=(11, 12, 13)),
        meta={"backend": "transformers"},
    )

    filename = tmp_path / "qwen35.safetensors"
    saved = provider.prompt_cache_save("qwen", str(filename))
    assert saved["meta"]["cache_schema"] == "tensor-list-cache/v1"
    assert saved["meta"]["cache_class"] == "Qwen3_5DynamicCache"

    provider.prompt_cache_clear()
    loaded = provider.prompt_cache_load(str(filename), key="loaded", make_default=False)
    loaded_state = provider._prompt_cache_store.get(loaded["key"])

    assert isinstance(loaded_state, _TransformersPromptCacheValue)
    loaded_cache = loaded_state.cache
    assert loaded_cache.__class__.__name__ == "Qwen3_5DynamicCache"
    assert loaded_state.prompt_tokens == (11, 12, 13)
    assert loaded_cache.get_seq_length() == 3
    assert loaded_cache.has_previous_state is True
    assert torch.equal(loaded_cache.key_cache[3], torch.full((1, 1, 3, 1), 3.0))
    assert torch.equal(loaded_cache.value_cache[3], torch.full((1, 1, 3, 1), 4.0))
    assert torch.equal(loaded_cache.conv_states[0], torch.full((1, 2, 4), 2.0))
    assert torch.equal(loaded_cache.recurrent_states[0], torch.full((1, 2, 3), 3.0))


def test_transformers_prompt_cache_save_load_preserves_mamba_tensor_state(tmp_path) -> None:
    modeling = pytest.importorskip("transformers.models.mamba.modeling_mamba")
    config_mod = pytest.importorskip("transformers.models.mamba.configuration_mamba")

    config = config_mod.MambaConfig(num_hidden_layers=2, hidden_size=8, state_size=3, expand=1, conv_kernel=2)
    cache = modeling.MambaCache(
        config,
        max_batch_size=1,
        dtype=torch.float32,
        device="cpu",
    )
    cache.conv_states[0].fill_(5.0)
    cache.ssm_states[1].fill_(7.0)

    provider = _new_provider()
    provider.model = "state-spaces/mamba-test"
    provider.model_instance.config = config
    provider._prompt_cache_store.set(
        "mamba",
        _TransformersPromptCacheValue(cache=cache, prompt_tokens=(21, 22)),
        meta={"backend": "transformers"},
    )

    filename = tmp_path / "mamba.safetensors"
    saved = provider.prompt_cache_save("mamba", str(filename))
    assert saved["meta"]["cache_schema"] == "tensor-list-cache/v1"
    assert saved["meta"]["cache_class"] == "MambaCache"

    provider.prompt_cache_clear()
    loaded = provider.prompt_cache_load(str(filename), key="loaded-mamba", make_default=False)
    loaded_state = provider._prompt_cache_store.get(loaded["key"])

    assert isinstance(loaded_state, _TransformersPromptCacheValue)
    loaded_cache = loaded_state.cache
    assert loaded_cache.__class__.__name__ == "MambaCache"
    assert loaded_state.prompt_tokens == (21, 22)
    assert torch.equal(loaded_cache.conv_states[0], torch.full_like(loaded_cache.conv_states[0], 5.0))
    assert torch.equal(loaded_cache.ssm_states[1], torch.full_like(loaded_cache.ssm_states[1], 7.0))
