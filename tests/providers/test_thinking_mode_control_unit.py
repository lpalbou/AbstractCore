import warnings
from types import SimpleNamespace

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.ollama_provider import OllamaProvider
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.vllm_provider import VLLMProvider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.providers.base import BaseProvider


def _install_fake_openai(monkeypatch) -> None:
    import abstractcore.providers.openai_provider as openai_provider_module

    class _FakeOpenAIClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_k: object()))
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClient, AsyncOpenAI=_FakeOpenAIClient)
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True, raising=False)
    monkeypatch.setattr(openai_provider_module, "openai", fake_openai, raising=False)


def test_vllm_thinking_sets_chat_template_kwargs_enable_thinking(monkeypatch):
    # Avoid any dependency on a running server during provider init.
    monkeypatch.setattr(VLLMProvider, "_validate_model", lambda self: None, raising=False)
    provider = VLLMProvider(model="qwen3-4b", base_url="http://127.0.0.1:8000/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    resp = provider.generate("hi", thinking="off", temperature=0)
    assert isinstance(resp, GenerateResponse)
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_requested") == "off"
    assert resp.metadata.get("thinking_effective") == "off"

    payload = captured["payload"]
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_thinking_none_is_alias_for_off(monkeypatch):
    monkeypatch.setattr(VLLMProvider, "_validate_model", lambda self: None, raising=False)
    provider = VLLMProvider(model="qwen3-4b", base_url="http://127.0.0.1:8000/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    resp = provider.generate("hi", thinking="none", temperature=0)
    assert isinstance(resp, GenerateResponse)
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_effective") == "off"

    payload = captured["payload"]
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_thinking_xhigh_is_accepted(monkeypatch):
    monkeypatch.setattr(VLLMProvider, "_validate_model", lambda self: None, raising=False)
    provider = VLLMProvider(model="gpt-5.2", base_url="http://127.0.0.1:8000/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="xhigh", temperature=0)

    payload = captured["payload"]
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_openai_thinking_maps_to_reasoning_effort_without_network(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None, raising=False)
    provider = OpenAIProvider(model="gpt-5.2")

    captured = {}

    def _capture_create(**call_params):
        captured["call_params"] = call_params
        return object()

    monkeypatch.setattr(provider.client.chat.completions, "create", _capture_create)
    monkeypatch.setattr(
        provider,
        "_format_response",
        lambda _resp: GenerateResponse(content="ok", model=provider.model, finish_reason="stop"),
    )

    resp = provider.generate("hi", thinking="xhigh", max_output_tokens=8, temperature=0)
    assert resp.content == "ok"
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_requested") == "xhigh"
    assert resp.metadata.get("thinking_effective") == "xhigh"

    call_params = captured["call_params"]
    assert call_params.get("reasoning_effort") == "xhigh"
    assert call_params.get("max_completion_tokens") == 8


def test_openai_pro_thinking_off_maps_to_min_supported_effort(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None, raising=False)
    provider = OpenAIProvider(model="gpt-5.2-pro")

    captured = {}

    def _capture_create(**call_params):
        captured["call_params"] = call_params
        return object()

    monkeypatch.setattr(provider.client.chat.completions, "create", _capture_create)
    monkeypatch.setattr(
        provider,
        "_format_response",
        lambda _resp: GenerateResponse(content="ok", model=provider.model, finish_reason="stop"),
    )

    with pytest.warns(RuntimeWarning):
        resp = provider.generate("hi", thinking="off", max_output_tokens=8, temperature=0)

    call_params = captured["call_params"]
    assert call_params.get("reasoning_effort") == "medium"
    assert call_params.get("max_completion_tokens") == 8
    assert isinstance(resp, GenerateResponse)
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_requested") == "off"
    assert resp.metadata.get("thinking_effective") == "medium"


def test_ollama_thinking_sets_payload_think_boolean(monkeypatch):
    provider = OllamaProvider(model="qwen3:4b-instruct-2507-q4_K_M", base_url="http://127.0.0.1:11434")

    captured = {}

    def _capture_single_generate(endpoint, payload, tools=None, media_metadata=None):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    resp = provider.generate("hi", thinking=False, temperature=0)
    assert isinstance(resp, GenerateResponse)
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_effective") == "off"

    payload = captured["payload"]
    assert payload.get("think") is False


def test_ollama_gpt_oss_thinking_level_sets_payload_think_string(monkeypatch):
    provider = OllamaProvider(model="gpt-oss:20b", base_url="http://127.0.0.1:11434")

    captured = {}

    def _capture_single_generate(endpoint, payload, tools=None, media_metadata=None):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    resp = provider.generate("hi", thinking="high", temperature=0)
    assert isinstance(resp, GenerateResponse)
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_effective") == "high"

    payload = captured["payload"]
    assert payload.get("think") == "high"


def test_ollama_payload_uses_metadata_sampling_defaults(monkeypatch):
    provider = OllamaProvider(model="google/gemma-4-E4B-it", base_url="http://127.0.0.1:11434")

    captured = {}

    def _capture_single_generate(endpoint, payload, tools=None, media_metadata=None):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi")

    options = captured["payload"]["options"]
    assert options["temperature"] == 1.0
    assert options["top_p"] == 0.95
    assert options["top_k"] == 64


def test_harmony_thinking_injects_reasoning_system_prompt(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    provider = OpenAICompatibleProvider(model="openai/gpt-oss-20b", base_url="http://127.0.0.1:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    resp = provider.generate("hi", thinking="high", temperature=0)
    assert isinstance(resp, GenerateResponse)
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_effective") == "high"

    payload = captured["payload"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"].strip() == "Reasoning: high"


def test_harmony_unsupported_thinking_level_maps_to_nearest(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    provider = OpenAICompatibleProvider(model="openai/gpt-oss-20b", base_url="http://127.0.0.1:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    with pytest.warns(RuntimeWarning):
        resp = provider.generate("hi", thinking="minimal", temperature=0)

    payload = captured["payload"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"].strip() == "Reasoning: low"
    assert isinstance(resp, GenerateResponse)
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_requested") == "minimal"
    assert resp.metadata.get("thinking_effective") == "low"


def test_openai_unsupported_thinking_level_maps_to_nearest(monkeypatch):
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None, raising=False)
    provider = OpenAIProvider(model="gpt-5")

    captured = {}

    def _capture_create(**call_params):
        captured["call_params"] = call_params
        return object()

    monkeypatch.setattr(provider.client.chat.completions, "create", _capture_create)
    monkeypatch.setattr(
        provider,
        "_format_response",
        lambda _resp: GenerateResponse(content="ok", model=provider.model, finish_reason="stop"),
    )

    with pytest.warns(RuntimeWarning):
        resp = provider.generate("hi", thinking="xhigh", max_output_tokens=8, temperature=0)

    call_params = captured["call_params"]
    assert call_params.get("reasoning_effort") == "high"
    assert call_params.get("max_completion_tokens") == 8
    assert isinstance(resp, GenerateResponse)
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_requested") == "xhigh"
    assert resp.metadata.get("thinking_effective") == "high"


def test_huggingface_gguf_qwen_thinking_level_warns_about_effort_scaling() -> None:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(provider, "unsloth/Qwen3.5-2B-GGUF")
    provider.provider = "huggingface"
    provider.model_type = "gguf"

    with pytest.warns(RuntimeWarning):
        provider._apply_thinking_request(
            thinking="high",
            prompt="hi",
            messages=None,
            system_prompt=None,
            kwargs={},
        )


def test_lmstudio_qwen3_5_thinking_off_sets_chat_template_enable_thinking_false(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="qwen/qwen3.5-9b", base_url="http://localhost:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="off", temperature=0)

    payload = captured["payload"]
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_lmstudio_qwen3_6_thinking_off_sets_chat_template_enable_thinking_false(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="Qwen/Qwen3.6-27B", base_url="http://localhost:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="off", temperature=0)

    payload = captured["payload"]
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_openai_compatible_qwen3_6_thinking_does_not_emit_template_kwargs_by_default(monkeypatch) -> None:
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    provider = OpenAICompatibleProvider(
        model="Qwen/Qwen3.6-27B",
        base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    )

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    with pytest.warns(RuntimeWarning, match="cannot enforce effort scaling"):
        provider.generate("hi", thinking="high", temperature=0)

    payload = captured["payload"]
    assert "chat_template_kwargs" not in payload
    assert "extra_body" not in payload
    assert "enableThinking" not in str(payload)
    assert "enable_thinking" not in str(payload)


def test_openai_compatible_qwen3_6_thinking_template_kwargs_require_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    provider = OpenAICompatibleProvider(
        model="Qwen/Qwen3.6-27B",
        base_url="http://127.0.0.1:1234/v1",
        supports_chat_template_kwargs=True,
    )

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="high", temperature=0)

    payload = captured["payload"]
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["chat_template_kwargs"]["enableThinking"] is True


def test_lmstudio_payload_uses_metadata_sampling_defaults(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="google/gemma-4-E4B-it", base_url="http://localhost:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi")

    payload = captured["payload"]
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 64


def _make_unloaded_mlx_provider(model: str) -> MLXProvider:
    provider = MLXProvider.__new__(MLXProvider)
    BaseProvider.__init__(provider, model)
    provider.provider = "mlx"
    return provider


def test_mlx_qwen_thinking_off_serializes_assistant_no_think_marker() -> None:
    provider = _make_unloaded_mlx_provider("mlx-community/Qwen3.6-27B-4bit")

    prompt, messages, system_prompt, kwargs, meta = provider._apply_thinking_request(
        thinking="off",
        prompt="Reply with exactly: OK",
        messages=None,
        system_prompt=None,
        kwargs={},
    )

    assert kwargs["_acore_mlx_enable_thinking"] is False
    assert meta is not None
    assert meta["thinking_effective"] == "off"
    assert meta["thinking_handled_enable_disable"] is True

    rendered = provider._build_prompt(
        prompt,
        messages,
        system_prompt,
        tools=None,
        enable_thinking=kwargs["_acore_mlx_enable_thinking"],
    )
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "/no_think" not in rendered


def test_mlx_qwen_thinking_level_warns_and_degrades_to_enabled() -> None:
    provider = _make_unloaded_mlx_provider("mlx-community/Qwen3.6-27B-4bit")

    with pytest.warns(RuntimeWarning, match="cannot enforce effort scaling"):
        _prompt, _messages, _system_prompt, kwargs, meta = provider._apply_thinking_request(
            thinking="high",
            prompt="Reply with exactly: OK",
            messages=None,
            system_prompt=None,
            kwargs={},
        )

    assert kwargs["_acore_mlx_enable_thinking"] is True
    assert meta is not None
    assert meta["thinking_requested"] == "high"
    assert meta["thinking_effective"] == "on"
    assert meta["thinking_handled_enable_disable"] is True
    assert meta["thinking_handled_level"] is False


# --- Qwen3.8 effort enforcement (asset-declared `thinking_control.effort_system_lines`) ---
#
# These sentences must stay byte-identical to Qwen/Qwen3.8-27B chat_template.jinja
# (the template's `reasoning_instructions` variable), which is what the asset entry mirrors.

QWEN38_LOW_LINE = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
QWEN38_XHIGH_LINE = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."


def test_mlx_qwen38_thinking_level_is_enforced_via_template_effort_line() -> None:
    provider = _make_unloaded_mlx_provider("mlx-community/Qwen3.8-27B-4bit")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prompt, messages, system_prompt, kwargs, meta = provider._apply_thinking_request(
            thinking="low",
            prompt="Reply with exactly: OK",
            messages=None,
            system_prompt="You are a terse assistant.",
            kwargs={},
        )
    assert not [w for w in caught if "effort scaling" in str(w.message)]

    assert kwargs["_acore_mlx_enable_thinking"] is True
    assert kwargs["_acore_mlx_reasoning_effort"] == "low"
    assert meta is not None
    assert meta["thinking_effective"] == "low"
    assert meta["thinking_level_effective"] == "low"
    assert meta["thinking_handled_level"] is True

    rendered = provider._build_prompt(
        prompt,
        messages,
        system_prompt,
        tools=None,
        enable_thinking=kwargs["_acore_mlx_enable_thinking"],
        reasoning_effort=kwargs["_acore_mlx_reasoning_effort"],
    )
    assert rendered.startswith(
        f"<|im_start|>system\n{QWEN38_LOW_LINE}\n\nYou are a terse assistant.<|im_end|>\n"
    )


def test_mlx_qwen38_thinking_medium_is_handled_and_renders_no_line() -> None:
    provider = _make_unloaded_mlx_provider("mlx-community/Qwen3.8-27B-4bit")

    prompt, messages, system_prompt, kwargs, meta = provider._apply_thinking_request(
        thinking="medium",
        prompt="Reply with exactly: OK",
        messages=None,
        system_prompt="You are a terse assistant.",
        kwargs={},
    )

    assert kwargs["_acore_mlx_reasoning_effort"] == "medium"
    assert meta is not None
    assert meta["thinking_effective"] == "medium"
    assert meta["thinking_handled_level"] is True

    rendered = provider._build_prompt(
        prompt,
        messages,
        system_prompt,
        tools=None,
        enable_thinking=kwargs["_acore_mlx_enable_thinking"],
        reasoning_effort=kwargs["_acore_mlx_reasoning_effort"],
    )
    # The template renders no instruction for medium; the system block is untouched.
    assert "Reasoning effort" not in rendered
    assert rendered.startswith("<|im_start|>system\nYou are a terse assistant.<|im_end|>\n")


def test_mlx_qwen38_thinking_level_without_system_prompt_emits_system_block() -> None:
    provider = _make_unloaded_mlx_provider("mlx-community/Qwen3.8-27B-4bit")

    prompt, messages, system_prompt, kwargs, _meta = provider._apply_thinking_request(
        thinking="xhigh",
        prompt="Reply with exactly: OK",
        messages=None,
        system_prompt=None,
        kwargs={},
    )

    rendered = provider._build_prompt(
        prompt,
        messages,
        system_prompt,
        tools=None,
        enable_thinking=kwargs["_acore_mlx_enable_thinking"],
        reasoning_effort=kwargs["_acore_mlx_reasoning_effort"],
    )
    assert rendered.startswith(f"<|im_start|>system\n{QWEN38_XHIGH_LINE}<|im_end|>\n<|im_start|>user\n")


def test_mlx_qwen38_thinking_off_renders_no_effort_line() -> None:
    provider = _make_unloaded_mlx_provider("mlx-community/Qwen3.8-27B-4bit")

    prompt, messages, system_prompt, kwargs, meta = provider._apply_thinking_request(
        thinking="off",
        prompt="Reply with exactly: OK",
        messages=None,
        system_prompt=None,
        kwargs={},
    )

    assert kwargs["_acore_mlx_enable_thinking"] is False
    assert "_acore_mlx_reasoning_effort" not in kwargs
    assert meta is not None
    assert meta["thinking_effective"] == "off"

    rendered = provider._build_prompt(
        prompt,
        messages,
        system_prompt,
        tools=None,
        enable_thinking=kwargs["_acore_mlx_enable_thinking"],
    )
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Reasoning effort" not in rendered


def test_mlx_qwen38_prefilled_system_bloc_degrades_level_honestly() -> None:
    # KV-mode CachedSession feeds a prefilled system bloc: the fragment renderer
    # must not reopen the system region, so no effort artifact can be applied.
    # The hook must decline the level and the base ladder must warn (ADR-0001).
    provider = _make_unloaded_mlx_provider("mlx-community/Qwen3.8-27B-4bit")

    with pytest.warns(RuntimeWarning, match="cannot enforce effort scaling"):
        _prompt, _messages, _system_prompt, kwargs, meta = provider._apply_thinking_request(
            thinking="low",
            prompt="Reply with exactly: OK",
            messages=None,
            system_prompt=None,
            kwargs={"prompt_cache_prefilled_modules": ("system",)},
        )

    assert "_acore_mlx_reasoning_effort" not in kwargs
    assert kwargs["_acore_mlx_enable_thinking"] is True
    assert meta is not None
    assert meta["thinking_effective"] == "on"
    assert meta["thinking_handled_level"] is False


def test_mlx_qwen38_level_merges_into_leading_system_message() -> None:
    # A conversation whose system turn arrives inside `messages` must get the
    # effort line merged into THAT turn — never a second consecutive system block.
    provider = _make_unloaded_mlx_provider("mlx-community/Qwen3.8-27B-4bit")

    rendered = provider._build_prompt_fragment(
        prompt="",
        messages=[
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
        system_prompt=None,
        tools=None,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="low",
    )
    assert rendered.count("<|im_start|>system") == 1
    assert rendered.startswith(
        f"<|im_start|>system\n{QWEN38_LOW_LINE}\n\nYou are a terse assistant.<|im_end|>\n"
    )


def test_lmstudio_qwen38_thinking_level_sets_top_level_reasoning_effort(monkeypatch) -> None:
    # The top-level OpenAI-standard `reasoning_effort` param is the only transport
    # LM Studio maps into the chat template on EVERY request shape (live-verified
    # 2026-08-19: low and xhigh rendered their template sentences on plain and on
    # messages+tools requests, while chat_template_kwargs was ignored and the
    # native REST enum rejected xhigh). A validating server that rejects the param
    # is handled by the 400 drop-and-retry net in OpenAICompatibleProvider.
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="qwen3.8-27b", base_url="http://localhost:1234/v1")

    for lvl in ("low", "medium", "xhigh"):
        new_kwargs, handling = provider._apply_provider_thinking_kwargs(enabled=True, level=lvl, kwargs={})
        assert new_kwargs["reasoning_effort"] == lvl
        assert "reasoning" not in new_kwargs  # native REST field not used for effort models
        ctk = new_kwargs["chat_template_kwargs"]
        assert ctk["enable_thinking"] is True
        assert ctk["reasoning_effort"] == lvl  # best-effort belt for builds that honor ctk
        tv = new_kwargs["lmstudio_template_vars"]
        assert "reasoning_effort" not in tv  # template vars spread to top level; no dup key
        assert handling.handled_enable_disable is True
        assert handling.handled_level is True

    # An explicit caller-provided reasoning_effort= wins over the mapped level.
    new_kwargs, _handling = provider._apply_provider_thinking_kwargs(
        enabled=True, level="low", kwargs={"reasoning_effort": "medium"}
    )
    assert new_kwargs["reasoning_effort"] == "medium"

    # After a 400 rejection latched, the claim is declined and the param not re-sent.
    provider._reasoning_effort_unsupported = True
    new_kwargs, handling = provider._apply_provider_thinking_kwargs(enabled=True, level="low", kwargs={})
    assert "reasoning_effort" not in new_kwargs
    assert handling.handled_level is False


def test_openai_compatible_reasoning_effort_400_rejection_detector(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="qwen3.8-27b", base_url="http://localhost:1234/v1")

    class _Resp:
        status_code = 400
        def json(self):
            return {"error": {"message": "Invalid enum value for reasoning_effort"}}
        text = '{"error": {"message": "Invalid enum value for reasoning_effort"}}'

    payload = {"model": "qwen3.8-27b", "reasoning_effort": "xhigh"}
    assert provider._is_reasoning_effort_rejection(_Resp(), payload) is True
    assert provider._is_reasoning_effort_rejection(_Resp(), {"model": "x"}) is False
    with pytest.warns(RuntimeWarning, match="NOT applied"):
        provider._mark_reasoning_effort_unsupported()


def _make_unloaded_hf_gguf_provider(model: str) -> HuggingFaceProvider:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(provider, model)
    provider.provider = "huggingface"
    provider.model_type = "gguf"
    return provider


def test_hf_gguf_qwen38_level_is_stashed_and_claimed_for_text_shapes() -> None:
    provider = _make_unloaded_hf_gguf_provider("lmstudio-community/Qwen3.8-27B-GGUF")

    new_kwargs, handling = provider._apply_provider_thinking_kwargs(
        enabled=True, level="low", kwargs={}, request_shape={}
    )
    assert new_kwargs["_acore_gguf_reasoning_effort"] == "low"
    assert handling.handled_enable_disable is True
    assert handling.handled_level is True


def test_hf_gguf_blocker_shapes_decline_all_claims() -> None:
    # response_model / media route through create_chat_completion, where the
    # trailing-assistant marker is a closed turn (live find 2026-08-19) and the
    # level has no transport: every claim must be declined so base warns.
    provider = _make_unloaded_hf_gguf_provider("lmstudio-community/Qwen3.8-27B-GGUF")

    for shape in ({"has_response_model": True}, {"has_media": True}):
        new_kwargs, handling = provider._apply_provider_thinking_kwargs(
            enabled=False, level=None, kwargs={}, request_shape=shape
        )
        assert "_acore_gguf_enable_thinking" not in new_kwargs
        assert handling.handled_enable_disable is False
        new_kwargs, handling = provider._apply_provider_thinking_kwargs(
            enabled=True, level="low", kwargs={}, request_shape=shape
        )
        assert "_acore_gguf_reasoning_effort" not in new_kwargs
        assert handling.handled_level is False


def test_hf_gguf_lane_predicate_parity_declines(monkeypatch) -> None:
    # The claim must be computed with the SAME predicate the control-plane gate
    # applies at generate time (V2-F2): env escape hatch, exotic message roles,
    # and content-parts payloads all fall back to create_chat_completion, where
    # neither control has a transport.
    provider = _make_unloaded_hf_gguf_provider("lmstudio-community/Qwen3.8-27B-GGUF")

    monkeypatch.setenv("ABSTRACTCORE_GGUF_CONTROL_PLANE", "0")
    new_kwargs, handling = provider._apply_provider_thinking_kwargs(
        enabled=True, level="low", kwargs={}, request_shape={}
    )
    assert "_acore_gguf_reasoning_effort" not in new_kwargs
    assert handling.handled_level is False
    monkeypatch.delenv("ABSTRACTCORE_GGUF_CONTROL_PLANE")

    tool_role_history = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "content": "a.txt"},
    ]
    parts_history = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    for msgs in (tool_role_history, parts_history):
        new_kwargs, handling = provider._apply_provider_thinking_kwargs(
            enabled=False, level=None, kwargs={}, request_shape={"messages": msgs}
        )
        assert "_acore_gguf_enable_thinking" not in new_kwargs
        assert handling.handled_enable_disable is False
        new_kwargs, handling = provider._apply_provider_thinking_kwargs(
            enabled=True, level="low", kwargs={}, request_shape={"messages": msgs}
        )
        assert handling.handled_level is False

    # Plain assistant/user history keeps the claim.
    ok_history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    new_kwargs, handling = provider._apply_provider_thinking_kwargs(
        enabled=True, level="low", kwargs={}, request_shape={"messages": ok_history}
    )
    assert new_kwargs["_acore_gguf_reasoning_effort"] == "low"
    assert handling.handled_level is True


def test_hf_gguf_chatml_render_injects_effort_line() -> None:
    provider = _make_unloaded_hf_gguf_provider("lmstudio-community/Qwen3.8-27B-GGUF")

    rendered = provider._gguf_render_chatml_prompt(
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ],
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="low",
    )
    assert rendered.count("<|im_start|>system") == 1
    assert rendered.startswith(f"<|im_start|>system\n{QWEN38_LOW_LINE}\n\nYou are terse.<|im_end|>\n")

    # No system message: the line opens its own leading system block.
    rendered = provider._gguf_render_chatml_prompt(
        messages=[{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )
    assert rendered.startswith(f"<|im_start|>system\n{QWEN38_XHIGH_LINE}<|im_end|>\n<|im_start|>user\n")

    # medium renders nothing — same as the official template.
    rendered = provider._gguf_render_chatml_prompt(
        messages=[{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="medium",
    )
    assert "Reasoning effort" not in rendered


def test_hf_transformers_qwen38_level_claim_and_cached_fragment_injection() -> None:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    BaseProvider.__init__(provider, "Qwen/Qwen3.8-27B")
    provider.provider = "huggingface"
    provider.model_type = "transformers"

    new_kwargs, handling = provider._apply_provider_thinking_kwargs(
        enabled=True, level="low", kwargs={}, request_shape={}
    )
    assert new_kwargs["_acore_hf_transformers_reasoning_effort"] == "low"
    assert handling.handled_level is True

    # Prefilled system bloc: decline (no artifact possible in the fragment).
    new_kwargs, handling = provider._apply_provider_thinking_kwargs(
        enabled=True, level="low",
        kwargs={"prompt_cache_prefilled_modules": ("system",)},
        request_shape={},
    )
    assert "_acore_hf_transformers_reasoning_effort" not in new_kwargs
    assert handling.handled_level is False

    # Cached-lane hand renderer: line lands at the top of the first system block.
    fragment = provider._transformers_build_prompt_fragment(
        prompt="hi",
        messages=None,
        system_prompt="You are terse.",
        tools=None,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="low",
    )
    assert QWEN38_LOW_LINE in fragment
    assert fragment.index(QWEN38_LOW_LINE) < fragment.index("You are terse.")


def test_vllm_qwen38_thinking_level_forwards_reasoning_effort_template_kwarg(monkeypatch) -> None:
    monkeypatch.setattr(VLLMProvider, "_validate_model", lambda self: None, raising=False)
    provider = VLLMProvider(model="Qwen/Qwen3.8-27B", base_url="http://127.0.0.1:8000/v1")

    new_kwargs, handling = provider._apply_provider_thinking_kwargs(enabled=True, level="medium", kwargs={})

    ctk = new_kwargs["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "medium"
    assert handling.handled_level is True


def test_lmstudio_seed_oss_thinking_high_sets_chat_template_thinking_budget(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="seed-oss-36b", base_url="http://localhost:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="high", temperature=0)

    payload = captured["payload"]
    assert payload["chat_template_kwargs"]["thinking_budget"] == 4096


def test_lmstudio_gpt_oss_thinking_level_injects_reasoning_system_prompt(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="openai/gpt-oss-20b", base_url="http://localhost:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="high", temperature=0)

    payload = captured["payload"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"].strip() == "Reasoning: high"


def test_thinking_output_field_and_think_tags_are_normalized_in_base_provider(monkeypatch):
    # GLM-4.6V models declare thinking_tags + thinking_output_field in model_capabilities.json.
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    provider = OpenAICompatibleProvider(model="glm-4.6v", base_url="http://127.0.0.1:1234/v1")

    def _fake_single_generate(payload):
        _ = payload
        return GenerateResponse(
            content="<think>r</think>\n\nfinal",
            model=provider.model,
            finish_reason="stop",
        )

    monkeypatch.setattr(provider, "_single_generate", _fake_single_generate)

    resp = provider.generate("hi", temperature=0)
    assert resp.content == "final"
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("reasoning") == "r"


def test_thinking_off_does_not_warn_for_non_reasoning_model(monkeypatch):
    import warnings

    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="google/gemma-3-4b", base_url="http://localhost:1234/v1")

    def _fake_single_generate(payload):
        _ = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _fake_single_generate)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider.generate("hi", thinking="none", temperature=0)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings


def test_qwen3_5_im_end_wrapper_is_stripped_from_visible_content(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    provider = OpenAICompatibleProvider(model="qwen3.5-4b-mlx@4bit", base_url="http://127.0.0.1:1234/v1")

    def _fake_single_generate(payload):
        _ = payload
        return GenerateResponse(content="Hello<|im_end|>\n", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _fake_single_generate)

    resp = provider.generate("hi", temperature=0)
    assert resp.content == "Hello"
