import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.ollama_provider import OllamaProvider
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.vllm_provider import VLLMProvider


def test_vllm_thinking_sets_chat_template_kwargs_enable_thinking(monkeypatch):
    # Avoid any dependency on a running server during provider init.
    monkeypatch.setattr(VLLMProvider, "_validate_model", lambda self: None, raising=False)
    provider = VLLMProvider(model="qwen3-4b", base_url="http://127.0.0.1:8000/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="off", temperature=0)

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

    provider.generate("hi", thinking="none", temperature=0)

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

    call_params = captured["call_params"]
    assert call_params.get("reasoning_effort") == "xhigh"
    assert call_params.get("max_completion_tokens") == 8


def test_openai_pro_thinking_off_maps_to_min_supported_effort(monkeypatch):
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
        provider.generate("hi", thinking="off", max_output_tokens=8, temperature=0)

    call_params = captured["call_params"]
    assert call_params.get("reasoning_effort") == "medium"
    assert call_params.get("max_completion_tokens") == 8


def test_ollama_thinking_sets_payload_think_boolean(monkeypatch):
    provider = OllamaProvider(model="qwen3:4b-instruct-2507-q4_K_M", base_url="http://127.0.0.1:11434")

    captured = {}

    def _capture_single_generate(endpoint, payload, tools=None, media_metadata=None):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking=False, temperature=0)

    payload = captured["payload"]
    assert payload.get("think") is False


def test_ollama_gpt_oss_thinking_level_sets_payload_think_string(monkeypatch):
    provider = OllamaProvider(model="gpt-oss:20b", base_url="http://127.0.0.1:11434")

    captured = {}

    def _capture_single_generate(endpoint, payload, tools=None, media_metadata=None):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="high", temperature=0)

    payload = captured["payload"]
    assert payload.get("think") == "high"


def test_harmony_thinking_injects_reasoning_system_prompt(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    provider = OpenAICompatibleProvider(model="openai/gpt-oss-20b", base_url="http://127.0.0.1:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="high", temperature=0)

    payload = captured["payload"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"].strip() == "Reasoning: high"


def test_lmstudio_qwen3_5_thinking_off_appends_no_think_token(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="qwen/qwen3.5-9b", base_url="http://localhost:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="off", temperature=0)

    payload = captured["payload"]
    user_messages = [m for m in payload["messages"] if isinstance(m, dict) and m.get("role") == "user"]
    assert user_messages, "expected at least one user message"
    assert "/no_think" in str(user_messages[-1].get("content") or "")


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
