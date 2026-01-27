import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.ollama_provider import OllamaProvider
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
