import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.vllm_provider import VLLMProvider


@pytest.mark.parametrize(
    ("provider_cls", "base_url"),
    [
        (LMStudioProvider, "http://localhost:1234/v1"),
        (OpenAICompatibleProvider, "http://localhost:8080/v1"),
        (VLLMProvider, "http://localhost:8000/v1"),
    ],
)
def test_openai_compatible_providers_send_native_tools_payload_when_supported(provider_cls, base_url, monkeypatch):
    provider = provider_cls(model="nvidia/nemotron-3-nano", base_url=base_url)

    tools = [
        {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
        }
    ]

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider._generate_internal(prompt="hi", tools=tools, stream=False)

    payload = captured["payload"]
    assert "tools" in payload
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "write_file"

