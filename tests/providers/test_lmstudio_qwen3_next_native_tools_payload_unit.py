import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import LMStudioProvider


def test_lmstudio_qwen3_next_sends_native_tools_payload(monkeypatch):
    """
    Regression test for qwen/qwen3-next-80b:
    - capabilities must classify it as native-tools for OpenAI-compatible servers
    - LMStudioProvider must include payload["tools"] / payload["tool_choice"]
    """
    # Avoid any dependency on a running LMStudio server during provider init.
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)

    provider = LMStudioProvider(model="qwen/qwen3-next-80b", base_url="http://localhost:1234/v1")

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
    assert payload["model"] == "qwen/qwen3-next-80b"
    assert "tools" in payload
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "write_file"


