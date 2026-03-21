from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import LMStudioProvider


def test_lmstudio_fallback_injects_user_message_when_missing(monkeypatch) -> None:
    """LM Studio templates can hard-fail when no user message exists (jinja: no user query).

    Ensure OpenAICompatibleProvider (LMStudioProvider) always sends at least one user message.
    """

    # Avoid any dependency on a running LMStudio server during provider init.
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)

    provider = LMStudioProvider(model="qwen/qwen3.5-9b", base_url="http://localhost:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider._generate_internal(prompt="", system_prompt="You are a helpful assistant.", stream=False)

    messages = captured["payload"]["messages"]
    assert any(isinstance(m, dict) and m.get("role") == "user" for m in messages)

