from __future__ import annotations

import pytest


@pytest.mark.integration
def test_openai_compatible_payload_contains_visual_context_and_user_request(monkeypatch, sample_media_files) -> None:
    """
    Level B integration test:
    - Uses real media processing (file -> MediaContent)
    - Uses the OpenAI-compatible provider message/payload builder
    - Stubs the actual HTTP request and the vision model call
    """
    from abstractcore.core.types import GenerateResponse
    from abstractcore.media.vision_fallback import VisionFallbackHandler
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    # No external network: avoid model discovery on init.
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)

    # No external vision model call: stub vision fallback output.
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, image_path, user_prompt=None: (
            "A solid red square on a plain background.",
            {"strategy": "caption", "backend": {"kind": "llm", "provider": "stub", "model": "stub"}},
        ),
    )

    provider = OpenAICompatibleProvider(
        model="qwen/qwen3-next-80b",
        base_url="http://127.0.0.1:1234/v1",
        api_key="EMPTY",
    )

    captured: dict = {}

    def fake_single_generate(self, payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop", gen_time=0)

    monkeypatch.setattr(OpenAICompatibleProvider, "_single_generate", fake_single_generate)

    user_request = "What is the dominant color in this image?"
    response = provider.generate(user_request, media=[str(sample_media_files["png"])], stream=False)
    assert response.content == "ok"
    assert isinstance(response.metadata, dict)
    assert isinstance(response.metadata.get("media_enrichment"), list)
    assert response.metadata["media_enrichment"]

    payload = captured["payload"]
    assert payload["model"] == "qwen/qwen3-next-80b"
    assert isinstance(payload.get("messages"), list)
    assert payload["messages"][-1]["role"] == "user"

    content = payload["messages"][-1]["content"]
    assert isinstance(content, str)
    assert "Visual context from attached image(s)" in content
    assert "A solid red square on a plain background." in content
    assert "Now answer the user's request:" in content
    assert content.strip().endswith(user_request)
