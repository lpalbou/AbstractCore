from __future__ import annotations

import pytest


@pytest.mark.integration
def test_openai_compatible_audio_speech_to_text_injects_context_and_emits_metadata(tmp_path, monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    # No external network: avoid model discovery on init.
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)

    class _StubAudioBackend:
        backend_id = "stub:audio"

        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, language=None, **kwargs) -> str:
            self.calls += 1
            return "hello world"

    stub_audio = _StubAudioBackend()

    class StubProvider(OpenAICompatibleProvider):
        @property
        def audio(self):
            return stub_audio

    provider = StubProvider(
        model="qwen/qwen3-next-80b",
        base_url="http://127.0.0.1:1234/v1",
        api_key="EMPTY",
    )

    captured: dict = {}

    def fake_single_generate(self, payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop", gen_time=0, metadata={})

    monkeypatch.setattr(OpenAICompatibleProvider, "_single_generate", fake_single_generate)

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    user_request = "What did they say?"
    response = provider.generate(user_request, media=[str(audio_path)], stream=False, audio_policy="speech_to_text")
    assert response.content == "ok"
    assert stub_audio.calls == 1

    payload = captured["payload"]
    assert payload["model"] == "qwen/qwen3-next-80b"
    assert isinstance(payload.get("messages"), list)
    assert payload["messages"][-1]["role"] == "user"

    content = payload["messages"][-1]["content"]
    assert isinstance(content, str)
    assert "Audio context from attached audio file(s)" in content
    assert "hello world" in content
    assert content.strip().endswith(user_request)

    assert isinstance(response.metadata, dict)
    assert isinstance(response.metadata.get("media_enrichment"), list)
    assert response.metadata["media_enrichment"]

