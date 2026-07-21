from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class _StubAudioBackend:
    backend_id: str = "stub:audio"
    calls: int = 0
    last_kwargs: dict = field(default_factory=dict)

    def transcribe(self, audio, language=None, **kwargs) -> str:
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        return "hello world"


@pytest.mark.basic
def test_audio_speech_to_text_policy_injects_transcript_and_removes_audio_media(tmp_path) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.base import BaseProvider

    stub_audio = _StubAudioBackend()

    class DummyProvider(BaseProvider):
        def __init__(self):
            super().__init__(model="qwen/qwen3-next-80b")
            self._audio_backend = stub_audio
            self.last_prompt = None
            self.last_media = None

        @property
        def audio(self):
            return self._audio_backend

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
            self.last_prompt = prompt
            self.last_media = media
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop", metadata={})

        def get_capabilities(self):
            return []

        def unload_model(self, model_name: str) -> None:
            return None

        def list_available_models(self, **kwargs):
            return []

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    provider = DummyProvider()
    resp = provider.generate("What did they say?", media=[str(audio_path)], audio_policy="speech_to_text")

    assert resp.content == "ok"
    assert stub_audio.calls == 1

    assert isinstance(provider.last_prompt, str)
    assert "Audio context from attached audio file(s)" in provider.last_prompt
    assert "Audio 1" in provider.last_prompt
    assert "hello world" in provider.last_prompt
    assert "Now answer the user's request:" in provider.last_prompt
    assert provider.last_prompt.strip().endswith("What did they say?")

    assert provider.last_media == []  # audio removed from provider-native media path

    assert isinstance(resp.metadata, dict)
    enrichments = resp.metadata.get("media_enrichment")
    assert isinstance(enrichments, list)
    assert enrichments
    entry = enrichments[0]
    assert entry.get("status") == "used"
    assert entry.get("input_modality") == "audio"
    assert entry.get("summary_kind") == "transcript"


@pytest.mark.basic
def test_stt_explicit_provider_override_does_not_inherit_route_model(tmp_path) -> None:
    # Same class as the 2026-07-17 voice-options leak, on the STT fallback lane:
    # an explicit stt_provider that contradicts the configured input.voice route
    # must not inherit the route's model (openai would otherwise be asked for a
    # faster-whisper model like "large-v3"). Route rows are one backend identity.
    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.base import BaseProvider

    stub_audio = _StubAudioBackend()

    class DummyProvider(BaseProvider):
        def __init__(self):
            super().__init__(model="qwen/qwen3-next-80b")
            self._audio_backend = stub_audio

        @property
        def audio(self):
            return self._audio_backend

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop", metadata={})

        def get_capabilities(self):
            return []

        def unload_model(self, model_name: str) -> None:
            return None

        def list_available_models(self, **kwargs):
            return []

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    provider = DummyProvider()
    provider._abstractcore_capability_defaults = {  # type: ignore[attr-defined]
        "input.voice": {
            "key": "input.voice",
            "provider": "faster-whisper",
            "model": "large-v3",
            "source": "gateway.capability_defaults",
        }
    }

    resp = provider.generate(
        "What did they say?",
        media=[str(audio_path)],
        audio_policy="speech_to_text",
        stt_provider="openai",
    )

    assert resp.content == "ok"
    assert stub_audio.calls == 1
    assert stub_audio.last_kwargs.get("provider") == "openai"
    assert "model" not in stub_audio.last_kwargs

    # Same explicit provider as the route keeps the route's model (one backend).
    stub_audio.calls = 0
    provider.generate(
        "What did they say?",
        media=[str(audio_path)],
        audio_policy="speech_to_text",
        stt_provider="faster-whisper",
    )
    assert stub_audio.calls == 1
    assert stub_audio.last_kwargs.get("provider") == "faster-whisper"
    assert stub_audio.last_kwargs.get("model") == "large-v3"

