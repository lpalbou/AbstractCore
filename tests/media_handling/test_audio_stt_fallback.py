from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _StubAudioBackend:
    backend_id: str = "stub:audio"
    calls: int = 0

    def transcribe(self, audio, language=None, **kwargs) -> str:
        self.calls += 1
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

