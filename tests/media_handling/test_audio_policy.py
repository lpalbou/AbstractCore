from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _StubAudioBackend:
    backend_id: str = "stub:audio"
    calls: int = 0

    def transcribe(self, audio, language=None, **kwargs) -> str:  # pragma: no cover
        self.calls += 1
        return "stub transcript"


class _DummyProvider:  # intentionally minimal; mixed in at runtime
    pass


@pytest.mark.basic
def test_default_audio_policy_is_native_only_and_does_not_silently_stt(tmp_path) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.exceptions import UnsupportedFeatureError
    from abstractcore.providers.base import BaseProvider

    stub_audio = _StubAudioBackend()

    class DummyProvider(BaseProvider):
        def __init__(self):
            super().__init__(model="qwen/qwen3-next-80b")
            self._audio_backend = stub_audio
            self.last_media = None

        @property
        def audio(self):
            return self._audio_backend

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
            self.last_media = media
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop", metadata={})

        def get_capabilities(self):
            return []

        def unload_model(self, model_name: str) -> None:
            return None

        def list_available_models(self, **kwargs):
            return []

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")  # AudioProcessor only needs a file ref for v0.

    provider = DummyProvider()

    with pytest.raises(UnsupportedFeatureError):
        provider.generate("What is in this audio?", media=[str(audio_path)], stream=False)

    assert stub_audio.calls == 0


@pytest.mark.basic
def test_native_only_allows_audio_when_model_capabilities_claim_support(tmp_path) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.base import BaseProvider

    class DummyProvider(BaseProvider):
        def __init__(self):
            super().__init__(model="qwen/qwen3-next-80b")
            # Simulate an audio-capable model.
            self.model_capabilities["audio_support"] = True
            self.last_media = None

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
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
    resp = provider.generate("What is in this audio?", media=[str(audio_path)], stream=False, audio_policy="native_only")
    assert resp.content == "ok"
    assert isinstance(provider.last_media, list)
    assert provider.last_media and provider.last_media[0].media_type.value == "audio"

