from __future__ import annotations

from typing import Any, Dict, List

from abstractcore.capabilities.registry import CapabilityRegistry


class _FakeResidencyBackend:
    backend_id = "fake"

    def __init__(self) -> None:
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def load_resident_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("load", dict(request)))
        return {"task": request.get("task"), "provider": request.get("provider"), "model": request.get("model"), "state": "resident"}

    def list_loaded_models(self, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        self.calls.append(("list_loaded", dict(filters or {})))
        return [{"task": (filters or {}).get("task"), "provider": "fake", "model": "loaded"}]

    def list_resident_models(self, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        self.calls.append(("list_resident", dict(filters or {})))
        return [{"task": (filters or {}).get("task"), "provider": "fake", "model": "resident"}]

    def unload_resident_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("unload", dict(request)))
        return {"task": request.get("task"), "provider": request.get("provider"), "model": request.get("model"), "state": "unloaded"}

    def tts(self, text: str, **kwargs: Any) -> bytes:
        _ = text, kwargs
        return b""

    def stt(self, audio: Any, **kwargs: Any) -> str:
        _ = audio, kwargs
        return ""

    def transcribe(self, audio: Any, **kwargs: Any) -> str:
        _ = audio, kwargs
        return ""

    def t2i(self, prompt: str, **kwargs: Any) -> bytes:
        _ = prompt, kwargs
        return b""

    def i2i(self, prompt: str, image: Any, **kwargs: Any) -> bytes:
        _ = prompt, image, kwargs
        return b""

    def t2v(self, prompt: str, **kwargs: Any) -> bytes:
        _ = prompt, kwargs
        return b""

    def i2v(self, image: Any, **kwargs: Any) -> bytes:
        _ = image, kwargs
        return b""


def test_residency_methods_are_exposed_on_python_capability_facades() -> None:
    owner = object()
    registry = CapabilityRegistry(owner)
    registry._plugins_loaded = True

    voice = _FakeResidencyBackend()
    audio = _FakeResidencyBackend()
    vision = _FakeResidencyBackend()
    registry.register_voice_backend(backend_id="fake-voice", factory=lambda _owner: voice)
    registry.register_audio_backend(backend_id="fake-audio", factory=lambda _owner: audio)
    registry.register_vision_backend(backend_id="fake-vision", factory=lambda _owner: vision)

    assert registry.voice.load_resident_model({"task": "tts", "provider": "cloned", "model": "omnivoice"})["state"] == "resident"
    assert registry.audio.load_resident_model({"task": "stt", "provider": "faster-whisper", "model": "base"})["state"] == "resident"
    assert registry.vision.load_resident_model({"task": "image_generation", "provider": "diffusers", "model": "sd15"})["state"] == "resident"

    assert registry.voice.list_loaded_models({"task": "tts"})[0]["model"] == "loaded"
    assert registry.audio.list_resident_models({"task": "stt"})[0]["model"] == "resident"
    assert registry.vision.unload_resident_model({"provider": "diffusers", "model": "sd15"})["state"] == "unloaded"

    assert voice.calls[0][0] == "load"
    assert audio.calls[0][1]["task"] == "stt"
    assert vision.calls[-1][0] == "unload"
