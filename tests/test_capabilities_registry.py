import importlib.metadata

import pytest

from abstractcore.core.interface import AbstractCoreInterface
from abstractcore.core.types import GenerateResponse
from abstractcore.capabilities.errors import CapabilityUnavailableError


class _FakeEntryPoint:
    def __init__(self, *, name: str, value: str, obj):
        self.name = name
        self.value = value
        self._obj = obj

    def load(self):
        return self._obj


class _EntryPoints:
    def __init__(self, eps):
        self._eps = list(eps)

    def select(self, *, group: str):
        if group == "abstractcore.capabilities_plugins":
            return list(self._eps)
        return []


class _DummyProvider(AbstractCoreInterface):
    def generate(self, prompt: str, **kwargs):
        return GenerateResponse(content=str(prompt))

    def get_capabilities(self):
        return []

    def unload_model(self, model_name: str) -> None:
        return None


@pytest.mark.basic
def test_capabilities_status_empty(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))

    llm = _DummyProvider(model="dummy")
    status = llm.capabilities.status()
    assert status["plugins_loaded"] is True
    assert status["capabilities"]["voice"]["available"] is False
    assert status["capabilities"]["voice"]["install_hint"] == 'pip install "abstractcore[voice]"'
    assert status["capabilities"]["vision"]["available"] is False
    assert status["capabilities"]["vision"]["install_hint"] == 'pip install "abstractcore[vision]"'


@pytest.mark.basic
def test_missing_capability_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))

    llm = _DummyProvider(model="dummy")
    with pytest.raises(CapabilityUnavailableError) as e:
        llm.voice.tts("hello")
    assert "voice:" in str(e.value)
    assert 'pip install "abstractcore[voice]"' in str(e.value)


@pytest.mark.basic
def test_missing_music_capability_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))

    llm = _DummyProvider(model="dummy")
    with pytest.raises(CapabilityUnavailableError) as e:
        llm.music.t2m("hello")
    assert "music:" in str(e.value)
    assert "pip install abstractmusic" in str(e.value)


@pytest.mark.basic
def test_plugin_registration_and_backend_resolution(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    assert llm.capabilities.status()["capabilities"]["voice"]["available"] is True
    assert llm.voice.tts("hello") == b"wav-bytes"
    assert llm.audio.transcribe(b"123") == "transcript"
    assert llm.voice.list_profiles()[0]["profile_id"] == "coral"
    assert llm.voice.list_tts_models() == ["tts-test"]
    assert llm.voice.voice_catalog()["active_model"] == "tts-test"
    assert llm.vision.list_provider_models(task="text_to_image") == [
        {"id": "image-test", "task": "text_to_image"}
    ]
    assert llm.music.t2m("hello") == b"mp3-bytes"


@pytest.mark.basic
def test_preferred_backend_beats_priority(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_multi_backend_plugin_ep()]))

    llm_default = _DummyProvider(model="dummy")
    assert llm_default.capabilities.status()["capabilities"]["voice"]["selected_backend"] == "b"

    llm_prefer_a = _DummyProvider(model="dummy", capabilities_preferred_backends={"voice": "a"})
    assert llm_prefer_a.capabilities.status()["capabilities"]["voice"]["selected_backend"] == "a"
    assert llm_prefer_a.voice.tts("hello") == b"a"


@pytest.mark.basic
def test_voice_facade_delegates_remote_voice_to_backend(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_remote_voice_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    assert llm.voice.tts("hello", voice="coral", format="wav", speed=1.1) == b"remote:coral:wav:1.1"


@pytest.mark.basic
def test_voice_facade_respects_provider_override_before_adapter_voice_fast_path(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_provider_override_voice_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    assert llm.voice.tts("hello", voice="alloy", provider="openai", format="wav") == b"backend:openai:alloy"


@pytest.mark.basic
def test_voice_facade_delegates_local_profile_voice_to_backend(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_local_profile_voice_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    assert (
        llm.voice.tts(
            "hello",
            voice="M1",
            provider="supertonic",
            model="supertonic-3",
            format="wav",
        )
        == b"backend:supertonic:supertonic-3:M1"
    )


@pytest.mark.basic
def test_voice_clone_uses_public_backend_clone_method(monkeypatch):
    def register(registry):
        class _Voice:
            backend_id = "abstractvoice:default"

            def clone(self, audio, **kwargs):
                return {
                    "audio": audio,
                    "name": kwargs.get("name"),
                    "reference_text": kwargs.get("reference_text"),
                    "provider": kwargs.get("provider"),
                }

            def tts(self, text: str, **kwargs):
                _ = text, kwargs
                return b"wav"

            def stt(self, audio, **kwargs):
                _ = audio, kwargs
                return "transcript"

        registry.register_voice_backend(backend_id="abstractvoice:default", factory=lambda _owner: _Voice(), priority=0)

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_FakeEntryPoint(name="fake", value="tests.fake_legacy_clone:register", obj=register)]))

    llm = _DummyProvider(model="dummy")
    out = llm.voice.clone(b"abc", name="voice-a", reference_text="hello", provider="omnivoice")
    assert out == {
        "audio": b"abc",
        "name": "voice-a",
        "reference_text": "hello",
        "provider": "omnivoice",
    }


@pytest.mark.basic
def test_voice_clone_uses_public_backend_clone_voice_alias(monkeypatch):
    def register(registry):
        class _Voice:
            backend_id = "alias-voice"

            def clone_voice(self, audio, **kwargs):
                return {"audio": audio, "model": kwargs.get("model"), "provider": kwargs.get("provider")}

            def tts(self, text: str, **kwargs):
                _ = text, kwargs
                return b"wav"

            def stt(self, audio, **kwargs):
                _ = audio, kwargs
                return "transcript"

        registry.register_voice_backend(backend_id="alias-voice", factory=lambda _owner: _Voice(), priority=0)

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_FakeEntryPoint(name="fake", value="tests.fake_clone_voice_alias:register", obj=register)]))

    llm = _DummyProvider(model="dummy")
    out = llm.voice.clone(b"abc", provider="f5-tts", model="f5-tts/F5TTS_v1_Base")
    assert out == {"audio": b"abc", "model": "f5-tts/F5TTS_v1_Base", "provider": "f5-tts"}


@pytest.mark.basic
def test_voice_list_cloning_models_uses_backend_discovery_surface(monkeypatch):
    def register(registry):
        class _Voice:
            backend_id = "fake-voice"

            def tts(self, text: str, **kwargs):
                _ = text, kwargs
                return b"wav"

            def stt(self, audio, **kwargs):
                _ = audio, kwargs
                return "transcript"

            def list_cloning_models(self, provider=None):
                if provider == "fake-clone":
                    return ["clone-test"]
                if provider:
                    return []
                return ["clone-test", "clone-backup"]

        registry.register_voice_backend(backend_id="fake-voice", factory=lambda _owner: _Voice(), priority=0)

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_FakeEntryPoint(name="fake", value="tests.fake_clone_discovery:register", obj=register)]))

    llm = _DummyProvider(model="dummy")
    assert llm.voice.list_cloning_models() == ["clone-test", "clone-backup"]
    assert llm.voice.list_cloning_models(provider="fake-clone") == ["clone-test"]
    assert llm.voice.list_models(kind="cloning", provider="fake-clone") == ["clone-test"]


def _make_fake_plugin_ep():
    def register(registry):
        class _Voice:
            backend_id = "fake-voice"

            def tts(self, text: str, **kwargs):
                return b"wav-bytes"

            def stt(self, audio, **kwargs):
                return "transcript"

            def list_profiles(self, *, kind: str = "tts"):
                return [{"profile_id": "coral", "kind": kind}]

            def list_tts_models(self):
                return ["tts-test"]

            def voice_catalog(self):
                return {
                    "kind": "tts",
                    "engine_id": "fake",
                    "active_profile": {"profile_id": "coral"},
                    "active_model": "tts-test",
                    "profiles": [{"profile_id": "coral"}],
                    "tts_models": ["tts-test"],
                }

        class _Audio:
            backend_id = "fake-audio"

            def transcribe(self, audio, **kwargs):
                return "transcript"

        class _Vision:
            backend_id = "fake-vision"

            def list_provider_models(self, *, task=None):
                return [{"id": "image-test", "task": task}]

            def t2i(self, prompt: str, **kwargs):
                return b"png-bytes"

            def i2i(self, prompt: str, image, **kwargs):
                return b"edited-png-bytes"

            def t2v(self, prompt: str, **kwargs):
                return b"mp4-bytes"

            def i2v(self, image, **kwargs):
                return b"mp4-bytes"

        class _Music:
            backend_id = "fake-music"

            def t2m(self, prompt: str, **kwargs):
                return b"mp3-bytes"

        registry.register_voice_backend(
            backend_id="fake-voice",
            factory=lambda _owner: _Voice(),
            priority=0,
            description="Fake voice backend for tests",
        )
        registry.register_audio_backend(
            backend_id="fake-audio",
            factory=lambda _owner: _Audio(),
            priority=0,
            description="Fake audio backend for tests",
        )
        registry.register_vision_backend(
            backend_id="fake-vision",
            factory=lambda _owner: _Vision(),
            priority=0,
            description="Fake vision backend for tests",
        )
        registry.register_music_backend(
            backend_id="fake-music",
            factory=lambda _owner: _Music(),
            priority=0,
            description="Fake music backend for tests",
        )

    return _FakeEntryPoint(name="fake", value="tests.fake_plugin:register", obj=register)


def _make_multi_backend_plugin_ep():
    def register(registry):
        class _VoiceA:
            backend_id = "a"

            def tts(self, text: str, **kwargs):
                return b"a"

            def stt(self, audio, **kwargs):
                return "a"

        class _VoiceB:
            backend_id = "b"

            def tts(self, text: str, **kwargs):
                return b"b"

            def stt(self, audio, **kwargs):
                return "b"

        # Lower priority, but explicitly preferred in one test.
        registry.register_voice_backend(backend_id="a", factory=lambda _owner: _VoiceA(), priority=0)
        registry.register_voice_backend(backend_id="b", factory=lambda _owner: _VoiceB(), priority=10)

    return _FakeEntryPoint(name="fake-multi", value="tests.fake_plugin_multi:register", obj=register)


def _make_remote_voice_plugin_ep():
    def register(registry):
        class _Adapter:
            def is_available(self):
                return True

            def synthesize_to_bytes_with_voice(self, text: str, *, format: str, voice: str, speed=None, instructions=None):
                _ = (text, instructions)
                return f"remote:{voice}:{format}:{speed}".encode()

        class _VM:
            tts_adapter = _Adapter()

            def get_cloned_voice(self, voice_id: str):
                _ = voice_id
                return None

        class _Voice:
            backend_id = "remote-voice"

            def _get_vm(self):
                return _VM()

            def tts(self, text: str, **kwargs):
                _ = text
                return f"remote:{kwargs.get('voice')}:{kwargs.get('format')}:{kwargs.get('speed')}".encode()

            def stt(self, audio, **kwargs):
                return "transcript"

        registry.register_voice_backend(
            backend_id="remote-voice",
            factory=lambda _owner: _Voice(),
            priority=0,
        )

    return _FakeEntryPoint(name="fake-remote-voice", value="tests.fake_remote_voice:register", obj=register)


def _make_provider_override_voice_plugin_ep():
    def register(registry):
        class _Adapter:
            engine_id = "piper"

            def is_available(self):
                return True

            def synthesize_to_bytes_with_voice(self, text: str, *, format: str, voice: str, speed=None, instructions=None):
                raise AssertionError("provider override should bypass the active adapter fast path")

        class _VM:
            tts_adapter = _Adapter()
            _tts_engine_name = "piper"

            def get_cloned_voice(self, voice_id: str):
                _ = voice_id
                return None

        class _Voice:
            backend_id = "provider-override-voice"

            def _get_vm(self):
                return _VM()

            def tts(self, text: str, **kwargs):
                _ = text
                return f"backend:{kwargs.get('provider')}:{kwargs.get('voice')}".encode()

            def stt(self, audio, **kwargs):
                return "transcript"

        registry.register_voice_backend(
            backend_id="provider-override-voice",
            factory=lambda _owner: _Voice(),
            priority=0,
        )

    return _FakeEntryPoint(name="fake-provider-override-voice", value="tests.fake_provider_override_voice:register", obj=register)


def _make_local_profile_voice_plugin_ep():
    def register(registry):
        class _Adapter:
            engine_id = "supertonic"

            def is_available(self):
                return True

            def synthesize_to_bytes_with_voice(self, text: str, *, format: str, voice: str, speed=None, instructions=None):
                _ = (text, format, speed, instructions)
                raise AssertionError("local provider profile voices must be routed through backend.tts")

        class _VM:
            tts_adapter = _Adapter()
            _tts_engine_name = "supertonic"

            def get_cloned_voice(self, voice_id: str):
                _ = voice_id
                return None

        class _Voice:
            backend_id = "local-profile-voice"

            def _get_vm(self):
                return _VM()

            def tts(self, text: str, **kwargs):
                _ = text
                return f"backend:{kwargs.get('provider')}:{kwargs.get('model')}:{kwargs.get('voice')}".encode()

            def stt(self, audio, **kwargs):
                return "transcript"

        registry.register_voice_backend(
            backend_id="local-profile-voice",
            factory=lambda _owner: _Voice(),
            priority=0,
        )

    return _FakeEntryPoint(name="fake-local-profile-voice", value="tests.fake_local_profile_voice:register", obj=register)
