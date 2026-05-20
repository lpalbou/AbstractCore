import importlib.metadata

import pytest

from abstractcore.core.multimodal_generation import MultimodalGenerateResponse
from abstractcore.core.types import GenerateResponse
from abstractcore.media.types import MediaContent, MediaType
from abstractcore.providers.base import BaseProvider


class _FakeEntryPoint:
    name = "fake"
    value = "tests.fake_multimodal_plugin:register"

    def __init__(self, obj):
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


class _FakeProvider(BaseProvider):
    def __init__(self):
        super().__init__(model="fake-model")
        self.provider = "fake"
        self.provider_calls = []
        self.plugin_calls = []

    def _generate_internal(
        self,
        prompt,
        messages=None,
        system_prompt=None,
        tools=None,
        media=None,
        stream=False,
        **kwargs,
    ):
        self.provider_calls.append(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "media": media,
                "stream": stream,
                "kwargs": dict(kwargs),
            }
        )
        return GenerateResponse(content=f"generated:{prompt}", model=self.model)

    def get_capabilities(self):
        return []

    def unload_model(self, model_name: str) -> None:
        return None

    def list_available_models(self, **kwargs):
        return [self.model]


class _NativeAsyncFakeProvider(_FakeProvider):
    async def _agenerate_internal(
        self, prompt, messages, system_prompt, tools, media, stream, **kwargs
    ):
        self.provider_calls.append(
            {
                "prompt": prompt,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "media": media,
                "stream": stream,
                "kwargs": dict(kwargs),
                "native_async": True,
            }
        )
        return GenerateResponse(content=f"async-generated:{prompt}", model=self.model)


def _make_plugin_ep():
    def register(registry):
        class _Voice:
            backend_id = "fake-voice"

            def __init__(self, owner):
                self.owner = owner

            def tts(self, text: str, *, voice=None, format="wav", **kwargs):
                self.owner.plugin_calls.append(("tts", text, voice, format, kwargs))
                return b"voice-bytes"

            def stt(self, audio, **kwargs):
                return "voice transcript"

            def clone(self, audio, *, name=None, reference_text=None, consent=None, **kwargs):
                self.owner.plugin_calls.append(
                    ("clone", audio, name, reference_text, consent, kwargs)
                )
                return {"voice_id": "voice-123", "name": name or "clone"}

        class _Audio:
            backend_id = "fake-audio"

            def __init__(self, owner):
                self.owner = owner

            def transcribe(self, audio, **kwargs):
                self.owner.plugin_calls.append(("transcribe", audio, kwargs))
                return "transcribed audio"

        class _Vision:
            backend_id = "fake-vision"

            def __init__(self, owner):
                self.owner = owner

            def t2i(self, prompt: str, **kwargs):
                self.owner.plugin_calls.append(("t2i", prompt, kwargs))
                return b"png-bytes"

            def i2i(self, prompt: str, image, *, mask=None, **kwargs):
                self.owner.plugin_calls.append(("i2i", prompt, image, mask, kwargs))
                return b"edited-png-bytes"

            def t2v(self, prompt: str, **kwargs):
                return b"mp4"

            def i2v(self, image, **kwargs):
                return b"mp4"

        class _Music:
            backend_id = "fake-music"

            def __init__(self, owner):
                self.owner = owner

            def t2m(self, prompt: str, *, lyrics=None, format="wav", **kwargs):
                self.owner.plugin_calls.append(("t2m", prompt, lyrics, format, kwargs))
                return b"music-bytes"

        registry.register_voice_backend(
            backend_id="fake-voice", factory=lambda owner: _Voice(owner)
        )
        registry.register_audio_backend(
            backend_id="fake-audio", factory=lambda owner: _Audio(owner)
        )
        registry.register_vision_backend(
            backend_id="fake-vision", factory=lambda owner: _Vision(owner)
        )
        registry.register_music_backend(
            backend_id="fake-music", factory=lambda owner: _Music(owner)
        )

    return _FakeEntryPoint(register)


@pytest.fixture()
def fake_plugins(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_plugin_ep()])
    )


@pytest.mark.basic
def test_generate_without_output_keeps_text_path_and_does_not_load_plugins(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: (_ for _ in ()).throw(AssertionError("plugins loaded")),
    )

    llm = _FakeProvider()
    response = llm.generate("hello")

    assert isinstance(response, GenerateResponse)
    assert response.content == "generated:hello"
    assert llm.provider_calls[0]["kwargs"].get("output") is None


@pytest.mark.basic
def test_output_image_without_media_calls_t2i_not_text_provider(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(
        "red cube", output={"modality": "image", "width": 64, "height": 64, "format": "png"}
    )

    assert isinstance(response, MultimodalGenerateResponse)
    assert response.outputs["image"][0].data == b"png-bytes"
    assert llm.provider_calls == []
    assert llm.plugin_calls[0][0] == "t2i"
    assert llm.plugin_calls[0][1] == "red cube"
    assert llm.plugin_calls[0][2]["width"] == 64
    assert "format" not in llm.plugin_calls[0][2]


@pytest.mark.basic
def test_output_image_with_one_image_media_infers_edit(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(
        "make it watercolor",
        media="source.png",
        output={"modality": "image", "width": 1024, "height": 1024},
    )

    assert response.outputs["image"][0].task == "image_edit"
    assert llm.provider_calls == []
    assert llm.plugin_calls[0] == ("i2i", "make it watercolor", "source.png", None, {})


@pytest.mark.basic
def test_output_image_with_source_and_mask_roles(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(
        "change only the masked region",
        media=[
            {"type": "image", "path": "source.png", "role": "source"},
            {"type": "image", "path": "mask.png", "role": "mask"},
        ],
        output="image",
    )

    assert response.outputs["image"][0].task == "image_edit"
    assert llm.plugin_calls[0] == (
        "i2i",
        "change only the masked region",
        "source.png",
        "mask.png",
        {},
    )


@pytest.mark.basic
def test_output_image_with_unroled_source_and_mask_infers_edit(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(
        "change only the masked region",
        media=[
            "source.png",
            {"type": "image", "path": "mask.png", "role": "mask"},
        ],
        output="image",
    )

    assert response.outputs["image"][0].task == "image_edit"
    assert llm.plugin_calls[0] == (
        "i2i",
        "change only the masked region",
        "source.png",
        "mask.png",
        {},
    )


@pytest.mark.basic
def test_output_image_with_ambiguous_images_raises(fake_plugins):
    llm = _FakeProvider()

    with pytest.raises(ValueError, match="Multiple image media items require explicit roles"):
        llm.generate("edit these", media=["a.png", "b.png"], output="image")


@pytest.mark.basic
def test_output_voice_with_text_calls_tts_without_text_model(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(
        text="Hello from AbstractCore.", output={"modality": "voice", "voice": "coral"}
    )

    assert response.outputs["voice"][0].task == "tts"
    assert response.outputs["voice"][0].data == b"voice-bytes"
    assert llm.provider_calls == []
    assert llm.plugin_calls[0][0:4] == ("tts", "Hello from AbstractCore.", "coral", "wav")


@pytest.mark.basic
def test_output_voice_tts_forwards_backend_kwargs(fake_plugins):
    llm = _FakeProvider()

    llm.generate(
        text="Hello from AbstractCore.",
        output={
            "modality": "voice",
            "voice": "coral",
            "format": "wav",
            "speed": 1.1,
            "provider": "supertonic",
            "model": "supertonic-3",
            "run_id": "run-1",
            "tags": {"case": "tts"},
            "metadata": {"source": "test"},
        },
    )

    assert llm.plugin_calls[0][0:4] == ("tts", "Hello from AbstractCore.", "coral", "wav")
    assert llm.plugin_calls[0][4]["speed"] == 1.1
    assert llm.plugin_calls[0][4]["provider"] == "supertonic"
    assert llm.plugin_calls[0][4]["model"] == "supertonic-3"
    assert llm.plugin_calls[0][4]["run_id"] == "run-1"
    assert llm.plugin_calls[0][4]["tags"] == {"case": "tts"}
    assert llm.plugin_calls[0][4]["metadata"] == {"source": "test"}


@pytest.mark.basic
def test_output_music_with_text_calls_music_generate(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(
        text="A bright analog synth hook.",
        output={
            "modality": "music",
            "lyrics": "[Instrumental]",
            "format": "wav",
            "duration_s": 8,
            "provider": "ace-step",
            "model": "ACE-Step/acestep-v15-xl-turbo-diffusers",
        },
    )

    assert response.outputs["music"][0].task == "music_generation"
    assert response.outputs["music"][0].data == b"music-bytes"
    assert response.outputs["music"][0].content_type == "audio/wav"
    assert llm.provider_calls == []
    assert llm.plugin_calls[0][0:4] == (
        "t2m",
        "A bright analog synth hook.",
        "[Instrumental]",
        "wav",
    )
    assert llm.plugin_calls[0][4]["duration_s"] == 8
    assert llm.plugin_calls[0][4]["provider"] == "ace-step"
    assert llm.plugin_calls[0][4]["model"] == "ACE-Step/acestep-v15-xl-turbo-diffusers"


@pytest.mark.basic
def test_output_voice_with_audio_media_infers_clone_resource(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(
        text="Optional transcript.",
        media={"type": "audio", "path": "reference.wav", "role": "clone_sample"},
        output={"modality": "voice", "name": "narrator", "consent": "consent-1"},
    )

    assert response.outputs == {}
    assert response.resources["voice"][0].resource_id == "voice-123"
    assert llm.provider_calls == []
    assert llm.plugin_calls[0][0:5] == (
        "clone",
        "reference.wav",
        "narrator",
        "Optional transcript.",
        "consent-1",
    )


@pytest.mark.basic
def test_output_voice_with_audio_and_voice_id_requires_explicit_task(fake_plugins):
    llm = _FakeProvider()

    with pytest.raises(ValueError, match="ambiguous"):
        llm.generate(
            text="Hello.",
            media={"type": "audio", "path": "reference.wav"},
            output={"modality": "voice", "voice": "existing"},
        )


@pytest.mark.basic
def test_output_voice_with_audio_and_voice_id_without_prompt_is_ambiguous(fake_plugins):
    llm = _FakeProvider()

    with pytest.raises(ValueError, match="ambiguous"):
        llm.generate(
            media={"type": "audio", "path": "reference.wav"},
            output={"modality": "voice", "voice": "existing"},
        )


@pytest.mark.basic
def test_task_only_output_specs_infer_modality(fake_plugins):
    llm = _FakeProvider()

    speech = llm.generate(text="Hello.", output={"task": "tts"})
    image = llm.generate("red square", output={"task": "t2i"})
    clone = llm.generate(
        media={"type": "audio", "path": "reference.wav"}, output={"task": "voice_clone"}
    )
    transcript = llm.generate(
        media={"type": "audio", "path": "meeting.wav"}, output={"task": "transcription"}
    )

    assert speech.outputs["voice"][0].task == "tts"
    assert image.outputs["image"][0].task == "image_generation"
    assert clone.resources["voice"][0].resource_id == "voice-123"
    assert transcript.text.content == "transcribed audio"


@pytest.mark.basic
@pytest.mark.parametrize(
    "output",
    [
        {"task": "text_generation"},
        {"modality": "text", "task": "text_generation"},
    ],
)
def test_text_generation_output_selectors_use_normal_text_path(fake_plugins, output):
    llm = _FakeProvider()

    response = llm.generate("Hello.", output=output)

    assert isinstance(response, GenerateResponse)
    assert response.content == "generated:Hello."
    assert llm.provider_calls[0]["kwargs"].get("output") is None
    assert llm.plugin_calls == []


@pytest.mark.basic
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        {"task": "text_generation"},
        {"modality": "text", "task": "text_generation"},
    ],
)
async def test_native_async_text_generation_output_selectors_use_normal_text_path(
    fake_plugins, output
):
    llm = _NativeAsyncFakeProvider()

    response = await llm.agenerate("Hello.", output=output)

    assert isinstance(response, GenerateResponse)
    assert response.content == "async-generated:Hello."
    assert llm.provider_calls[0]["native_async"] is True
    assert llm.provider_calls[0]["kwargs"].get("output") is None
    assert llm.plugin_calls == []


@pytest.mark.basic
def test_output_text_with_audio_and_no_prompt_transcribes(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(media={"type": "audio", "path": "meeting.wav"}, output="text")

    assert response.text.content == "transcribed audio"
    assert llm.provider_calls == []
    assert llm.plugin_calls[0] == ("transcribe", "meeting.wav", {})


@pytest.mark.basic
def test_multi_output_single_text_feeds_image_and_voice(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate(
        "write a short story",
        output=[
            {"id": "story", "modality": "text"},
            {"modality": "image"},
            {"modality": "voice", "format": "mp3"},
        ],
    )

    assert response.text.content == "generated:write a short story"
    assert response.outputs["image"][0].data == b"png-bytes"
    assert response.outputs["voice"][0].data == b"voice-bytes"
    assert len(llm.provider_calls) == 1
    assert llm.plugin_calls[0][0:2] == ("t2i", "generated:write a short story")
    assert llm.plugin_calls[1][0:4] == ("tts", "generated:write a short story", None, "mp3")


@pytest.mark.basic
def test_streaming_non_text_output_raises(fake_plugins):
    llm = _FakeProvider()

    with pytest.raises(ValueError, match="stream=True"):
        llm.generate("hello", output="voice", stream=True)


@pytest.mark.basic
def test_unknown_output_kw_still_reaches_provider(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate("hello", output="json")

    assert response.content == "generated:hello"
    assert llm.provider_calls[0]["kwargs"]["output"] == "json"


@pytest.mark.basic
def test_empty_output_list_still_reaches_provider(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate("hello", output=[])

    assert response.content == "generated:hello"
    assert llm.provider_calls[0]["kwargs"]["output"] == []


@pytest.mark.basic
def test_generated_item_records_backend_and_provider(fake_plugins):
    llm = _FakeProvider()

    response = llm.generate("red cube", output="image")

    item = response.outputs["image"][0]
    assert item.backend_id == "fake-vision"
    assert item.provider == "fake-vision"


@pytest.mark.basic
@pytest.mark.asyncio
async def test_async_output_voice_preserves_media_and_routes_like_sync(fake_plugins):
    llm = _FakeProvider()

    response = await llm.agenerate(
        text="Optional transcript.",
        media={"type": "audio", "path": "reference.wav", "role": "clone_sample"},
        output="voice",
    )

    assert response.resources["voice"][0].resource_id == "voice-123"
    assert llm.provider_calls == []
    assert llm.plugin_calls[0][0:4] == ("clone", "reference.wav", None, "Optional transcript.")


@pytest.mark.basic
@pytest.mark.asyncio
async def test_native_async_output_routes_through_multimodal_dispatch(fake_plugins):
    llm = _NativeAsyncFakeProvider()

    response = await llm.agenerate("red cube", output="image")

    assert response.outputs["image"][0].data == b"png-bytes"
    assert llm.provider_calls == []
    assert llm.plugin_calls[0][0] == "t2i"


@pytest.mark.basic
def test_media_dict_public_shape_preserves_role():
    media = MediaContent.from_dict({"type": "Image", "path": "source.png", "role": "source"})

    assert media.media_type is MediaType.IMAGE
    assert media.file_path == "source.png"
    assert media.metadata["role"] == "source"
