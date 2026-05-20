import pytest

from abstractcore.core.output_specs import (
    is_output_request,
    normalize_output_spec,
    normalize_output_specs,
    output_has_generated_media,
    output_plugin_kwargs,
    output_requires_non_chat_dispatch,
    strip_runtime_output_metadata,
)
from abstractcore.providers.base import BaseProvider


@pytest.mark.basic
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("text", True),
        (" audio ", True),
        ("image", True),
        ("music", True),
        ("t2m", True),
        ("json", False),
        ([], False),
        (["image", "voice"], True),
        (["image", "json"], False),
        ({"modality": "image"}, True),
        ({"type": "voice"}, True),
        ({"output": "text"}, True),
        ({"task": "tts"}, True),
        ({"task": "text_to_music"}, True),
        ({"task": "text_generation"}, True),
        ({"modality": "audio"}, False),
        ({"type": "audio"}, False),
        ({"output": "audio"}, False),
        ({"task": "audio"}, False),
    ],
)
def test_public_selector_matches_base_provider_wrapper(value, expected):
    assert is_output_request(value) is expected
    assert BaseProvider._is_acore_output_request(value) is expected


@pytest.mark.basic
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("audio", [{"modality": "voice", "task": "tts"}]),
        ("transcript", [{"modality": "text", "task": "transcription"}]),
        ("image_generation", [{"modality": "image", "task": "image_generation"}]),
        ("music", [{"modality": "music", "task": "music_generation"}]),
        ("text_to_music", [{"modality": "music", "task": "music_generation"}]),
        ({"task": "t2i"}, [{"task": "image_generation", "modality": "image"}]),
        ({"task": "t2m"}, [{"task": "music_generation", "modality": "music"}]),
        ({"task": "image_to_image"}, [{"task": "image_edit", "modality": "image"}]),
        ({"task": "audio"}, [{"task": "tts", "modality": "voice"}]),
        ({"task": "text_generation"}, [{"task": "text_generation", "modality": "text"}]),
        (
            {"modality": "text", "task": "text_generation"},
            [{"modality": "text", "task": "text_generation"}],
        ),
        (
            {"output": "text_generation"},
            [{"output": "text_generation", "modality": "text", "task": "text_generation"}],
        ),
        (
            {"type": "voice", "format": "wav"},
            [{"type": "voice", "format": "wav", "modality": "voice"}],
        ),
        (
            [{"modality": "text"}, {"modality": "voice", "format": "mp3"}],
            [{"modality": "text"}, {"modality": "voice", "format": "mp3"}],
        ),
    ],
)
def test_public_normalizer_matches_base_provider_wrapper(value, expected):
    assert normalize_output_specs(value) == expected
    assert BaseProvider._normalize_output_specs(value) == expected


@pytest.mark.basic
def test_normalize_output_spec_copies_input_dict():
    original = {"type": "image", "run_id": "run-1"}

    normalized = normalize_output_spec(original)

    assert normalized == {"type": "image", "run_id": "run-1", "modality": "image"}
    assert original == {"type": "image", "run_id": "run-1"}


@pytest.mark.basic
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("image", True),
        ("voice", True),
        ("music", True),
        (["text", "image"], True),
        ({"task": "voice_clone"}, False),
        ({"task": "clone"}, False),
        ([{"modality": "text"}, {"task": "voice_clone"}], False),
        ("text", False),
        ({"task": "text_generation"}, False),
        ({"modality": "text", "task": "text_generation"}, False),
        ("transcription", False),
        ({"task": "transcription"}, False),
        ("json", False),
        ({"modality": "audio"}, False),
    ],
)
def test_output_has_generated_media(value, expected):
    assert output_has_generated_media(value) is expected


@pytest.mark.basic
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("image", True),
        ("voice", True),
        ("music", True),
        ("transcription", True),
        ({"task": "transcription"}, True),
        ("text", False),
        ({"task": "text_generation"}, False),
        ({"modality": "text", "task": "text_generation"}, False),
        ("json", False),
    ],
)
def test_output_requires_non_chat_dispatch(value, expected):
    assert output_requires_non_chat_dispatch(value) is expected


@pytest.mark.basic
def test_strip_runtime_output_metadata_removes_storage_fields_without_mutating():
    original = [
        {
            "modality": "image",
            "run_id": "run-1",
            "tags": {"tenant": "demo"},
            "artifact_id": "img-1",
        },
        {"modality": "voice", "format": "wav"},
    ]

    stripped = strip_runtime_output_metadata(original)

    assert stripped == [{"modality": "image"}, {"modality": "voice", "format": "wav"}]
    assert original[0]["run_id"] == "run-1"


@pytest.mark.basic
def test_output_plugin_kwargs_can_strip_runtime_metadata_for_runtime_callers():
    spec = {
        "modality": "voice",
        "task": "tts",
        "voice": "coral",
        "format": "wav",
        "run_id": "run-1",
        "tags": {"tenant": "demo"},
        "artifact_id": "voice-1",
        "metadata": {"source": "test"},
    }

    assert BaseProvider._output_plugin_kwargs(spec) == {
        "voice": "coral",
        "format": "wav",
        "run_id": "run-1",
        "tags": {"tenant": "demo"},
        "artifact_id": "voice-1",
        "metadata": {"source": "test"},
    }
    assert output_plugin_kwargs(spec, strip_runtime_metadata=True) == {
        "voice": "coral",
        "format": "wav",
        "metadata": {"source": "test"},
    }
