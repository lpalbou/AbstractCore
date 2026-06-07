"""Public helpers for AbstractCore multimodal output selectors."""

from __future__ import annotations

from typing import Any

GenerationOutputSpec = dict[str, Any]

OUTPUT_STRING_VALUES = {
    "text",
    "transcript",
    "transcription",
    "image",
    "upscale",
    "image_upscale",
    "upscale_image",
    "video",
    "t2v",
    "i2v",
    "text_to_video",
    "image_to_video",
    "video_generation",
    "voice",
    "speech",
    "tts",
    "audio",
    "sound",
    "sfx",
    "text_to_audio",
    "sound_generation",
    "music",
    "song",
    "t2m",
    "text_to_music",
    "music_generation",
}

OUTPUT_DICT_VALUES = {
    "text",
    "transcript",
    "transcription",
    "text_generation",
    "image",
    "image_generation",
    "image_edit",
    "upscale",
    "image_upscale",
    "upscale_image",
    "t2i",
    "i2i",
    "image_to_image",
    "video",
    "video_generation",
    "t2v",
    "i2v",
    "text_to_video",
    "image_to_video",
    "voice",
    "speech",
    "tts",
    "voice_clone",
    "clone",
    "sound",
    "sfx",
    "text_to_audio",
    "sound_generation",
    "music",
    "song",
    "t2m",
    "text_to_music",
    "music_generation",
    "lyrics_to_music",
}

OUTPUT_MODALITY_ALIASES = {
    "text_generation": ("text", "text_generation"),
    "speech": ("voice", "tts"),
    "tts": ("voice", "tts"),
    "audio": ("voice", "tts"),
    "transcript": ("text", "transcription"),
    "transcription": ("text", "transcription"),
    "t2i": ("image", "image_generation"),
    "image_generation": ("image", "image_generation"),
    "i2i": ("image", "image_edit"),
    "image_to_image": ("image", "image_edit"),
    "image_edit": ("image", "image_edit"),
    "upscale": ("image", "image_upscale"),
    "image_upscale": ("image", "image_upscale"),
    "upscale_image": ("image", "image_upscale"),
    "video": ("video", "video_generation"),
    "video_generation": ("video", "video_generation"),
    "t2v": ("video", "text_to_video"),
    "text_to_video": ("video", "text_to_video"),
    "i2v": ("video", "image_to_video"),
    "image_to_video": ("video", "image_to_video"),
    "sound": ("music", "text_to_audio"),
    "sfx": ("music", "text_to_audio"),
    "text_to_audio": ("music", "text_to_audio"),
    "sound_generation": ("music", "text_to_audio"),
    "music": ("music", "music_generation"),
    "song": ("music", "music_generation"),
    "t2m": ("music", "music_generation"),
    "text_to_music": ("music", "music_generation"),
    "music_generation": ("music", "music_generation"),
    "lyrics_to_music": ("music", "music_generation"),
}

OUTPUT_TASK_ALIASES = {
    "speech": "tts",
    "audio": "tts",
    "clone": "voice_clone",
    "sound": "text_to_audio",
    "sfx": "text_to_audio",
    "sound_generation": "text_to_audio",
    "transcript": "transcription",
    "t2i": "image_generation",
    "i2i": "image_edit",
    "image_to_image": "image_edit",
    "upscale": "image_upscale",
    "upscale_image": "image_upscale",
    "t2v": "text_to_video",
    "i2v": "image_to_video",
    "song": "music_generation",
    "t2m": "music_generation",
    "text_to_music": "music_generation",
    "lyrics_to_music": "music_generation",
}

OUTPUT_TASK_MODALITIES = {
    "text_generation": "text",
    "transcription": "text",
    "image_generation": "image",
    "image_edit": "image",
    "image_upscale": "image",
    "video_generation": "video",
    "text_to_video": "video",
    "image_to_video": "video",
    "tts": "voice",
    "voice_clone": "voice",
    "text_to_audio": "music",
    "music_generation": "music",
}

OUTPUT_PLUGIN_EXCLUDE_KEYS = {
    "id",
    "modality",
    "type",
    "output",
    "task",
    "source",
    "prompt",
    "text",
    "media",
    "input_media",
    "role",
}

RUNTIME_OUTPUT_METADATA_KEYS = {
    "run_id",
    "tags",
    "artifact_id",
}


def is_output_request(value: Any) -> bool:
    """Return True when ``value`` is AbstractCore's multimodal output selector."""

    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in OUTPUT_STRING_VALUES
    if isinstance(value, (list, tuple)):
        return bool(value) and all(is_output_request(item) for item in value)
    if isinstance(value, dict):
        raw = value.get("modality", value.get("type", value.get("output")))
        task = value.get("task")
        values = {str(v).strip().lower() for v in (raw, task) if isinstance(v, str) and v.strip()}
        return bool(values & OUTPUT_DICT_VALUES)
    return False


def normalize_output_specs(value: Any) -> list[GenerationOutputSpec]:
    """Normalize one output selector or a list/tuple of selectors."""

    if isinstance(value, (list, tuple)):
        return [normalize_output_spec(item) for item in value]
    return [normalize_output_spec(value)]


def normalize_output_spec(value: Any) -> GenerationOutputSpec:
    """Normalize an AbstractCore output selector without changing dispatch policy."""

    spec: GenerationOutputSpec
    if isinstance(value, str):
        spec = {"modality": value}
    elif isinstance(value, dict):
        spec = dict(value)
        spec["modality"] = spec.get("modality", spec.get("type", spec.get("output")))
    else:
        raise ValueError("output must be a string, dict, or list of output specs")

    modality = str(spec.get("modality") or "").strip().lower()
    task = str(spec.get("task") or "").strip().lower()

    if modality in OUTPUT_MODALITY_ALIASES:
        modality, default_task = OUTPUT_MODALITY_ALIASES[modality]
        task = task or default_task

    if task in OUTPUT_TASK_ALIASES:
        task = OUTPUT_TASK_ALIASES[task]

    if not modality and task in OUTPUT_TASK_MODALITIES:
        modality = OUTPUT_TASK_MODALITIES[task]

    spec["modality"] = modality
    if task:
        spec["task"] = task
    return spec


def output_has_generated_media(value: Any) -> bool:
    """Return True when ``value`` requests generated non-text media."""

    if not is_output_request(value):
        return False
    for spec in normalize_output_specs(value):
        modality = str(spec.get("modality") or "").strip().lower()
        task = str(spec.get("task") or "").strip().lower()
        if modality == "voice" and task == "voice_clone":
            continue
        if modality != "text":
            return True
    return False


def output_requires_non_chat_dispatch(value: Any) -> bool:
    """Return True when selector dispatch should skip the normal chat/text path."""

    if not is_output_request(value):
        return False
    for spec in normalize_output_specs(value):
        modality = str(spec.get("modality") or "").strip().lower()
        task = str(spec.get("task") or "").strip().lower()
        if modality != "text" or task == "transcription":
            return True
    return False


def strip_runtime_output_metadata(value: Any) -> Any:
    """Return ``value`` without runtime-only artifact metadata keys."""

    if isinstance(value, (list, tuple)):
        return [strip_runtime_output_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value

    spec = dict(value)
    for key in RUNTIME_OUTPUT_METADATA_KEYS:
        spec.pop(key, None)
    return spec


def output_plugin_kwargs(
    spec: GenerationOutputSpec,
    *,
    exclude: set[str] | None = None,
    strip_runtime_metadata: bool = False,
) -> dict[str, Any]:
    """Return backend kwargs from a normalized output spec."""

    excluded = set(OUTPUT_PLUGIN_EXCLUDE_KEYS)
    if exclude:
        excluded.update(exclude)
    if strip_runtime_metadata:
        excluded.update(RUNTIME_OUTPUT_METADATA_KEYS)
    return {k: v for k, v in spec.items() if k not in excluded and v is not None}


__all__ = [
    "GenerationOutputSpec",
    "OUTPUT_DICT_VALUES",
    "OUTPUT_STRING_VALUES",
    "RUNTIME_OUTPUT_METADATA_KEYS",
    "is_output_request",
    "normalize_output_spec",
    "normalize_output_specs",
    "output_has_generated_media",
    "output_plugin_kwargs",
    "output_requires_non_chat_dispatch",
    "strip_runtime_output_metadata",
]
