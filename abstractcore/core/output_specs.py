"""Public helpers for AbstractCore multimodal output selectors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


GenerationOutputSpec = Dict[str, Any]

OUTPUT_STRING_VALUES = {
    "text",
    "transcript",
    "transcription",
    "image",
    "voice",
    "speech",
    "tts",
    "audio",
}

OUTPUT_DICT_VALUES = {
    "text",
    "transcript",
    "transcription",
    "text_generation",
    "image",
    "image_generation",
    "image_edit",
    "t2i",
    "i2i",
    "image_to_image",
    "voice",
    "speech",
    "tts",
    "voice_clone",
    "clone",
}

OUTPUT_MODALITY_ALIASES = {
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
}

OUTPUT_TASK_ALIASES = {
    "speech": "tts",
    "audio": "tts",
    "clone": "voice_clone",
    "transcript": "transcription",
    "t2i": "image_generation",
    "i2i": "image_edit",
    "image_to_image": "image_edit",
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


def normalize_output_specs(value: Any) -> List[GenerationOutputSpec]:
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

    if not modality:
        if task in {"tts", "voice_clone"}:
            modality = "voice"
        elif task in {"transcription"}:
            modality = "text"
        elif task in {"image_generation", "image_edit"}:
            modality = "image"

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
    exclude: Optional[Set[str]] = None,
    strip_runtime_metadata: bool = False,
) -> Dict[str, Any]:
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
