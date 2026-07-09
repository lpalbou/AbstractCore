"""Shared capability routing defaults.

This module defines the small, JSON-safe contract used by AbstractCore and
AbstractGateway to describe default provider/model routing for framework
capabilities.  It intentionally does not know how to load a model or invoke a
plugin; it only normalizes the configuration shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


CAPABILITY_DEFAULTS_VERSION = 1

CAPABILITY_KINDS = ("input", "output", "embedding", "rerank")
CAPABILITY_MODALITIES = ("text", "image", "video", "voice", "sound", "music", "scene3d")
CAPABILITY_ROUTE_TASKS = (
    "text_to_image",
    "image_to_image",
    "image_upscale",
    "text_to_video",
    "image_to_video",
    "text_to_scene3d",
    "image_to_scene3d",
)

_KIND_ALIASES = {
    "in": "input",
    "inputs": "input",
    "understand": "input",
    "understanding": "input",
    "out": "output",
    "outputs": "output",
    "generate": "output",
    "generation": "output",
    "embed": "embedding",
    "embeddings": "embedding",
    "vector": "embedding",
    "vectors": "embedding",
    "rank": "rerank",
    "ranking": "rerank",
    "reranker": "rerank",
    "rerankers": "rerank",
}

_MODALITY_ALIASES = {
    "speech": "voice",
    "tts": "voice",
    "stt": "voice",
    "sfx": "sound",
    "sound_effect": "sound",
    "sound_effects": "sound",
    "audio": "sound",
    "3d": "scene3d",
    "3d_scene": "scene3d",
    "scene_3d": "scene3d",
    "scene-3d": "scene3d",
    "scene": "scene3d",
}

_TASK_ALIASES = {
    "t2i": "text_to_image",
    "image_generation": "text_to_image",
    "generate_image": "text_to_image",
    "i2i": "image_to_image",
    "image_edit": "image_to_image",
    "edit_image": "image_to_image",
    "upscale": "image_upscale",
    "upscaler": "image_upscale",
    "upscale_image": "image_upscale",
    "image_upscaling": "image_upscale",
    "t2v": "text_to_video",
    "video_generation": "text_to_video",
    "generate_video": "text_to_video",
    "i2v": "image_to_video",
    "video_from_image": "image_to_video",
    "image_video": "image_to_video",
    "t23d": "text_to_scene3d",
    "text2scene3d": "text_to_scene3d",
    "text_to_3d": "text_to_scene3d",
    "i23d": "image_to_scene3d",
    "image2scene3d": "image_to_scene3d",
    "image_to_3d": "image_to_scene3d",
    "image_to_scene": "image_to_scene3d",
}


@dataclass(frozen=True)
class CapabilityDefaultSpec:
    """One routable framework capability row."""

    key: str
    kind: str
    modality: str
    label: str
    task: str
    package_hint: Optional[str] = None
    option_examples: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "modality": self.modality,
            "label": self.label,
            "task": self.task,
        }
        if self.package_hint:
            out["package_hint"] = self.package_hint
        if self.option_examples:
            out["option_examples"] = dict(self.option_examples)
        return out


@dataclass
class CapabilityRouteDefault:
    """Default routing target for one capability route."""

    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    reasoning: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def configured(self) -> bool:
        return bool(self.provider or self.model or self.base_url or self.reasoning or self.options)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.provider:
            out["provider"] = self.provider
        if self.model:
            out["model"] = self.model
        if self.base_url:
            out["base_url"] = self.base_url
        if self.reasoning:
            out["reasoning"] = self.reasoning
        if self.options:
            out["options"] = dict(self.options)
        return out


@dataclass
class CapabilityDefaultsConfig:
    """Versioned collection of capability routing defaults."""

    version: int = CAPABILITY_DEFAULTS_VERSION
    routes: Dict[str, CapabilityRouteDefault] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version or CAPABILITY_DEFAULTS_VERSION),
            "routes": {key: route.to_dict() for key, route in sorted(self.routes.items()) if route.configured()},
        }


def normalize_kind(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    raw = _KIND_ALIASES.get(raw, raw)
    if raw not in CAPABILITY_KINDS:
        raise ValueError(
            f"Unknown capability route kind: {value!r}. "
            "Expected input, output, embedding, or rerank."
        )
    return raw


def normalize_direction(value: Any) -> str:
    """Compatibility alias for callers not yet renamed to route kind."""
    return normalize_kind(value)


def normalize_modality(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    raw = _MODALITY_ALIASES.get(raw, raw)
    if raw not in CAPABILITY_MODALITIES:
        raise ValueError(
            f"Unknown capability modality: {value!r}. "
            "Expected text, image, video, voice, sound, music, or scene3d."
        )
    return raw


def normalize_task(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    raw = _TASK_ALIASES.get(raw, raw)
    if raw not in CAPABILITY_ROUTE_TASKS:
        raise ValueError(
            f"Unknown capability route task: {value!r}. "
            "Expected text_to_image, image_to_image, image_upscale, text_to_video, image_to_video, "
            "text_to_scene3d, or image_to_scene3d."
        )
    return raw


def capability_route_key(kind: Any, modality: Any, task: Any = None) -> str:
    base = f"{normalize_kind(kind)}.{normalize_modality(modality)}"
    if task is None or str(task or "").strip() == "":
        return base
    return f"{base}.{normalize_task(task)}"


def split_capability_route(value: Any, modality: Any = None) -> Tuple[str, str]:
    if modality is not None:
        return normalize_kind(value), normalize_modality(modality)

    raw = str(value or "").strip()
    if "." in raw:
        left, right = raw.split(".", 1)
        return normalize_kind(left), normalize_modality(right)
    if ":" in raw:
        left, right = raw.split(":", 1)
        return normalize_kind(left), normalize_modality(right)
    raise ValueError("Capability route must be written as kind.modality, for example output.text.")


def split_capability_default_route(value: Any, modality: Any = None, task: Any = None) -> Tuple[str, str, Optional[str]]:
    """Split a persisted capability default route.

    Defaults may be broad (`output.image`) or task-specific
    (`output.image.image_to_image`). Model capability routes intentionally keep
    using `split_capability_route` so static model metadata stays broad.
    """

    if modality is not None:
        normalized_task = normalize_task(task) if task is not None and str(task or "").strip() else None
        return normalize_kind(value), normalize_modality(modality), normalized_task

    raw = str(value or "").strip()
    separator = "." if "." in raw else ":" if ":" in raw else ""
    if not separator:
        raise ValueError("Capability route must be written as kind.modality, for example output.text.")
    parts = [part.strip() for part in raw.replace(":", ".").split(".") if part.strip()]
    if len(parts) == 2:
        return normalize_kind(parts[0]), normalize_modality(parts[1]), None
    if len(parts) == 3:
        return normalize_kind(parts[0]), normalize_modality(parts[1]), normalize_task(parts[2])
    raise ValueError(
        "Capability default route must be written as kind.modality or kind.modality.task, "
        "for example output.image.image_to_image."
    )


def clean_capability_route_default(value: Any) -> CapabilityRouteDefault:
    if isinstance(value, CapabilityRouteDefault):
        return value
    data = value if isinstance(value, Mapping) else {}
    provider = _clean_optional_string(data.get("provider") or data.get("provider_id"))
    model = _clean_optional_string(data.get("model") or data.get("model_id"))
    base_url = _clean_optional_string(data.get("base_url"))
    reasoning = _clean_optional_string(data.get("reasoning"))
    options_raw = data.get("options")
    options = dict(options_raw) if isinstance(options_raw, Mapping) else {}

    # Backward-compatible convenience: unknown scalar fields become options so
    # plugin-specific defaults such as voice/profile are not lost.
    for key, raw in data.items():
        if key in {
            "provider",
            "provider_id",
            "model",
            "model_id",
            "base_url",
            "reasoning",
            "options",
            "key",
            "source",
            "kind",
            "modality",
            "task",
            "label",
            "package_hint",
            "option_examples",
        }:
            continue
        if isinstance(key, str) and key.strip():
            options.setdefault(key.strip(), raw)

    return CapabilityRouteDefault(
        provider=provider,
        model=model,
        base_url=base_url,
        reasoning=reasoning,
        options=options,
    )


def capability_defaults_from_dict(value: Any) -> CapabilityDefaultsConfig:
    if isinstance(value, CapabilityDefaultsConfig):
        return value
    data = value if isinstance(value, Mapping) else {}
    routes_raw = data.get("routes") if isinstance(data.get("routes"), Mapping) else data
    routes: Dict[str, CapabilityRouteDefault] = {}
    for key_raw, route_raw in dict(routes_raw or {}).items():
        try:
            kind, modality, task = split_capability_default_route(key_raw)
            key = capability_route_key(kind, modality, task)
            route = clean_capability_route_default(route_raw)
        except Exception:
            continue
        if route.configured():
            routes[key] = route
    version = data.get("version", CAPABILITY_DEFAULTS_VERSION)
    try:
        version_i = int(version)
    except Exception:
        version_i = CAPABILITY_DEFAULTS_VERSION
    return CapabilityDefaultsConfig(version=version_i, routes=routes)


def iter_capability_default_specs() -> Iterable[CapabilityDefaultSpec]:
    specs = [
        ("input", "text", "Text Input", "text_understanding", None, {}),
        ("input", "image", "Image Input", "image_understanding", "abstractvision or a vision-capable LLM", {}),
        ("input", "video", "Video Input", "video_understanding", "abstractvideo or a video-capable LLM", {}),
        ("input", "voice", "Voice Input", "speech_to_text", "abstractvoice", {"language": "en"}),
        ("input", "sound", "Sound Input", "audio_understanding", "abstractsound or abstractmusic", {}),
        ("input", "music", "Music Input", "music_understanding", "abstractmusic or a music-capable LLM", {}),
        ("input", "scene3d", "3D Scene Input", "scene3d_understanding", "abstract3d", {}),
        ("output", "text", "Text Output", "text_generation", None, {}),
        ("output", "image", "Image Output", "image_generation", "abstractvision", {}),
        ("output", "image.text_to_image", "Image Generation", "text_to_image", "abstractvision", {}),
        ("output", "image.image_to_image", "Image Edit", "image_to_image", "abstractvision", {}),
        ("output", "image.image_upscale", "Image Restore / Upscale", "image_upscale", "abstractvision", {"resolution": "2x", "softness": 0.25}),
        ("output", "video", "Video Output", "video_generation", "abstractvideo or abstractvision", {}),
        ("output", "video.text_to_video", "Video Generation", "text_to_video", "abstractvideo or abstractvision", {}),
        ("output", "video.image_to_video", "Image To Video", "image_to_video", "abstractvideo or abstractvision", {}),
        ("output", "voice", "Voice Output", "text_to_speech", "abstractvoice", {"voice": "default"}),
        ("output", "sound", "Sound Effects Output", "sound_generation", "abstractsound or abstractmusic", {}),
        ("output", "music", "Music Output", "music_generation", "abstractmusic", {}),
        ("output", "scene3d", "3D Scene Output", "scene3d_generation", "abstract3d", {}),
        ("output", "scene3d.text_to_scene3d", "Text To 3D", "text_to_scene3d", "abstract3d", {}),
        ("output", "scene3d.image_to_scene3d", "Image To 3D", "image_to_scene3d", "abstract3d", {}),
        ("embedding", "text", "Text Embeddings", "text_embedding", "abstractcore.embeddings", {}),
        ("embedding", "image", "Image Embeddings", "image_embedding", "abstractcore.embeddings or abstractvision", {}),
        ("rerank", "text", "Text Rerank", "text_rerank", "future reranker manager", {}),
    ]
    for kind, modality_raw, label, task, package_hint, option_examples in specs:
        modality, route_task = (
            str(modality_raw).split(".", 1) if "." in str(modality_raw) else (str(modality_raw), None)
        )
        yield CapabilityDefaultSpec(
            key=capability_route_key(kind, modality, route_task),
            kind=kind,
            modality=modality,
            label=label,
            task=task,
            package_hint=package_hint,
            option_examples=option_examples,
        )


def capability_default_specs_dict() -> Dict[str, Dict[str, Any]]:
    return {spec.key: spec.to_dict() for spec in iter_capability_default_specs()}


def _clean_optional_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
