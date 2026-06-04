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
    options: Dict[str, Any] = field(default_factory=dict)

    def configured(self) -> bool:
        return bool(self.provider or self.model or self.base_url or self.options)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.provider:
            out["provider"] = self.provider
        if self.model:
            out["model"] = self.model
        if self.base_url:
            out["base_url"] = self.base_url
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


def capability_route_key(kind: Any, modality: Any) -> str:
    return f"{normalize_kind(kind)}.{normalize_modality(modality)}"


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


def clean_capability_route_default(value: Any) -> CapabilityRouteDefault:
    if isinstance(value, CapabilityRouteDefault):
        return value
    data = value if isinstance(value, Mapping) else {}
    provider = _clean_optional_string(data.get("provider") or data.get("provider_id"))
    model = _clean_optional_string(data.get("model") or data.get("model_id"))
    base_url = _clean_optional_string(data.get("base_url"))
    options_raw = data.get("options")
    options = dict(options_raw) if isinstance(options_raw, Mapping) else {}

    # Backward-compatible convenience: unknown scalar fields become options so
    # plugin-specific defaults such as voice/profile are not lost.
    for key, raw in data.items():
        if key in {"provider", "provider_id", "model", "model_id", "base_url", "options"}:
            continue
        if isinstance(key, str) and key.strip():
            options.setdefault(key.strip(), raw)

    return CapabilityRouteDefault(provider=provider, model=model, base_url=base_url, options=options)


def capability_defaults_from_dict(value: Any) -> CapabilityDefaultsConfig:
    if isinstance(value, CapabilityDefaultsConfig):
        return value
    data = value if isinstance(value, Mapping) else {}
    routes_raw = data.get("routes") if isinstance(data.get("routes"), Mapping) else data
    routes: Dict[str, CapabilityRouteDefault] = {}
    for key_raw, route_raw in dict(routes_raw or {}).items():
        try:
            kind, modality = split_capability_route(key_raw)
            key = capability_route_key(kind, modality)
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
        ("output", "video", "Video Output", "video_generation", "abstractvideo or abstractvision", {}),
        ("output", "voice", "Voice Output", "text_to_speech", "abstractvoice", {"voice": "default"}),
        ("output", "sound", "Sound Effects Output", "sound_generation", "abstractsound or abstractmusic", {}),
        ("output", "music", "Music Output", "music_generation", "abstractmusic", {}),
        ("output", "scene3d", "3D Scene Output", "scene3d_generation", "abstract3d", {}),
        ("embedding", "text", "Text Embeddings", "text_embedding", "abstractcore.embeddings", {}),
        ("embedding", "image", "Image Embeddings", "image_embedding", "abstractcore.embeddings or abstractvision", {}),
        ("rerank", "text", "Text Rerank", "text_rerank", "future reranker manager", {}),
    ]
    for kind, modality, label, task, package_hint, option_examples in specs:
        yield CapabilityDefaultSpec(
            key=capability_route_key(kind, modality),
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
