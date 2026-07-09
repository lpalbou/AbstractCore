"""Internal request and route helpers for ``generate(...)``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..config.capability_defaults import (
    capability_route_key,
    clean_capability_route_default,
    split_capability_default_route,
)
from .output_specs import normalize_output_specs


def _clean_optional_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _clean_reasoning(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        text = value.strip().lower()
        return text or None
    return None


def _normalize_media_items(media: Any) -> list[Any]:
    if media is None:
        return []
    if isinstance(media, (str, bytes, bytearray, dict)) or hasattr(media, "media_type"):
        return [media]
    if isinstance(media, Sequence):
        return list(media)
    return [media]


def _media_role(item: Any) -> Optional[str]:
    role = None
    if isinstance(item, dict):
        role = item.get("role")
        if role is None and isinstance(item.get("metadata"), dict):
            role = item["metadata"].get("role")
    elif hasattr(item, "metadata"):
        metadata = getattr(item, "metadata", None)
        if isinstance(metadata, dict):
            role = metadata.get("role")
    return _clean_optional_text(role.lower() if isinstance(role, str) else role)


def _media_hint(item: Any) -> Optional[str]:
    value = None
    if isinstance(item, dict):
        value = item.get("kind") or item.get("hint")
        if value is None and isinstance(item.get("metadata"), dict):
            value = item["metadata"].get("kind") or item["metadata"].get("hint")
    elif hasattr(item, "metadata"):
        metadata = getattr(item, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("kind") or metadata.get("hint")
    return _clean_optional_text(str(value).lower() if value is not None else None)


def _media_type(item: Any, *, fallback: Optional[str] = None) -> Optional[str]:
    if isinstance(item, dict):
        raw = item.get("media_type", item.get("mediaType"))
        if raw is None:
            raw = item.get("type")
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"image", "audio", "video", "document", "text"}:
                return lowered
        mime = item.get("mime_type", item.get("mimeType", item.get("mime")))
        if mime is None:
            mime = item.get("content_type", item.get("contentType"))
        path = item.get("file_path", item.get("filePath", item.get("path")))
    elif hasattr(item, "media_type"):
        raw = getattr(item, "media_type", None)
        value = getattr(raw, "value", raw)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        mime = getattr(item, "mime_type", None)
        path = getattr(item, "file_path", None)
    elif isinstance(item, (bytes, bytearray)):
        return fallback
    else:
        mime = None
        path = item if isinstance(item, str) else None

    if isinstance(mime, str):
        lowered = mime.lower()
        if lowered.startswith("image/"):
            return "image"
        if lowered.startswith("audio/"):
            return "audio"
        if lowered.startswith("video/"):
            return "video"
        if lowered.startswith("text/"):
            return "text"

    if isinstance(path, str):
        lowered = path.lower()
        if lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            return "image"
        if lowered.endswith((".wav", ".mp3", ".ogg", ".m4a", ".flac")):
            return "audio"
        if lowered.endswith((".mp4", ".mov", ".avi", ".mkv")):
            return "video"
        if lowered.endswith((".txt", ".md", ".json")):
            return "text"
    return fallback


@dataclass
class GenerateRequest:
    """Normalized semantic request for ``generate(...)``."""

    text: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    media: list[Any] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        media_types: list[str] = []
        for item in self.media:
            media_type = _media_type(item, fallback="audio" if isinstance(item, (bytes, bytearray)) else None)
            if isinstance(media_type, str) and media_type:
                media_types.append(media_type)
        return {
            "text": self.text,
            "has_messages": bool(self.messages),
            "message_count": len(self.messages),
            "media_count": len(self.media),
            "media_types": media_types,
        }


@dataclass
class ResolvedGenerateRouteEntry:
    """Call-scoped routing decision for one capability route."""

    route_key: str
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    reasoning: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)
    source: str = "not_configured"
    field_sources: dict[str, str] = field(default_factory=dict)
    denied_reason: Optional[str] = None

    def configured(self) -> bool:
        return bool(self.provider or self.model or self.base_url or self.reasoning or self.options)

    def apply_to_output_spec(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        routed = dict(spec)
        for key, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("base_url", self.base_url),
        ):
            if key not in routed and isinstance(value, str) and value.strip():
                routed[key] = value.strip()
        for key, value in self.options.items():
            if isinstance(key, str) and key.strip() and key not in routed and value is not None:
                routed[key.strip()] = value
        return routed

    def to_summary(self) -> dict[str, Any]:
        summary = {
            "route_key": self.route_key,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "reasoning": self.reasoning,
            "options": dict(self.options),
            "source": self.source,
            "field_sources": dict(self.field_sources),
            "denied_reason": self.denied_reason,
        }
        return {k: v for k, v in summary.items() if v not in (None, {}, [])}


@dataclass
class ResolvedGenerateRoute:
    """Normalized call-scoped route truth for ``generate(...)``."""

    request: GenerateRequest
    output_specs: list[dict[str, Any]] = field(default_factory=list)
    text_route: Optional[ResolvedGenerateRouteEntry] = None
    input_routes: list[ResolvedGenerateRouteEntry] = field(default_factory=list)
    output_routes: list[ResolvedGenerateRouteEntry] = field(default_factory=list)
    reasoning: Optional[str] = None
    reasoning_source: Optional[str] = None

    def output_summary(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for spec in self.output_specs:
            if not isinstance(spec, dict):
                continue
            item = {
                "modality": _clean_optional_text(spec.get("modality")),
                "task": _clean_optional_text(spec.get("task")),
                "provider": _clean_optional_text(spec.get("provider")),
                "model": _clean_optional_text(spec.get("model")),
                "base_url": _clean_optional_text(spec.get("base_url")),
            }
            out.append({k: v for k, v in item.items() if v is not None})
        return out

    def to_summary(self) -> dict[str, Any]:
        return {
            "request": self.request.to_summary(),
            "outputs": self.output_summary(),
            "text_route": self.text_route.to_summary() if self.text_route is not None else None,
            "input_routes": [entry.to_summary() for entry in self.input_routes],
            "output_routes": [entry.to_summary() for entry in self.output_routes],
            "reasoning": self.reasoning,
            "reasoning_source": self.reasoning_source,
        }


def normalize_generate_request(
    *,
    prompt: Any = "",
    request: Any = None,
    text: Any = None,
    messages: Any = None,
    media: Any = None,
) -> GenerateRequest:
    """Normalize prompt-first and request-first shapes to one semantic request."""

    request_text = ""
    request_messages: list[dict[str, Any]] = []
    request_media: list[Any] = []

    if isinstance(request, GenerateRequest):
        request_text = str(request.text or "")
        request_messages = [dict(item) for item in request.messages if isinstance(item, dict)]
        request_media = list(request.media)
    elif isinstance(request, Mapping):
        raw_text = request.get("text")
        request_text = "" if raw_text is None else str(raw_text)
        raw_messages = request.get("messages")
        if isinstance(raw_messages, list):
            request_messages = [dict(item) for item in raw_messages if isinstance(item, dict)]
        request_media = _normalize_media_items(request.get("media"))
    elif request is not None:
        raise ValueError("request must be a GenerateRequest or a mapping.")

    if not request_text:
        if text is not None:
            request_text = str(text)
        elif prompt is not None:
            request_text = str(prompt)

    if not request_messages and isinstance(messages, list):
        request_messages = [dict(item) for item in messages if isinstance(item, dict)]

    if not request_media:
        request_media = _normalize_media_items(media)

    return GenerateRequest(
        text=request_text,
        messages=request_messages,
        media=request_media,
    )


def _normalize_default_row(value: Any) -> Optional[dict[str, Any]]:
    if not value:
        return None
    try:
        cleaned = clean_capability_route_default(value)
    except Exception:
        return None
    if not cleaned.configured():
        return None
    row = cleaned.to_dict()
    source = value.get("source") if isinstance(value, dict) else None
    if isinstance(source, str) and source.strip():
        row["source"] = source.strip()
    return row


def _is_explicit_not_configured(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    source = value.get("source")
    return isinstance(source, str) and source.strip() == "not_configured"


def resolve_capability_default_route(
    route_key: str,
    *,
    scoped_routes: Optional[Mapping[str, Any]] = None,
    resolver: Optional[Callable[..., Any]] = None,
    config_file: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Resolve one capability default row with exact-key then broad fallback semantics."""

    try:
        kind, modality, task = split_capability_default_route(route_key)
    except Exception:
        return None

    normalized_key = capability_route_key(kind, modality, task)
    broad_key = capability_route_key(kind, modality)

    def _from_scoped(key: str) -> Optional[dict[str, Any]]:
        if not isinstance(scoped_routes, Mapping):
            return None
        row = scoped_routes.get(key)
        if row is None and key == capability_route_key("output", "text"):
            row = scoped_routes.get(capability_route_key("input", "text"))
        if _is_explicit_not_configured(row):
            return {"key": key, "source": "not_configured"}
        return _normalize_default_row(row)

    for key in (normalized_key, broad_key):
        scoped = _from_scoped(key)
        if scoped is not None:
            scoped.setdefault("key", key)
            scoped.setdefault("source", "scoped")
            return scoped

    if callable(resolver):
        attempts: list[tuple[Any, ...]] = []
        if task is not None:
            attempts.append((kind, modality, task))
        attempts.append((kind, modality))
        attempts.append((normalized_key,))
        attempts.append((broad_key,))
        for args in attempts:
            try:
                row = resolver(*args)
            except TypeError:
                continue
            except Exception:
                row = None
            if _is_explicit_not_configured(row):
                return {"key": args[0] if len(args) == 1 else normalized_key, "source": "not_configured"}
            normalized = _normalize_default_row(row)
            if normalized is not None:
                normalized.setdefault("key", args[0] if len(args) == 1 else normalized_key)
                normalized.setdefault("source", "resolver")
                return normalized

    try:
        from ..config.manager import ConfigurationManager, get_config_manager

        manager = (
            ConfigurationManager(config_file=config_file.strip(), apply_env=False)
            if isinstance(config_file, str) and config_file.strip()
            else get_config_manager()
        )
        row = manager.get_capability_default(kind, modality, task)
        normalized = _normalize_default_row(row)
        if normalized is not None:
            normalized.setdefault("key", normalized_key)
            return normalized
    except Exception:
        return None

    if task is not None:
        try:
            from ..config.manager import ConfigurationManager, get_config_manager

            manager = (
                ConfigurationManager(config_file=config_file.strip(), apply_env=False)
                if isinstance(config_file, str) and config_file.strip()
                else get_config_manager()
            )
            row = manager.get_capability_default(kind, modality)
            normalized = _normalize_default_row(row)
            if normalized is not None:
                normalized.setdefault("key", broad_key)
                return normalized
        except Exception:
            return None
    return None


def _field_sources(route_key: str, explicit: Mapping[str, Any], default_row: Optional[Mapping[str, Any]]) -> dict[str, str]:
    sources: dict[str, str] = {}
    default_source = str(default_row.get("source") or "default").strip() if isinstance(default_row, Mapping) else "default"
    for key in ("provider", "model", "base_url", "reasoning"):
        value = explicit.get(key)
        if isinstance(value, str) and value.strip():
            sources[key] = "explicit"
        elif isinstance(default_row, Mapping):
            default_value = default_row.get(key)
            if isinstance(default_value, str) and default_value.strip():
                sources[key] = default_source
    if isinstance(default_row, Mapping):
        options = default_row.get("options")
        if isinstance(options, Mapping) and options:
            sources["options"] = default_source
    if not sources:
        sources["route"] = "not_configured"
    sources.setdefault("route", default_source if default_row else "not_configured")
    return sources


def _route_entry(
    route_key: str,
    *,
    explicit: Optional[Mapping[str, Any]] = None,
    default_row: Optional[Mapping[str, Any]] = None,
) -> ResolvedGenerateRouteEntry:
    explicit_row = dict(explicit or {})
    default_dict = dict(default_row or {})
    sources = _field_sources(route_key, explicit_row, default_dict)
    options = dict(default_dict.get("options") or {}) if isinstance(default_dict.get("options"), Mapping) else {}
    return ResolvedGenerateRouteEntry(
        route_key=route_key,
        provider=_clean_optional_text(explicit_row.get("provider")) or _clean_optional_text(default_dict.get("provider")),
        model=_clean_optional_text(explicit_row.get("model")) or _clean_optional_text(default_dict.get("model")),
        base_url=_clean_optional_text(explicit_row.get("base_url")) or _clean_optional_text(default_dict.get("base_url")),
        reasoning=_clean_reasoning(explicit_row.get("reasoning")) or _clean_reasoning(default_dict.get("reasoning")),
        options=options,
        source=sources.get("route", "not_configured"),
        field_sources=sources,
    )


def _output_route_key(spec: Mapping[str, Any], request: GenerateRequest) -> Optional[str]:
    modality = str(spec.get("modality") or "").strip().lower()
    task = str(spec.get("task") or "").strip().lower().replace("-", "_")
    source_images = [
        item
        for item in request.media
        if _media_type(item, fallback="image" if isinstance(item, (bytes, bytearray)) else None) == "image"
        and _media_role(item) == "source"
    ]

    if modality == "text":
        if task == "transcription":
            return None
        return "output.text"
    if modality == "image":
        if task in {"image_upscale", "upscale_image"}:
            return "output.image.image_upscale"
        if task in {"image_edit", "image_to_image"}:
            return "output.image.image_to_image"
        if task in {"", "image_generation", "text_to_image"}:
            return "output.image.image_to_image" if source_images else "output.image.text_to_image"
    if modality == "video":
        if task in {"image_to_video", "i2v"}:
            return "output.video.image_to_video"
        if task in {"", "video_generation", "text_to_video"}:
            return "output.video.image_to_video" if source_images else "output.video.text_to_video"
    if modality == "voice":
        return "output.voice"
    if modality == "music":
        if task == "text_to_audio":
            return "output.sound"
        return "output.music"
    if modality == "scene3d":
        if task in {"image_to_scene3d", "i23d"}:
            return "output.scene3d.image_to_scene3d"
        if task in {"", "scene3d_generation", "text_to_scene3d", "t23d"}:
            return "output.scene3d.image_to_scene3d" if source_images else "output.scene3d.text_to_scene3d"
    return None


def _input_route_keys(request: GenerateRequest, output_specs: Iterable[Mapping[str, Any]]) -> list[str]:
    keys: list[str] = []
    if request.text or request.messages:
        keys.append("input.text")
    audio_for_transcription = any(
        isinstance(spec, Mapping)
        and str(spec.get("modality") or "").strip().lower() == "text"
        and str(spec.get("task") or "").strip().lower() == "transcription"
        for spec in output_specs
    )
    for item in request.media:
        media_type = _media_type(item, fallback="audio" if isinstance(item, (bytes, bytearray)) else None)
        hint = _media_hint(item)
        if media_type == "image":
            keys.append("input.image")
        elif media_type == "video":
            keys.append("input.video")
        elif media_type == "audio":
            if audio_for_transcription or hint in {"voice", "speech"}:
                keys.append("input.voice")
            elif hint == "music":
                keys.append("input.music")
            else:
                keys.append("input.sound")
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _infer_output_specs(request: GenerateRequest, output: Any) -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in normalize_output_specs(output)]
    has_audio = any(
        _media_type(item, fallback="audio" if isinstance(item, (bytes, bytearray)) else None) == "audio"
        for item in request.media
    )
    has_source_image = any(
        _media_type(item, fallback="image" if isinstance(item, (bytes, bytearray)) else None) == "image"
        and _media_role(item) == "source"
        for item in request.media
    )
    has_any_image = any(
        _media_type(item, fallback="image" if isinstance(item, (bytes, bytearray)) else None) == "image"
        for item in request.media
    )
    for spec in specs:
        modality = str(spec.get("modality") or "").strip().lower()
        task = str(spec.get("task") or "").strip().lower()
        if modality == "text" and not task and has_audio and not request.text.strip():
            spec["task"] = "transcription"
            continue
        if modality == "image" and not task and has_source_image:
            spec["task"] = "image_edit"
            continue
        if modality == "video" and not task and has_any_image:
            spec["task"] = "image_to_video"
            continue
        if modality == "voice" and not task:
            spec["task"] = "voice_clone" if has_audio else "tts"
            continue
        if modality == "music" and not task:
            spec["task"] = "music_generation"
            continue
        if modality == "scene3d" and not task:
            spec["task"] = "image_to_scene3d" if has_source_image else "scene3d_generation"
    return specs


def resolve_generate_route(
    *,
    request: GenerateRequest,
    output: Any = None,
    scoped_routes: Optional[Mapping[str, Any]] = None,
    resolver: Optional[Callable[..., Any]] = None,
    config_file: Optional[str] = None,
    explicit_text_route: Optional[Mapping[str, Any]] = None,
    explicit_input_routes: Optional[Mapping[str, Mapping[str, Any]]] = None,
    explicit_reasoning: Any = None,
) -> ResolvedGenerateRoute:
    """Resolve one call-scoped route record for ``generate(...)``."""

    output_specs = _infer_output_specs(request, output) if output is not None else []
    text_default = resolve_capability_default_route(
        "input.text",
        scoped_routes=scoped_routes,
        resolver=resolver,
        config_file=config_file,
    )
    text_route = _route_entry("input.text", explicit=explicit_text_route, default_row=text_default)
    if not (request.text or request.messages):
        text_route = None

    input_routes: list[ResolvedGenerateRouteEntry] = []
    explicit_inputs = explicit_input_routes if isinstance(explicit_input_routes, Mapping) else {}
    for route_key in _input_route_keys(request, output_specs):
        if text_route is not None and route_key == text_route.route_key:
            continue
        default_row = resolve_capability_default_route(
            route_key,
            scoped_routes=scoped_routes,
            resolver=resolver,
            config_file=config_file,
        )
        input_routes.append(_route_entry(route_key, explicit=explicit_inputs.get(route_key), default_row=default_row))

    output_routes: list[ResolvedGenerateRouteEntry] = []
    patched_specs: list[dict[str, Any]] = []
    for spec in output_specs:
        route_key = _output_route_key(spec, request)
        if route_key is None:
            patched_specs.append(dict(spec))
            continue
        default_row = resolve_capability_default_route(
            route_key,
            scoped_routes=scoped_routes,
            resolver=resolver,
            config_file=config_file,
        )
        entry = _route_entry(route_key, explicit=spec, default_row=default_row)
        output_routes.append(entry)
        patched_specs.append(entry.apply_to_output_spec(spec))

    reasoning = _clean_reasoning(explicit_reasoning)
    reasoning_source = "explicit" if reasoning is not None else None
    if reasoning is None and text_route is not None and text_route.reasoning is not None:
        reasoning = text_route.reasoning
        reasoning_source = text_route.field_sources.get("reasoning", text_route.source)

    return ResolvedGenerateRoute(
        request=request,
        output_specs=patched_specs,
        text_route=text_route,
        input_routes=input_routes,
        output_routes=output_routes,
        reasoning=reasoning,
        reasoning_source=reasoning_source,
    )


__all__ = [
    "GenerateRequest",
    "ResolvedGenerateRoute",
    "ResolvedGenerateRouteEntry",
    "normalize_generate_request",
    "resolve_capability_default_route",
    "resolve_generate_route",
]
