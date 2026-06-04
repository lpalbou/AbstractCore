"""
Route-aware model capability metadata and filtering.

AbstractCore keeps raw model metadata in ``assets/model_capabilities.json``.
This module turns that metadata into two public views:

* route keys such as ``input.image`` and ``output.text`` for precise setup and
  discovery flows;
* the legacy input/output enums used by existing provider and server callers.

The route keys intentionally share the same vocabulary as capability defaults
(``<kind>.<modality>``). They identify what a model can accept or produce, but
they do not describe provider readiness, download state, UI actions, or runtime
configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

from ..architectures.detection import get_model_capabilities
from ..config.capability_defaults import CAPABILITY_KINDS, CAPABILITY_MODALITIES, capability_route_key, split_capability_route


class ModelInputCapability(Enum):
    """
    Enumeration of broad input data types that models can process.

    ``AUDIO`` is retained as a compatibility umbrella. New setup and discovery
    code should prefer route keys such as ``input.voice`` and ``input.sound``
    when the modality distinction matters.
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    VOICE = "voice"
    SOUND = "sound"
    MUSIC = "music"
    SCENE3D = "scene3d"


class ModelOutputCapability(Enum):
    """
    Enumeration of broad output data types that models can produce.

    ``EMBEDDINGS`` keeps the historical API value. The route-key equivalent is
    ``embedding.text`` or another ``embedding.<modality>`` route.
    """

    TEXT = "text"
    EMBEDDINGS = "embeddings"
    IMAGE = "image"
    VIDEO = "video"
    VOICE = "voice"
    SOUND = "sound"
    MUSIC = "music"
    SCENE3D = "scene3d"
    RERANK = "rerank"


CapabilityRouteFilter = Optional[Union[str, Sequence[str]]]


_EMBEDDING_MODEL_NAME_PATTERNS = (
    "embedding",
    "embed",
    "embeddings",
    "text-embedding",
    "sentence-transformer",
    "all-minilm",
    "nomic-embed",
    "granite-embedding",
    "qwen3-embedding",
    "embeddinggemma",
)

_ROUTE_ALIASES = {
    "embedding": "embedding.text",
    "embeddings": "embedding.text",
    "output.embedding": "embedding.text",
    "output.embeddings": "embedding.text",
}

_INPUT_ROUTE_CAPABILITY_MAP = {
    "input.text": ModelInputCapability.TEXT,
    "input.image": ModelInputCapability.IMAGE,
    "input.video": ModelInputCapability.VIDEO,
    "input.voice": ModelInputCapability.VOICE,
    "input.sound": ModelInputCapability.SOUND,
    "input.music": ModelInputCapability.MUSIC,
    "input.scene3d": ModelInputCapability.SCENE3D,
}

_OUTPUT_ROUTE_CAPABILITY_MAP = {
    "output.text": ModelOutputCapability.TEXT,
    "output.image": ModelOutputCapability.IMAGE,
    "output.video": ModelOutputCapability.VIDEO,
    "output.voice": ModelOutputCapability.VOICE,
    "output.sound": ModelOutputCapability.SOUND,
    "output.music": ModelOutputCapability.MUSIC,
    "output.scene3d": ModelOutputCapability.SCENE3D,
}

_INPUT_CAPABILITY_ORDER = (
    ModelInputCapability.TEXT,
    ModelInputCapability.IMAGE,
    ModelInputCapability.AUDIO,
    ModelInputCapability.VIDEO,
    ModelInputCapability.VOICE,
    ModelInputCapability.SOUND,
    ModelInputCapability.MUSIC,
    ModelInputCapability.SCENE3D,
)

_OUTPUT_CAPABILITY_ORDER = (
    ModelOutputCapability.TEXT,
    ModelOutputCapability.EMBEDDINGS,
    ModelOutputCapability.IMAGE,
    ModelOutputCapability.VIDEO,
    ModelOutputCapability.VOICE,
    ModelOutputCapability.SOUND,
    ModelOutputCapability.MUSIC,
    ModelOutputCapability.SCENE3D,
    ModelOutputCapability.RERANK,
)


def normalize_capability_route(route: str) -> str:
    """
    Normalize a route key to canonical ``<kind>.<modality>`` form.

    Raises:
        ValueError: if the value is not a valid capability route.
    """

    raw = str(route or "").strip().lower().replace(":", ".")
    if not raw:
        raise ValueError("capability route must be a non-empty string")
    raw = _ROUTE_ALIASES.get(raw, raw)
    kind, modality = split_capability_route(raw)
    return capability_route_key(kind, modality)


def normalize_capability_route_filter(routes: CapabilityRouteFilter) -> List[str]:
    """
    Normalize a discovery filter value.

    ``routes`` may be a single route string, a comma-separated route string, or
    a sequence containing either form. Duplicates are removed while preserving
    caller order.
    """

    if routes is None:
        return []

    tokens: List[str] = []
    if isinstance(routes, str):
        values: Iterable[Any] = (routes,)
    else:
        values = routes

    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            tokens.extend(part.strip() for part in value.split(","))
        else:
            tokens.append(str(value).strip())

    normalized: List[str] = []
    seen: Set[str] = set()
    for token in tokens:
        if not token:
            continue
        route = normalize_capability_route(token)
        if route not in seen:
            normalized.append(route)
            seen.add(route)
    return normalized


def _add_route(routes: Set[str], route: str) -> None:
    routes.add(normalize_capability_route(route))


def _iter_modalities(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (part.strip() for part in value.split(","))
    if isinstance(value, Iterable):
        return value
    return (value,)


def _normalize_record_capability_routes(value: Any) -> Set[str]:
    routes: Set[str] = set()
    if isinstance(value, Mapping):
        for kind_or_route, modalities in value.items():
            raw_key = str(kind_or_route or "").strip()
            if not raw_key:
                continue
            if "." in raw_key or ":" in raw_key or raw_key.lower() in _ROUTE_ALIASES:
                try:
                    _add_route(routes, raw_key)
                except ValueError:
                    continue
                continue
            for modality in _iter_modalities(modalities):
                try:
                    _add_route(routes, capability_route_key(raw_key, modality))
                except ValueError:
                    continue
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for route in value:
            try:
                _add_route(routes, str(route))
            except ValueError:
                continue
    return routes


def _is_embedding_model(model_name: str, capabilities: Mapping[str, Any]) -> bool:
    if capabilities.get("model_type") == "embedding":
        return True
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in _EMBEDDING_MODEL_NAME_PATTERNS)


def _has_output_route(routes: Iterable[str]) -> bool:
    return any(
        route.startswith("output.")
        or route.startswith("embedding.")
        or route.startswith("rerank.")
        for route in routes
    )


def _derive_route_compatibility(
    model_name: str,
    capabilities: Mapping[str, Any],
    explicit_routes: Set[str],
) -> Set[str]:
    routes = set(explicit_routes)
    has_explicit_routes = bool(routes)

    if not has_explicit_routes:
        _add_route(routes, "input.text")

    if _is_embedding_model(model_name, capabilities):
        _add_route(routes, "embedding.text")
    elif not has_explicit_routes or not _has_output_route(routes):
        _add_route(routes, "output.text")

    if capabilities.get("vision_support", False):
        _add_route(routes, "input.image")

    video_mode = capabilities.get("video_input_mode")
    if isinstance(video_mode, str) and video_mode.strip().lower() in {"frames", "native"}:
        _add_route(routes, "input.video")
    elif capabilities.get("video_support", False):
        _add_route(routes, "input.video")

    audio_capabilities = capabilities.get("audio_input_capabilities")
    added_audio_route = False
    if isinstance(audio_capabilities, Iterable) and not isinstance(audio_capabilities, (str, bytes)):
        for audio_capability in audio_capabilities:
            item = str(audio_capability or "").strip().lower()
            if item in {"speech", "voice", "stt", "asr"}:
                _add_route(routes, "input.voice")
                added_audio_route = True
            elif item in {"sound", "audio", "sfx"}:
                _add_route(routes, "input.sound")
                added_audio_route = True
            elif item == "music":
                _add_route(routes, "input.music")
                added_audio_route = True

    if capabilities.get("audio_support", False) and not added_audio_route and not any(
        route in routes for route in ("input.voice", "input.sound", "input.music")
    ):
        _add_route(routes, "input.sound")

    return routes


def _ordered_routes(routes: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    route_set = set(routes)
    for kind in CAPABILITY_KINDS:
        for modality in CAPABILITY_MODALITIES:
            route = capability_route_key(kind, modality)
            if route in route_set:
                ordered.append(route)
                route_set.remove(route)
    ordered.extend(sorted(route_set))
    return ordered


def capability_routes_to_profile(routes: Iterable[str]) -> Dict[str, List[str]]:
    """Group route keys by kind for JSON-friendly summaries."""

    profile: Dict[str, List[str]] = {}
    for route in _ordered_routes(routes):
        try:
            kind, modality = split_capability_route(route)
        except ValueError:
            continue
        values = profile.setdefault(kind, [])
        if modality not in values:
            values.append(modality)
    return profile


def get_model_capability_routes(model_name: str) -> List[str]:
    """
    Return normalized capability routes for ``model_name``.

    The returned value is an ordered list of route keys. Route presence is the
    stable filtering signal; raw model metadata stores only modality sets.
    """

    try:
        capabilities = get_model_capabilities(model_name)
    except Exception:
        capabilities = {}

    explicit_routes = _normalize_record_capability_routes(capabilities.get("capability_routes"))
    return _ordered_routes(_derive_route_compatibility(model_name, capabilities, explicit_routes))


def model_supports_capability_route(model_name: str, route: str) -> bool:
    """Return ``True`` when ``model_name`` supports the normalized route key."""

    normalized = normalize_capability_route(route)
    return normalized in get_model_capability_routes(model_name)


def model_matches_capability_routes(model_name: str, required_routes: CapabilityRouteFilter) -> bool:
    """Return ``True`` when ``model_name`` supports all required route keys."""

    routes = normalize_capability_route_filter(required_routes)
    if not routes:
        return True
    model_routes = get_model_capability_routes(model_name)
    return all(route in model_routes for route in routes)


def _coerce_input_capabilities(
    capabilities: Optional[Sequence[Union[ModelInputCapability, str]]]
) -> List[ModelInputCapability]:
    coerced: List[ModelInputCapability] = []
    for capability in capabilities or []:
        if isinstance(capability, ModelInputCapability):
            value = capability
        else:
            value = ModelInputCapability(str(capability).strip().lower())
        if value not in coerced:
            coerced.append(value)
    return coerced


def _coerce_output_capabilities(
    capabilities: Optional[Sequence[Union[ModelOutputCapability, str]]]
) -> List[ModelOutputCapability]:
    coerced: List[ModelOutputCapability] = []
    for capability in capabilities or []:
        if isinstance(capability, ModelOutputCapability):
            value = capability
        else:
            value = ModelOutputCapability(str(capability).strip().lower())
        if value not in coerced:
            coerced.append(value)
    return coerced


def get_model_input_capabilities(model_name: str) -> List[ModelInputCapability]:
    """
    Determine the input capability enum view for a model.

    Prefer ``get_model_capability_routes`` for new code that needs precise
    voice/sound/music distinctions.
    """

    try:
        capabilities = get_model_capabilities(model_name)
        routes = get_model_capability_routes(model_name)
    except Exception:
        return [ModelInputCapability.TEXT]

    model_caps: Set[ModelInputCapability] = set()
    for route, capability in _INPUT_ROUTE_CAPABILITY_MAP.items():
        if route in routes:
            model_caps.add(capability)

    if capabilities.get("audio_support", False) or any(
        route in routes for route in ("input.voice", "input.sound", "input.music")
    ):
        model_caps.add(ModelInputCapability.AUDIO)

    return [capability for capability in _INPUT_CAPABILITY_ORDER if capability in model_caps]


def get_model_output_capabilities(model_name: str) -> List[ModelOutputCapability]:
    """
    Determine the output capability enum view for a model.

    The historical ``EMBEDDINGS`` enum value represents any ``embedding.*``
    route.
    """

    try:
        routes = get_model_capability_routes(model_name)
    except Exception:
        return [ModelOutputCapability.TEXT]

    model_caps: Set[ModelOutputCapability] = set()
    for route, capability in _OUTPUT_ROUTE_CAPABILITY_MAP.items():
        if route in routes:
            model_caps.add(capability)
    if any(route.startswith("embedding.") for route in routes):
        model_caps.add(ModelOutputCapability.EMBEDDINGS)
    if any(route.startswith("rerank.") for route in routes):
        model_caps.add(ModelOutputCapability.RERANK)

    if not model_caps:
        model_caps.add(ModelOutputCapability.TEXT)

    return [capability for capability in _OUTPUT_CAPABILITY_ORDER if capability in model_caps]


def model_matches_input_capabilities(
    model_name: str,
    required_capabilities: Optional[Sequence[Union[ModelInputCapability, str]]],
) -> bool:
    """Check if a model supports all required input capabilities."""

    required = _coerce_input_capabilities(required_capabilities)
    if not required:
        return True

    # Keep broad audio filters compatible with legacy metadata. Specific
    # voice/sound/music filters still resolve through route-derived enum values.
    model_caps = set(get_model_input_capabilities(model_name))
    return set(required).issubset(model_caps)


def model_matches_output_capabilities(
    model_name: str,
    required_capabilities: Optional[Sequence[Union[ModelOutputCapability, str]]],
) -> bool:
    """Check if a model supports all required output capabilities."""

    required = _coerce_output_capabilities(required_capabilities)
    if not required:
        return True

    model_caps = set(get_model_output_capabilities(model_name))
    return set(required).issubset(model_caps)


def filter_models_by_capabilities(
    models: List[str],
    input_capabilities: Optional[Sequence[Union[ModelInputCapability, str]]] = None,
    output_capabilities: Optional[Sequence[Union[ModelOutputCapability, str]]] = None,
    capability_routes: CapabilityRouteFilter = None,
) -> List[str]:
    """
    Filter models by precise route keys and/or legacy enum requirements.

    ``capability_routes`` is the preferred filter for setup/discovery workflows,
    for example ``["input.image", "output.text"]``.
    """

    filtered_models: List[str] = []
    required_routes = normalize_capability_route_filter(capability_routes)

    for model_name in models:
        try:
            if required_routes and not model_matches_capability_routes(model_name, required_routes):
                continue
            if input_capabilities and not model_matches_input_capabilities(model_name, input_capabilities):
                continue
            if output_capabilities and not model_matches_output_capabilities(model_name, output_capabilities):
                continue
            filtered_models.append(model_name)
        except Exception:
            continue

    return filtered_models


def get_capability_summary(model_name: str) -> Dict[str, Any]:
    """
    Get a comprehensive summary of a model's input, output, and route metadata.
    """

    input_caps = get_model_input_capabilities(model_name)
    output_caps = get_model_output_capabilities(model_name)
    routes = _ordered_routes(get_model_capability_routes(model_name))

    return {
        "model_name": model_name,
        "input_capabilities": [cap.value for cap in input_caps],
        "output_capabilities": [cap.value for cap in output_caps],
        "capability_routes": capability_routes_to_profile(routes),
        "capability_route_keys": routes,
        "is_multimodal": len(input_caps) > 1,
        "is_embedding_model": ModelOutputCapability.EMBEDDINGS in output_caps,
    }
