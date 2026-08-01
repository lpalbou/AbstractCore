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
    # Provenance marker for routes written by the fresh-install seed
    # (`RECOMMENDED_CAPABILITY_DEFAULT_ROUTES`). Informational only: it never
    # gates behaviour (file existence gates the seed), it lets surfaces say
    # "recommended default" instead of implying an operator chose the value.
    seeded: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "version": int(self.version or CAPABILITY_DEFAULTS_VERSION),
            "routes": {key: route.to_dict() for key, route in sorted(self.routes.items()) if route.configured()},
        }
        if self.seeded:
            out["seeded"] = str(self.seeded)
        return out


def normalize_kind(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    raw = _KIND_ALIASES.get(raw, raw)
    if raw not in CAPABILITY_KINDS:
        raise ValueError(
            f"Unknown capability route kind: {value!r}. "
            "Expected input, output, embedding, or rerank."
        )
    return raw


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


RECOMMENDED_SEED_VERSION = "recommended-v1"

# Fresh-install recommended defaults (operator ruling 2026-08-01): a new
# install should WORK out of the box on the framework's recommended local
# stack rather than refuse until configured. Seeded ONLY when the config file
# does not exist yet — never merged into an existing store (an operator who
# cleared a route meant it), and never on the corrupt-file fallback (those
# settings are recoverable and must not be replaced by recommendations).
# Ordinary rows once written: fully visible in every grid, overridable and
# clearable from either entry point, and always beaten by request pins.
# Text stores at input.text (the canonical storage key; output.text derives).
RECOMMENDED_CAPABILITY_DEFAULT_ROUTES: Dict[str, CapabilityRouteDefault] = {
    # Text: the 4-BIT quantized build (operator ruling 2026-08-01). The ROUTE
    # stores the bare LM Studio id because that is what the server serves when
    # a single quant is installed; the 4-bit choice is pinned by the download
    # artifact reference below, which is what actually fetches the weights.
    "input.text": CapabilityRouteDefault(provider="lmstudio", model="qwen/qwen3.5-9b"),
    "output.voice": CapabilityRouteDefault(provider="supertonic", model="supertonic-3"),
    "output.image": CapabilityRouteDefault(provider="mlx-gen", model="AbstractFramework/flux.2-klein-4b-8bit"),
}

# Per-provider artifact references that FETCH each recommended model — the
# download surface resolves these, never the route's served id. Quantization
# intent lives here (`@4bit`), because served ids drop the suffix when only
# one quant is installed while download refs must name the exact artifact.
RECOMMENDED_MODEL_DOWNLOADS: Dict[str, Dict[str, str]] = {
    "input.text": {"provider": "lmstudio", "artifact": "qwen/qwen3.5-9b@4bit"},
    "output.voice": {"provider": "supertonic", "artifact": "supertonic-3"},
    "output.image": {"provider": "mlx-gen", "artifact": "AbstractFramework/flux.2-klein-4b-8bit"},
}


# The `--only` vocabulary: the words the operator says ("text", "voice",
# "image") mapped to the route keys the recommendation actually writes. One
# table so the CLI, the Gateway endpoint and both console-TUIs offer the same
# three words and can never disagree about which row each one means.
RECOMMENDED_SELECTORS: Dict[str, str] = {
    "text": "input.text",
    "voice": "output.voice",
    "image": "output.image",
}


def recommended_selector_for_route(key: str) -> str:
    """The `--only` word for a recommended route key (`""` if it has none)."""
    for selector, route_key in RECOMMENDED_SELECTORS.items():
        if route_key == key:
            return selector
    return ""


def plan_recommended_capability_defaults(
    routes: Mapping[str, CapabilityRouteDefault],
    *,
    only: Optional[Iterable[str]] = None,
    force: bool = False,
) -> Tuple[Dict[str, Any], ...]:
    """What `apply-recommended` WOULD do to `routes`, one entry per route.

    THE SEED WILL NOT DO THIS. `seed_recommended_capability_defaults` runs only
    when the store file has never existed, deliberately: an operator who
    cleared a route meant it. That safety left a hole an operator fell into --
    "I asked for qwen3.5-9b everywhere and I see qwen3-0.6b" (2026-08-01) --
    because nothing in the product could say "make this machine match the
    recommendation". This plan is that action, and it stays honest about the
    difference between filling a gap and overruling a choice:

      `apply`      the route is empty -> the recommendation is written
      `already`    the route already names the recommended provider/model
      `kept`       the operator configured something else -> UNTOUCHED unless
                   `force`, and reported so they can see what was skipped
      `overwrite`  `force`, and the operator's provider/model is replaced

    FIELD-PRESERVING like every other writer here: only `provider` and `model`
    come from the recommendation. A pinned `base_url`, a reasoning effort and
    plugin options (`{voice: M2}`) belong to the operator's machine, not to the
    recommendation, and survive every outcome above.
    """

    wanted: Optional[set] = None
    if only is not None:
        wanted = set()
        for token in only:
            name = str(token or "").strip().lower()
            if not name:
                continue
            key = RECOMMENDED_SELECTORS.get(name, name)
            if key not in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES:
                raise ValueError(
                    f"Unknown recommended selector: {token!r}. "
                    f"Expected one of {', '.join(sorted(RECOMMENDED_SELECTORS))}."
                )
            wanted.add(key)

    plan: list = []
    for key, recommended in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES.items():
        if wanted is not None and key not in wanted:
            continue
        current = routes.get(key)
        before = current.to_dict() if isinstance(current, CapabilityRouteDefault) else {}
        configured = bool(before.get("provider") or before.get("model"))
        matches = (
            str(before.get("provider") or "") == str(recommended.provider or "")
            and str(before.get("model") or "") == str(recommended.model or "")
        )
        if matches:
            action = "already"
        elif not configured:
            action = "apply"
        elif force:
            action = "overwrite"
        else:
            action = "kept"

        after = dict(before)
        if action in {"apply", "overwrite"}:
            after["provider"] = recommended.provider
            after["model"] = recommended.model
            after = {k: v for k, v in after.items() if v not in (None, "", {})}
        plan.append(
            {
                "key": key,
                "selector": recommended_selector_for_route(key),
                "action": action,
                "changed": action in {"apply", "overwrite"},
                "recommended": recommended.to_dict(),
                "before": before,
                "after": after,
                "download": dict(RECOMMENDED_MODEL_DOWNLOADS.get(key, {})),
            }
        )
    return tuple(plan)


def seed_recommended_capability_defaults(config: CapabilityDefaultsConfig) -> CapabilityDefaultsConfig:
    """Apply the fresh-install recommendation to an empty defaults config.

    Only fills routes that are not already configured (defensive — the caller
    gates on file absence, so in practice all three are empty) and stamps the
    provenance marker so surfaces can label the values as recommended.
    """
    for key, route in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES.items():
        existing = config.routes.get(key)
        if existing is None or not existing.configured():
            config.routes[key] = CapabilityRouteDefault(
                provider=route.provider, model=route.model,
                base_url=route.base_url, reasoning=route.reasoning,
                options=dict(route.options),
            )
    config.seeded = RECOMMENDED_SEED_VERSION
    return config


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
    seeded_raw = data.get("seeded")
    seeded = str(seeded_raw).strip() if isinstance(seeded_raw, str) and seeded_raw.strip() else None
    return CapabilityDefaultsConfig(version=version_i, routes=routes, seeded=seeded)


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


# ---------------------------------------------------------------------------
# THE HIERARCHY, DERIVED ONCE
# ---------------------------------------------------------------------------
#
# `output.image` is not a remnant and not a sibling of `output.image.*` -- it is
# their PARENT: the answer for every image task that has no row of its own.
# Setting it is the simple path (one model for generate/edit/upscale), and it is
# what the fresh-install seed writes. A `.task` row overrides it for that task.
#
# Four surfaces render this grid (web console, both console-TUIs, the CLI) and
# every one of them used to draw the parent as a flat sibling ABOVE its own
# children with a red "not configured" -- which is what made an operator ask
# whether the row was dead code. The parent/child facts are derived HERE, once,
# and travel on the payload, so no surface re-derives them and they cannot drift.


def capability_route_broad_key(key: Any) -> Optional[str]:
    """The modality-cell key a task row falls back to, or ``None``.

    ``output.image.image_upscale`` -> ``output.image``; a 2-part key is already
    the modality cell and has no parent.
    """

    parts = [part for part in str(key or "").strip().split(".") if part]
    if len(parts) < 3:
        return None
    return f"{parts[0]}.{parts[1]}"


def capability_route_task_keys(key: Any) -> Tuple[str, ...]:
    """The task rows that override one modality cell, in grid order.

    ``output.image`` -> the three `output.image.*` rows. Empty for a modality
    with no persistable sub-task (voice/sound/music, every input route): those
    cells ARE the primary key, which is why the row shape can never be deleted.
    """

    parent = str(key or "").strip()
    if not parent or capability_route_broad_key(parent) is not None:
        return ()
    return tuple(
        spec.key
        for spec in iter_capability_default_specs()
        if capability_route_broad_key(spec.key) == parent
    )


def capability_route_tasks_cover_broad(key: Any, routes: Any) -> bool:
    """True when every task row under ``key`` is configured, so broad is unreachable.

    PROVABLE, not cosmetic: `_OUTPUT_ROUTE_TABLE`'s 3-part keys for a modality
    are exactly that modality's task rows, so once all of them carry a route,
    `capability_route_key_for_output` can never return the modality cell for
    that modality and nothing reads it. The grid may then say "not needed"
    instead of flagging an unset parent as a problem.

    ``routes`` is any mapping of route key -> row in the JSON-safe shape
    `list_capability_defaults()` produces (a row marked
    ``source: "not_configured"`` counts as unset), or route key ->
    `CapabilityRouteDefault`.
    """

    task_keys = capability_route_task_keys(key)
    if not task_keys or not isinstance(routes, Mapping):
        return False
    return all(_route_row_is_configured(routes.get(task_key)) for task_key in task_keys)


def _route_row_is_configured(row: Any) -> bool:
    if isinstance(row, CapabilityRouteDefault):
        return row.configured()
    if not isinstance(row, Mapping):
        return False
    if str(row.get("source") or "").strip() == "not_configured":
        return False
    if "configured" in row:
        return bool(row.get("configured"))
    return bool(
        row.get("provider")
        or row.get("model")
        or row.get("base_url")
        or row.get("reasoning")
        or row.get("options")
    )


# ---------------------------------------------------------------------------
# THE ONE TABLE: generation-task vocabulary -> capability default route key
# ---------------------------------------------------------------------------
#
# TWO ENTRY POINTS, ONE STORE. AbstractCore owns the only store of per-modality
# provider/model defaults; AbstractGateway is a CRUD surface over it. Both entry
# points therefore have to agree on ONE question: given a generation request,
# WHICH route key holds its default? That question is answered here and nowhere
# else. `abstractcore.core.generate_contract` (the execution path) and
# AbstractRuntime's LLM client (the stream lane) both delegate to this table --
# each used to carry its own copy, and the copies had already drifted.
#
# WHY THE input.X / output.X GRID (and not a flat per-task namespace):
#   `kind` is the direction of the MEDIA the route handles, not the direction of
#   the user's intent. That is the only reading under which every modality lands
#   in exactly one cell:
#       input.image   image understanding (vision-in)
#       output.image  image generation / edit / upscale
#       input.voice   speech-to-text  -- the AUDIO is the input; the text result
#                     is not a generated-media output at all, which is why
#                     `text/transcription` maps to NO output route below and is
#                     routed through `input.voice` by `_input_route_keys`
#       output.voice  text-to-speech
#       input.sound / input.music / input.video   understanding
#       output.sound / output.music / output.video / output.scene3d  generation
#   The grid also matches the multimodal COVERAGE logic already in the manager
#   ("input.image covered by input.text" when the text model is vision-capable),
#   which only makes sense on an input/output grid.
#
# BROAD vs TASK-SPECIFIC -- A PARENT AND ITS OVERRIDES, NOT TWO FLAT NAMESPACES.
# A modality cell (`output.image`) is always valid and is the fallback: it is the
# answer for EVERY image task that has no row of its own, which makes setting it
# the simple path ("one model for generate/edit/upscale") and is why the
# fresh-install seed writes `output.image` rather than three task rows. A `.task`
# suffix overrides it for that task, WHOLESALE (a row is one coherent backend
# identity -- never field-merged with its parent), but ONLY for the seven tasks in
# `CAPABILITY_ROUTE_TASKS` -- those are the ones the store can actually persist.
# Anything else (tts, stt, music_generation, ...) resolves at the modality cell.
# Emitting a `<kind>.<modality>.<task>` key for a task outside that tuple mints a
# key the store can never hold; the tuple below is the guard against that.
#
# FOUR of the seven output modalities have NO task rows at all (voice, sound,
# music, and every `input.*`/`embedding.*`/`rerank.*` cell): for them the modality
# cell is the PRIMARY key, not a fallback. Only image/video/scene3d carry both
# levels. That is why the broad row shape cannot be deleted -- deleting it would
# delete `output.voice`.
#
# A MODALITY-LEVEL QUESTION ("which image backend does this host use?") must
# resolve the SAME WAY execution does -- canonical task row first, modality cell
# second -- or advertising and execution disagree. Ask
# `capability_route_keys_for_output(modality)` for that pair rather than reading
# the modality cell directly; two advertising readers did the latter and reported
# `openai` for a host whose three image task rows all named mlx-gen.
#
# `has_source_image` picks the image/video/scene3d variant a bare request means:
# with a source image attached, "generate" is really an edit/i2v/i23d.
#
# Rows are matched IN ORDER; `_ANY_TASK` is a catch-all alias for "every task not
# claimed by an earlier row of the same modality".
_ANY_TASK = "*"

_OUTPUT_ROUTE_TABLE: Tuple[Tuple[str, Tuple[str, ...], str, Optional[str]], ...] = (
    # (modality, accepted task aliases, route key, route key when a source image is attached)
    # `transcription` must be listed BEFORE the text catch-all: a transcription
    # is not a text GENERATION, and claiming `output.text` for it would hand the
    # STT call the operator's chat model.
    ("text", ("transcription",), "", None),
    ("text", (_ANY_TASK,), "output.text", None),
    ("image", ("image_upscale", "upscale_image", "image_upscaling", "upscale"), "output.image.image_upscale", None),
    ("image", ("image_edit", "image_to_image", "i2i", "edit_image"), "output.image.image_to_image", None),
    ("image", ("", "image_generation", "text_to_image", "t2i"), "output.image.text_to_image", "output.image.image_to_image"),
    ("video", ("image_to_video", "i2v", "video_from_image", "video_edit"), "output.video.image_to_video", None),
    ("video", ("", "video_generation", "text_to_video", "t2v"), "output.video.text_to_video", "output.video.image_to_video"),
    # Voice/sound/music have no persistable sub-task, so both directions resolve
    # at the modality cell. `stt` appears here because a caller may label the
    # spec `modality=voice`; the canonical STT spec is `text/transcription`
    # above, whose default comes from the `input.voice` INPUT route.
    ("voice", ("stt", "transcribe", "transcription", "speech_to_text", "asr"), "input.voice", None),
    # `voice_clone` is the task `_infer_output_specs` assigns to a bare
    # `output="voice"` request that carries reference audio, and it is in the
    # public output vocabulary (`OUTPUT_TASK_MODALITIES`). It SYNTHESISES voice,
    # so it is an output.voice route exactly like tts -- the reference audio is
    # a conditioning input, not a second modality. (Adversary catch: the first
    # cut of this table omitted it, which silently dropped the operator's voice
    # default for every clone request and handed it back to the plugin's
    # env-or-openai fallback -- the dm#28 429 incident.)
    ("voice", ("", "tts", "text_to_speech", "speech", "speak", "voice_clone", "clone"), "output.voice", None),
    ("music", ("text_to_audio", "sound_generation", "text_to_sound", "sfx", "sound_effect"), "output.sound", None),
    ("music", ("", "music_generation", "text_to_music", "t2m"), "output.music", None),
    ("sound", ("", "sound_generation", "text_to_sound", "sfx", "sound_effect"), "output.sound", None),
    ("scene3d", ("image_to_scene3d", "i23d", "image_to_3d"), "output.scene3d.image_to_scene3d", None),
    (
        "scene3d",
        ("", "scene3d_generation", "text_to_scene3d", "t23d", "text_to_3d"),
        "output.scene3d.text_to_scene3d",
        "output.scene3d.image_to_scene3d",
    ),
)


def capability_route_key_for_output(
    modality: Any,
    task: Any = None,
    *,
    has_source_image: bool = False,
) -> Optional[str]:
    """Return the capability default route key for one generation output spec.

    THE ONE TABLE (see `_OUTPUT_ROUTE_TABLE` above). Returns ``None`` when the
    spec names no routable generation -- notably `text/transcription`, whose
    default lives on the `input.voice` INPUT route because the audio, not the
    text, is what the route provisions.
    """

    modality_s = str(modality or "").strip().lower().replace("-", "_")
    task_s = str(task or "").strip().lower().replace("-", "_")
    for row_modality, aliases, route_key, source_image_key in _OUTPUT_ROUTE_TABLE:
        if row_modality != modality_s:
            continue
        if task_s not in aliases and _ANY_TASK not in aliases:
            continue
        if has_source_image and source_image_key:
            return source_image_key
        return route_key or None
    return None


def capability_route_keys_for_output(
    modality: Any,
    task: Any = None,
    *,
    has_source_image: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """The (exact, broad-fallback) route-key pair for one generation output spec.

    The broad key is the modality cell (`output.image`); it is ``None`` when the
    exact key already IS the modality cell, so callers never look the same key
    up twice.
    """

    exact = capability_route_key_for_output(modality, task, has_source_image=has_source_image)
    if not exact:
        return None, None
    parts = [part for part in exact.split(".") if part]
    if len(parts) < 3:
        return exact, None
    return exact, f"{parts[0]}.{parts[1]}"


# THE TEXT-GENERATION ROUTE, BY NAME. `output.text` is the canonical read; the
# store canonicalizes it to `input.text`, which stays readable so a config that
# carries only the storage key still resolves. Both keys name the same cell.
TEXT_ROUTE_KEY = "output.text"
TEXT_ROUTE_STORAGE_KEY = "input.text"
TEXT_ROUTE_KEYS: Tuple[str, ...] = (TEXT_ROUTE_KEY, TEXT_ROUTE_STORAGE_KEY)


def capability_default_reasoning(routes: Any) -> Optional[str]:
    """The configured reasoning effort for the text-generation route.

    `routes` is a mapping of route key -> route row, in the JSON-safe shape
    `list_capability_defaults()` and `get_capability_default()` produce. The
    canonical key answers first, the storage key second; a row explicitly marked
    ``source: "not_configured"`` never answers.

    This is the ONE definition of where a reasoning default lives, so callers
    resolve it by asking rather than by re-deriving the key order.
    """

    if not isinstance(routes, Mapping):
        return None
    for key in TEXT_ROUTE_KEYS:
        row = routes.get(key)
        if not isinstance(row, Mapping):
            continue
        if str(row.get("source") or "").strip() == "not_configured":
            continue
        value = _clean_optional_string(row.get("reasoning"))
        if value:
            return value.lower()
    return None


def _clean_optional_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
