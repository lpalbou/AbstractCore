import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest


def _load_model_capabilities() -> Dict[str, Any]:
    assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
    path = assets_dir / "model_capabilities.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "model_capabilities.json must parse to an object"
    return data


def _load_model_capabilities_raw_pairs() -> list[tuple[str, Any]]:
    assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
    path = assets_dir / "model_capabilities.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=lambda pairs: pairs)
    assert isinstance(data, list), "model_capabilities.json must parse to an object"
    return data


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == value and bool(value.strip())


def _require_int(value: Any, *, name: str) -> int:
    assert isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer"
    return int(value)


def _validate_output_wrappers(label: str, wrappers: Any) -> None:
    assert isinstance(wrappers, dict), f"{label}.output_wrappers must be an object"
    extra_keys = set(wrappers) - {"start", "end"}
    assert not extra_keys, f"{label}.output_wrappers has unknown keys: {sorted(extra_keys)}"
    assert any(k in wrappers for k in ("start", "end")), f"{label}.output_wrappers must include 'start' and/or 'end'"
    for k in ("start", "end"):
        if k in wrappers:
            assert _non_empty_str(wrappers.get(k)), f"{label}.output_wrappers[{k!r}] must be a non-empty string"


def _validate_thinking_tags(label: str, tags: Any) -> None:
    assert isinstance(tags, (list, tuple)), f"{label}.thinking_tags must be a 2-item list/tuple"
    assert len(tags) == 2, f"{label}.thinking_tags must have length 2"
    assert _non_empty_str(tags[0]), f"{label}.thinking_tags[0] must be non-empty"
    assert _non_empty_str(tags[1]), f"{label}.thinking_tags[1] must be non-empty"


def _validate_reasoning_levels(label: str, levels: Any) -> None:
    assert isinstance(levels, list) and levels, f"{label}.reasoning_levels must be a non-empty list when set"
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    for level in levels:
        assert _non_empty_str(level), f"{label}.reasoning_levels contains invalid: {level!r}"
        assert level in allowed, f"{label}.reasoning_levels must be subset of {sorted(allowed)}"


# Typed thinking-control surface keys (see abstractcore/architectures/thinking_controls.py).
THINKING_CONTROL_SURFACE_KEYS = {
    "prompt_disable_token",
    "template_kwarg",
    "assistant_prefill_disable",
    "budget_template_kwarg",
    "low_effort_template_kwarg",
    "request_param",
}


def _validate_thinking_control(label: str, value: Any) -> None:
    """thinking_control must be a typed object; legacy untyped strings are forbidden in shipped assets."""
    assert isinstance(value, dict), (
        f"{label}.thinking_control must be a typed object declaring control surfaces "
        f"(e.g. {{'template_kwarg': 'enable_thinking'}}); legacy string form is forbidden"
    )
    assert value, f"{label}.thinking_control must declare at least one control surface"
    extra = set(value) - THINKING_CONTROL_SURFACE_KEYS
    assert not extra, f"{label}.thinking_control contains unknown surface keys: {sorted(extra)}"
    for key, surface in value.items():
        assert isinstance(surface, str) and surface.strip(), (
            f"{label}.thinking_control[{key!r}] must be a non-empty string"
        )


def _validate_audio_input_capabilities(label: str, capabilities: Any) -> None:
    assert isinstance(capabilities, list) and capabilities, (
        f"{label}.audio_input_capabilities must be a non-empty list when set"
    )
    allowed = {"speech", "sound", "music"}
    normalized = []
    for capability in capabilities:
        assert _non_empty_str(capability), (
            f"{label}.audio_input_capabilities contains invalid: {capability!r}"
        )
        value = capability.strip().lower()
        assert value in allowed, f"{label}.audio_input_capabilities must be subset of {sorted(allowed)}"
        normalized.append(value)
    assert len(normalized) == len(set(normalized)), f"{label}.audio_input_capabilities must not contain duplicates"


_CAPABILITY_ROUTE_KINDS = {"input", "output", "embedding", "rerank"}
_CAPABILITY_ROUTE_MODALITIES = {"text", "image", "video", "voice", "sound", "music", "scene3d"}


def _validate_capability_routes(label: str, routes: Any, cfg: Mapping[str, Any]) -> None:
    assert isinstance(routes, dict) and routes, f"{label}.capability_routes must be a non-empty object when set"
    normalized_routes: list[str] = []
    for kind, modalities in routes.items():
        assert _non_empty_str(kind), f"{label}.capability_routes contains invalid kind: {kind!r}"
        assert kind in _CAPABILITY_ROUTE_KINDS, (
            f"{label}.capability_routes route kind {kind!r} must be one of {sorted(_CAPABILITY_ROUTE_KINDS)}"
        )
        assert isinstance(modalities, list) and modalities, (
            f"{label}.capability_routes[{kind!r}] must be a non-empty modality list"
        )
        normalized_modalities: list[str] = []
        for modality in modalities:
            assert _non_empty_str(modality), (
                f"{label}.capability_routes[{kind!r}] contains invalid modality: {modality!r}"
            )
            assert modality in _CAPABILITY_ROUTE_MODALITIES, (
                f"{label}.capability_routes[{kind!r}] modality {modality!r} must be one of "
                f"{sorted(_CAPABILITY_ROUTE_MODALITIES)}"
            )
            normalized_modalities.append(modality)
            normalized_routes.append(f"{kind}.{modality}")
        assert len(normalized_modalities) == len(set(normalized_modalities)), (
            f"{label}.capability_routes[{kind!r}] must not contain duplicate modalities"
        )

    assert len(normalized_routes) == len(set(normalized_routes)), (
        f"{label}.capability_routes must not contain duplicate route keys"
    )
    route_set = set(normalized_routes)
    if "input.image" in route_set:
        assert cfg.get("vision_support") is True, f"{label}.vision_support must be true when input.image is routed"
    if "input.video" in route_set:
        assert cfg.get("video_input_mode") in {"frames", "native"} or cfg.get("video_support") is True, (
            f"{label}.video_input_mode must be frames/native when input.video is routed"
        )
    audio_routes = {"input.voice", "input.sound", "input.music"} & route_set
    if audio_routes:
        assert cfg.get("audio_support") is True, (
            f"{label}.audio_support must be true when audio input routes are declared: {sorted(audio_routes)}"
        )


def _validate_inference_parameters(label: str, params: Any) -> None:
    assert isinstance(params, dict), f"{label}.inference_parameters must be an object"
    allowed = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "repeat_penalty",
        "enable_thinking",
        "clear_thinking",
    }
    extra = set(params) - allowed
    assert not extra, f"{label}.inference_parameters has unknown keys: {sorted(extra)}"
    for key, value in params.items():
        if key in {"enable_thinking", "clear_thinking"}:
            assert isinstance(value, bool), f"{label}.inference_parameters[{key!r}] must be boolean"
            continue
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{label}.inference_parameters[{key!r}] must be numeric"
        )
        if key in {"temperature", "top_p", "min_p", "typical_p", "repeat_penalty"}:
            assert float(value) >= 0, f"{label}.inference_parameters[{key!r}] must be non-negative"
        if key == "top_k":
            assert int(value) > 0, f"{label}.inference_parameters['top_k'] must be positive"


def _validate_model_entry_v0(*, model_key: str, cfg: Mapping[str, Any]) -> None:
    label = f"models[{model_key}]"

    required_keys = {
        "canonical_name",
        "aliases",
        "max_tokens",
        "max_output_tokens",
        "tool_support",
        "structured_output",
        "parallel_tools",
        "max_tools",
        "vision_support",
        "audio_support",
        "video_support",
        "video_input_mode",
    }

    # NOTE: Keep this allowlist strict to catch typos and accidental drift.
    # Add new research-only keys either under an existing bucket (e.g. `benchmarks`)
    # or by explicitly extending this allowlist + docs.
    optional_keys = {
        "active_parameters",
        "adaptive_resolution",
        "adaptive_windowing",
        "agentic_capabilities",
        "agentic_coding",
        "architecture",
        "architecture_updates",
        "arxiv",
        "aspect_ratio_support",
        "attention_layers",
        "attention_mechanism",
        "audio_input_capabilities",
        "base_image_tokens",
        "base_model",
        "base_tokens_per_resolution",
        "benchmarks",
        "capability_routes",
        "conversation_template",
        "default_system_prompt",
        "detail_levels",
        "document_understanding",
        "embedding_dimension",
        "embedding_size",
        "embedding_support",
        "expert_hidden_size",
        "experts",
        "experts_activated",
        "fine_tunable",
        "fim_support",
        "fixed_resolution",
        "frontend_replication",
        "function_calling",
        "gpu_memory_required",
        "image_patch_size",
        "image_resolutions",
        "image_tokenization_method",
        "image_tokens_per_image",
        "inference_parameters",
        "interleaved_generation",
        "languages",
        "license",
        "low_detail_tokens",
        "mamba2_layers",
        "mamba2_state_size",
        "matryoshka_dims",
        "max_image_dimension",
        "max_image_resolution",
        "max_image_tokens",
        "max_resolution",
        "memory_footprint",
        "message_format",
        "min_dimension_warning",
        "min_resolution",
        "model_class",
        "model_type",
        "native_function_calling",
        "notes",
        "ocr_languages",
        "optimized_for_glyph",
        "output_wrappers",
        "pixel_divisor",
        "pixel_grouping",
        "positional_encoding",
        "preprocessing",
        "processor_class",
        "python_execution",
        "quantization_method",
        "reasoning_configurable",
        "reasoning_levels",
        "reasoning_paradigm",
        "reasoning_parser",
        "release_date",
        "repository",
        "requires_processor",
        "response_format",
        "shared_expert_hidden_size",
        "shared_experts",
        "short_side_resize_target",
        "source",
        "spatial_perception",
        "status",
        "supported_resolutions",
        "tensor_type",
        "terminal_tasks",
        "text_image_processing",
        "thinking_budget",
        "thinking_control",
        "thinking_control_mode",
        "thinking_disable_supported",
        "reasoning_output",
        "thinking_format",
        "thinking_modes",
        "thinking_output_field",
        "thinking_paradigm",
        "thinking_support",
        "thinking_tags",
        "max_effort_supported",
        "tile_size",
        "token_cap",
        "token_formula",
        "tokens_per_tile",
        "token_param_name",
        "tool_calling_format",
        "tool_calling_parser",
        "total_parameters",
        "transformer_layers",
        "transformers_version_min",
        "trust_remote_code",
        "ui_generation",
        "unsupported_parameters",
        "video_support",
        "vision_encoder",
        "visual_agent",
        "visual_coding",
        "web_browsing",
    }

    allowed_keys = required_keys | optional_keys

    extra = set(cfg) - allowed_keys
    assert not extra, f"{label} contains unknown keys: {sorted(extra)}"

    missing = required_keys - set(cfg)
    assert not missing, f"{label} is missing required keys: {sorted(missing)}"

    canonical_name = cfg.get("canonical_name")
    assert _non_empty_str(canonical_name), f"{label}.canonical_name must be a non-empty string"

    aliases = cfg.get("aliases")
    assert isinstance(aliases, list), f"{label}.aliases must be a list"
    normalized_aliases: list[str] = []
    for a in aliases:
        assert _non_empty_str(a), f"{label}.aliases entries must be non-empty strings: {a!r}"
        normalized_aliases.append(a.strip().lower())
    assert len(normalized_aliases) == len(set(normalized_aliases)), f"{label}.aliases must not contain duplicates"

    # If canonical_name differs from the entry key, ensure it still resolves to this entry via aliases.
    if canonical_name != model_key:
        assert str(canonical_name).strip().lower() in set(normalized_aliases), (
            f"{label}.canonical_name differs from entry key; canonical_name must be included in aliases "
            f"so resolve_model_alias() can map it back to {model_key!r}"
        )

    max_tokens = _require_int(cfg.get("max_tokens"), name=f"{label}.max_tokens")
    assert max_tokens > 0, f"{label}.max_tokens must be > 0"

    max_output_tokens_raw = cfg.get("max_output_tokens")
    if max_output_tokens_raw is None:
        assert cfg.get("model_type") != "embedding", (
            f"{label}.max_output_tokens may be null only when no primary source publishes a hard generation cap"
        )
    else:
        max_output_tokens = _require_int(max_output_tokens_raw, name=f"{label}.max_output_tokens")
        assert max_output_tokens >= 0, f"{label}.max_output_tokens must be >= 0"
        if max_output_tokens == 0:
            assert cfg.get("model_type") == "embedding", (
                f"{label}.max_output_tokens==0 is only allowed for embedding models (model_type='embedding')"
            )

    tool_support = cfg.get("tool_support")
    assert tool_support in {"native", "prompted", "none"}, (
        f"{label}.tool_support must be one of: native, prompted, none"
    )
    structured_output = cfg.get("structured_output")
    assert structured_output in {"native", "prompted", "none"}, (
        f"{label}.structured_output must be one of: native, prompted, none"
    )

    parallel_tools = cfg.get("parallel_tools")
    assert isinstance(parallel_tools, bool), f"{label}.parallel_tools must be boolean"

    max_tools = _require_int(cfg.get("max_tools"), name=f"{label}.max_tools")
    assert max_tools == -1 or max_tools >= 0, f"{label}.max_tools must be -1 or >= 0"
    if tool_support == "none":
        assert max_tools == 0, f"{label}.max_tools must be 0 when tool_support='none'"
        assert parallel_tools is False, f"{label}.parallel_tools must be false when tool_support='none'"

    for key in ("vision_support", "audio_support", "video_support"):
        value = cfg.get(key)
        assert isinstance(value, bool), f"{label}.{key} must be boolean"

    audio_input_capabilities = cfg.get("audio_input_capabilities")
    if audio_input_capabilities is not None:
        _validate_audio_input_capabilities(label, audio_input_capabilities)
        assert cfg.get("audio_support") is True, (
            f"{label}.audio_support must be true when audio_input_capabilities is set"
        )

    capability_routes = cfg.get("capability_routes")
    if capability_routes is not None:
        _validate_capability_routes(label, capability_routes, cfg)

    video_support = bool(cfg.get("video_support"))
    video_mode = cfg.get("video_input_mode")
    assert video_mode in {"none", "frames", "native"}, (
        f"{label}.video_input_mode must be one of: none, frames, native"
    )

    if video_mode == "native":
        assert video_support is True, f"{label}.video_support must be true when video_input_mode='native'"
        assert cfg.get("vision_support") is True, f"{label}.vision_support must be true when video_input_mode='native'"
    elif video_mode == "frames":
        assert video_support is False, f"{label}.video_support must be false when video_input_mode='frames'"
        assert cfg.get("vision_support") is True, f"{label}.vision_support must be true when video_input_mode='frames'"
    else:
        assert video_support is False, f"{label}.video_support must be false when video_input_mode='none'"

    output_wrappers = cfg.get("output_wrappers")
    if output_wrappers is not None:
        _validate_output_wrappers(label, output_wrappers)

    thinking_tags = cfg.get("thinking_tags")
    if thinking_tags is not None:
        _validate_thinking_tags(label, thinking_tags)

    for key in ("thinking_output_field", "thinking_format", "tool_calling_format"):
        value = cfg.get(key)
        if value is not None:
            assert _non_empty_str(value), f"{label}.{key} must be a non-empty string when set"

    thinking_control = cfg.get("thinking_control")
    if thinking_control is not None:
        _validate_thinking_control(label, thinking_control)

    reasoning_output = cfg.get("reasoning_output")
    if reasoning_output is not None:
        assert isinstance(reasoning_output, bool), f"{label}.reasoning_output must be boolean"

    thinking_support = cfg.get("thinking_support")
    if thinking_support is not None:
        assert isinstance(thinking_support, bool), f"{label}.thinking_support must be boolean"

    thinking_budget = cfg.get("thinking_budget")
    if thinking_budget is not None:
        assert isinstance(thinking_budget, bool), f"{label}.thinking_budget must be boolean"

    thinking_disable_supported = cfg.get("thinking_disable_supported")
    if thinking_disable_supported is not None:
        assert isinstance(thinking_disable_supported, bool), (
            f"{label}.thinking_disable_supported must be boolean"
        )

    thinking_control_mode = cfg.get("thinking_control_mode")
    if thinking_control_mode is not None:
        assert _non_empty_str(thinking_control_mode), f"{label}.thinking_control_mode must be a non-empty string when set"
        assert thinking_control_mode in {"adaptive", "budget"}, (
            f"{label}.thinking_control_mode must be one of: adaptive, budget"
        )

    max_effort_supported = cfg.get("max_effort_supported")
    if max_effort_supported is not None:
        assert isinstance(max_effort_supported, bool), f"{label}.max_effort_supported must be boolean"

    fim_support = cfg.get("fim_support")
    if fim_support is not None:
        assert isinstance(fim_support, bool), f"{label}.fim_support must be boolean"

    reasoning_levels = cfg.get("reasoning_levels")
    if reasoning_levels is not None:
        _validate_reasoning_levels(label, reasoning_levels)

    response_format = cfg.get("response_format")
    if response_format is not None:
        assert response_format in {"harmony"}, f"{label}.response_format must be one of: harmony"

    unsupported_parameters = cfg.get("unsupported_parameters")
    if unsupported_parameters is not None:
        assert isinstance(unsupported_parameters, list), f"{label}.unsupported_parameters must be a list"
        allowed_params = {"temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"}
        for p in unsupported_parameters:
            assert _non_empty_str(p), f"{label}.unsupported_parameters entries must be non-empty strings: {p!r}"
            assert p in allowed_params, f"{label}.unsupported_parameters entry {p!r} not in allowed set {sorted(allowed_params)}"
        assert len(unsupported_parameters) == len(set(unsupported_parameters)), (
            f"{label}.unsupported_parameters must not contain duplicates"
        )

    inference_parameters = cfg.get("inference_parameters")
    if inference_parameters is not None:
        _validate_inference_parameters(label, inference_parameters)

    token_param_name = cfg.get("token_param_name")
    if token_param_name is not None:
        assert token_param_name in {"max_tokens", "max_completion_tokens"}, (
            f"{label}.token_param_name must be one of: max_tokens, max_completion_tokens"
        )


@pytest.mark.basic
def test_model_capabilities_json_has_unique_model_keys():
    data = _load_model_capabilities_raw_pairs()
    models_pairs = None
    for key, value in data:
        if key == "models":
            models_pairs = value
            break

    assert isinstance(models_pairs, list), "top-level key 'models' must exist and be an object"

    model_key_to_count: defaultdict[str, int] = defaultdict(int)
    for key, _value in models_pairs:
        assert isinstance(key, str) and key.strip(), f"model key must be a non-empty string: {key!r}"
        model_key_to_count[key] += 1

    duplicates = sorted(key for key, count in model_key_to_count.items() if count > 1)
    assert not duplicates, f"model_capabilities.json contains duplicate model keys: {duplicates}"


@pytest.mark.basic
def test_model_capabilities_json_has_required_top_level_sections():
    data = _load_model_capabilities()

    assert "models" in data, "top-level key 'models' is required"
    assert isinstance(data["models"], dict), "'models' must be an object"
    assert data["models"], "'models' must not be empty"

    assert "default_capabilities" in data, "top-level key 'default_capabilities' is required"
    assert isinstance(data["default_capabilities"], dict), "'default_capabilities' must be an object"


@pytest.mark.basic
def test_model_capabilities_schema_declares_route_metadata_contract():
    assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
    path = assets_dir / "model_capabilities.schema.json"
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    defs = schema.get("$defs")
    assert isinstance(defs, dict), "model_capabilities.schema.json must define $defs"
    assert "capabilityRoutes" in defs, "schema must define capabilityRoutes"
    assert "capabilityRouteKind" in defs, "schema must define capabilityRouteKind"
    assert set(defs.get("capabilityRouteKind", {}).get("enum") or []) == _CAPABILITY_ROUTE_KINDS
    route_modality_items = defs.get("capabilityRouteModalities", {}).get("items")
    assert isinstance(route_modality_items, dict), "schema must constrain capability route modality items"
    assert set(route_modality_items.get("enum") or []) == _CAPABILITY_ROUTE_MODALITIES


@pytest.mark.basic
def test_model_entries_conform_to_v0_template():
    data = _load_model_capabilities()
    models = data["models"]

    for model_key, cfg in models.items():
        assert _non_empty_str(model_key), f"model key must be a non-empty string: {model_key!r}"
        assert isinstance(cfg, dict), f"models[{model_key!r}] must be an object"
        _validate_model_entry_v0(model_key=model_key, cfg=cfg)


@pytest.mark.basic
def test_model_capabilities_aliases_are_unique_across_models():
    data = _load_model_capabilities()
    models = data["models"]

    alias_to_models: dict[str, list[str]] = defaultdict(list)
    for model_key, cfg in models.items():
        if not isinstance(cfg, dict):
            continue
        aliases = cfg.get("aliases", [])
        if not isinstance(aliases, list):
            continue
        for a in aliases:
            if not isinstance(a, str) or not a.strip():
                continue
            alias_to_models[a.strip().lower()].append(model_key)

    duplicates = {a: sorted(set(v)) for a, v in alias_to_models.items() if len(set(v)) > 1}
    assert not duplicates, (
        "Duplicate aliases across models are ambiguous and make alias resolution order-dependent.\n"
        f"Duplicates: {duplicates}"
    )


@pytest.mark.basic
def test_anthropic_thinking_models_declare_thinking_control_mode():
    """Prevent model-name heuristics in Anthropic provider for thinking controls."""
    data = _load_model_capabilities()
    models = data["models"]

    missing: list[str] = []
    for model_key, cfg in models.items():
        if not isinstance(cfg, dict):
            continue
        if "claude" not in str(model_key).lower():
            continue
        if cfg.get("thinking_support") is not True:
            continue
        if "thinking_control_mode" not in cfg:
            missing.append(model_key)

    assert not missing, f"Anthropic thinking models missing thinking_control_mode: {sorted(missing)}"


@pytest.mark.basic
def test_default_capabilities_conform_to_v0_template():
    data = _load_model_capabilities()
    default_caps = data["default_capabilities"]

    # Default capabilities should be a valid v0 model entry (minus identity/alias fields).
    # Use a sentinel model key for consistent error messages.
    cfg = dict(default_caps)
    cfg.setdefault("canonical_name", "default")
    cfg.setdefault("aliases", [])

    _validate_model_entry_v0(model_key="default", cfg=cfg)


@pytest.mark.basic
def test_generic_vision_model_conforms_to_v0_template():
    data = _load_model_capabilities()
    generic = data.get("generic_vision_model")
    assert isinstance(generic, dict), "top-level key 'generic_vision_model' must exist and be an object"
    _validate_model_entry_v0(model_key="generic_vision_model", cfg=generic)


@pytest.mark.basic
def test_tool_support_level_registries_match_enums():
    data = _load_model_capabilities()

    tool_support_levels = data.get("tool_support_levels")
    assert isinstance(tool_support_levels, dict), "top-level key 'tool_support_levels' must be an object"
    assert set(tool_support_levels) == {"native", "prompted", "none"}, "tool_support_levels must define native/prompted/none"

    structured_levels = data.get("structured_output_levels")
    assert isinstance(structured_levels, dict), "top-level key 'structured_output_levels' must be an object"
    assert set(structured_levels) == {"native", "prompted", "none"}, (
        "structured_output_levels must define native/prompted/none"
    )


@pytest.mark.basic
def test_parameter_constraint_fields_are_well_formed():
    """Registry lint for the wire-contract fields (parameter filtering).

    Providers enforce these on live payloads, so malformed shapes become live
    request bugs:
    - `unsupported_parameters` must be a list of non-empty strings (a comma-joined
      STRING would turn membership checks into substring matching).
    - `token_param_name` must be one of the two known API spellings.
    - A model declaring `max_tokens` in unsupported_parameters MUST declare a
      different token_param_name — the payload filter renames the cap (never
      drops it), so without the rename the declared-rejected key would still be
      sent and 400 on every call.
    """
    data = _load_model_capabilities()
    models = data.get("models", {})

    for model_key, entry in models.items():
        if not isinstance(entry, dict):
            continue
        blocked = entry.get("unsupported_parameters")
        if blocked is not None:
            assert isinstance(blocked, list), (
                f"{model_key}: unsupported_parameters must be a LIST, got {type(blocked).__name__}"
            )
            for item in blocked:
                assert isinstance(item, str) and item.strip(), (
                    f"{model_key}: unsupported_parameters entries must be non-empty strings: {item!r}"
                )

        token_param = entry.get("token_param_name")
        if token_param is not None:
            assert token_param in {"max_tokens", "max_completion_tokens"}, (
                f"{model_key}: token_param_name must be max_tokens or max_completion_tokens, got {token_param!r}"
            )

        if isinstance(blocked, list) and "max_tokens" in blocked:
            assert token_param == "max_completion_tokens", (
                f"{model_key}: declares max_tokens unsupported but no alternate token_param_name — "
                f"the payload filter renames the cap (never drops it), so this shape would still "
                f"send the rejected key"
            )
