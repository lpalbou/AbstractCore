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
    allowed = {"low", "medium", "high"}
    for level in levels:
        assert _non_empty_str(level), f"{label}.reasoning_levels contains invalid: {level!r}"
        assert level in allowed, f"{label}.reasoning_levels must be subset of {sorted(allowed)}"


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
        "base_image_tokens",
        "base_model",
        "base_tokens_per_resolution",
        "benchmarks",
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
        "thinking_format",
        "thinking_modes",
        "thinking_output_field",
        "thinking_paradigm",
        "thinking_support",
        "thinking_tags",
        "tile_size",
        "token_cap",
        "token_formula",
        "tokens_per_tile",
        "tool_calling_format",
        "tool_calling_parser",
        "total_parameters",
        "transformer_layers",
        "transformers_version_min",
        "trust_remote_code",
        "ui_generation",
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

    max_output_tokens = _require_int(cfg.get("max_output_tokens"), name=f"{label}.max_output_tokens")
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

    for key in ("thinking_output_field", "thinking_control", "thinking_format", "tool_calling_format"):
        value = cfg.get(key)
        if value is not None:
            assert _non_empty_str(value), f"{label}.{key} must be a non-empty string when set"

    thinking_support = cfg.get("thinking_support")
    if thinking_support is not None:
        assert isinstance(thinking_support, bool), f"{label}.thinking_support must be boolean"

    thinking_budget = cfg.get("thinking_budget")
    if thinking_budget is not None:
        assert isinstance(thinking_budget, bool), f"{label}.thinking_budget must be boolean"

    fim_support = cfg.get("fim_support")
    if fim_support is not None:
        assert isinstance(fim_support, bool), f"{label}.fim_support must be boolean"

    reasoning_levels = cfg.get("reasoning_levels")
    if reasoning_levels is not None:
        _validate_reasoning_levels(label, reasoning_levels)

    response_format = cfg.get("response_format")
    if response_format is not None:
        assert response_format in {"harmony"}, f"{label}.response_format must be one of: harmony"


@pytest.mark.basic
def test_model_capabilities_json_has_required_top_level_sections():
    data = _load_model_capabilities()

    assert "models" in data, "top-level key 'models' is required"
    assert isinstance(data["models"], dict), "'models' must be an object"
    assert data["models"], "'models' must not be empty"

    assert "default_capabilities" in data, "top-level key 'default_capabilities' is required"
    assert isinstance(data["default_capabilities"], dict), "'default_capabilities' must be an object"


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
