import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import pytest


def _load_architecture_formats() -> Dict[str, Any]:
    assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
    path = assets_dir / "architecture_formats.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "architecture_formats.json must parse to an object"
    return data


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == value and bool(value.strip())


def _iter_patterns(architectures: Mapping[str, Any]) -> Iterable[Tuple[str, str]]:
    for arch_name, cfg in architectures.items():
        if not isinstance(cfg, dict):
            continue
        patterns = cfg.get("patterns", [])
        if not isinstance(patterns, list):
            continue
        for p in patterns:
            if isinstance(p, str):
                yield arch_name, p


@pytest.mark.basic
def test_architecture_formats_json_has_required_top_level_sections():
    data = _load_architecture_formats()

    assert "architectures" in data, "top-level key 'architectures' is required"
    assert isinstance(data["architectures"], dict), "'architectures' must be an object"

    assert "message_formats" in data, "top-level key 'message_formats' is required"
    assert isinstance(data["message_formats"], dict), "'message_formats' must be an object"
    assert data["message_formats"], "'message_formats' must not be empty"

    assert "tool_formats" in data, "top-level key 'tool_formats' is required"
    assert isinstance(data["tool_formats"], dict), "'tool_formats' must be an object"
    assert data["tool_formats"], "'tool_formats' must not be empty"


@pytest.mark.basic
def test_architecture_formats_json_registries_are_well_formed():
    data = _load_architecture_formats()
    message_formats = data["message_formats"]
    tool_formats = data["tool_formats"]

    for registry_name, registry in (("message_formats", message_formats), ("tool_formats", tool_formats)):
        for key, desc in registry.items():
            assert _non_empty_str(key), f"{registry_name} keys must be non-empty strings: {key!r}"
            assert key.lower() == key, f"{registry_name} key must be lowercase: {key!r}"
            assert " " not in key, f"{registry_name} key must not contain spaces: {key!r}"
            assert _non_empty_str(desc), f"{registry_name}[{key!r}] must be a non-empty description string"


@pytest.mark.basic
def test_architecture_entries_conform_to_v0_template():
    data = _load_architecture_formats()
    architectures = data["architectures"]
    message_formats = data["message_formats"]
    tool_formats = data["tool_formats"]

    required_keys = {"description", "message_format", "tool_format", "patterns"}
    prefix_suffix_keys = {
        "system_prefix",
        "system_suffix",
        "user_prefix",
        "user_suffix",
        "assistant_prefix",
        "assistant_suffix",
    }
    optional_keys = {
        "tool_prefix",
        "output_wrappers",
        "thinking_tags",
        "thinking_output_field",
        "thinking_control",
        "thinking_format",
        "tool_calling_format",
        "default_tool_support",
        "reasoning_support",
        "reasoning_levels",
    }
    allowed_keys = required_keys | prefix_suffix_keys | optional_keys

    for arch_name, cfg in architectures.items():
        assert _non_empty_str(arch_name), f"architecture key must be a non-empty string: {arch_name!r}"
        assert isinstance(cfg, dict), f"architectures[{arch_name!r}] must be an object"

        extra = set(cfg) - allowed_keys
        assert not extra, f"architectures[{arch_name}] contains unknown keys: {sorted(extra)}"

        missing = required_keys - set(cfg)
        assert not missing, f"architectures[{arch_name}] is missing required keys: {sorted(missing)}"

        assert _non_empty_str(cfg.get("description")), f"architectures[{arch_name}].description must be non-empty"

        msg_fmt = cfg.get("message_format")
        assert _non_empty_str(msg_fmt), f"architectures[{arch_name}].message_format must be non-empty"
        assert msg_fmt in message_formats, (
            f"architectures[{arch_name}].message_format={msg_fmt!r} must be declared in top-level message_formats"
        )

        tool_fmt = cfg.get("tool_format")
        assert _non_empty_str(tool_fmt), f"architectures[{arch_name}].tool_format must be non-empty"
        assert tool_fmt in tool_formats, (
            f"architectures[{arch_name}].tool_format={tool_fmt!r} must be declared in top-level tool_formats"
        )

        patterns = cfg.get("patterns")
        assert isinstance(patterns, list), f"architectures[{arch_name}].patterns must be a list"
        assert patterns, f"architectures[{arch_name}].patterns must be non-empty"
        for p in patterns:
            assert _non_empty_str(p), f"architectures[{arch_name}].patterns contains invalid pattern: {p!r}"
            assert p.lower() == p, f"architectures[{arch_name}].patterns must be lowercase: {p!r}"
            assert not any(ch.isspace() for ch in p), (
                f"architectures[{arch_name}].patterns must not contain whitespace: {p!r}"
            )
            assert len(p) >= 3, (
                f"architectures[{arch_name}].patterns contains overly-generic pattern (len<3): {p!r}"
            )

        # Conditional prefix/suffix requirements:
        # - OpenAI-style chat formats are structured and do not require tokenized prompt formatting.
        # - All other formats should provide explicit role wrappers (even if empty strings).
        if str(msg_fmt).strip().lower() != "openai_chat":
            for k in prefix_suffix_keys:
                assert k in cfg, f"architectures[{arch_name}] missing {k!r} (required for message_format={msg_fmt!r})"
                assert isinstance(cfg.get(k), str), f"architectures[{arch_name}].{k} must be a string"
        else:
            for k in prefix_suffix_keys:
                if k in cfg:
                    assert isinstance(cfg.get(k), str), f"architectures[{arch_name}].{k} must be a string"

        tool_prefix = cfg.get("tool_prefix")
        if tool_prefix is not None:
            # Tool wrapper tokens frequently include trailing newlines; we only require non-empty content.
            assert isinstance(tool_prefix, str) and tool_prefix.strip(), (
                f"architectures[{arch_name}].tool_prefix must be a non-empty string when set"
            )

        default_tool_support = cfg.get("default_tool_support")
        if default_tool_support is not None:
            assert _non_empty_str(default_tool_support), (
                f"architectures[{arch_name}].default_tool_support must be a non-empty string"
            )
            assert default_tool_support in {"native", "prompted", "none"}, (
                f"architectures[{arch_name}].default_tool_support must be one of: native, prompted, none"
            )

        output_wrappers = cfg.get("output_wrappers")
        if output_wrappers is not None:
            assert isinstance(output_wrappers, dict), f"architectures[{arch_name}].output_wrappers must be an object"
            extra_keys = set(output_wrappers) - {"start", "end"}
            assert not extra_keys, (
                f"architectures[{arch_name}].output_wrappers has unknown keys: {sorted(extra_keys)}"
            )
            assert any(k in output_wrappers for k in ("start", "end")), (
                f"architectures[{arch_name}].output_wrappers must include 'start' and/or 'end'"
            )
            for k in ("start", "end"):
                if k in output_wrappers:
                    assert _non_empty_str(output_wrappers.get(k)), (
                        f"architectures[{arch_name}].output_wrappers[{k!r}] must be a non-empty string"
                    )

        thinking_tags = cfg.get("thinking_tags")
        if thinking_tags is not None:
            assert isinstance(thinking_tags, (list, tuple)), (
                f"architectures[{arch_name}].thinking_tags must be a 2-item list/tuple"
            )
            assert len(thinking_tags) == 2, f"architectures[{arch_name}].thinking_tags must have length 2"
            assert _non_empty_str(thinking_tags[0]), f"architectures[{arch_name}].thinking_tags[0] must be non-empty"
            assert _non_empty_str(thinking_tags[1]), f"architectures[{arch_name}].thinking_tags[1] must be non-empty"

        for key in ("thinking_output_field", "thinking_control", "thinking_format", "tool_calling_format"):
            value = cfg.get(key)
            if value is not None:
                assert _non_empty_str(value), f"architectures[{arch_name}].{key} must be a non-empty string when set"

        reasoning_support = cfg.get("reasoning_support")
        if reasoning_support is not None:
            assert isinstance(reasoning_support, bool), f"architectures[{arch_name}].reasoning_support must be boolean"

        reasoning_levels = cfg.get("reasoning_levels")
        if reasoning_levels is not None:
            assert isinstance(reasoning_levels, list) and reasoning_levels, (
                f"architectures[{arch_name}].reasoning_levels must be a non-empty list when set"
            )
            allowed = {"low", "medium", "high"}
            for level in reasoning_levels:
                assert _non_empty_str(level), f"architectures[{arch_name}].reasoning_levels contains invalid: {level!r}"
                assert level in allowed, (
                    f"architectures[{arch_name}].reasoning_levels must be subset of {sorted(allowed)}"
                )


@pytest.mark.basic
def test_patterns_are_unique_across_architectures():
    data = _load_architecture_formats()
    architectures = data["architectures"]

    pattern_to_arches: Dict[str, list[str]] = defaultdict(list)
    for arch_name, p in _iter_patterns(architectures):
        if not isinstance(p, str) or not p.strip():
            continue
        pattern_to_arches[p.lower()].append(arch_name)

    duplicates = {p: sorted(set(v)) for p, v in pattern_to_arches.items() if len(set(v)) > 1}
    assert not duplicates, (
        "architecture detection relies on 'longest match wins' and does not resolve ties.\n"
        "Duplicate patterns across architectures are ambiguous and order-dependent.\n"
        f"Duplicates: {duplicates}"
    )


@pytest.mark.basic
def test_pattern_specificity_sanity_warnings():
    data = _load_architecture_formats()
    architectures = data["architectures"]

    # Warn (do not fail) for very short alphabetic patterns; they are often intentional (family fallbacks),
    # but are high-risk because substring matching can cause surprising detection if no specific pattern exists.
    for arch_name, p in _iter_patterns(architectures):
        pat = str(p).strip()
        if len(pat) <= 3 and pat.isalpha():
            warnings.warn(
                f"architecture_formats.json: pattern {pat!r} for architecture {arch_name!r} is very generic; "
                "prefer more specific substrings (e.g., 'family-3.1', 'family3:8b') when possible.",
                UserWarning,
            )
