"""
Typed thinking-control surfaces resolved from asset registries.

`thinking_control` in `assets/architecture_formats.json` / `assets/model_capabilities.json`
declares *which control surfaces a model's template or API exposes*. Historically this was a
single untyped string, which conflated four different kinds of controls (prompt tokens,
chat-template variables, provider API params, budgets) and caused the generic disable
fallback to append template-variable names (e.g. ``enable_thinking``) as literal prompt
text. The typed object form makes the surface kind explicit:

```json
"thinking_control": {
  "prompt_disable_token": "/nothink",
  "template_kwarg": "enable_thinking",
  "assistant_prefill_disable": "<think>\n\n</think>\n\n",
  "budget_template_kwarg": "thinking_budget",
  "low_effort_template_kwarg": "low_effort",
  "request_param": "reasoning_effort",
  "effort_template_kwarg": "reasoning_effort",
  "effort_system_lines": {"xhigh": "…", "medium": "", "low": "…"}
}
```

All keys are optional; a model may declare several surfaces (Qwen3 legitimately has three).
Providers own the *transport* (how a surface is sent on a given stack); assets own the
*model knowledge* (which surfaces exist).

Semantics:
- ``prompt_disable_token``: literal token appended to the user prompt to disable thinking
  (GLM ``/nothink``, Qwen3 ``/no_think``). The ONLY kind the generic prompt fallback may append.
- ``template_kwarg``: boolean chat-template variable (e.g. ``enable_thinking``) sent via
  ``chat_template_kwargs`` / tokenizer render kwargs by providers that support it.
- ``assistant_prefill_disable``: assistant-generation-prompt prefill that disables thinking
  (Qwen ``<think>\\n\\n</think>\\n\\n`` empty think block).
- ``budget_template_kwarg``: integer chat-template variable controlling a thinking budget
  (Seed-OSS ``thinking_budget``).
- ``low_effort_template_kwarg``: boolean chat-template variable reducing reasoning effort
  while keeping thinking enabled (Nemotron ``low_effort``).
- ``request_param``: provider-native request parameter name (informational; recorded so the
  control surface is declared even before a provider consumes it).
- ``effort_template_kwarg``: string chat-template variable selecting a reasoning-effort
  level (Qwen3.8 ``reasoning_effort``), sent via ``chat_template_kwargs`` by providers whose
  backend renders the template.
- ``effort_system_lines``: level -> exact instruction sentence the model's own chat template
  prepends to the system block for that effort level. Lets locally-serializing providers
  (MLX) reproduce the template's effort control byte-for-byte. An EMPTY string value means
  "level supported, template renders no text" (Qwen3.8 ``medium``). Merged as a whole map
  (model capabilities replace the architecture's map, not per-level).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

_SURFACE_KEYS = (
    "prompt_disable_token",
    "template_kwarg",
    "assistant_prefill_disable",
    "budget_template_kwarg",
    "low_effort_template_kwarg",
    "request_param",
    "effort_template_kwarg",
)

_MAPPING_SURFACE_KEYS = ("effort_system_lines",)


@dataclass(frozen=True)
class ThinkingControlSurfaces:
    """Resolved thinking-control surfaces for one model (model caps override architecture)."""

    prompt_disable_token: Optional[str] = None
    template_kwarg: Optional[str] = None
    assistant_prefill_disable: Optional[str] = None
    budget_template_kwarg: Optional[str] = None
    low_effort_template_kwarg: Optional[str] = None
    request_param: Optional[str] = None
    effort_template_kwarg: Optional[str] = None
    effort_system_lines: Optional[Mapping[str, str]] = None

    def any_declared(self) -> bool:
        return any(getattr(self, f.name) is not None for f in fields(self))


def _coerce_surface_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    # Prefill markers are whitespace-significant; only reject empty/whitespace-only values.
    return value if value.strip() else None


def _coerce_surface_mapping(value: Any) -> Optional[Mapping[str, str]]:
    """Coerce a level->text surface map. Values may be EMPTY strings ("supported, renders
    no text"); non-string keys/values and empty keys are dropped."""
    if not isinstance(value, Mapping):
        return None
    coerced: dict = {}
    for key, text in value.items():
        if not (isinstance(key, str) and key.strip() and isinstance(text, str)):
            continue
        coerced[key.strip().lower()] = text
    return coerced or None


def _merge_source(merged: dict, raw: Any, *, source_name: str) -> None:
    if raw is None:
        return

    if isinstance(raw, str):
        # Legacy untyped string. Tolerated for user-supplied assets only:
        # a leading "/" is unambiguous (prompt token); anything else cannot be
        # disambiguated (template kwarg vs API param) and is ignored loudly.
        token = raw.strip()
        if token.startswith("/"):
            warnings.warn(
                f"#FALLBACK: legacy string thinking_control {token!r} in {source_name}; "
                "interpreting as prompt_disable_token. Migrate to the typed object form "
                '({"prompt_disable_token": "..."}).',
                RuntimeWarning,
                stacklevel=4,
            )
            merged["prompt_disable_token"] = token
        elif token:
            warnings.warn(
                f"#FALLBACK: legacy string thinking_control {token!r} in {source_name} is not a "
                "prompt token and cannot be applied safely; ignoring it. Declare the surface kind "
                'explicitly (e.g. {"template_kwarg": "enable_thinking"}).',
                RuntimeWarning,
                stacklevel=4,
            )
        return

    if isinstance(raw, Mapping):
        for key in _SURFACE_KEYS:
            value = _coerce_surface_value(raw.get(key))
            if value is not None:
                merged[key] = value
        for key in _MAPPING_SURFACE_KEYS:
            value = _coerce_surface_mapping(raw.get(key))
            if value is not None:
                merged[key] = value


def resolve_thinking_control_surfaces(
    *,
    model_capabilities: Optional[Mapping[str, Any]] = None,
    architecture_format: Any = None,
) -> ThinkingControlSurfaces:
    """Resolve typed thinking-control surfaces.

    Precedence: per-key merge with model capabilities overriding the architecture entry
    (a model may add or override individual surfaces declared by its family).
    ``architecture_format`` may be a single mapping or a sequence of mappings applied
    in order (later entries override earlier ones per key).
    """
    merged: dict = {}
    arch_sources: list = []
    if isinstance(architecture_format, Mapping):
        arch_sources = [architecture_format]
    elif isinstance(architecture_format, (list, tuple)):
        arch_sources = [src for src in architecture_format if isinstance(src, Mapping)]
    for src in arch_sources:
        _merge_source(merged, src.get("thinking_control"), source_name="architecture_formats.json")
    if isinstance(model_capabilities, Mapping):
        _merge_source(merged, model_capabilities.get("thinking_control"), source_name="model_capabilities.json")
    return ThinkingControlSurfaces(**merged)


__all__ = [
    "ThinkingControlSurfaces",
    "resolve_thinking_control_surfaces",
]
