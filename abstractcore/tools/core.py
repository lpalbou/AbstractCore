"""
Core tool definitions and abstractions.
"""

import re
import warnings
from typing import Dict, Any, List, Optional, Callable, Union, get_args, get_origin
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


_MAX_TOOL_DESCRIPTION_CHARS = 200
_MAX_TOOL_WHEN_TO_USE_CHARS = 240
_MAX_TOOL_EXAMPLES = 3

# What a good description looks like — quoted verbatim into every over-length
# report so the person who has to fix it does not have to go find the rule.
_DESCRIPTION_RULE = (
    "A tool description is ONE short sentence saying what the tool does and when to reach for it "
    f"(max {_MAX_TOOL_DESCRIPTION_CHARS} chars). Put detail in `when_to_use` "
    f"(max {_MAX_TOOL_WHEN_TO_USE_CHARS}) or the tool's own docs."
)


def _first_non_empty_line(text: Optional[str]) -> str:
    if not text:
        return ""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _normalize_one_line(text: Optional[str]) -> str:
    """Collapse whitespace (including newlines) into a single, prompt-friendly line."""
    return " ".join(str(text or "").split()).strip()


def _validate_tool_metadata(*, name: str, description: str, when_to_use: Optional[str], examples: List[Dict[str, Any]]) -> None:
    if not description:
        raise ValueError(f"Tool '{name}': description must be a non-empty string")
    if len(description) > _MAX_TOOL_DESCRIPTION_CHARS:
        raise ValueError(
            f"Tool '{name}': description is too long ({len(description)} chars; max {_MAX_TOOL_DESCRIPTION_CHARS}). "
            "Keep it to a single short sentence; put detailed guidance in `when_to_use` or docs."
        )
    if when_to_use is not None and len(when_to_use) > _MAX_TOOL_WHEN_TO_USE_CHARS:
        raise ValueError(
            f"Tool '{name}': when_to_use is too long ({len(when_to_use)} chars; max {_MAX_TOOL_WHEN_TO_USE_CHARS}). "
            "Keep it to a single short sentence."
        )
    if len(examples) > _MAX_TOOL_EXAMPLES:
        raise ValueError(f"Tool '{name}': too many examples ({len(examples)}; max {_MAX_TOOL_EXAMPLES}).")


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Tokens whose trailing period is NOT a sentence end. Without this guard a
# description opening "Reads e.g. JSON files…" packs to the stub "Reads e.g."
_ABBREVIATIONS = frozenset({
    "e.g", "eg", "i.e", "ie", "etc", "incl", "excl", "vs", "approx", "ca", "cf", "al",
    "resp", "min", "max", "sec", "msec", "no", "fig", "st", "mr", "mrs", "ms", "dr", "prof",
})


def _sentence_end_offsets(text: str):
    """Offsets just past each PLAUSIBLE sentence terminator in `text`."""
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        if text[end - 1] == ".":
            token = re.split(r"[\s(\[\"']", text[: end - 1])[-1].lower().rstrip(".")
            # An abbreviation or a lone initial ("J. Smith") is not a sentence end.
            if token in _ABBREVIATIONS or len(token) <= 1:
                continue
        yield end


def _pack_leading_sentences(text: str, limit: int) -> str:
    """Longest run of WHOLE leading sentences that fits in `limit`.

    Whole sentences, never a fragment: a description cut mid-sentence can INVERT
    its own meaning — "Deletes the record permanently. Do not use this unless the
    user has explicitly confirmed." cut to 60 chars reads "Deletes the record
    permanently. Do not use this unless", which tells the model the opposite of
    what the author meant. A tool description is the only thing the model has
    when it decides what to call, so a half-sentence is not an acceptable fit.

    Returns "" when even the first sentence exceeds `limit`; the caller then
    falls back to a word-boundary cut, which at least never splits a word.
    """
    packed = ""
    for end in _sentence_end_offsets(text):
        candidate = text[:end].strip()
        if len(candidate) > limit:
            break
        packed = candidate
    return packed


def _truncate_on_word_boundary(text: str, limit: int) -> str:
    """Last resort: cut at a word boundary and mark the cut with an ellipsis."""
    if limit <= 1:
        return text[:limit]
    room = limit - 1  # the ellipsis character costs one
    head = text[:room]
    spaced = head.rsplit(" ", 1)[0] if " " in head else head
    return f"{spaced.rstrip().rstrip(',;:')}…"



# Words that mark a sentence as a guardrail rather than a description. Kept
# deliberately broad and lowercase-matched: a false positive costs a slightly
# awkward description, a false negative costs a destructive tool shown to a
# model with its warning deleted.
_GUARDRAIL_MARKERS = (
    "do not", "don't", "never", "only use", "only call", "unless",
    "cannot be undone", "can't be undone", "irreversible", "permanent",
    "destructive", "danger", "caution", "warning", "careful", "confirm",
    "requires approval", "requires permission", "at your own risk",
    "will delete", "will remove", "will overwrite", "data loss",
)


def _contains_guardrail_language(text: str) -> bool:
    """True when `text` reads like a safety caveat rather than a description."""
    low = (text or "").lower()
    return any(marker in low for marker in _GUARDRAIL_MARKERS)


def _pack_trailing_guardrail_sentences(one_line: str, budget: int) -> str:
    """The guardrail sentences from the tail, as many as fit in `budget`.

    Walks sentences from the END so the caveat closest to the author's final
    word — usually the strongest one — is the one that survives.
    """
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", one_line or "") if s.strip()]
    picked: list[str] = []
    total = 0
    for sentence in reversed(parts):
        if not _contains_guardrail_language(sentence):
            continue
        cost = len(sentence) + (1 if picked else 0)
        if total + cost > budget:
            break
        picked.insert(0, sentence)
        total += cost
    return " ".join(picked)


def adapt_external_description(
    description: Optional[str],
    *,
    tool_name: str,
    source: str,
) -> str:
    """Fit a description WE DO NOT OWN to the cap, and report it for correction.

    The policy, decided 2026-08-07. Three things are true at once and the design
    has to honour all three:

      1. The cap is real. It is a forcing function that keeps the tool catalog
         readable by the model, and every one of core's own tools already meets
         it. Raising it to accommodate the worst MCP server on the internet
         would degrade every prompt we render.
      2. We cannot edit an MCP server's source. A hard failure at registration
         would cost the caller the whole tool — and, for a server that ships one
         wordy tool among twenty, potentially the whole server.
      3. A dropped tool is a silent capability loss (ADR 0001). So is a silently
         mangled description: the model then chooses tools on text nobody wrote.

    So: the tool KEEPS WORKING on the best-fitting whole-sentence prefix, and the
    overflow is reported through `warnings.warn` with everything needed to write
    a better one-liner — the tool, its source, the length, what was kept, what
    was dropped, and the rule.

    This is deliberately NOT wired into `ToolDefinition.__post_init__`. Our own
    `@tool` definitions must keep failing hard: an author who can edit the
    source should fix the description, not have it quietly rewritten.
    """
    one_line = _normalize_one_line(description)

    if not one_line:
        placeholder = f"Tool '{tool_name}' (no description supplied by {source})."
        warnings.warn(
            f"Tool '{tool_name}' from {source} has NO description. It will still be offered to the "
            f"model, but with a placeholder the model cannot choose on. {_DESCRIPTION_RULE}",
            RuntimeWarning,
            stacklevel=3,
        )
        return _normalize_one_line(placeholder)[:_MAX_TOOL_DESCRIPTION_CHARS]

    if len(one_line) <= _MAX_TOOL_DESCRIPTION_CHARS:
        return one_line

    # Whole sentences when we can; a word-boundary cut only when the very first
    # sentence is itself over budget (nothing whole exists to keep).
    kept = _pack_leading_sentences(one_line, _MAX_TOOL_DESCRIPTION_CHARS) or (
        _truncate_on_word_boundary(one_line, _MAX_TOOL_DESCRIPTION_CHARS)
    )

    dropped = one_line[len(kept.rstrip("…").rstrip()):].strip()

    # SAFETY TEXT IS NEVER THE PART WE DROP.
    #
    # Keeping leading sentences is right for meaning and wrong for danger: a tool
    # description says what it does first and when NOT to use it last. Dropping
    # the tail therefore removes exactly the guardrail. Measured on a
    # realistically-shaped MCP description, the model received
    #
    #   "Permanently deletes the customer record ... purges the archived copies."
    #
    # and lost "Do not use this unless the user has explicitly confirmed ...
    # because it cannot be undone." That is worse than the bug this function
    # replaced: a dropped tool cannot be called at all, whereas this one is
    # callable with its brakes removed, and the description is the ONLY thing the
    # model uses to decide whether to fire it.
    #
    # So when the overflow carries guardrail language, rebuild around it: first
    # sentence for what the tool does, plus the caveat, and accept a word-boundary
    # cut on the FIRST part rather than losing the warning.
    if dropped and _contains_guardrail_language(dropped):
        caveat = _pack_trailing_guardrail_sentences(
            one_line, _MAX_TOOL_DESCRIPTION_CHARS)
        if caveat:
            budget = _MAX_TOOL_DESCRIPTION_CHARS - len(caveat) - 1
            lead = _pack_leading_sentences(one_line, budget) or (
                _truncate_on_word_boundary(one_line, budget) if budget > 0 else ""
            )
            rebuilt = f"{lead} {caveat}".strip() if lead else caveat
            if len(rebuilt) <= _MAX_TOOL_DESCRIPTION_CHARS:
                warnings.warn(
                    f"Tool '{tool_name}' from {source}: description is "
                    f"{len(one_line)} chars against a {_MAX_TOOL_DESCRIPTION_CHARS} "
                    f"cap, AND the overflow contained a safety caveat. AbstractCore "
                    f"kept the caveat and shortened the explanation instead — a "
                    f"model must never be shown a destructive tool with its warning "
                    f"removed. The description is still not what its author wrote. "
                    f"Fix it at the source.\n"
                    f"  sent to the model ({len(rebuilt)} chars): {rebuilt!r}\n"
                    f"  {_DESCRIPTION_RULE}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return rebuilt
    warnings.warn(
        f"Tool '{tool_name}' from {source}: description is {len(one_line)} chars; the cap is "
        f"{_MAX_TOOL_DESCRIPTION_CHARS}. The tool STILL WORKS — AbstractCore kept the leading text "
        f"that fits and dropped the rest, but the model now chooses this tool on a shortened "
        f"description. Fix it at the source.\n"
        f"  kept    ({len(kept)} chars): {kept!r}\n"
        f"  dropped ({len(dropped)} chars): {_truncate_on_word_boundary(dropped, 160)!r}\n"
        f"  {_DESCRIPTION_RULE}",
        RuntimeWarning,
        stacklevel=3,
    )
    return kept


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by LLM"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None

    # Enhanced metadata for better LLM guidance
    tags: List[str] = field(default_factory=list)
    when_to_use: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)

    # Host-side POLICY attribute (entity-topology item 7, G1): an act-only tool's
    # durable record carries only the ACT-FRAME (id + reason + gist), never the
    # returned words — hosts key their observe/ledger/result channels on this flag.
    # First-class bool (not a tag) so the dataclass default IS the policy:
    # undeclared = normal durable tool. MAINTAINER-CONFIRMED DEFAULT (2026-07-10):
    # False — by default EVERYTHING is recorded (ledger-transparency principle);
    # act-only is a narrow, explicitly-declared exception for diary-class tools,
    # never a category default. Deliberately NEVER serialized into native provider
    # payloads (strict servers reject unknown fields; enforcement is host-side
    # anyway) — if the model should know, say it in `description`.
    act_only: bool = False

    def __post_init__(self) -> None:
        # Normalize to a single line for prompt-friendly catalogs.
        self.name = str(self.name or "").strip()
        self.description = _normalize_one_line(self.description)
        self.when_to_use = _normalize_one_line(self.when_to_use) if self.when_to_use else None
        self.tags = list(self.tags) if isinstance(self.tags, list) else []
        self.examples = list(self.examples) if isinstance(self.examples, list) else []
        self.act_only = bool(self.act_only)
        _validate_tool_metadata(
            name=self.name,
            description=self.description,
            when_to_use=self.when_to_use,
            examples=self.examples,
        )

    @classmethod
    def from_external(
        cls,
        *,
        name: str,
        description: Optional[str],
        parameters: Dict[str, Any],
        source: str,
        **kwargs: Any,
    ) -> 'ToolDefinition':
        """Build a definition from metadata WE DO NOT OWN (MCP servers, raw dicts).

        The one entry point that says "I am importing a tool, not authoring one".
        Over-long descriptions are adapted and reported (see
        `adapt_external_description`) instead of raising, so a wordy third-party
        tool is never lost. `when_to_use` gets the same treatment against its own
        cap. Everything else still validates normally — a nameless tool is still
        an error, because there is nothing to adapt it to.

        Authoring a tool in this repo? Use `@tool`. It fails hard on purpose.
        """
        adapted = adapt_external_description(description, tool_name=name, source=source)

        when_to_use = kwargs.pop("when_to_use", None)
        normalized_when = _normalize_one_line(when_to_use) if when_to_use else None
        if normalized_when and len(normalized_when) > _MAX_TOOL_WHEN_TO_USE_CHARS:
            warnings.warn(
                f"Tool '{name}' from {source}: when_to_use is {len(normalized_when)} chars; the cap "
                f"is {_MAX_TOOL_WHEN_TO_USE_CHARS}. Kept the leading text that fits. "
                f"Keep it to a single short sentence.",
                RuntimeWarning,
                stacklevel=2,
            )
            packed = _pack_leading_sentences(normalized_when, _MAX_TOOL_WHEN_TO_USE_CHARS)
            normalized_when = packed or _truncate_on_word_boundary(
                normalized_when, _MAX_TOOL_WHEN_TO_USE_CHARS
            )

        examples = kwargs.pop("examples", None) or []
        if len(examples) > _MAX_TOOL_EXAMPLES:
            warnings.warn(
                f"Tool '{name}' from {source}: {len(examples)} examples supplied; the cap is "
                f"{_MAX_TOOL_EXAMPLES}. Kept the first {_MAX_TOOL_EXAMPLES}.",
                RuntimeWarning,
                stacklevel=2,
            )
            examples = list(examples)[:_MAX_TOOL_EXAMPLES]

        return cls(
            name=name,
            description=adapted,
            parameters=parameters,
            when_to_use=normalized_when,
            examples=list(examples),
            **kwargs,
        )

    @classmethod
    def from_function(cls, func: Callable, *, external_source: Optional[str] = None) -> 'ToolDefinition':
        """Create tool definition from a function.

        `external_source`: set it when the function is IMPORTED rather than
        authored here (a third-party callable handed straight to `generate()`).
        An over-long docstring first line is then adapted and reported rather
        than raised — a raise here used to travel up into the handler's `except`
        and cost the caller the tool. Left unset, the cap still fails hard.
        """
        import inspect
        import types

        # Extract function name and docstring
        name = func.__name__
        # Tool `description` must be short; use the first docstring line (not the whole docstring).
        description = _first_non_empty_line(func.__doc__) or "No description provided"
        description = _normalize_one_line(description)
        if description != "No description provided":
            if external_source:
                description = adapt_external_description(
                    description, tool_name=name, source=external_source
                )
            else:
                _validate_tool_metadata(name=name, description=description, when_to_use=None, examples=[])

        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = {}

        # PEP 563 / `from __future__ import annotations`: annotations may be STRINGS
        # (e.g. "bool", "Optional[int]") rather than real types. Resolve them once via
        # get_type_hints so type inference is truthful; without this, every tool defined in a
        # module that uses `from __future__ import annotations` (e.g. common_tools.py) silently
        # falls through to {"type": "string"}, which breaks schema-aware argument coercion for
        # boolean/int flags (backlog 039).
        try:
            import typing

            resolved_hints = typing.get_type_hints(func)
        except Exception:
            resolved_hints = {}

        # Fallback map for bare string annotations when get_type_hints cannot resolve them.
        _STRING_ANNOTATION_SCHEMAS = {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
        }

        def _schema_for_annotation(annotation: Any) -> Dict[str, Any]:
            if annotation in {inspect._empty, None}:
                return {"type": "string"}

            # Bare string annotation (PEP 563) that get_type_hints did not resolve.
            if isinstance(annotation, str):
                key = annotation.strip()
                if key in _STRING_ANNOTATION_SCHEMAS:
                    return dict(_STRING_ANNOTATION_SCHEMAS[key])
                lowered = key.lower()
                # Optional[T]/Union[..., None] and container spellings as text.
                if lowered.startswith(("optional[", "list[", "dict[", "tuple[", "set[", "sequence[")):
                    inner = key[key.find("[") + 1 : key.rfind("]")].strip()
                    if lowered.startswith("optional["):
                        return _schema_for_annotation(inner.split(",")[0].strip())
                    if lowered.startswith(("dict[",)):
                        return {"type": "object"}
                    item = inner.split(",")[0].strip()
                    return {"type": "array", "items": _schema_for_annotation(item)}
                return {"type": "string"}

            if annotation is str:
                return {"type": "string"}
            if annotation is int:
                return {"type": "integer"}
            if annotation is float:
                return {"type": "number"}
            if annotation is bool:
                return {"type": "boolean"}

            origin = get_origin(annotation)
            args = get_args(annotation)

            # Optional[T] (Union[T, None]) => schema(T)
            if origin in {Union, getattr(types, "UnionType", object())}:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    return _schema_for_annotation(non_none[0])

            if origin in {list, List, tuple, set}:
                item_schema = _schema_for_annotation(args[0]) if args else {"type": "string"}
                return {"type": "array", "items": item_schema}

            if origin in {dict, Dict}:
                return {"type": "object"}

            # Fall back to string for unknown / complex annotations.
            return {"type": "string"}

        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type

            # Try to infer type from annotation. Prefer the resolved type hint (handles PEP 563
            # stringized annotations); fall back to the raw annotation on the parameter.
            annotation = resolved_hints.get(param_name, param.annotation)
            if annotation != param.empty:
                param_info = _schema_for_annotation(annotation)

            if param.default != param.empty:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            function=func
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        # Preserve a human/LLM-friendly field ordering (also improves UX when dumped as JSON).
        result: Dict[str, Any] = {"name": self.name, "description": self.description}

        if self.when_to_use:
            result["when_to_use"] = self.when_to_use

        result["parameters"] = self.parameters

        # Expose required args explicitly for hosts that render ToolDefinitions directly
        # (e.g. AbstractRuntime prompt payloads / debug traces). This is additive metadata:
        # providers still rely on JSON Schema `required` when building native tool payloads.
        try:
            required: List[str] = []
            for param_name, meta in (self.parameters or {}).items():
                if not isinstance(param_name, str) or not param_name.strip():
                    continue
                # Convention in this repo: absence of `default` means "required".
                if not isinstance(meta, dict) or "default" not in meta:
                    required.append(param_name)
            required.sort()
            if required:
                result["required_args"] = required
        except Exception:
            # Best-effort only; never break tool serialization.
            pass

        # Include enhanced metadata if available
        if self.tags:
            result["tags"] = self.tags
        if self.examples:
            result["examples"] = self.examples

        # Host-side policy attribute: additive — emitted only when set, so normal
        # tools serialize exactly as before. Never copied into native provider
        # payloads (see UniversalToolHandler.prepare_tools_for_native).
        if self.act_only:
            result["act_only"] = True

        return result


@dataclass
class ToolCall:
    """Represents a tool call from the LLM"""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result of tool execution"""
    call_id: str
    output: Any
    error: Optional[str] = None
    success: bool = True


@dataclass
class ToolCallResponse:
    """Response containing content and tool calls"""
    content: str
    tool_calls: List[ToolCall]
    raw_response: Any = None

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls"""
        return bool(self.tool_calls)


def tool(
    func=None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    when_to_use: Optional[str] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    hide_args: Optional[List[str]] = None,
    act_only: bool = False,
):
    """
    Enhanced decorator to convert a function into a tool with rich metadata.

    Usage:
        @tool
        def my_function(param: str) -> str:
            "Does something"
            return result

        # Or with enhanced metadata
        @tool(
            name="custom",
            description="Custom tool",
            tags=["utility", "helper"],
            when_to_use="When you need to perform X operation",
            examples=[
                {
                    "description": "Basic usage",
                    "arguments": {"param": "value"}
                }
            ]
        )
        def my_function(param: str) -> str:
            return result

        # Pass to generate like this:
        llm.generate("Do something", tools=[my_function])
    """
    def decorator(f):
        tool_name = name or f.__name__
        tool_description = description or _first_non_empty_line(f.__doc__) or f"Execute {tool_name}"
        tool_description = _normalize_one_line(tool_description)

        # Create tool definition from function and customize
        tool_def = ToolDefinition.from_function(f)
        tool_def.name = tool_name
        tool_def.description = tool_description

        # Add enhanced metadata
        tool_def.tags = tags or []
        tool_def.when_to_use = _normalize_one_line(when_to_use) if when_to_use else None
        tool_def.examples = list(examples) if isinstance(examples, list) else []
        tool_def.act_only = bool(act_only)

        # Optionally hide parameters from the exported schema (LLM-facing), while
        # keeping them accepted by the underlying Python callable for backwards
        # compatibility (e.g. legacy callers still passing deprecated kwargs).
        hidden = [str(a).strip() for a in (hide_args or []) if str(a).strip()]
        if hidden:
            for arg in hidden:
                if arg not in tool_def.parameters:
                    continue
                # Avoid hiding required args (no default), which would make the
                # tool schema incomplete for tool-call generation.
                if "default" not in tool_def.parameters.get(arg, {}):
                    raise ValueError(f"Tool '{tool_def.name}': cannot hide required arg '{arg}'")
                tool_def.parameters.pop(arg, None)

        _validate_tool_metadata(
            name=tool_def.name,
            description=tool_def.description,
            when_to_use=tool_def.when_to_use,
            examples=tool_def.examples,
        )

        # Attach tool definition to function for easy access
        f._tool_definition = tool_def
        # Public alias (docs/examples use this name).
        f.tool_definition = tool_def
        f.tool_name = tool_name

        return f

    if func is None:
        # Called with arguments: @tool(name="custom")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)
