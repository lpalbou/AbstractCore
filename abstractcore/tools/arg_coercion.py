"""Schema-aware tool-argument type coercion (backlog 039).

Tool calls parsed from prompted / XML-ish / code-block formats preserve raw *string*
values (see `parser.py`). Without a central coercion step, a string like ``"false"`` is
truthy in Python, so security-sensitive flags silently invert:

- ``allow_dangerous="false"`` would *enable* the dangerous path,
- ``preview_only="false"`` would *skip* writing an edit,
- ``use_regex="false"`` would turn a literal replace into a regex compile.

This module coerces each argument to its **declared schema type** at dispatch, returning
``(coerced_arguments, warnings)``. It follows two hard rules from the framework's policy:

1. **Never silently default** (ADR-0026 posture): an unrecognized value for a typed field
   raises :class:`ArgumentCoercionError` so the caller fails loudly instead of guessing.
2. **Never mutate in place**: a new dict is returned; the input is untouched.

It is intentionally conservative for container types (``array``/``object``): it upgrades a
JSON-encoded string into the structured value when unambiguous, but otherwise leaves the
value untouched so tools that accept string-or-structured inputs keep working.

Design is general-purpose: it reads the JSON-Schema-style ``{"type": ...}`` that
``ToolDefinition.parameters`` already carries, so any tool benefits without per-tool code.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

# Explicit, closed token sets. Anything outside these for a boolean field is an error,
# not a silent default — a weak model that emits "maybe" must be told, not guessed for.
_TRUE_TOKENS = {"true", "1", "yes", "y", "on", "enabled"}
_FALSE_TOKENS = {"false", "0", "no", "n", "off", "disabled"}


class ArgumentCoercionError(ValueError):
    """Raised when an argument cannot be coerced to its declared schema type."""


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        raise ArgumentCoercionError(
            f"expected a boolean but got {value!r}; "
            f"use one of true/false/1/0/yes/no/on/off"
        )
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ArgumentCoercionError(f"expected a boolean but got integer {value!r}")
    raise ArgumentCoercionError(f"expected a boolean but got {type(value).__name__}")


def _coerce_int(value: Any) -> int:
    # A real bool is not a valid integer argument here (avoids masking a wrong-type call).
    if isinstance(value, bool):
        raise ArgumentCoercionError(f"expected an integer but got boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ArgumentCoercionError(f"expected an integer but got non-integral float {value!r}")
    if isinstance(value, str):
        token = value.strip()
        try:
            return int(token)
        except ValueError:
            try:
                f = float(token)
            except ValueError:
                raise ArgumentCoercionError(f"expected an integer but got {value!r}")
            if f.is_integer():
                return int(f)
            raise ArgumentCoercionError(f"expected an integer but got non-integral {value!r}")
    raise ArgumentCoercionError(f"expected an integer but got {type(value).__name__}")


def _coerce_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ArgumentCoercionError(f"expected a number but got boolean {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        try:
            return float(token)
        except ValueError:
            raise ArgumentCoercionError(f"expected a number but got {value!r}")
    raise ArgumentCoercionError(f"expected a number but got {type(value).__name__}")


def _maybe_json(value: str) -> Any:
    token = value.strip()
    if not token:
        return None
    try:
        return json.loads(token)
    except (ValueError, TypeError):
        return None


def coerce_arguments(
    parameters: Dict[str, Any],
    arguments: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce ``arguments`` to the types declared in ``parameters``.

    Returns a ``(coerced, warnings)`` tuple. ``warnings`` entries are ``#FALLBACK``-tagged
    strings naming the argument and the coercion applied. Raises
    :class:`ArgumentCoercionError` for a typed field whose value cannot be coerced.

    Only keys present in both ``arguments`` and ``parameters`` are considered; unknown keys
    are passed through unchanged (name canonicalization / kwarg filtering owns those).
    """
    if not isinstance(arguments, dict) or not arguments:
        return (dict(arguments) if isinstance(arguments, dict) else {}, [])
    if not isinstance(parameters, dict) or not parameters:
        return (dict(arguments), [])

    out: Dict[str, Any] = dict(arguments)
    warnings: List[str] = []

    for key, value in list(out.items()):
        schema = parameters.get(key)
        if not isinstance(schema, dict):
            continue
        schema_type = str(schema.get("type") or "").strip().lower()
        if not schema_type:
            continue

        # An explicit null for a typed field means "not provided": callers
        # (models, flow-composed tool calls) routinely emit e.g.
        # {"head_limit": null} for optional parameters. JSON Schema treats
        # null as a distinct type, so coercing it to int/bool would be
        # invention and raising turns a well-formed optional into a hard
        # failure (live case: coding-agent's gate listing sent
        # head_limit=None and every run died at dispatch). Drop the key so
        # the tool's own default applies — same outcome as omission.
        if value is None and schema_type != "string":
            out.pop(key)
            warnings.append(
                f"#FALLBACK: dropped argument '{key}' (explicit null for {schema_type}; tool default applies)"
            )
            continue

        try:
            if schema_type == "boolean":
                if isinstance(value, bool):
                    continue
                new_value: Any = _coerce_bool(value)
            elif schema_type == "integer":
                if isinstance(value, int) and not isinstance(value, bool):
                    continue
                new_value = _coerce_int(value)
            elif schema_type == "number":
                if isinstance(value, float):
                    continue
                new_value = _coerce_number(value)
            elif schema_type == "string":
                if isinstance(value, str) or value is None:
                    continue
                # Only stringify SCALARS (a weak model may emit 3 where "3" is expected). Never
                # stringify a list/dict — turning a container into its Python repr
                # (e.g. "['a@b']") silently corrupts the argument. Leave containers untouched:
                # the tool either accepts the structured value or the mismatch surfaces honestly.
                if isinstance(value, (bool, int, float)):
                    new_value = str(value)
                else:
                    continue
            elif schema_type == "array":
                if isinstance(value, list) or not isinstance(value, str):
                    continue
                parsed = _maybe_json(value)
                if not isinstance(parsed, list):
                    # Conservative: leave string-or-list tools to their own parsing.
                    continue
                new_value = parsed
            elif schema_type == "object":
                if isinstance(value, dict) or not isinstance(value, str):
                    continue
                parsed = _maybe_json(value)
                if not isinstance(parsed, dict):
                    continue
                new_value = parsed
            else:
                continue
        except ArgumentCoercionError as exc:
            raise ArgumentCoercionError(f"argument '{key}': {exc}") from exc

        if new_value != value or type(new_value) is not type(value):
            out[key] = new_value
            warnings.append(
                f"#FALLBACK: coerced argument '{key}' to {schema_type} ({value!r} -> {new_value!r})"
            )

    return out, warnings


def coerce_arguments_for_callable(
    func: Any,
    arguments: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce arguments using the schema attached to a decorated tool callable.

    Prefers the callable's ``_tool_definition`` (attached by the ``@tool`` decorator); falls
    back to deriving a definition from the signature. Returns ``(arguments, [])`` unchanged
    when no schema can be resolved.
    """
    parameters: Dict[str, Any] = {}
    tool_def = getattr(func, "_tool_definition", None)
    if tool_def is not None and isinstance(getattr(tool_def, "parameters", None), dict):
        parameters = tool_def.parameters
    else:
        try:
            from .core import ToolDefinition

            parameters = ToolDefinition.from_function(func).parameters
        except Exception:
            parameters = {}
    if not parameters:
        return (dict(arguments) if isinstance(arguments, dict) else {}, [])
    return coerce_arguments(parameters, arguments)


__all__ = ["ArgumentCoercionError", "coerce_arguments", "coerce_arguments_for_callable"]
