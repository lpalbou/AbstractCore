"""Wire-safe tool names for strict native tool APIs.

Problem (abstractagent find, 2026-07-13): MCP tools are namespaced
``mcp::server::tool`` (see ``abstractcore.mcp.naming``). Native tool
declarations put that name on the wire verbatim, and strict endpoints
(OpenAI: ``^[a-zA-Z0-9_-]{1,64}$``; Anthropic is equally strict) reject the
WHOLE request with a 400 — one MCP tool in the list kills the call.

Contract:
- ``wire_safe_tool_name`` is a pure, deterministic function. Names that are
  already wire-safe pass through byte-identical (zero behavior change for
  every existing builtin tool). Unsafe names are sanitized and carry a short
  hash of the ORIGINAL name so two different originals can never collide on
  the wire (``mcp::a::b`` vs a literal ``mcp__a__b`` tool stay distinct).
- The reverse direction needs no stored state: given the allowed ORIGINAL
  names, recompute each one's wire name and match. ``resolve_wire_tool_name``
  does exactly that at the single response-normalization choke point.
- Only NATIVE declarations are aliased. The prompted lane renders original
  names in prompt text (no schema validates them) and stays untouched.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

# OpenAI function-name contract; Anthropic's is compatible with this subset.
_WIRE_SAFE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_INVALID_RUN_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_MAX_WIRE_LEN = 64
_HASH_LEN = 8


def is_wire_safe_tool_name(name: str) -> bool:
    return bool(_WIRE_SAFE_RE.match(str(name or "")))


def wire_safe_tool_name(name: str) -> str:
    """Deterministic wire-safe alias for a tool name.

    Already-safe names return unchanged. Unsafe names have invalid character
    runs collapsed to ``_`` and an 8-hex sha1 tail of the ORIGINAL name
    appended (collision-proof and reversible by recomputation).
    """
    raw = str(name or "")
    if is_wire_safe_tool_name(raw):
        return raw
    sanitized = _INVALID_RUN_RE.sub("_", raw).strip("_") or "tool"
    tag = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:_HASH_LEN]
    budget = _MAX_WIRE_LEN - _HASH_LEN - 1
    if len(sanitized) > budget:
        sanitized = sanitized[:budget].rstrip("_")
    return f"{sanitized}_{tag}"


def resolve_wire_tool_name(name: str, allowed_original_names: Iterable[str]) -> Optional[str]:
    """Map a wire name the model returned back to its ORIGINAL tool name.

    Returns the original name when ``name`` is the wire alias of exactly one
    allowed original, else None. Pure recomputation — no stored alias map to
    go stale across calls/processes.
    """
    candidate = str(name or "").strip()
    if not candidate:
        return None
    for original in allowed_original_names:
        if not isinstance(original, str) or not original:
            continue
        if original == candidate:
            return original
        if not is_wire_safe_tool_name(original) and wire_safe_tool_name(original) == candidate:
            return original
    return None


def build_wire_name_map(names: Iterable[str]) -> Dict[str, str]:
    """original -> wire map for a declaration batch (unchanged names included)."""
    return {str(n): wire_safe_tool_name(str(n)) for n in names if isinstance(n, str) and n}


def wire_safe_tool_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy chat history and encode tool names for strict native APIs.

    Native endpoints validate names replayed in assistant ``tool_calls`` and
    tool-message ``name`` fields just as strictly as current declarations.
    Runtime history remains in its original namespace; only this outbound copy
    uses deterministic aliases.
    """
    safe = deepcopy(messages)
    for message in safe:
        name = message.get("name")
        if isinstance(name, str) and name:
            message["name"] = wire_safe_tool_name(name)
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict):
                call_name = function.get("name")
                if isinstance(call_name, str) and call_name:
                    function["name"] = wire_safe_tool_name(call_name)
            direct_name = call.get("name")
            if isinstance(direct_name, str) and direct_name:
                call["name"] = wire_safe_tool_name(direct_name)
    return safe
