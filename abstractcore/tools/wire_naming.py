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
from dataclasses import is_dataclass, replace
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


def map_namespaced_tool_name(raw: str, allowed_names: Iterable[str]) -> Optional[str]:
    """Recover the intended tool behind a namespaced or decorated name.

    Models drift into OpenAI-style namespace prefixes (`functions.browser_probe`)
    and dotted spellings of underscore names (`browser.probe`) — observed live on
    Qwen3.8-Flash-Next (session acode-912712976c0c). Priority reads the WHOLE
    name most faithfully first, so `browser.probe` prefers a roster name
    `browser_probe` over a shorter roster name `probe`:

      1. exact match;
      2. whole name with dots replaced by underscores;
      3. progressively strip leading dot-segments (`functions.x` -> `x`), first
         remainder — or its dots->underscores spelling — in the roster wins;
      4. longest allowed name occurring as a standalone token inside the raw
         string (legacy wrapped-name recovery, e.g. "{function-name: write_file}").

    Returns None when nothing maps; callers must not guess further.
    """
    s = str(raw or "").strip()
    allowed = {
        str(n).strip() for n in (allowed_names or [])
        if isinstance(n, str) and str(n).strip()
    }
    if not s or not allowed:
        return None
    if s in allowed:
        return s
    if "." in s:
        underscored = s.replace(".", "_")
        if underscored in allowed:
            return underscored
        parts = s.split(".")
        for i in range(1, len(parts)):
            remainder = ".".join(parts[i:])
            if remainder in allowed:
                return remainder
            remainder_us = remainder.replace(".", "_")
            if remainder_us in allowed:
                return remainder_us
    try:
        candidates = [
            n for n in allowed
            if re.search(r"(^|[^\w])" + re.escape(n) + r"([^\w]|$)", s)
        ]
        if candidates:
            return max(candidates, key=lambda n: (len(n), n))
    except Exception:
        return None
    return None


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


def restore_tool_name(name: str, allowed_original_names: Iterable[str]) -> Optional[str]:
    """The declared tool a returned name means, under the response-normalization rules.

    Exact match, then the wire alias of an allowed original
    (``resolve_wire_tool_name``), then a namespaced/decorated spelling
    (``map_namespaced_tool_name``). None when nothing in the allowlist maps:
    the caller must leave the name as returned (the host refuses it).
    """
    allowed = [n for n in (allowed_original_names or []) if isinstance(n, str) and n]
    candidate = str(name or "").strip()
    if not candidate or not allowed:
        return None
    if candidate in allowed:
        return candidate
    return resolve_wire_tool_name(candidate, allowed) or map_namespaced_tool_name(candidate, allowed)


def restore_tool_call_names(
    tool_calls: Any,
    allowed_original_names: Iterable[str],
    *,
    cache: Optional[Dict[str, Optional[str]]] = None,
) -> "tuple[Any, List[str]]":
    """Restore declared names on provider-emitted (native) tool calls, nothing else.

    Backlog 0856: the non-streaming passthrough restored wire aliases, streamed
    native calls did not, so a host saw ``mcp_RAGnarok_query_b6ff8949`` and
    refused it. This rewrites ONLY the name, wherever the call carries one (a
    top-level ``name`` and/or ``function.name``), so it is safe on complete
    calls and on partial deltas alike: ids, indexes and (possibly partial)
    argument strings are untouched, a delta without a name passes through, and
    the caller's dicts are copied, never mutated.

    Returns ``(calls, unresolved_names)``. A name nothing in THIS request's
    allowlist maps to is left as returned and listed in ``unresolved_names``.
    ``cache`` (one dict per request) memoizes lookups across a stream's
    chunks: some providers repeat the full name on every partial delta.
    """
    allowed = [n for n in (allowed_original_names or []) if isinstance(n, str) and n]
    if not isinstance(tool_calls, list) or not tool_calls or not allowed:
        return tool_calls, []

    unresolved: List[str] = []

    def _restored(raw: Any) -> Any:
        if not isinstance(raw, str) or not raw.strip():
            return raw
        key = raw.strip()
        if cache is not None and key in cache:
            mapped = cache[key]
        else:
            mapped = restore_tool_name(raw, allowed)
            if cache is not None:
                cache[key] = mapped
        if mapped is None:
            unresolved.append(raw.strip())
            return raw
        return mapped

    out: List[Any] = []
    changed = False
    for call in tool_calls:
        if isinstance(call, dict):
            new_call = call
            if isinstance(call.get("name"), str):
                restored = _restored(call["name"])
                if restored != call["name"]:
                    new_call = dict(call)
                    new_call["name"] = restored
            function = call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                restored = _restored(function["name"])
                if restored != function["name"]:
                    new_call = dict(new_call) if new_call is call else new_call
                    new_call["function"] = {**function, "name": restored}
            changed = changed or new_call is not call
            out.append(new_call)
            continue
        raw_name = getattr(call, "name", None)
        restored = _restored(raw_name)
        if restored != raw_name and is_dataclass(call) and not isinstance(call, type):
            out.append(replace(call, name=restored))
            changed = True
        else:
            out.append(call)
    return (out if changed else tool_calls), list(dict.fromkeys(unresolved))


def unresolved_tool_name_warning(name: str, allowed_original_names: Iterable[str]) -> str:
    """The warning both streaming lanes attach when a returned name maps to no declared tool."""
    available = ", ".join(sorted(n for n in allowed_original_names if isinstance(n, str) and n))
    return f"Tool call '{name}' does not match any available tool (available: {available})."
