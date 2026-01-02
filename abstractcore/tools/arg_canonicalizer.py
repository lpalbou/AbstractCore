"""Tool argument canonicalization (aliases -> canonical).

AbstractCore owns tool-call parsing/rewriting. This module provides a small,
central place to normalize common argument-name drift that appears in LLM
generated calls, while keeping the runtime/tool implementations stable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..utils.jsonish import loads_dict_like


def _loads_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = loads_dict_like(value)
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


def canonicalize_tool_arguments(tool_name: str, arguments: Any) -> Dict[str, Any]:
    """Return a canonical argument dict for a tool call (best-effort)."""
    name = str(tool_name or "").strip()
    args = _loads_dict(arguments) or {}

    if not name or not args:
        return args

    if name == "read_file":
        return _canonicalize_read_file_args(args)

    return args


def _canonicalize_read_file_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(arguments)

    if "start_line" not in out:
        if "start_line_one_indexed" in out:
            out["start_line"] = out.get("start_line_one_indexed")
        elif "start" in out:
            out["start_line"] = out.get("start")

    if "end_line" not in out:
        if "end_line_one_indexed_inclusive" in out:
            out["end_line"] = out.get("end_line_one_indexed_inclusive")
        elif "end" in out:
            out["end_line"] = out.get("end")

    for k in ("start_line_one_indexed", "end_line_one_indexed_inclusive", "start", "end"):
        out.pop(k, None)

    return out


__all__ = ["canonicalize_tool_arguments"]
