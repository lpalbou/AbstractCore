from __future__ import annotations

from typing import Optional, Tuple


MCP_TOOL_PREFIX = "mcp::"


def namespaced_tool_name(*, server_id: str, tool_name: str) -> str:
    sid = str(server_id or "").strip()
    tn = str(tool_name or "").strip()
    if not sid:
        raise ValueError("server_id must be non-empty")
    if not tn:
        raise ValueError("tool_name must be non-empty")
    return f"{MCP_TOOL_PREFIX}{sid}::{tn}"


def parse_namespaced_tool_name(name: str) -> Optional[Tuple[str, str]]:
    text = str(name or "").strip()
    if not text.startswith(MCP_TOOL_PREFIX):
        return None
    rest = text[len(MCP_TOOL_PREFIX) :]
    sid, sep, tn = rest.partition("::")
    if sep != "::" or not sid or not tn:
        return None
    return sid, tn

