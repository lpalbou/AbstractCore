from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set

from .naming import namespaced_tool_name


def _first_non_empty_line(text: Optional[str]) -> str:
    if not text:
        return ""
    for line in str(text).splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _normalize_one_line(text: Optional[str]) -> str:
    return " ".join(str(text or "").split()).strip()


def _short_description(text: Optional[str], *, max_chars: int = 200) -> str:
    """DEPRECATED — kept only for callers outside this module.

    Silently cut at `max_chars - 3` and glued on "...", which can invert a
    description's meaning and told nobody. Use
    `abstractcore.tools.core.adapt_external_description`, which keeps whole
    sentences and reports the overflow for correction.
    """
    one = _normalize_one_line(_first_non_empty_line(text))
    if not one:
        return ""
    if len(one) <= max_chars:
        return one
    if max_chars <= 3:
        return one[:max_chars]
    return one[: max_chars - 3].rstrip() + "..."


@dataclass(frozen=True)
class McpServerInfo:
    server_id: str
    url: str
    transport: str = "streamable_http"
    version: Optional[str] = None


class McpToolClient(Protocol):
    def list_tools(self, *, cursor: Optional[str] = None) -> List[Dict[str, Any]]: ...

    def call_tool(self, *, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...

    def close(self) -> None: ...


def _normalize_mcp_parameters(input_schema: Any) -> Dict[str, Any]:
    """Convert an MCP inputSchema into AbstractCore's compact parameter map.

    AbstractCore's tool prompts and native tool formatting treat a parameter as required
    when its schema dict omits a `default`. Therefore, for MCP schemas:
    - required properties => ensure `default` is absent
    - optional properties => set `default` to None (unless a default is already provided)
    """
    if not isinstance(input_schema, dict):
        return {}

    props = input_schema.get("properties")
    if not isinstance(props, dict):
        return {}

    required_raw = input_schema.get("required")
    required: Set[str] = set()
    if isinstance(required_raw, list):
        required = {str(x) for x in required_raw if isinstance(x, str) and x.strip()}

    out: Dict[str, Any] = {}
    for key, schema in props.items():
        if not isinstance(key, str) or not key.strip():
            continue
        meta: Dict[str, Any]
        if isinstance(schema, dict):
            meta = dict(schema)
        else:
            meta = {}

        if key in required:
            meta.pop("default", None)
        else:
            meta.setdefault("default", None)

        out[key] = meta

    return out


def mcp_tool_to_abstractcore_tool_spec(
    tool: Dict[str, Any],
    *,
    server: McpServerInfo,
) -> Dict[str, Any]:
    """Convert a single MCP tool entry into an AbstractCore tool spec dict."""
    raw_name = tool.get("name")
    name = str(raw_name or "").strip()
    if not name:
        raise ValueError("MCP tool is missing a valid name")

    # An MCP server's description is not ours to edit, and MCP servers routinely
    # ship paragraph-length ones. Adapt it to the cap on a sentence boundary and
    # report the overflow so the integrator can ask upstream for a better line —
    # never a silent mid-sentence cut, never a dropped tool.
    from ..tools.core import adapt_external_description

    raw_description = tool.get("description") or tool.get("title") or ""
    description = adapt_external_description(
        str(raw_description or ""),
        tool_name=name,
        source=f"MCP server '{server.server_id}'",
    )

    parameters = _normalize_mcp_parameters(tool.get("inputSchema"))

    return {
        "name": namespaced_tool_name(server_id=server.server_id, tool_name=name),
        "description": description,
        "parameters": parameters,
        "tags": ["mcp", f"mcp_server:{server.server_id}"],
        "origin": {
            "type": "mcp",
            "server_id": server.server_id,
            "tool_name": name,
            "transport": server.transport,
            "url": server.url,
            "version": server.version,
        },
    }


class McpToolSource:
    """Discover tools from an MCP server and return AbstractCore-compatible tool specs."""

    def __init__(
        self,
        *,
        server_id: str,
        client: McpToolClient,
        transport: str = "streamable_http",
        origin_url: Optional[str] = None,
    ) -> None:
        sid = str(server_id or "").strip()
        if not sid:
            raise ValueError("McpToolSource requires a non-empty server_id")
        transport_norm = str(transport or "").strip() or "streamable_http"

        url = str(origin_url or "").strip() if origin_url else ""
        if not url:
            candidate = getattr(client, "url", None)
            if isinstance(candidate, str) and candidate.strip():
                url = candidate.strip()
        if not url:
            url = f"{transport_norm}://{sid}"

        self._server = McpServerInfo(server_id=sid, url=url, transport=transport_norm)
        self._client = client

    @property
    def server(self) -> McpServerInfo:
        return self._server

    def list_tool_specs(self) -> List[Dict[str, Any]]:
        """Every tool the server advertises — and a warning for each one we could not take.

        The skip stays best-effort (one malformed entry must not cost the whole
        server), but it is no longer SILENT: a skipped tool is a capability the
        caller asked for and will never see (ADR 0001). `logger.warning` is not
        a channel here — abstractcore leaves the root logger at ERROR and its own
        loggers NOTSET, so such a record is never even created.
        """
        tools = self._client.list_tools()
        specs: List[Dict[str, Any]] = []
        for t in tools:
            try:
                specs.append(mcp_tool_to_abstractcore_tool_spec(t, server=self._server))
            except Exception as exc:
                name = ""
                if isinstance(t, dict):
                    name = str(t.get("name") or t.get("title") or "").strip()
                warnings.warn(
                    f"MCP server '{self._server.server_id}' advertised a tool "
                    f"{('named ' + repr(name)) if name else '(unnamed)'} that AbstractCore could "
                    f"NOT import: {exc}. It will not be offered to the model — the model cannot "
                    f"call a tool it never sees.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
        return specs
