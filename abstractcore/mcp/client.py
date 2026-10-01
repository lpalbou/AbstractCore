from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from ..utils.truncation import preview_text


_DEFAULT_ACCEPT = "application/json, text/event-stream"

# The MCP protocol revision this client asks for in `initialize`. A server that
# speaks another revision answers with its own; the client then uses that one.
DEFAULT_PROTOCOL_VERSION = "2025-11-25"

# `tools/list` is paginated (`nextCursor`); a server that keeps returning new
# cursors past this many pages is treated as broken rather than looped forever.
MAX_TOOL_PAGES = 100


class McpError(RuntimeError):
    """Base error for MCP client failures."""


class McpHttpError(McpError):
    """Raised when an MCP HTTP request fails (non-2xx, invalid JSON)."""


class McpRpcError(McpError):
    """Raised when an MCP JSON-RPC response contains an error object."""

    def __init__(self, *, code: int, message: str, data: Any = None):
        super().__init__(f"MCP JSON-RPC error {code}: {message}")
        self.code = int(code)
        self.message = str(message)
        self.data = data


class McpProtocolError(McpError):
    """Raised when an MCP response is malformed or violates JSON-RPC expectations."""


def default_client_version() -> str:
    for pkg in ("abstractcore", "AbstractCore"):
        try:
            v = str(metadata.version(pkg) or "").strip()
        except Exception:
            v = ""
        if v:
            return v
    return "0.0.0"


def initialize_params(*, protocol_version: str, client_name: str, client_version: str) -> Dict[str, Any]:
    """The `initialize` request params both transports send."""
    return {
        "protocolVersion": protocol_version,
        # Match the MCP reference client's envelope shape (server-side validators often
        # expect these keys to exist even when the values are null).
        "capabilities": {
            "experimental": None,
            "sampling": None,
            "elicitation": None,
            "roots": None,
            "tasks": None,
        },
        "clientInfo": {"name": client_name, "version": client_version or "0.0.0"},
    }


def tools_page(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """(tools, nextCursor) from one `tools/list` result."""
    tools = result.get("tools")
    if not isinstance(tools, list):
        raise McpProtocolError("MCP tools/list result missing tools list")
    out = [t for t in tools if isinstance(t, dict)]
    raw = result.get("nextCursor")
    nxt = str(raw) if isinstance(raw, str) and raw != "" else None
    return out, nxt


def collect_tool_pages(
    fetch: Callable[[Optional[str]], Dict[str, Any]], *, cursor: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Every tool of a paginated `tools/list`, following `nextCursor` from `cursor`."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    current = cursor
    for _ in range(MAX_TOOL_PAGES):
        tools, nxt = tools_page(fetch(current))
        out.extend(tools)
        if nxt is None:
            return out
        if nxt in seen:
            raise McpProtocolError(f"MCP tools/list returned the cursor {nxt!r} twice; stopping")
        seen.add(nxt)
        current = nxt
    raise McpProtocolError(f"MCP tools/list returned more than {MAX_TOOL_PAGES} pages; stopping")


def _answer_in_event(data_lines: List[str], want: Any) -> Optional[Dict[str, Any]]:
    """The JSON-RPC message answering request `want` in one SSE event's data lines, if any."""
    if not data_lines:
        return None
    try:
        obj = json.loads("\n".join(data_lines))
    except Exception:
        return None
    for msg in obj if isinstance(obj, list) else [obj]:
        if not isinstance(msg, dict):
            continue
        if "result" not in msg and "error" not in msg:
            continue  # a server notification or request, not our answer
        if str(msg.get("id")) == str(want) or (msg.get("id") is None and msg.get("error") is not None):
            return msg
    return None


@dataclass(frozen=True)
class McpJsonRpcRequest:
    jsonrpc: str
    id: int
    method: str
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id, "method": self.method}
        if self.params is not None:
            payload["params"] = self.params
        return payload


class McpClient:
    """A minimal MCP JSON-RPC client using HTTP POST (Streamable HTTP transport).

    This is intentionally small: it focuses on the tools surface (`tools/list`, `tools/call`).
    """

    @staticmethod
    def _ensure_accept_header(client: httpx.Client) -> None:
        existing = str(client.headers.get("Accept") or "").strip()
        if not existing:
            client.headers["Accept"] = _DEFAULT_ACCEPT
            return

        # httpx defaults to "*/*"; some MCP servers require both application/json and
        # text/event-stream in Accept for streamable HTTP.
        if existing == "*/*":
            client.headers["Accept"] = _DEFAULT_ACCEPT
            return

        has_json = "application/json" in existing
        has_sse = "text/event-stream" in existing
        if has_json and has_sse:
            return

        # Preserve any custom values while guaranteeing the required types appear.
        parts = [p.strip() for p in existing.split(",") if p.strip()]
        # Drop wildcard because some servers treat it as insufficient.
        parts = [p for p in parts if p != "*/*"]

        required = ["application/json", "text/event-stream"]
        out: List[str] = []
        for r in required:
            if any(r in p for p in parts):
                continue
            out.append(r)
        out.extend(parts)
        client.headers["Accept"] = ", ".join(out) if out else _DEFAULT_ACCEPT

    def __init__(
        self,
        *,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: Optional[float] = 30.0,
        protocol_version: Optional[str] = None,
        session_id: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        client_name: str = "abstractcore.mcp",
        client_version: Optional[str] = None,
    ) -> None:
        self._url = str(url or "").strip()
        if not self._url:
            raise ValueError("McpClient requires a non-empty url")

        self._timeout_s = timeout_s
        self._client = client or httpx.Client(headers=headers, timeout=timeout_s)
        self._owns_client = client is None
        self._id_iter = itertools.count(1)
        self._session_id: Optional[str] = str(session_id).strip() if session_id else None
        self._requested_protocol_version = (
            str(protocol_version).strip() if protocol_version else DEFAULT_PROTOCOL_VERSION
        )
        self._client_name = str(client_name or "abstractcore.mcp").strip() or "abstractcore.mcp"
        self._client_version = str(client_version).strip() if client_version else default_client_version()
        self._init_attempted = False
        self._initialized = False
        self._init_result: Optional[Dict[str, Any]] = None

        # Ensure streamable HTTP compatibility by default.
        self._ensure_accept_header(self._client)
        if protocol_version:
            self._client.headers["MCP-Protocol-Version"] = str(protocol_version).strip()
        if self._session_id:
            self._client.headers["MCP-Session-Id"] = self._session_id

    @property
    def url(self) -> str:
        return self._url

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def initialize_result(self) -> Optional[Dict[str, Any]]:
        """The server's `initialize` result (serverInfo, protocolVersion, capabilities), once known."""
        return self._init_result

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "McpClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _capture_session(self, resp: httpx.Response) -> None:
        # Capture MCP session id if the server provides one (streamable HTTP sessions).
        sid = str(resp.headers.get("MCP-Session-Id") or "").strip()
        if sid and sid != self._session_id:
            self._session_id = sid
            self._client.headers["MCP-Session-Id"] = sid

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code < 200 or resp.status_code >= 300:
            body = (resp.text or "").strip()
            raise McpHttpError(f"MCP HTTP {resp.status_code}: {preview_text(body, max_chars=500)}")

    def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        try:
            resp = self._client.post(self._url, json=payload)
        except Exception as e:
            raise McpHttpError(f"MCP request failed: {e}") from e
        self._capture_session(resp)
        self._raise_for_status(resp)
        return resp

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        want = payload.get("id")
        try:
            with self._client.stream("POST", self._url, json=payload) as resp:
                self._capture_session(resp)
                content_type = str(resp.headers.get("Content-Type") or "").lower()
                if resp.status_code < 200 or resp.status_code >= 300 or "text/event-stream" not in content_type:
                    resp.read()
                else:
                    # Streamable HTTP may answer with an SSE stream that carries the response
                    # (possibly after server notifications) and may stay open afterwards:
                    # read events until the one answering this request id.
                    data_lines: List[str] = []
                    for line in resp.iter_lines():
                        if line != "":
                            field, _, value = line.partition(":")
                            if field == "data":
                                data_lines.append(value[1:] if value.startswith(" ") else value)
                            continue
                        found = _answer_in_event(data_lines, want)
                        data_lines = []
                        if found is not None:
                            return found
                    found = _answer_in_event(data_lines, want)
                    if found is not None:
                        return found
                    raise McpProtocolError(f"MCP event stream ended without a response to request {want}")
        except McpError:
            raise
        except Exception as e:
            raise McpHttpError(f"MCP request failed: {e}") from e

        self._raise_for_status(resp)
        try:
            data = resp.json()
        except Exception as e:
            raise McpHttpError(f"MCP response is not valid JSON: {e}") from e

        if not isinstance(data, dict):
            raise McpProtocolError("MCP JSON-RPC response must be an object")
        return data

    def initialize(self) -> Dict[str, Any]:
        """Run the MCP handshake: `initialize`, then the `notifications/initialized` notification.

        Captures the server's `MCP-Session-Id` (sent on every later request) and the negotiated
        protocol version (sent as `MCP-Protocol-Version`). Returns the server's initialize result.
        Errors propagate (McpRpcError when the server refuses initialize).
        """
        self._init_attempted = True
        result = self._request_no_init(
            method="initialize",
            params=initialize_params(
                protocol_version=self._requested_protocol_version,
                client_name=self._client_name,
                client_version=self._client_version,
            ),
        )
        self._init_result = dict(result)
        negotiated = str(result.get("protocolVersion") or "").strip()
        if negotiated:
            self._client.headers["MCP-Protocol-Version"] = negotiated
        self.notify(method="notifications/initialized")
        self._initialized = True
        return dict(result)

    def _ensure_initialized(self) -> None:
        if self._initialized or self._init_attempted:
            return
        try:
            self.initialize()
        except McpRpcError as e:
            # Some non-conformant servers do not implement initialize; allow continuing.
            if int(getattr(e, "code", 0)) == -32601:
                self._initialized = True
                return
            raise

    def notify(self, *, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no id; the server answers 202 Accepted with no body)."""
        mid = str(method or "").strip()
        if not mid:
            raise ValueError("MCP notify requires a non-empty method")
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": mid}
        if params is not None:
            payload["params"] = params
        self._post(payload)

    def request(self, *, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if str(method or "").strip() != "initialize":
            self._ensure_initialized()
        return self._request_no_init(method=method, params=params)

    def _request_no_init(self, *, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mid = str(method or "").strip()
        if not mid:
            raise ValueError("MCP request requires a non-empty method")

        req = McpJsonRpcRequest(jsonrpc="2.0", id=next(self._id_iter), method=mid, params=params)
        resp = self._post_json(req.to_dict())

        if resp.get("jsonrpc") != "2.0":
            raise McpProtocolError("MCP response missing jsonrpc='2.0'")

        if "error" in resp and resp["error"] is not None:
            err = resp["error"]
            if isinstance(err, dict):
                raise McpRpcError(
                    code=int(err.get("code") or 0),
                    message=str(err.get("message") or "Unknown error"),
                    data=err.get("data"),
                )
            raise McpRpcError(code=-32000, message=str(err), data=None)

        resp_id = resp.get("id")
        if resp_id is None:
            raise McpProtocolError("MCP response missing id")
        if str(resp_id) != str(req.id):
            raise McpProtocolError(f"MCP response id mismatch (expected {req.id}, got {resp_id})")

        result = resp.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError("MCP response missing result object")
        return result

    def list_tools_page(self, *, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """One `tools/list` page: (tools, nextCursor or None)."""
        params = {"cursor": str(cursor)} if cursor is not None else None
        return tools_page(self.request(method="tools/list", params=params))

    def list_tools(self, *, cursor: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every tool the server lists, following `nextCursor` (from `cursor` when given)."""
        return collect_tool_pages(
            lambda c: self.request(method="tools/list", params={"cursor": c} if c is not None else None),
            cursor=cursor,
        )

    def call_tool(self, *, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tool_name = str(name or "").strip()
        if not tool_name:
            raise ValueError("MCP tools/call requires a non-empty tool name")
        args = dict(arguments or {})

        result = self.request(method="tools/call", params={"name": tool_name, "arguments": args})
        if not isinstance(result, dict):
            raise McpProtocolError("MCP tools/call result must be an object")
        return result
