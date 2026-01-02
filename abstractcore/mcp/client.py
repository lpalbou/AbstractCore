from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Optional

import httpx


_DEFAULT_ACCEPT = "application/json, text/event-stream"


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
    ) -> None:
        self._url = str(url or "").strip()
        if not self._url:
            raise ValueError("McpClient requires a non-empty url")

        self._timeout_s = timeout_s
        self._client = client or httpx.Client(headers=headers, timeout=timeout_s)
        self._owns_client = client is None
        self._id_iter = itertools.count(1)
        self._session_id: Optional[str] = str(session_id).strip() if session_id else None

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

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "McpClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = self._client.post(self._url, json=payload)
        except Exception as e:
            raise McpHttpError(f"MCP request failed: {e}") from e

        # Capture MCP session id if the server provides one (streamable HTTP sessions).
        sid = str(resp.headers.get("MCP-Session-Id") or "").strip()
        if sid and sid != self._session_id:
            self._session_id = sid
            self._client.headers["MCP-Session-Id"] = sid

        if resp.status_code < 200 or resp.status_code >= 300:
            body = (resp.text or "").strip()
            raise McpHttpError(f"MCP HTTP {resp.status_code}: {body[:500]}")

        try:
            data = resp.json()
        except Exception as e:
            raise McpHttpError(f"MCP response is not valid JSON: {e}") from e

        if not isinstance(data, dict):
            raise McpProtocolError("MCP JSON-RPC response must be an object")
        return data

    def request(self, *, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    def list_tools(self, *, cursor: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Optional[Dict[str, Any]] = None
        if cursor is not None:
            params = {"cursor": str(cursor)}

        result = self.request(method="tools/list", params=params)
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise McpProtocolError("MCP tools/list result missing tools list")
        out: List[Dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                out.append(t)
        return out

    def call_tool(self, *, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tool_name = str(name or "").strip()
        if not tool_name:
            raise ValueError("MCP tools/call requires a non-empty tool name")
        args = dict(arguments or {})

        result = self.request(method="tools/call", params={"name": tool_name, "arguments": args})
        if not isinstance(result, dict):
            raise McpProtocolError("MCP tools/call result must be an object")
        return result
