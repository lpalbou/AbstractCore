from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Deque, Dict, List, Optional, Sequence

from .client import McpError, McpJsonRpcRequest, McpProtocolError, McpRpcError


_DEFAULT_PROTOCOL_VERSION = "2025-11-25"


@dataclass(frozen=True)
class McpStdioServerParameters:
    command: List[str]
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None


class McpStdioClient:
    """A minimal MCP JSON-RPC client using stdio (spawn a subprocess).

    This client is synchronous and intentionally small: it focuses on the tools surface
    (`tools/list`, `tools/call`) and includes a best-effort MCP initialization handshake.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_s: Optional[float] = 30.0,
        protocol_version: Optional[str] = None,
        client_name: str = "abstractcore.mcp",
        client_version: Optional[str] = None,
    ) -> None:
        cmd = [str(c) for c in (command or []) if str(c).strip()]
        if not cmd:
            raise ValueError("McpStdioClient requires a non-empty command")

        self._timeout_s = float(timeout_s) if timeout_s is not None else None
        self._id_iter = itertools.count(1)
        self._init_attempted = False
        self._initialized = False

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(cwd) if cwd else None,
            env=self._merge_env(env),
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise McpError("Failed to start MCP stdio subprocess with pipes")

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._responses: Dict[int, Dict[str, Any]] = {}
        self._global_error: Optional[Dict[str, Any]] = None
        self._closed = False
        self._stderr_tail: Deque[str] = deque(maxlen=200)

        self._protocol_version = str(protocol_version).strip() if protocol_version else _DEFAULT_PROTOCOL_VERSION
        self._client_name = str(client_name or "abstractcore.mcp").strip() or "abstractcore.mcp"
        self._client_version = str(client_version).strip() if client_version else self._default_client_version()

        self._stdout_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    @staticmethod
    def _default_client_version() -> str:
        for pkg in ("abstractcore", "AbstractCore"):
            try:
                v = str(metadata.version(pkg) or "").strip()
            except Exception:
                v = ""
            if v:
                return v
        return "0.0.0"

    @staticmethod
    def _merge_env(env: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if env is None:
            return None
        merged = dict(os.environ)
        for k, v in env.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if v is None:
                continue
            merged[str(k)] = str(v)
        return merged

    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()

        try:
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            if self._proc.stdout:
                try:
                    self._proc.stdout.close()
                except Exception:
                    pass
        finally:
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def __enter__(self) -> "McpStdioClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _read_stdout_loop(self) -> None:
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                text = (line or "").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue

                msg_id = msg.get("id")
                if msg_id is None:
                    # JSON-RPC error may legally have id=null; surface as a global error.
                    if msg.get("error") is not None:
                        with self._cond:
                            self._global_error = dict(msg)
                            self._cond.notify_all()
                    continue
                try:
                    mid = int(msg_id)
                except Exception:
                    continue

                with self._cond:
                    self._responses[mid] = dict(msg)
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def _read_stderr_loop(self) -> None:
        try:
            if self._proc.stderr is None:
                return
            for line in self._proc.stderr:
                text = (line or "").rstrip()
                if not text:
                    continue
                with self._cond:
                    self._stderr_tail.append(text)
        except Exception:
            return

    def _send(self, payload: Dict[str, Any]) -> None:
        if self._proc.stdin is None:
            raise McpError("MCP stdio process stdin is closed")
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except Exception as e:
            raise McpError(f"Failed to write to MCP stdio process: {e}") from e

    def _wait_for(self, req_id: int) -> Dict[str, Any]:
        deadline: Optional[float]
        if self._timeout_s is None:
            deadline = None
        else:
            deadline = time.time() + self._timeout_s

        with self._cond:
            while True:
                if req_id in self._responses:
                    return self._responses.pop(req_id)
                if self._global_error is not None:
                    err = self._global_error
                    self._global_error = None
                    raise McpProtocolError(f"MCP stdio global error: {err}")
                if self._closed:
                    tail = "\n".join(list(self._stderr_tail)[-20:])
                    raise McpError(f"MCP stdio process closed unexpectedly.\n\nstderr tail:\n{tail}")

                if deadline is None:
                    self._cond.wait(timeout=0.25)
                    continue

                remaining = deadline - time.time()
                if remaining <= 0:
                    tail = "\n".join(list(self._stderr_tail)[-20:])
                    raise McpError(f"MCP stdio request timed out after {self._timeout_s}s.\n\nstderr tail:\n{tail}")
                self._cond.wait(timeout=min(0.25, remaining))

    def _request_no_init(self, *, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mid = str(method or "").strip()
        if not mid:
            raise ValueError("MCP request requires a non-empty method")

        req = McpJsonRpcRequest(jsonrpc="2.0", id=next(self._id_iter), method=mid, params=params)
        self._send(req.to_dict())
        resp = self._wait_for(req.id)

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

    def _ensure_initialized(self) -> None:
        if self._initialized or self._init_attempted:
            return
        self._init_attempted = True

        params: Dict[str, Any] = {
            "protocolVersion": self._protocol_version,
            # Match the MCP reference client's envelope shape (server-side validators often
            # expect these keys to exist even when the values are null).
            "capabilities": {
                "experimental": None,
                "sampling": None,
                "elicitation": None,
                "roots": None,
                "tasks": None,
            },
            "clientInfo": {"name": self._client_name, "version": self._client_version or "0.0.0"},
        }

        try:
            self._request_no_init(method="initialize", params=params)
            # Best-effort "initialized" notification.
            try:
                self.notify(method="initialized", params={})
            except Exception:
                pass
            self._initialized = True
        except McpRpcError as e:
            # Some non-conformant servers may not implement initialize; allow continuing.
            if int(getattr(e, "code", 0)) == -32601:
                self._initialized = True
                return
            raise

    def notify(self, *, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        mid = str(method or "").strip()
        if not mid:
            raise ValueError("MCP notify requires a non-empty method")
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": mid}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def request(self, *, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if str(method or "").strip() != "initialize":
            self._ensure_initialized()
        return self._request_no_init(method=method, params=params)

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
