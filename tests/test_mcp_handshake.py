"""MCP handshake + pagination against strict fake servers (HTTP and stdio).

The fakes behave like conformant MCP servers that refuse sloppy clients:
- every request but `initialize` is refused until the client sent the
  `notifications/initialized` notification (JSON-RPC -32002);
- the HTTP fake issues an `MCP-Session-Id` on initialize and answers 400 to a
  later request that does not carry it;
- `tools/list` is paginated with `nextCursor`, and the HTTP fake answers the
  second page as a `text/event-stream` (a server notification first, then the
  response) the way Streamable HTTP servers may.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractcore.mcp import McpClient, McpStdioClient, create_mcp_client
from abstractcore.mcp.client import McpProtocolError, McpRpcError

TOOLS = [
    {"name": "add", "description": "Add two integers.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "echo", "description": "Echo a text.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "clock", "description": "Tell the time.", "inputSchema": {"type": "object", "properties": {}}},
]


class _FakeHttpMcp:
    def __init__(self, *, refuse_initialize: bool = False, loop_cursor: bool = False) -> None:
        self.log: List[Dict[str, Any]] = []
        self.sessions: Dict[str, bool] = {}
        self.refuse_initialize = refuse_initialize
        self.loop_cursor = loop_cursor
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # quiet
                return

            def _send(self, status: int, body: Any = None, headers: Dict[str, str] | None = None, sse: bool = False) -> None:
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                if body is None:
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if sse:
                    events = body
                    raw = "".join(f"event: message\ndata: {json.dumps(e)}\n\n" for e in events).encode()
                    self.send_header("Content-Type", "text/event-stream")
                else:
                    raw = json.dumps(body).encode()
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self) -> None:  # noqa: N802
                accept = self.headers.get("Accept") or ""
                if "application/json" not in accept or "text/event-stream" not in accept:
                    self._send(406, {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": "Not Acceptable"}})
                    return
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                sid = self.headers.get("MCP-Session-Id")
                fake.log.append(
                    {"method": req.get("method"), "id": req.get("id"), "session": sid,
                     "protocol": self.headers.get("MCP-Protocol-Version"), "params": req.get("params")}
                )
                method, rid = req.get("method"), req.get("id")
                if method == "initialize":
                    if fake.refuse_initialize:
                        self._send(200, {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "Unsupported protocol version"}})
                        return
                    new_sid = f"sess-{len(fake.sessions) + 1}"
                    fake.sessions[new_sid] = False
                    self._send(
                        200,
                        {"jsonrpc": "2.0", "id": rid, "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {"name": "fake-http", "version": "1.2.3"}}},
                        headers={"MCP-Session-Id": new_sid},
                    )
                    return
                if sid not in fake.sessions:
                    self._send(400, {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": "Bad Request: missing or unknown MCP-Session-Id"}})
                    return
                if rid is None:
                    if method == "notifications/initialized":
                        fake.sessions[sid] = True
                    self._send(202)
                    return
                if not fake.sessions[sid]:
                    self._send(200, {"jsonrpc": "2.0", "id": rid, "error": {"code": -32002, "message": "Server not initialized"}})
                    return
                if method == "tools/list":
                    cursor = (req.get("params") or {}).get("cursor")
                    if cursor is None:
                        self._send(200, {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS[:2], "nextCursor": "page-2"}})
                    elif fake.loop_cursor:
                        self._send(200, {"jsonrpc": "2.0", "id": rid, "result": {"tools": [], "nextCursor": "page-2"}})
                    else:
                        self._send(
                            200,
                            [
                                {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "listing"}},
                                {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS[2:]}},
                            ],
                            sse=True,
                        )
                    return
                self._send(200, {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_FakeHttpMcp":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


STDIO_FAKE = r'''
import json, sys
TOOLS = %s
ready = False
log = open(sys.argv[1], "a") if len(sys.argv) > 1 else None
def send(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if log:
        log.write(json.dumps({"method": req.get("method"), "id": req.get("id")}) + "\n"); log.flush()
    m, rid = req.get("method"), req.get("id")
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": req["params"]["protocolVersion"],
              "capabilities": {"tools": {}}, "serverInfo": {"name": "fake-stdio", "version": "0.1"}}})
        continue
    if rid is None:
        if m == "notifications/initialized":
            ready = True
        continue
    if not ready:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32002, "message": "Server not initialized"}})
        continue
    if m == "tools/list":
        cursor = (req.get("params") or {}).get("cursor")
        if cursor is None:
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS[:1], "nextCursor": "c1"}})
        elif cursor == "c1":
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS[1:2], "nextCursor": "c2"}})
        else:
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS[2:]}})
        continue
    send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}})
''' % json.dumps(TOOLS)


def _stdio_fake(tmp_path: Path) -> Path:
    path = tmp_path / "fake_mcp_stdio.py"
    path.write_text(STDIO_FAKE, encoding="utf-8")
    return path


# ---------------------------------------------------------------- HTTP


def test_http_initialize_handshake_session_and_paged_tools() -> None:
    with _FakeHttpMcp() as fake, McpClient(url=fake.url, timeout_s=5) as client:
        info = client.initialize()
        assert info["serverInfo"] == {"name": "fake-http", "version": "1.2.3"}
        assert client.session_id == "sess-1"
        assert client.initialize_result["protocolVersion"] == "2025-06-18"

        tools = client.list_tools()
        assert [t["name"] for t in tools] == ["add", "echo", "clock"]

    methods = [e["method"] for e in fake.log]
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/list"]
    init = fake.log[0]
    assert init["session"] is None
    assert init["params"]["protocolVersion"] == "2025-11-25"
    assert init["params"]["clientInfo"]["name"] == "abstractcore.mcp"
    # Every request after initialize carries the session id and the negotiated version.
    assert all(e["session"] == "sess-1" for e in fake.log[1:])
    assert all(e["protocol"] == "2025-06-18" for e in fake.log[1:])
    assert fake.log[3]["params"] == {"cursor": "page-2"}


def test_http_tools_list_initializes_lazily() -> None:
    with _FakeHttpMcp() as fake, McpClient(url=fake.url, timeout_s=5) as client:
        assert [t["name"] for t in client.list_tools()] == ["add", "echo", "clock"]
    assert [e["method"] for e in fake.log][:2] == ["initialize", "notifications/initialized"]


def test_http_factory_client_handshakes() -> None:
    with _FakeHttpMcp() as fake:
        client = create_mcp_client(config={"url": fake.url, "headers": {"X-Test": "1"}}, timeout_s=5)
        try:
            assert client.initialize()["serverInfo"]["name"] == "fake-http"
            assert len(client.list_tools()) == 3
        finally:
            client.close()


def test_http_refused_initialize_raises() -> None:
    with _FakeHttpMcp(refuse_initialize=True) as fake, McpClient(url=fake.url, timeout_s=5) as client:
        with pytest.raises(McpRpcError) as err:
            client.initialize()
        assert "Unsupported protocol version" in str(err.value)


def test_http_repeated_cursor_stops_loudly() -> None:
    with _FakeHttpMcp(loop_cursor=True) as fake, McpClient(url=fake.url, timeout_s=5) as client:
        with pytest.raises(McpProtocolError, match="twice"):
            client.list_tools()


def test_http_list_tools_page_returns_cursor() -> None:
    with _FakeHttpMcp() as fake, McpClient(url=fake.url, timeout_s=5) as client:
        tools, nxt = client.list_tools_page()
        assert [t["name"] for t in tools] == ["add", "echo"] and nxt == "page-2"


# ---------------------------------------------------------------- stdio


def test_stdio_initialize_handshake_and_paged_tools(tmp_path: Path) -> None:
    script = _stdio_fake(tmp_path)
    log = tmp_path / "stdio.log"
    with McpStdioClient(command=[sys.executable, "-u", str(script), str(log)], timeout_s=5) as client:
        info = client.initialize()
        assert info["serverInfo"] == {"name": "fake-stdio", "version": "0.1"}
        assert [t["name"] for t in client.list_tools()] == ["add", "echo", "clock"]
    methods = [json.loads(line)["method"] for line in log.read_text().splitlines()]
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/list", "tools/list"]


def test_stdio_lazy_initialize_sends_notifications_initialized(tmp_path: Path) -> None:
    script = _stdio_fake(tmp_path)
    client = create_mcp_client(config={"transport": "stdio", "command": [sys.executable, "-u", str(script)]}, timeout_s=5)
    try:
        assert [t["name"] for t in client.list_tools()] == ["add", "echo", "clock"]
        assert client.initialize_result["serverInfo"]["name"] == "fake-stdio"
    finally:
        client.close()
