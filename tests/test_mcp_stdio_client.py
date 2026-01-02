from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _write_stdio_stub(path: Path) -> None:
    path.write_text(
        """
import json
import sys

initialized = False

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = (line or "").strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    if not isinstance(req, dict):
        continue

    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params") if isinstance(req.get("params"), dict) else {}

    if req.get("jsonrpc") != "2.0" or not method:
        send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid Request"}})
        continue

    if req_id is None:
        # Notification
        continue

    if method == "initialize":
        initialized = True
        send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": params.get("protocolVersion") or "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stub"},
                },
            }
        )
        continue

    if not initialized:
        send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32002, "message": "Not initialized"}})
        continue

    if method == "tools/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "add",
                            "description": "Add two integers.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                "required": ["a", "b"],
                            },
                        }
                    ]
                },
            }
        )
        continue

    if method == "tools/call":
        name = str(params.get("name") or "")
        args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name != "add":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": "Unknown tool"}], "isError": True},
                }
            )
            continue
        a = int(args.get("a") or 0)
        b = int(args.get("b") or 0)
        send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(a + b)}], "isError": False},
            }
        )
        continue

    send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})
""".lstrip()
    )


def test_mcp_stdio_client_list_and_call(tmp_path: Path) -> None:
    from abstractcore.mcp import McpStdioClient

    server = tmp_path / "mcp_stdio_stub.py"
    _write_stdio_stub(server)

    with McpStdioClient(command=[sys.executable, "-u", str(server)]) as client:
        tools = client.list_tools()
        assert [t.get("name") for t in tools] == ["add"]

        result = client.call_tool(name="add", arguments={"a": 2, "b": 3})
        assert result.get("isError") is False
        content = result.get("content")
        assert isinstance(content, list)
        assert content and content[0].get("type") == "text"
        assert content[0].get("text") == "5"


def test_create_mcp_client_stdio_factory(tmp_path: Path) -> None:
    from abstractcore.mcp import McpStdioClient, create_mcp_client

    server = tmp_path / "mcp_stdio_stub.py"
    _write_stdio_stub(server)

    client = create_mcp_client(config={"transport": "stdio", "command": [sys.executable, "-u", str(server)]})
    try:
        assert isinstance(client, McpStdioClient)
        tools = client.list_tools()
        assert [t.get("name") for t in tools] == ["add"]
    finally:
        try:
            client.close()
        except Exception:
            pass

