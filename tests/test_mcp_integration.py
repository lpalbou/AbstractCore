import json
from typing import Any, Dict, List

import httpx
import pytest


def _mcp_wsgi_app(environ: Dict[str, Any], start_response) -> List[bytes]:
    """In-process MCP stub using httpx.WSGITransport (no sockets required)."""
    method = environ.get("REQUEST_METHOD")
    if method != "POST":
        start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
        return [b"method not allowed"]

    accept = str(environ.get("HTTP_ACCEPT") or "")
    if "application/json" not in accept or "text/event-stream" not in accept:
        start_response("406 Not Acceptable", [("Content-Type", "application/json")])
        return [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32000,
                        "message": "Not Acceptable: Client must accept both application/json and text/event-stream",
                    },
                    "id": None,
                }
            ).encode("utf-8")
        ]

    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except Exception:
        length = 0

    body = environ.get("wsgi.input").read(length) if environ.get("wsgi.input") else b""
    try:
        req = json.loads(body.decode("utf-8"))
    except Exception:
        start_response("400 Bad Request", [("Content-Type", "application/json")])
        return [json.dumps({"error": "invalid json"}).encode("utf-8")]

    if not isinstance(req, dict):
        start_response("400 Bad Request", [("Content-Type", "application/json")])
        return [json.dumps({"error": "request must be object"}).encode("utf-8")]

    jsonrpc = req.get("jsonrpc")
    req_id = req.get("id")
    rpc_method = req.get("method")
    params = req.get("params") if isinstance(req.get("params"), dict) else {}

    if jsonrpc != "2.0" or not rpc_method:
        start_response("400 Bad Request", [("Content-Type", "application/json")])
        return [json.dumps({"error": "invalid jsonrpc envelope"}).encode("utf-8")]

    tools: List[Dict[str, Any]] = [
        {
            "name": "add",
            "description": "Add two integers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer"},
                    "b": {"type": "integer", "description": "Second integer"},
                    "note": {"type": "string", "description": "Optional note"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "search",
            "description": "Search with optional filters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "tag": {"type": "string"},
                            "limit": {"type": "integer", "default": 5},
                        },
                    },
                },
                "required": ["query"],
            },
        },
    ]

    resp: Dict[str, Any]
    if rpc_method == "tools/list":
        resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
    elif rpc_method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name == "add":
            a = int(arguments.get("a") or 0)
            b = int(arguments.get("b") or 0)
            value = a + b
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(value)}], "isError": False},
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": "Unknown tool"}], "isError": True},
            }
    else:
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {rpc_method}"},
        }

    start_response("200 OK", [("Content-Type", "application/json")])
    return [json.dumps(resp).encode("utf-8")]


def test_mcp_list_tools_and_schema_normalization() -> None:
    from abstractcore.mcp import McpClient, McpToolSource, parse_namespaced_tool_name

    transport = httpx.WSGITransport(app=_mcp_wsgi_app)
    with httpx.Client(transport=transport, base_url="http://mcp.test") as http_client:
        client = McpClient(url="http://mcp.test/", client=http_client)
        source = McpToolSource(server_id="srv", client=client)
        specs = source.list_tool_specs()

        by_name = {s["name"]: s for s in specs}
        assert "mcp::srv::add" in by_name
        assert "mcp::srv::search" in by_name

        add = by_name["mcp::srv::add"]
        assert add["description"]
        assert add["origin"]["type"] == "mcp"
        assert add["origin"]["server_id"] == "srv"
        assert add["origin"]["tool_name"] == "add"
        assert add["origin"]["url"] == "http://mcp.test/"

        params = add["parameters"]
        assert params["a"]["type"] == "integer"
        assert "default" not in params["a"]
        assert "default" not in params["b"]
        assert params["note"]["default"] is None

        parsed = parse_namespaced_tool_name("mcp::srv::add")
        assert parsed == ("srv", "add")


def test_mcp_tools_call() -> None:
    from abstractcore.mcp import McpClient

    transport = httpx.WSGITransport(app=_mcp_wsgi_app)
    with httpx.Client(transport=transport, base_url="http://mcp.test") as http_client:
        client = McpClient(url="http://mcp.test/", client=http_client)
        result = client.call_tool(name="add", arguments={"a": 2, "b": 3})
        assert result.get("isError") is False
        content = result.get("content")
        assert isinstance(content, list)
        assert content and content[0].get("type") == "text"
        assert content[0].get("text") == "5"


def test_mcp_error_id_null_raises_rpc_error() -> None:
    from abstractcore.mcp.client import McpClient, McpRpcError

    def _error_app(environ: Dict[str, Any], start_response) -> List[bytes]:
        start_response("200 OK", [("Content-Type", "application/json")])
        return [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": "Bad Request: Unsupported protocol version"},
                    "id": None,
                }
            ).encode("utf-8")
        ]

    transport = httpx.WSGITransport(app=_error_app)
    with httpx.Client(transport=transport, base_url="http://mcp.test") as http_client:
        client = McpClient(url="http://mcp.test/", client=http_client, protocol_version="2025-11-25")
        with pytest.raises(McpRpcError):
            client.list_tools()
