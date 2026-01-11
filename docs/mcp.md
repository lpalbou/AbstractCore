# MCP (Model Context Protocol)

AbstractCore treats MCP as a **tool-server protocol** (not an LLM provider).

The `abstractcore.mcp` module provides:
- a minimal MCP JSON-RPC client (Streamable HTTP) → `abstractcore.mcp.McpClient`
- a minimal MCP stdio client (spawn a subprocess) → `abstractcore.mcp.McpStdioClient`
- tool discovery (`tools/list`) and conversion into AbstractCore tool specs → `abstractcore.mcp.McpToolSource`

## What you can do today

### 1) Discover tools from an MCP server

```python
from abstractcore.mcp import McpClient, McpToolSource

client = McpClient(url="http://localhost:3000")  # MCP streamable HTTP endpoint
source = McpToolSource(server_id="local", client=client)
tools = source.list_tool_specs()
```

Each returned tool spec is an AbstractCore-compatible dict you can pass to `tools=[...]` in
`generate()`/`agenerate()`. Tool names are namespaced as:

`mcp::<server_id>::<tool_name>`

See `abstractcore/abstractcore/mcp/naming.py`.

### 2) Execute MCP tools in your host/runtime

AbstractCore’s default execution path is **passthrough** (`execute_tools=False`): the model can
request tool calls and you execute them in your host/runtime.

The built-in `abstractcore.tools.registry.execute_tools()` executes Python callables registered in
the (deprecated) global registry; it does **not** automatically route MCP tool calls. For MCP, your
host/runtime should detect names starting with `mcp::` and dispatch them to an MCP client.

```python
from abstractcore.mcp import McpClient, parse_namespaced_tool_name

client = McpClient(url="http://localhost:3000")

def execute_mcp_tool_call(call: dict) -> dict:
    parsed = parse_namespaced_tool_name(call.get("name", ""))
    if not parsed:
        raise ValueError("Not an MCP tool call")
    server_id, tool_name = parsed
    return client.call_tool(name=tool_name, arguments=call.get("arguments") or {})
```

## Transports supported

### Streamable HTTP

`McpClient` posts JSON-RPC to the server URL. It automatically sets an `Accept` header compatible
with streamable HTTP (`application/json, text/event-stream`) and will capture `MCP-Session-Id`
responses when provided.

See `abstractcore/abstractcore/mcp/client.py`.

### stdio

`McpStdioClient` spawns an MCP server subprocess and communicates over stdin/stdout with JSON-RPC,
including a best-effort initialization handshake.

See `abstractcore/abstractcore/mcp/stdio_client.py`.

## Configuration helpers

`create_mcp_client(config=...)` supports both HTTP and stdio config shapes:

```python
from abstractcore.mcp import create_mcp_client

client = create_mcp_client(config={"url": "http://localhost:3000"})
client = create_mcp_client(config={"transport": "stdio", "command": ["my-mcp-server", "--stdio"]})
```

See `abstractcore/abstractcore/mcp/factory.py`.

## Current limitations

- MCP is currently a **library-level** integration (tool discovery + clients). AbstractCore’s HTTP
  server does not expose MCP management endpoints.
- Tool execution routing for `mcp::...` names is host/runtime responsibility.

