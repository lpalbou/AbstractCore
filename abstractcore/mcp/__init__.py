"""
MCP (Model Context Protocol) integration.

This package provides:
- A minimal JSON-RPC client for MCP servers (Streamable HTTP transport).
- Tool discovery via `tools/list`.
- Schema normalization into AbstractCore-compatible tool specs (dicts accepted by `tools=...`).

MCP is treated as a *tool server protocol* (not an LLM provider). Tool execution remains
explicitly host/runtime-owned via a ToolExecutor boundary.
"""

from .client import McpClient
from .factory import create_mcp_client
from .naming import MCP_TOOL_PREFIX, namespaced_tool_name, parse_namespaced_tool_name
from .stdio_client import McpStdioClient, McpStdioServerParameters
from .tool_source import McpServerInfo, McpToolSource, mcp_tool_to_abstractcore_tool_spec

__all__ = [
    "McpClient",
    "McpStdioClient",
    "McpStdioServerParameters",
    "McpServerInfo",
    "McpToolSource",
    "MCP_TOOL_PREFIX",
    "create_mcp_client",
    "namespaced_tool_name",
    "parse_namespaced_tool_name",
    "mcp_tool_to_abstractcore_tool_spec",
]
