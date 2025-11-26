# 008: MCP Integration for AbstractCore

**Status**: Proposed (Backlog)
**Priority**: P2 - Medium
**Effort**: Medium (12-20 hours)
**Dependency**: Requires `002-async-await-support.md` to be completed first
**Target Version**: 2.7.0 (after async support in 2.6.0)

---

## Executive Summary

Enable AbstractCore to connect to MCP (Model Context Protocol) servers developed by Anthropic, using a clean "tool source" pattern that integrates with AbstractCore's existing tool system. MCP servers provide external tools (filesystem, GitHub, databases) that work with ANY AbstractCore provider.

**Key Benefits:**
- Access to growing ecosystem of MCP servers
- Clean integration via existing tool system
- Works with all 6 providers (OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace)
- Simple API: < 5 lines to connect and use MCP tools

---

## Dependency Note

**MCP depends on async support** (002-async-await-support.md) because:
1. Official MCP SDK (`mcp` package by Anthropic) is async-native
2. MCP server connections involve network I/O (stdio, HTTP)
3. Tool execution via MCP servers is async (JSON-RPC requests)
4. Implementing sync wrappers would be awkward and error-prone

**Recommendation**: Implement async support first (v2.6.0), then MCP integration (v2.7.0).

---

## Critical Insight: Simpler is Better

A previous proposal suggested a full "MCPProvider" class that treats MCP servers as "models". This is **over-engineered** because:

1. **MCP servers provide tools, not LLM capabilities** - they don't generate text
2. **Forcing MCP into the Provider abstraction** requires a confusing "underlying_llm" concept
3. **AbstractCore already has excellent tool injection** - we just need MCP as a tool source

**Simpler Approach**: MCP as a **tool source** that works with ANY existing provider.

---

## Recommended Design

### Usage Pattern (Target API)

```python
from abstractcore import create_llm
from abstractcore.mcp import MCPServer

# Connect to MCP servers (async context manager)
async with MCPServer.stdio(command=["npx", "mcp-server-filesystem", "/tmp"]) as filesystem:
    async with MCPServer.http(url="http://localhost:3000/mcp") as github:

        # Get tools from MCP servers (auto-converted to ToolDefinition)
        mcp_tools = filesystem.tools + github.tools

        # Use with ANY provider - no special "MCP provider" needed
        llm = create_llm("openai", model="gpt-4o")
        response = await llm.generate(
            "List files in /tmp and create a GitHub issue",
            tools=mcp_tools
        )
```

### Alternative: Session Integration

```python
from abstractcore.core.session import BasicSession
from abstractcore.mcp import MCPServer

async with MCPServer.stdio(command=["npx", "mcp-server-filesystem"]) as fs:
    session = BasicSession(
        provider=create_llm("anthropic", model="claude-3-5-sonnet"),
        tools=fs.tools  # MCP tools injected alongside regular tools
    )
    response = await session.generate("List all Python files")
```

---

## Implementation Architecture

### File Structure

```
abstractcore/
├── mcp/                         # NEW module
│   ├── __init__.py              # Exports: MCPServer
│   ├── server.py                # MCPServer class (wrapper around official SDK)
│   ├── tool_adapter.py          # Convert MCP tools → ToolDefinition
│   └── config.py                # MCPServerConfig dataclass
└── tools/
    └── core.py                  # Existing (no changes needed)
```

### Core Components

#### 1. MCPServer Class (`abstractcore/mcp/server.py`)

Wrapper around official Anthropic `mcp` Python SDK. Provides:
- Async context manager for server lifecycle (`async with`)
- Factory methods for stdio and HTTP transports
- Tool discovery via `list_tools()`
- Automatic conversion to AbstractCore `ToolDefinition`

```python
class MCPServer:
    """MCP Server connection wrapper using official mcp SDK."""

    @classmethod
    def stdio(cls, command: List[str], name: str = "stdio", **kwargs) -> "MCPServer":
        """Factory for stdio transport (local servers)."""

    @classmethod
    def http(cls, url: str, name: str = "http", **kwargs) -> "MCPServer":
        """Factory for HTTP transport (remote servers)."""

    async def connect(self):
        """Establish connection and discover tools."""

    @property
    def tools(self) -> List[ToolDefinition]:
        """Get discovered tools as AbstractCore ToolDefinitions."""
```

#### 2. Tool Adapter (`abstractcore/mcp/tool_adapter.py`)

Converts MCP tool schemas to AbstractCore `ToolDefinition`:
- Maps MCP `inputSchema` → AbstractCore `parameters`
- Creates async executor wrapper for each tool
- Adds `mcp_` prefix to avoid name conflicts
- Populates metadata (description, tags, when_to_use)

---

## Implementation Plan

### Phase 1: Core MCP Module (4-6 hours)

1. Create `abstractcore/mcp/` module
2. Implement `MCPServer` class with stdio transport
3. Implement tool adapter
4. Add `mcp>=1.0.0` as optional dependency

### Phase 2: Tool Integration (2-4 hours)

1. Test MCP tools with existing providers
2. Handle async tool execution
3. Add tool caching

### Phase 3: HTTP/SSE Transport (2-3 hours)

1. Implement HTTP transport
2. Add authentication/headers support
3. Test with remote servers

### Phase 4: Configuration & CLI (2-3 hours)

1. Add MCP config to `abstractcore.json`
2. CLI commands: `--add-mcp-server`, `--list-mcp-servers`, `--test-mcp-server`

### Phase 5: Testing (3-4 hours)

1. Unit tests for tool adapter
2. Integration tests with `@modelcontextprotocol/server-filesystem`
3. Multi-provider testing

**Total Estimated Time**: 13-19 hours

---

## Files to Modify

| File | Action | Purpose |
|------|--------|---------|
| `abstractcore/mcp/__init__.py` | CREATE | Module exports |
| `abstractcore/mcp/server.py` | CREATE | MCPServer class |
| `abstractcore/mcp/tool_adapter.py` | CREATE | Tool conversion |
| `abstractcore/mcp/config.py` | CREATE | Configuration dataclass |
| `pyproject.toml` | MODIFY | Add `mcp` optional dependency |
| `abstractcore/config/manager.py` | MODIFY | Add MCP config support |
| `abstractcore/cli/main.py` | MODIFY | Add MCP CLI commands |
| `tests/mcp/` | CREATE | Test suite |

---

## Key Design Decisions

### 1. Use Official MCP SDK as Dependency

**Rationale**: The `mcp` package on PyPI (Anthropic's official SDK) implements JSON-RPC 2.0, transport handling, and protocol details. Reinventing this adds complexity and maintenance burden.

### 2. MCP as Tool Source, Not Provider

**Rationale**: MCP servers provide tools, not LLM capabilities. Forcing MCP into the Provider abstraction requires awkward "underlying_llm" concepts. The simpler approach works with ANY provider.

### 3. Async Context Manager Pattern

**Rationale**: MCP connections have lifecycle (connect/disconnect). The `async with` pattern from the official SDK ensures proper cleanup and is familiar to Python developers.

### 4. Tool Name Prefixing (`mcp_`)

**Rationale**: Prevents conflicts between MCP tools and local tools with the same name. Makes tool origin clear in logs and traces.

### 5. Optional Dependency

**Rationale**: Not all AbstractCore users need MCP. Keep core package lightweight, add `pip install abstractcore[mcp]` for those who need it.

---

## What This Approach Avoids

1. **No MCPProvider class** - MCP doesn't fit the Provider abstraction
2. **No "mcp://server" model syntax** - confusing and unnecessary
3. **No JSON-RPC implementation** - use official SDK
4. **No complex streaming architecture** - MCP tool calls are simple request/response
5. **No "underlying_llm" concept** - use MCP tools with any provider directly

---

## Success Criteria

1. Connect to MCP server in < 5 lines of code
2. MCP tools work with all 6 existing providers
3. Zero breaking changes to existing APIs
4. < 500 lines of new code total
5. Full test coverage with real MCP servers

---

## References

- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Anthropic's official implementation
- [MCP Protocol Specification](https://modelcontextprotocol.io/) - Anthropic's protocol spec
- [OpenAI Agents SDK MCP Support](https://openai.github.io/openai-agents-python/mcp/) - Example of framework consuming MCP

---

**Document Version**: 1.0
**Created**: 2025-11-25
**Author**: Senior Architect Review
**Status**: Ready for Implementation (after async support)
