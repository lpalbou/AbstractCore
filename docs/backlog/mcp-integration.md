# MCP Integration Proposal for AbstractCore

**Status**: 📋 Proposal  
**Priority**: High  
**Complexity**: Moderate (2-3 weeks)  
**Author**: System Analysis  
**Date**: October 2025  

---

## Executive Summary

This proposal outlines a clean, simple, and efficient implementation of Model Context Protocol (MCP) capabilities within AbstractCore. The integration leverages AbstractCore's existing modular architecture to provide seamless access to MCP servers and tools while maintaining the library's design principles of simplicity and universality.

**Key Benefits:**
- Standardized access to external tools and data sources
- Dynamic tool discovery and registration
- Seamless integration with existing AbstractCore patterns
- Future-proof architecture aligned with industry standards

---

## Understanding MCP

### What is MCP?

The Model Context Protocol (MCP) is an open standard introduced by Anthropic in November 2024 that standardizes how AI systems connect with external resources. It addresses the "N×M integration problem" by providing a unified protocol for AI models to access:

- **Tools**: Functions that perform actions (database queries, API calls, file operations)
- **Resources**: Structured data sources (documents, databases, APIs)  
- **Prompts**: Predefined templates for consistent AI behavior

### MCP Architecture

```
┌─────────────────┐    JSON-RPC 2.0    ┌─────────────────┐
│   MCP Client    │ ←──────────────────→ │   MCP Server    │
│ (AbstractCore)  │                     │ (External Tool) │
└─────────────────┘                     └─────────────────┘
```

**Key Components:**
- **MCP Server**: Exposes tools, resources, and prompts via JSON-RPC 2.0
- **MCP Client**: Embedded in AI applications, manages server communication
- **Transport Layer**: stdio, HTTP, WebSocket support

### State-of-the-Art MCP Libraries

**Official Libraries:**
- `@modelcontextprotocol/sdk` (TypeScript/JavaScript) - Official Anthropic SDK
- `mcp-python` - Community Python implementation
- `python-mcp` - Alternative Python client/server library

**Transport Options:**
- **stdio**: Process-based communication (most common)
- **HTTP**: REST-like endpoints with JSON-RPC
- **WebSocket**: Real-time bidirectional communication

---

## AbstractCore Architecture Analysis

### Current Strengths for MCP Integration

AbstractCore's architecture provides excellent foundations for MCP integration:

#### 1. **Universal Tool System**
```python
# Existing tool architecture
@tool
def my_function(param: str) -> str:
    """Tool description"""
    return result

# Tools are registered in ToolRegistry
# Tools are injected into system prompts
# Tools work across all providers (native + prompted)
```

#### 2. **Provider Registry Pattern**
```python
# Existing provider registration
class ProviderInfo:
    name: str
    display_name: str
    provider_class: Type
    supported_features: List[str]
    # ... other metadata

# Dynamic provider loading and discovery
registry.register_provider(provider_info)
```

#### 3. **Configuration System**
```python
# Centralized configuration at ~/.abstractcore/config/abstractcore.json
from abstractcore.config import get_config_manager
config_manager = get_config_manager()
provider, model = config_manager.get_app_default('cli')
```

#### 4. **System Prompt Management**
```python
# Tools are automatically injected into system prompts
enhanced_system_prompt = system_prompt
if tools and self.tool_handler.supports_prompted:
    tool_prompt = self.tool_handler.format_tools_prompt(tools)
    enhanced_system_prompt += f"\n\n{tool_prompt}"
```

### Architecture Alignment

AbstractCore's design patterns align perfectly with MCP concepts:

| AbstractCore Pattern | MCP Equivalent | Integration Point |
|---------------------|----------------|-------------------|
| `ToolDefinition` | MCP Tool Schema | Direct mapping |
| `ToolRegistry` | MCP Tool Discovery | Dynamic registration |
| `BaseProvider` | MCP Client Interface | Inheritance |
| System Prompt Injection | MCP Tool Context | Existing mechanism |
| Provider Registry | MCP Server Registry | Extension |

---

## Proposed Implementation

### Design Philosophy

**Core Principles:**
1. **Leverage Existing Architecture**: Build on AbstractCore's proven patterns
2. **Minimal Disruption**: No breaking changes to existing APIs
3. **Clean Separation**: MCP as a specialized provider type
4. **Configuration-Driven**: Use existing config system for MCP servers
5. **Security-First**: Implement proper authentication and sandboxing

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AbstractCore                             │
├─────────────────────────────────────────────────────────────┤
│  Factory (create_llm)                                       │
│    ↓                                                        │
│  Provider Registry                                          │
│    ├── OpenAIProvider                                       │
│    ├── AnthropicProvider                                    │
│    ├── OllamaProvider                                       │
│    └── MCPProvider ← NEW                                    │
│         ├── MCP Client                                      │
│         ├── Server Registry                                 │
│         └── Tool Bridge                                     │
├─────────────────────────────────────────────────────────────┤
│  Tool System                                                │
│    ├── ToolRegistry (existing)                             │
│    ├── UniversalToolHandler (existing)                     │
│    └── MCP Tool Adapter ← NEW                              │
├─────────────────────────────────────────────────────────────┤
│  Configuration System                                       │
│    └── MCP Server Configuration ← NEW                      │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Strategy

#### Phase 1: Core MCP Provider (Week 1)

**1.1 MCP Provider Class**
```python
# abstractcore/providers/mcp_provider.py
class MCPProvider(BaseProvider):
    """MCP provider for accessing external tools via MCP servers."""
    
    def __init__(self, model: str, **kwargs):
        # model format: "mcp://server-name" or "mcp://localhost:3000"
        super().__init__(model, **kwargs)
        self.server_id = self._parse_server_id(model)
        self.mcp_client = MCPClient(self.server_id)
        self.tools = self._discover_tools()
    
    def _generate_internal(self, prompt, **kwargs):
        # Use discovered tools with existing tool system
        return super()._generate_internal(prompt, tools=self.tools, **kwargs)
```

**Justification**: Inheriting from `BaseProvider` ensures full compatibility with AbstractCore's architecture. The "model" parameter becomes the MCP server identifier, following AbstractCore's pattern of using model names for resource identification.

**1.2 MCP Client Implementation**
```python
# abstractcore/mcp/client.py
class MCPClient:
    """JSON-RPC 2.0 client for MCP servers."""
    
    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self.transport = self._create_transport()
        
    def _create_transport(self):
        if self.config.transport == "stdio":
            return StdioTransport(self.config.command)
        elif self.config.transport == "http":
            return HTTPTransport(self.config.url)
        # ... other transports
    
    async def list_tools(self) -> List[MCPTool]:
        """Discover available tools from MCP server."""
        response = await self.transport.call("tools/list")
        return [MCPTool.from_dict(tool) for tool in response["tools"]]
```

**Justification**: Separate client class follows single responsibility principle. Supporting multiple transports (stdio, HTTP) aligns with MCP specification and provides flexibility.

**1.3 Tool Bridge**
```python
# abstractcore/mcp/tool_bridge.py
class MCPToolBridge:
    """Converts MCP tools to AbstractCore ToolDefinitions."""
    
    @staticmethod
    def convert_mcp_tool(mcp_tool: MCPTool) -> ToolDefinition:
        """Convert MCP tool schema to AbstractCore ToolDefinition."""
        return ToolDefinition(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=mcp_tool.inputSchema,
            function=lambda **kwargs: MCPToolBridge._execute_mcp_tool(
                mcp_tool, kwargs
            )
        )
```

**Justification**: Bridge pattern isolates MCP-specific logic while leveraging AbstractCore's existing tool system. This ensures MCP tools work identically to native tools.

#### Phase 2: Configuration Integration (Week 2)

**2.1 MCP Server Configuration**
```python
# abstractcore/mcp/config.py
@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: Literal["stdio", "http", "websocket"]
    command: Optional[List[str]] = None  # For stdio
    url: Optional[str] = None  # For HTTP/WebSocket
    env: Optional[Dict[str, str]] = None
    timeout: int = 30
    auth: Optional[Dict[str, Any]] = None
```

**2.2 Configuration File Extension**
```json
// ~/.abstractcore/config/abstractcore.json
{
  "mcp": {
    "servers": {
      "filesystem": {
        "transport": "stdio",
        "command": ["python", "-m", "mcp_filesystem_server"],
        "timeout": 30
      },
      "github": {
        "transport": "http", 
        "url": "http://localhost:3001/mcp",
        "auth": {"type": "bearer", "token": "..."}
      }
    },
    "default_timeout": 30,
    "auto_discover": true
  }
}
```

**2.3 CLI Integration**
```bash
# New CLI commands
abstractcore --add-mcp-server filesystem stdio "python -m mcp_filesystem"
abstractcore --add-mcp-server github http "http://localhost:3001/mcp"
abstractcore --list-mcp-servers
abstractcore --test-mcp-server filesystem
abstractcore --set-app-default cli mcp mcp://filesystem
```

**Justification**: Extends existing configuration system without breaking changes. CLI commands follow AbstractCore's established patterns for consistency.

#### Phase 3: Advanced Features (Week 3)

**3.1 HTTP Server Integration**
```python
# abstractcore/server/app.py - New endpoints
@app.get("/mcp/servers")
async def list_mcp_servers():
    """List configured MCP servers."""
    
@app.post("/mcp/servers/{server_id}/connect")
async def connect_mcp_server(server_id: str):
    """Connect to an MCP server."""
    
@app.get("/mcp/servers/{server_id}/tools")
async def list_mcp_tools(server_id: str):
    """List tools from an MCP server."""
```

**3.2 Dynamic Tool Registration**
```python
# abstractcore/mcp/registry.py
class MCPServerRegistry:
    """Registry for managing MCP server connections."""
    
    def __init__(self):
        self.servers: Dict[str, MCPClient] = {}
        self.tool_registry = get_registry()
    
    async def register_server(self, config: MCPServerConfig):
        """Register and connect to an MCP server."""
        client = MCPClient(config)
        await client.connect()
        
        # Discover and register tools
        mcp_tools = await client.list_tools()
        for mcp_tool in mcp_tools:
            tool_def = MCPToolBridge.convert_mcp_tool(mcp_tool)
            self.tool_registry.register(tool_def)
```

**Justification**: Dynamic registration enables runtime tool discovery, a key MCP advantage. Integration with existing `ToolRegistry` ensures consistency.

### Usage Examples

**Basic Usage:**
```python
from abstractcore import create_llm

# Connect to filesystem MCP server
llm = create_llm("mcp", model="mcp://filesystem")
response = llm.generate("List files in my Documents folder")

# Tools are automatically discovered and available
```

**Configuration-Driven:**
```python
# After configuring via CLI
llm = create_llm("mcp", model="mcp://github")
response = llm.generate("Create an issue about the bug in authentication")
```

**Mixed Usage:**
```python
from abstractcore import BasicSession

# Use MCP tools alongside regular tools
session = BasicSession(
    create_llm("openai", model="gpt-4o"),
    tools=get_mcp_tools("filesystem", "github") + [custom_tool]
)
```

---

## Security Considerations

### Threat Model

MCP introduces several security risks that must be addressed:

1. **Prompt Injection**: Malicious MCP tools could manipulate AI responses
2. **Data Exfiltration**: MCP servers could leak sensitive information
3. **Remote Code Execution**: Malicious servers could execute arbitrary code
4. **Privilege Escalation**: MCP tools might access unauthorized resources

### Security Measures

**1. Server Allowlisting**
```python
# Only allow explicitly configured MCP servers
class MCPSecurityManager:
    def __init__(self):
        self.allowed_servers = self._load_allowed_servers()
    
    def validate_server(self, server_id: str) -> bool:
        return server_id in self.allowed_servers
```

**2. Tool Execution Sandboxing**
```python
# Sandbox MCP tool execution
class MCPToolExecutor:
    def __init__(self, sandbox_config: SandboxConfig):
        self.sandbox = create_sandbox(sandbox_config)
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        return await self.sandbox.execute(tool_call)
```

**3. Authentication & Authorization**
```python
# Support multiple auth methods
class MCPAuthManager:
    def authenticate(self, server_config: MCPServerConfig) -> AuthToken:
        if server_config.auth["type"] == "oauth2":
            return self._oauth2_flow(server_config.auth)
        elif server_config.auth["type"] == "api_key":
            return self._api_key_auth(server_config.auth)
```

**4. Request/Response Validation**
```python
# Validate all MCP communications
class MCPValidator:
    def validate_request(self, request: MCPRequest) -> bool:
        # Validate JSON-RPC format, parameter types, etc.
        
    def validate_response(self, response: MCPResponse) -> bool:
        # Validate response format, sanitize content
```

**Justification**: Layered security approach addresses multiple threat vectors. Sandboxing prevents privilege escalation, while validation prevents injection attacks.

---

## Implementation Complexity Analysis

### Difficulty Assessment: **MODERATE**

**Easy Components (1-2 days each):**
- MCP Provider skeleton inheriting from BaseProvider
- Basic JSON-RPC 2.0 client implementation  
- Tool conversion (MCP → AbstractCore ToolDefinition)
- Configuration file extension

**Medium Components (3-5 days each):**
- Multi-transport support (stdio, HTTP, WebSocket)
- Dynamic tool discovery and registration
- Error handling and retry logic
- CLI command integration

**Hard Components (5-7 days each):**
- Security implementation (sandboxing, auth, validation)
- Streaming support for real-time MCP communication
- Advanced features (resource management, caching)
- Comprehensive testing and documentation

**Total Estimate: 2-3 weeks**

### Risk Mitigation

**Technical Risks:**
- **MCP Protocol Changes**: Use official libraries where possible
- **Performance Impact**: Implement connection pooling and caching
- **Security Vulnerabilities**: Comprehensive security review and testing

**Integration Risks:**
- **Breaking Changes**: Maintain backward compatibility
- **Configuration Complexity**: Provide sensible defaults and validation
- **Tool Conflicts**: Implement namespace management

---

## Benefits and Justification

### Strategic Benefits

**1. Industry Alignment**
- MCP is becoming the standard for AI-tool integration
- Early adoption provides competitive advantage
- Future-proofs AbstractCore's architecture

**2. Ecosystem Expansion**
- Access to growing ecosystem of MCP servers
- Enables integration with popular tools (GitHub, Slack, databases)
- Reduces custom integration development

**3. Developer Experience**
- Standardized tool discovery and usage
- Consistent API across different external services
- Simplified configuration and management

### Technical Benefits

**1. Architectural Consistency**
- Leverages existing AbstractCore patterns
- Maintains clean separation of concerns
- No breaking changes to existing APIs

**2. Scalability**
- Dynamic tool registration supports growth
- Connection pooling handles multiple servers
- Caching improves performance

**3. Maintainability**
- Clear abstraction boundaries
- Comprehensive error handling
- Extensive testing and documentation

---

## Alternative Approaches Considered

### 1. MCP as Middleware Layer

**Approach**: Create MCP client as middleware between AbstractCore and MCP servers.

**Pros**: 
- Clean separation of concerns
- Reusable across providers
- Easier to test in isolation

**Cons**:
- Additional complexity layer
- Potential performance overhead
- Less integrated with AbstractCore patterns

**Decision**: Rejected in favor of provider approach for better integration.

### 2. MCP Tools as Prompted Tools Only

**Approach**: Fetch MCP tools at startup and inject as prompted tools.

**Pros**:
- Minimal code changes
- Leverages existing architecture
- Fast implementation

**Cons**:
- No real-time MCP communication
- Static tool discovery only
- Misses key MCP benefits

**Decision**: Considered as Phase 1 fallback, but full provider approach preferred.

### 3. Extend Existing Providers

**Approach**: Add MCP support to each existing provider.

**Pros**:
- No new provider needed
- Direct integration

**Cons**:
- Code duplication across providers
- Violates single responsibility principle
- Harder to maintain

**Decision**: Rejected in favor of dedicated MCP provider.

---

## Success Metrics

### Technical Metrics
- **Integration Time**: < 3 weeks implementation
- **Performance Impact**: < 10% overhead for MCP operations
- **Test Coverage**: > 90% for MCP-specific code
- **Security Audit**: Pass external security review

### User Experience Metrics
- **Configuration Time**: < 5 minutes to add MCP server
- **Tool Discovery**: Automatic discovery of 100% of server tools
- **Error Recovery**: Graceful handling of server failures
- **Documentation**: Complete API and usage documentation

### Adoption Metrics
- **Community Usage**: Track MCP server configurations
- **Issue Reports**: Monitor MCP-related issues
- **Performance**: Monitor MCP operation latency
- **Feedback**: Collect user experience feedback

---

## Conclusion

The proposed MCP integration represents a strategic enhancement to AbstractCore that aligns with industry standards while leveraging the library's existing architectural strengths. The implementation is moderate in complexity but provides significant value through:

1. **Clean Architecture**: Builds on proven AbstractCore patterns
2. **Strategic Positioning**: Positions AbstractCore as MCP-ready
3. **Developer Experience**: Provides seamless access to MCP ecosystem
4. **Future-Proofing**: Enables integration with emerging MCP tools

The phased implementation approach minimizes risk while delivering incremental value. The security-first design addresses MCP's known vulnerabilities, ensuring enterprise-ready deployment.

**Recommendation**: Proceed with implementation using the proposed provider-based approach, prioritizing security and maintaining AbstractCore's design principles of simplicity and universality.

---

## Next Steps

1. **Stakeholder Review**: Review proposal with AbstractCore maintainers
2. **Technical Spike**: Implement basic MCP client prototype
3. **Security Review**: Conduct threat modeling and security design
4. **Implementation Plan**: Create detailed implementation timeline
5. **Community Feedback**: Gather input from AbstractCore community

---

*This proposal represents a comprehensive analysis of MCP integration options for AbstractCore, prioritizing clean architecture, security, and developer experience while maintaining the library's core design principles.*
