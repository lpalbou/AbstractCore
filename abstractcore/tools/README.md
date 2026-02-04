# AbstractCore Tools Module

## Purpose

The `abstractcore/tools/` module provides a **universal tool calling system** that enables Large Language Models to interact with external functions and APIs across all providers. It abstracts away provider-specific implementations and offers a unified interface for tool definition, registration, parsing, execution, and format transformation.

This module is the cornerstone of AbstractCore's **provider-agnostic architecture**, allowing models that lack native tool support to use tools through intelligent prompting, while seamlessly leveraging native tool APIs when available.

## Installation

- Tool parsing/rewriting is part of core `abstractcore`.
- The built-in toolset in `abstractcore.tools.common_tools` (notably `fetch_url` and `web_search`) requires `pip install "abstractcore[tools]"` so runtime dependencies like BeautifulSoup/requests are available.

## Quick Reference

### Tool Format Quick Reference

| Format | Syntax | Used By | Auto-Detect |
|--------|--------|---------|-------------|
| Native | OpenAI JSON | OpenAI, Anthropic | ✅ Provider API |
| Special Token | `<\|tool_call\|>...`| Qwen3, GLM-4.5+ | ✅ Model name |
| Function Call | `<function_call>...` | LLaMA3 | ✅ Model name |
| XML | `<tool_call>...` | Claude (prompted), Mixtral | ✅ Content pattern |
| Code Block | ` ```tool_code...``` ` | Gemma, CodeLlama | ✅ Content pattern |

### Tool Type Selection

| Need | Use | Example |
|------|-----|---------|
| File operations | `list_files`, `read_file`, `write_file`, `edit_file`, `search_files` | List Python files, read config, update code |
| Web operations | `web_search`, `fetch_url` | Search DuckDuckGo, fetch API data |
| System commands | `execute_command` | Run tests, git operations |
| Custom logic | `@tool` decorator | Business rules, API integrations |

## Common Tasks

- **How do I define a tool?** → See [Defining Tools](#defining-tools)
- **How do I register a tool?** → See [Registering Tools](#registering-tools)
- **How do I execute tools?** → See [Executing Tools](#executing-tools)
- **How do I use tools with an LLM?** → See [Using with LLM](#using-with-llm)
- **How do I convert tool formats?** → See [Syntax Rewriting](#3-syntax_rewriterpy---format-transformation)
- **How do I handle tool errors?** → See [Error Handling](#2-validation-in-tools)
- **What built-in tools exist?** → See [Common Tools](#7-common_toolspy---built-in-tools)
- **How do I parse tool calls?** → See [Parser](#2-parserpy---tool-call-extraction)

## Architecture Position

**Layer**: Core Infrastructure
**Dependencies**:
- `abstractcore/architectures/` - Model architecture detection and capabilities
- `abstractcore/events/` - Event emission for tool execution lifecycle
- Standard library: `json`, `re`, `logging`, `dataclasses`, `typing`

**Used By**:
- All provider implementations (`abstractcore/providers/`)
- Session management (`abstractcore/session/`)
- Agent implementations
- User applications requiring tool calling

---

## Component Structure

The module consists of 7 specialized files:

```
abstractcore/tools/
├── core.py                  # Core data structures (ToolDefinition, ToolCall, ToolResult)
├── parser.py                # Tool call detection and parsing from LLM responses
├── syntax_rewriter.py       # Universal syntax transformation between formats
├── tag_rewriter.py          # Real-time tag rewriting for streaming scenarios
├── registry.py              # Global tool registry and execution management
├── handler.py               # Orchestration layer (UniversalToolHandler)
└── common_tools.py          # Built-in utility tools (file ops, web, commands)
```

---

## Detailed Components

### 1. core.py - Data Structures

**Purpose**: Defines the fundamental data classes for the tool system.

**Key Classes**:

```python
@dataclass
class ToolDefinition:
    """Complete specification of a tool"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None

    # Enhanced metadata
    tags: List[str] = field(default_factory=list)
    when_to_use: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_function(cls, func: Callable) -> 'ToolDefinition':
        """Auto-generate tool definition from function signature"""

@dataclass
class ToolCall:
    """Represents a tool call request from LLM"""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None

@dataclass
class ToolResult:
    """Result of tool execution"""
    call_id: str
    output: Any
    error: Optional[str] = None
    success: bool = True

@dataclass
class ToolCallResponse:
    """Combined response with content and tool calls"""
    content: str
    tool_calls: List[ToolCall]
    raw_response: Any = None

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
```

**Decorator**:
```python
@tool(
    name="custom_name",
    description="What this tool does",
    tags=["category", "type"],
    when_to_use="When you need to...",
    examples=[
        {
            "description": "Basic usage",
            "arguments": {"param": "value"}
        }
    ]
)
def my_function(param: str) -> str:
    """Function docstring"""
    return result
```

The `@tool` decorator automatically:
- Extracts function signature and type hints
- Converts to ToolDefinition
- Attaches metadata for rich LLM guidance
- Makes function directly usable with `llm.generate(tools=[my_function])`

---

### 2. parser.py - Tool Call Extraction

**Purpose**: Architecture-aware detection and parsing of tool calls from model responses.

**Key Features**:
- Automatic format detection based on model architecture
- Multiple parser strategies (special tokens, XML, JSON, code blocks)
- Robust fallback parsing with error recovery
- Tool call syntax cleaning

**Tool Formats Supported**:
- SPECIAL_TOKEN: `<|tool_call|>...JSON...</|tool_call|>` (Qwen3, GLM-4.5+)
- FUNCTION_CALL: `<function_call>...JSON...</function_call>` (LLaMA3)
- XML_WRAPPED: `<tool_call>...JSON...</tool_call>` (XML-based)
- TOOL_CODE: ` ```tool_code\n...JSON...``` ` (Gemma)
- RAW_JSON: Plain JSON objects
- NATIVE: API-level tool calls (OpenAI, Anthropic)

**Key Functions**: `detect_tool_calls()`, `parse_tool_calls()`, `format_tool_prompt()`, `clean_tool_syntax()`

→ See [architectures/](../architectures/README.md) for architecture-specific tool formats and detection patterns

---

### 3. syntax_rewriter.py - Format Transformation

**Purpose**: Universal syntax conversion for agent and CLI compatibility.

**Key Class**:

```python
class ToolCallSyntaxRewriter:
    """Convert tool calls between different formats"""

    def __init__(
        self,
        target_format: Union[str, SyntaxFormat],
        custom_config: Optional[CustomFormatConfig] = None,
        model_name: Optional[str] = None
    ):
        """Initialize with target format"""

    def rewrite_content(
        self,
        content: str,
        detected_tool_calls: Optional[List[Any]] = None
    ) -> str:
        """Rewrite tool call syntax to target format"""

    def convert_to_openai_format(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Convert to OpenAI API format"""
```

**Supported Target Formats**:
- `PASSTHROUGH` - No changes (for OpenAI models)
- `OPENAI` - Full OpenAI structure
- `CODEX` - OpenAI format optimized for Codex
- `QWEN3` - Qwen3 special token format
- `LLAMA3` - LLaMA3 function call format
- `GEMMA` - Gemma code block format
- `XML` - XML-wrapped format
- `CUSTOM` - User-defined format with template

**Custom Format Configuration**:
```python
@dataclass
class CustomFormatConfig:
    start_tag: str
    end_tag: str
    json_wrapper: bool = True
    add_ids: bool = False
    format_template: Optional[str] = None
```

**Example**:
```python
# Convert Qwen3 format to OpenAI format
rewriter = ToolCallSyntaxRewriter(SyntaxFormat.OPENAI, model_name="qwen3:4b")

qwen_response = """
<|tool_call|>
{"name": "search", "arguments": {"query": "test"}}
</|tool_call|>
"""

cleaned = rewriter.rewrite_content(qwen_response)
# Returns: Clean content without tool call syntax
# Tool calls accessible via: rewriter.convert_to_openai_format(tool_calls)
```

**Convenience Functions**:
```python
create_openai_rewriter(model_name: Optional[str] = None)
create_codex_rewriter(model_name: Optional[str] = None)
create_passthrough_rewriter()
create_custom_rewriter(start_tag, end_tag, ...)
auto_detect_format(model, user_agent, custom_headers)
```

---

### 4. tag_rewriter.py - Real-Time Tag Customization

**Purpose**: Real-time tool call tag rewriting for streaming scenarios and agentic CLIs.

**Key Class**:

```python
class ToolCallTagRewriter:
    """Real-time tag rewriter for streaming"""

    def __init__(self, target_tags: ToolCallTags):
        """Initialize with target tag configuration"""

    def rewrite_text(self, text: str) -> str:
        """Rewrite tool call tags in complete text"""

    def rewrite_streaming_chunk(self, chunk: str, buffer: str = "") -> Tuple[str, str]:
        """Rewrite tags in streaming chunks with buffer management"""

    def is_tool_call(self, text: str) -> bool:
        """Check if text contains a tool call"""

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract all tool calls from text"""
```

**Tag Configuration**:
```python
@dataclass
class ToolCallTags:
    start_tag: str
    end_tag: str
    preserve_json: bool = True
    auto_format: bool = True  # Automatically add angle brackets
```

**Predefined Configurations**:
```python
PREDEFINED_TAGS = {
    "qwen3": ToolCallTags("<|tool_call|>", "</|tool_call|>"),
    "llama3": ToolCallTags("<function_call>", "</function_call>"),
    "xml": ToolCallTags("<tool_call>", "</tool_call>"),
    "gemma": ToolCallTags("```tool_code", "```"),
    "codex": ToolCallTags("<|tool_call|>", "</|tool_call|>"),
    "openai": ToolCallTags("<|tool_call|>", "</|tool_call|>"),
}
```

**Streaming Strategy**:
The rewriter uses an **immediate rewriting strategy**:
1. Rewrite start tags immediately when detected
2. Buffer content until end tag is found
3. Rewrite end tag when complete tool call detected
4. Minimizes latency for agency loops
5. Avoids double-tagging with intelligent detection

**Example**:
```python
# Create rewriter for custom CLI
rewriter = ToolCallTagRewriter(
    ToolCallTags(start_tag="<TOOL>", end_tag="</TOOL>")
)

# Rewrite complete text
text = "<|tool_call|>{'name': 'search'}|</|tool_call|>"
rewritten = rewriter.rewrite_text(text)
# Returns: "<TOOL>{'name': 'search'}</TOOL>"

# Streaming with buffer management
chunk1 = "I'll search for that. <|tool_"
rewritten1, buffer = rewriter.rewrite_streaming_chunk(chunk1, "")
# Returns: ("", "<|tool_") - buffering incomplete tool call

chunk2 = "call|>{'name': 'search'}|</|tool_call|>"
rewritten2, buffer = rewriter.rewrite_streaming_chunk(chunk2, buffer)
# Returns: ("I'll search for that. <TOOL>{'name': 'search'}</TOOL>", "")
```

---

### 5. registry.py - Tool Management

**Purpose**: Centralized tool registration, discovery, and execution with event emission.

**Key Class**:

```python
class ToolRegistry:
    """Registry for managing available tools"""

    def register(self, tool: Union[ToolDefinition, Callable]) -> ToolDefinition:
        """Register a tool (accepts ToolDefinition or function)"""

    def unregister(self, name: str) -> bool:
        """Remove a tool from registry"""

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name"""

    def list_tools(self) -> List[ToolDefinition]:
        """Get all registered tools"""

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call with error handling"""

    def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls"""
```

**Global Registry Functions**:
```python
def get_registry() -> ToolRegistry:
    """Access global tool registry"""

def register_tool(tool: Union[ToolDefinition, Callable]) -> ToolDefinition:
    """Register tool in global registry"""

def execute_tool(tool_call: ToolCall) -> ToolResult:
    """Execute tool using global registry"""

def clear_registry():
    """Clear all tools (useful for testing)"""
```

Note: Global tool registration is **deprecated**. Prefer passing tools explicitly to `generate(...)`
and executing tool calls in your host/runtime via a `ToolExecutor`. If you still need legacy
global registration (e.g. for older code paths), use `register_tool(...)`.

**Event Emission**:
The registry emits events for monitoring:
- `BEFORE_TOOL_EXECUTION` - Before tool execution starts
- `TOOL_COMPLETED` - After tool execution (success or failure)
- Includes tool name, arguments, duration, result/error

**Example**:
```python
# Register a function as a tool (legacy global registry; deprecated)
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

register_tool(calculate_sum)

# Execute tool call
tool_call = ToolCall(
    name="calculate_sum",
    arguments={"a": 5, "b": 3}
)

result = execute_tool(tool_call)
print(result.output)  # 8
print(result.success)  # True

# List all tools
registry = get_registry()
print(registry.get_tool_names())  # ["calculate_sum", ...]
```

---

### 6. handler.py - Execution Orchestration

**Purpose**: Universal tool handler that works across all models and providers.

**Key Class**:

```python
class UniversalToolHandler:
    """Universal tool handler for all models"""

    def __init__(self, model_name: str):
        """Initialize with model-specific capabilities"""

    def format_tools_prompt(
        self,
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]]
    ) -> str:
        """Format tools for prompted models (architecture-specific)"""

    def prepare_tools_for_native(
        self,
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Convert tools to native API format (OpenAI-compatible)"""

    def parse_response(
        self,
        response: Union[str, Dict[str, Any]],
        mode: str = "prompted"
    ) -> ToolCallResponse:
        """Parse model response for tool calls"""
```

**Automatic Capability Detection**:
```python
handler = UniversalToolHandler("qwen3:4b")

print(handler.supports_native)    # False (no native API)
print(handler.supports_prompted)  # True (prompted tool support)
print(handler.architecture)       # "qwen3"
```

**Tool Format Conversion**:
The handler automatically converts tools to the appropriate format:

```python
# Define tools in any format
tools = [
    calculate_sum,  # Function with @tool decorator
    ToolDefinition(...),  # Explicit ToolDefinition
    {"name": "search", "description": "...", "parameters": {...}}  # Dict
]

# For prompted models
prompt = handler.format_tools_prompt(tools)
# Returns: Architecture-specific tool prompt (Qwen3, LLaMA3, etc.)

# For native API models
native_tools = handler.prepare_tools_for_native(tools)
# Returns: OpenAI-compatible tool definitions
```

**Response Parsing**:
```python
# Parse prompted response
response = """
<|tool_call|>
{"name": "search", "arguments": {"query": "test"}}
</|tool_call|>
"""

parsed = handler.parse_response(response, mode="prompted")
print(parsed.content)      # Clean content without tool syntax
print(parsed.tool_calls)   # [ToolCall(name="search", ...)]

# Parse native API response
native_response = {
    "content": "Here are the results",
    "tool_calls": [
        {
            "id": "call_123",
            "function": {
                "name": "search",
                "arguments": '{"query": "test"}'
            }
        }
    ]
}

parsed = handler.parse_response(native_response, mode="native")
print(parsed.tool_calls)  # [ToolCall(name="search", arguments={"query": "test"})]
```

**Convenience Function**:
```python
def create_handler(model_name: str) -> UniversalToolHandler:
    """Create handler for specific model"""
```

---

### 7. common_tools.py - Built-in Tools

**Purpose**: Production-ready utility tools for common operations.

**Available Tools**:

#### File Operations

```python
@tool
def list_files(
    directory_path: str = ".",
    pattern: str = "*",
    recursive: bool = False,
    include_hidden: bool = False,
    head_limit: Optional[int] = 50
) -> str:
    """List files with pattern matching (case-insensitive)"""

@tool
def search_files(
    pattern: str,
    path: str = ".",
    output_mode: str = "content",
    head_limit: Optional[int] = 20,
    file_pattern: str = "*",
    case_sensitive: bool = False,
    multiline: bool = False
) -> str:
    """Search file contents using regex"""

@tool
def read_file(
    file_path: str,
    start_line: int = 1,
    end_line: Optional[int] = None
) -> str:
    """Read file contents with optional line range"""

@tool
def write_file(
    file_path: str,
    content: str,
    mode: str = "w"
) -> str:
    """Write content to file with directory creation"""

@tool
def edit_file(
    file_path: str,
    pattern: str,
    replacement: str,
    use_regex: bool = False,
    max_replacements: int = -1,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    preview_only: bool = False
) -> str:
    """Edit files using pattern matching"""
```

#### Web Operations

```python
@tool
def web_search(
    query: str,
    num_results: int = 5,
    safe_search: str = "moderate",
    region: str = "us-en",
    time_range: Optional[str] = None
) -> str:
    """Search web using DuckDuckGo (no API key required) and return JSON results"""

@tool
def fetch_url(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Union[Dict[str, Any], str]] = None,
    timeout: int = 45,
    include_binary_preview: bool = False,
    keep_links: bool = True,
    user_agent: str = "AbstractCore-FetchTool/1.0",
    include_full_content: bool = True
) -> str:
    """Fetch and parse content from URLs"""
```

#### Command Execution

```python
@tool
def execute_command(
    command: str,
    working_directory: str = None,
    timeout: int = 300,
    capture_output: bool = True,
    require_confirmation: bool = False,
    allow_dangerous: bool = False
) -> str:
    """Execute shell commands with security controls"""
```

**Security Features**:
- Pattern-based blocking of destructive commands
- System-critical path protection
- Risk assessment and confirmation
- Comprehensive validation with multiple safety layers
- Only bypassed with explicit `allow_dangerous=True`

**Rich Metadata**:
All built-in tools include:
- Detailed descriptions
- Tags for categorization
- `when_to_use` guidance
- Multiple usage examples
- Parameter documentation

---

## Tool Call Formats

AbstractCore supports 5 major tool call formats:

### 1. Native Format (API-Level)

**Used By**: OpenAI, Anthropic, Gemini (via their APIs)

**Format**: Provider-specific API structures

**OpenAI Example**:
```python
{
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}'
            }
        }
    ]
}
```

**Advantages**:
- Most reliable (server-side enforcement)
- Structured and validated
- Includes call IDs for tracking

### 2. Special Token Format

**Used By**: Qwen3, Qwen2.5, GLM-4.5+

**Format**:
```
<|tool_call|>
{"name": "function_name", "arguments": {"param": "value"}}
</|tool_call|>
```

**Parsing Strategy**:
- Robust multi-pattern detection
- Handles malformed tags
- Supports multiple field names (`name`, `command`, `function`)
- Automatic JSON repair for common errors

### 3. JSON Format (Function Call)

**Used By**: LLaMA 3, LLaMA 3.1, LLaMA 3.2

**Format**:
```
<function_call>
{"name": "function_name", "arguments": {"param": "value"}}
</function_call>
```

**Parsing Strategy**:
- Standard XML-style tag parsing
- JSON extraction and validation
- Field normalization

### 4. XML Format

**Used By**: Claude (prompted), Mixtral (prompted)

**Format**:
```xml
<tool_call>
{"name": "function_name", "arguments": {"param": "value"}}
</tool_call>
```

**Parsing Strategy**:
- XML tag detection
- JSON content extraction
- Compatible with nested structures

### 5. Pythonic Format (Code Block)

**Used By**: Gemma, CodeLlama

**Format**:
````
```tool_code
{"name": "function_name", "arguments": {"param": "value"}}
```
````

**Parsing Strategy**:
- Code block detection
- JSON parsing first
- Fallback to Python function call parsing

---

## Usage Patterns

### Defining Tools

**Method 1: Using @tool Decorator**
```python
from abstractcore.tools import tool

@tool(
    description="Add two numbers",
    tags=["math", "arithmetic"],
    when_to_use="When you need to perform addition",
    examples=[
        {
            "description": "Add 5 and 3",
            "arguments": {"a": 5, "b": 3}
        }
    ]
)
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b
```

**Method 2: Explicit ToolDefinition**
```python
from abstractcore.tools import ToolDefinition

tool_def = ToolDefinition(
    name="add",
    description="Add two numbers",
    parameters={
        "a": {"type": "integer", "description": "First number"},
        "b": {"type": "integer", "description": "Second number"}
    },
    function=lambda a, b: a + b
)
```

**Method 3: From Function**
```python
def multiply(x: int, y: int) -> int:
    """Multiply two numbers"""
    return x * y

tool_def = ToolDefinition.from_function(multiply)
```

### Registering Tools

**Manual Registration (legacy global registry; deprecated)**:
```python
from abstractcore.tools import register_tool, get_registry

# Register a function
register_tool(my_function)

# Register a ToolDefinition
register_tool(tool_def)

# Access registry
registry = get_registry()
print(registry.get_tool_names())
```

### Executing Tools

**Direct Execution**:
```python
from abstractcore.tools import execute_tool, ToolCall

tool_call = ToolCall(
    name="add",
    arguments={"a": 5, "b": 3},
    call_id="call_123"
)

result = execute_tool(tool_call)

if result.success:
    print(result.output)  # 8
else:
    print(result.error)
```

**Batch Execution**:
```python
from abstractcore.tools import execute_tools

tool_calls = [
    ToolCall(name="add", arguments={"a": 5, "b": 3}),
    ToolCall(name="multiply", arguments={"x": 4, "y": 2})
]

results = execute_tools(tool_calls)
for result in results:
    print(result.output)
```

### Using with LLM

**Simple Usage**:
```python
from abstractcore import create_llm
from abstractcore.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location"""
    return f"Weather in {location}: Sunny, 72°F"

llm = create_llm("openai", model="gpt-4")

response = llm.generate(
    prompt="What's the weather in San Francisco?",
    tools=[get_weather]
)

print(response.content)
```

**With Tool Execution Loop**:
```python
from abstractcore import create_llm
from abstractcore.tools import tool, execute_tools

@tool
def search_web(query: str) -> str:
    """Search the web"""
    # Implementation
    return results

llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")

# Initial request
response = llm.generate(
    prompt="Find the latest AI news",
    tools=[search_web]
)

# Execute tool calls if present
if response.tool_calls:
    tool_results = execute_tools(response.tool_calls)

    # Continue conversation with results
    final_response = llm.generate(
        prompt="Here are the search results: " + str(tool_results[0].output),
        conversation_history=[...]
    )

    print(final_response.content)
```

---

## Integration Points

### Provider Integration

Providers use the tool system through `UniversalToolHandler`:

```python
from abstractcore.tools import create_handler

class MyProvider(BaseProvider):
    def __init__(self, model: str):
        self.tool_handler = create_handler(model)

    def generate(self, prompt: str, tools: List = None, **kwargs):
        if tools and self.tool_handler.supports_native:
            # Use native API
            api_tools = self.tool_handler.prepare_tools_for_native(tools)
            response = self._call_api(prompt, tools=api_tools)

        elif tools and self.tool_handler.supports_prompted:
            # Use prompted approach
            system_prompt = self.tool_handler.format_tools_prompt(tools)
            response = self._call_api(system_prompt + "\n" + prompt)

        # Parse response
        parsed = self.tool_handler.parse_response(
            response,
            mode="native" if self.tool_handler.supports_native else "prompted"
        )

        return parsed
```

### Session Integration

Sessions manage tool execution across conversation turns:

```python
from abstractcore.session import Session
from abstractcore.tools import execute_tools

session = Session(llm, tools=[tool1, tool2])

# Automatic tool execution
response = session.generate("Use the search tool")
# Session automatically detects and executes tool calls

# Manual control
response = session.generate("Use the search tool", auto_execute_tools=False)
if response.tool_calls:
    results = execute_tools(response.tool_calls)
    # Continue with results...
```

### Agent Integration

Agents leverage tools for autonomous operation:

```python
from abstractcore.agents import ReActAgent
from abstractcore.tools import tool

@tool
def calculator(expression: str) -> float:
    """Evaluate mathematical expressions"""
    return eval(expression)

agent = ReActAgent(
    llm=llm,
    tools=[calculator, web_search, read_file],
    max_iterations=10
)

result = agent.run("Calculate the GDP growth rate from the data in report.csv")
# Agent automatically uses tools to solve the task
```

---

## Best Practices

### Tool Design

**1. Clear, Descriptive Names**
```python
# Good
@tool
def search_web(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo"""

# Bad
@tool
def srch(q: str, n: int = 5) -> str:
    """Search"""
```

**2. Comprehensive Descriptions**
```python
@tool(
    description="Search the web for information using DuckDuckGo (no API key required)",
    when_to_use="When you need current information not in your training data",
    examples=[
        {
            "description": "Search for news",
            "arguments": {"query": "latest AI developments", "num_results": 5}
        }
    ]
)
def search_web(query: str, num_results: int = 5) -> str:
    """Implementation"""
```

**3. Type Hints**
```python
from typing import Optional, List, Dict

@tool
def process_data(
    data: List[Dict[str, Any]],
    filter_key: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Type hints enable automatic parameter detection"""
```

**4. Error Handling**
```python
@tool
def read_file(file_path: str) -> str:
    """Read file contents with proper error handling"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

**5. Deterministic Outputs**
```python
# Good - Consistent format
@tool
def get_user_info(user_id: int) -> Dict[str, Any]:
    """Returns structured user data"""
    return {"id": user_id, "name": "...", "email": "..."}

# Bad - Inconsistent format
@tool
def get_user_info(user_id: int) -> str:
    """Returns arbitrary text that may vary"""
    return f"User: {name} ({email})" if found else "Not found"
```

### Error Handling

**1. Validation in Tools**
```python
@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

**2. Safe Execution**
```python
from abstractcore.tools import execute_tool, ToolCall

tool_call = ToolCall(name="divide", arguments={"a": 10, "b": 0})
result = execute_tool(tool_call)

if not result.success:
    print(f"Tool error: {result.error}")
    # Handle error gracefully
```

**3. Retry Logic**
```python
def execute_with_retry(tool_call: ToolCall, max_retries: int = 3):
    for attempt in range(max_retries):
        result = execute_tool(tool_call)
        if result.success:
            return result

        if "timeout" in result.error.lower():
            time.sleep(2 ** attempt)  # Exponential backoff
            continue

        # Non-retryable error
        break

    return result
```

### Async Execution

**1. Async Tool Functions**
```python
import asyncio

@tool
async def fetch_data(url: str) -> str:
    """Async tool for non-blocking I/O"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

**2. Parallel Tool Execution**
```python
import asyncio
from abstractcore.tools import ToolCall, get_registry

async def execute_tools_parallel(tool_calls: List[ToolCall]) -> List[ToolResult]:
    """Execute multiple tools in parallel"""
    registry = get_registry()

    async def execute_async(call):
        tool = registry.get_tool(call.name)
        if asyncio.iscoroutinefunction(tool.function):
            result = await tool.function(**call.arguments)
        else:
            result = await asyncio.to_thread(tool.function, **call.arguments)
        return ToolResult(call_id=call.call_id, output=result, success=True)

    results = await asyncio.gather(*[execute_async(call) for call in tool_calls])
    return list(results)
```

---

## Common Pitfalls

### 1. Tool Execution Errors

**Problem**: Tool not found
```python
# Error
result = execute_tool(ToolCall(name="nonexistent_tool", arguments={}))
# Returns: ToolResult(success=False, error="Tool 'nonexistent_tool' not found")
```

**Solution**: Check registration
```python
from abstractcore.tools import get_registry

registry = get_registry()
if "my_tool" not in registry:
    register_tool(my_tool)
```

### 2. Format Mismatches

**Problem**: Parser not detecting tool calls
```python
# Model returns unexpected format
response = "I'll use the search function: search('query')"
tool_calls = parse_tool_calls(response)  # Returns []
```

**Solution**: Use `_parse_any_format` for robust detection
```python
from abstractcore.tools.parser import _parse_any_format

tool_calls = _parse_any_format(response)
# Or update tool prompt to guide model to correct format
```

### 3. Argument Type Mismatches

**Problem**: Arguments don't match function signature
```python
@tool
def calculate(a: int, b: int) -> int:
    return a + b

tool_call = ToolCall(name="calculate", arguments={"a": "5", "b": "3"})
result = execute_tool(tool_call)
# Error: TypeError - expects int, got str
```

**Solution**: Add type coercion
```python
@tool
def calculate(a: int, b: int) -> int:
    """Calculate with type coercion"""
    try:
        a = int(a)
        b = int(b)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Arguments must be convertible to integers: {e}")
    return a + b
```

### 4. Missing Tool Metadata

**Problem**: Model doesn't use tools effectively
```python
# Poor metadata
@tool
def process(data: str) -> str:
    """Process data"""
    return result
```

**Solution**: Add rich metadata
```python
@tool(
    description="Process customer data and extract key insights",
    tags=["data", "analysis", "customers"],
    when_to_use="When you need to analyze customer information and extract patterns",
    examples=[
        {
            "description": "Analyze customer feedback",
            "arguments": {"data": "Customer feedback: ..."}
        }
    ]
)
def process(data: str) -> str:
    """Process customer data"""
    return result
```

### 5. Tool Naming Conflicts

**Problem**: Multiple tools with the same name
```python
from abstractcore.tools import tool

@tool(name="search")
def search_web(query: str) -> str:
    """Web search"""
    return web_results

@tool(name="search")
def search_database(query: str) -> str:  # Conflicts with search_web
    """Database search"""
    return db_results
```

**Solution**: Use unique names
```python
from abstractcore.tools import tool

@tool(name="search_web")
def search_web(query: str) -> str:
    """Search the web"""
    return web_results

@tool(name="search_database")
def search_database(query: str) -> str:
    """Search database"""
    return db_results
```

---

## Testing Strategy

### Unit Testing Tools

```python
import pytest
from abstractcore.tools import tool, ToolDefinition, execute_tool, ToolCall

def test_tool_definition_from_function():
    """Test automatic tool definition generation"""

    def sample_tool(param: str, count: int = 5) -> str:
        """A sample tool"""
        return param * count

    tool_def = ToolDefinition.from_function(sample_tool)

    assert tool_def.name == "sample_tool"
    assert tool_def.description == "A sample tool"
    assert "param" in tool_def.parameters
    assert "count" in tool_def.parameters
    assert tool_def.parameters["count"]["default"] == 5

def test_tool_execution():
    """Test tool execution and error handling"""

    @tool
    def divide(a: float, b: float) -> float:
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

    # Test successful execution
    result = execute_tool(ToolCall(name="divide", arguments={"a": 10, "b": 2}))
    assert result.success
    assert result.output == 5.0

    # Test error handling
    result = execute_tool(ToolCall(name="divide", arguments={"a": 10, "b": 0}))
    assert not result.success
    assert "Division by zero" in result.error
```

### Integration Testing

```python
def test_tool_with_llm():
    """Test tool integration with LLM"""
    from abstractcore import create_llm

    @tool
    def get_timestamp() -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    llm = create_llm("openai", model="gpt-4o-mini")

    response = llm.generate(
        prompt="What's the current timestamp?",
        tools=[get_timestamp]
    )

    # Verify tool was called
    assert response.tool_calls
    assert response.tool_calls[0].name == "get_timestamp"

    # Execute and verify result
    result = execute_tool(response.tool_calls[0])
    assert result.success
    assert "T" in result.output  # ISO format contains T
```

### Parser Testing

```python
def test_tool_call_parsing():
    """Test parsing across different formats"""
    from abstractcore.tools.parser import parse_tool_calls

    # Test Qwen3 format
    qwen_response = """
    <|tool_call|>
    {"name": "search", "arguments": {"query": "test"}}
    </|tool_call|>
    """
    tool_calls = parse_tool_calls(qwen_response, model_name="qwen3:4b")
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "search"
    assert tool_calls[0].arguments == {"query": "test"}

    # Test LLaMA3 format
    llama_response = """
    <function_call>
    {"name": "search", "arguments": {"query": "test"}}
    </function_call>
    """
    tool_calls = parse_tool_calls(llama_response, model_name="llama3:8b")
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "search"
```

---

## Public API

### Recommended Imports

```python
# Core data structures
from abstractcore.tools import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolCallResponse,
    tool  # Decorator
)

# Registry and execution
from abstractcore.tools import (
    ToolRegistry,
    get_registry,
    register_tool,
    execute_tool,
    execute_tools,
    clear_registry,
    register  # Decorator for auto-registration
)

# Parsing and formatting
from abstractcore.tools import (
    detect_tool_calls,
    parse_tool_calls,
    format_tool_prompt,
    clean_tool_syntax
)

# Handler
from abstractcore.tools import (
    UniversalToolHandler,
    create_handler
)

# Syntax rewriting
from abstractcore.tools import (
    ToolCallSyntaxRewriter,
    SyntaxFormat,
    CustomFormatConfig,
    create_openai_rewriter,
    create_codex_rewriter,
    create_passthrough_rewriter,
    auto_detect_format
)

# Tag rewriting
from abstractcore.tools import (
    ToolCallTagRewriter,
    ToolCallTags,
    get_predefined_tags,
    create_tag_rewriter
)

# Built-in tools
from abstractcore.tools import (
    list_files,
    search_files,
    read_file,
    write_file,
    edit_file,
    web_search,
    fetch_url,
    execute_command
)
```

### Minimal Usage

```python
from abstractcore import create_llm
from abstractcore.tools import tool

@tool
def my_function(param: str) -> str:
    """My custom tool"""
    return f"Processed: {param}"

llm = create_llm("openai", model="gpt-4")
response = llm.generate("Use my_function with 'test'", tools=[my_function])
```

---

## Architecture Highlights

**1. Universal Compatibility**: Works with any LLM through automatic capability detection

**2. Provider Agnostic**: Single API works across OpenAI, Anthropic, local models, etc.

**3. Format Agnostic**: Handles native APIs and 5+ prompted formats seamlessly

**4. Production Ready**: Built-in tools with security, error handling, and rich metadata

**5. Extensible**: Easy to add custom tools, formats, and parsers

**6. Event-Driven**: Full lifecycle monitoring through event emission

**7. Streaming Support**: Real-time tag rewriting for streaming scenarios

**8. Type-Safe**: Full type hints and runtime validation

---

This tools module represents AbstractCore's commitment to **universal tool calling** - making function calling accessible to every model, regardless of whether it has native support or not, through intelligent abstraction and format handling.

## Related Modules

**Direct dependencies**:
- [`core/`](../core/README.md) - Base tool abstractions
- [`exceptions/`](../exceptions/README.md) - Tool execution errors
- [`events/`](../events/README.md) - Tool execution lifecycle events
- [`utils/`](../utils/README.md) - Web utilities, validation, logging

**Used by**:
- [`providers/`](../providers/README.md) - Tool execution in generation
- [`processing/`](../processing/README.md) - Web search and research tools
- [`server/`](../server/README.md) - Tool execution API endpoints
- [`apps/`](../apps/README.md) - Application-specific tools

**Related systems**:
- [`architectures/`](../architectures/README.md) - Tool support capability detection
- [`assets/`](../assets/README.md) - Model tool capabilities
- [`config/`](../config/README.md) - Tool execution configuration
- [`structured/`](../structured/README.md) - Tool parameter validation
