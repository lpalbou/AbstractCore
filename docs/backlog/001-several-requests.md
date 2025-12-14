# AbstractCore Feature Requests

## FR-001: Support Qwen3 Thinking Mode Control

**Date:** 2025-12-13
**Priority:** High
**Status:** Open

### Problem

Qwen3 models have a "thinking" mode enabled by default. When thinking mode is on, the model generates reasoning in a `<think>...</think>` block before the actual response. Different providers handle this differently.

### Provider-Specific Behavior

#### Ollama
- **Parameter**: `think: true/false` in API request
- **Response**: When thinking enabled, response has `thinking` field separate from `content`
- **Evidence**:
  ```bash
  # With thinking disabled - returns content directly
  curl http://localhost:11434/api/chat -d '{
    "model": "qwen3:1.7b-q4_K_M",
    "messages": [{"role": "user", "content": "Say hi"}],
    "think": false
  }'
  # Response: {"message": {"content": "Hello! How can I assist you today?"}}
  
  # With thinking enabled (default)
  # Response: {"message": {"content": "", "thinking": "Okay, the user wants..."}}
  ```

#### vLLM
- **Parameter**: `chat_template_kwargs: {"enable_thinking": false}` in API request
- **Alternative**: Custom chat template `--chat-template ./qwen3_nonthinking.jinja`
- **Parsing**: `--enable-reasoning --reasoning-parser deepseek_r1` (or `qwen3` in v0.9.0+)
- **Response**: `reasoning_content` field separate from `content`

#### SGLang
- **Parameter**: `chat_template_kwargs: {"enable_thinking": false}` in API request
- **Alternative**: Custom chat template `--chat-template ./qwen3_nonthinking.jinja`
- **Parsing**: `--reasoning-parser qwen3`
- **Response**: `reasoning_content` field separate from `content`

#### Hugging Face Transformers
- **Parameter**: `enable_thinking=True/False` in `tokenizer.apply_chat_template()`
- **Response**: Thinking content wrapped in `<think>...</think>` tokens (151667/151668)

#### MLX-LM
- **Status**: Documentation not yet updated for Qwen3
- **Expected**: Similar to Transformers via chat template

#### LM Studio
- **Status**: Supports Qwen3 but documentation not specific about thinking mode
- **Expected**: May use chat template or model-specific settings

### Soft Switch (All Providers)
Users can add `/think` or `/no_think` to prompts to dynamically control thinking per-turn when `enable_thinking=True`.

### Proposed Implementation for AbstractCore

1. **Add `think` parameter to provider base class:**
   ```python
   class BaseProvider:
       def generate(self, ..., think: Optional[bool] = None):
           # Provider-specific handling
   ```

2. **Provider-specific implementations:**
   
   **Ollama** (`providers/ollama.py`):
   ```python
   if think is not None:
       payload["think"] = think
   
   # Handle response
   content = response.get("message", {}).get("content", "")
   thinking = response.get("message", {}).get("thinking", "")
   if not content and thinking:
       content = thinking  # Fallback
   ```
   
   **vLLM/SGLang** (OpenAI-compatible):
   ```python
   if think is not None:
       extra_body["chat_template_kwargs"] = {"enable_thinking": think}
   
   # Handle response
   content = choice.message.content
   reasoning = getattr(choice.message, "reasoning_content", None)
   ```

3. **Auto-detect Qwen3 and set sensible defaults:**
   ```python
   def _should_disable_thinking(model_name: str) -> bool:
       # For tool-calling scenarios, thinking mode adds latency
       return "qwen3" in model_name.lower()
   ```

4. **Expose thinking content in response:**
   ```python
   @dataclass
   class GenerateResponse:
       content: str
       thinking: Optional[str] = None  # New field
       # ...
   ```

### Testing Required

AbstractCore team should test with:
- [ ] Ollama: qwen3:1.7b, qwen3:4b, qwen3:8b
- [ ] vLLM: Qwen/Qwen3-8B with `--reasoning-parser qwen3`
- [ ] SGLang: Qwen/Qwen3-8B with `--reasoning-parser qwen3`
- [ ] Transformers: Direct model loading

### References

- [Qwen3 HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3 vLLM Docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [Qwen3 SGLang Docs](https://qwen.readthedocs.io/en/latest/deployment/sglang.html)
- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md) - `think` parameter

---

## FR-002: ToolRegistry.register() Should Use @tool Decorator Metadata

**Date:** 2025-12-14
**Priority:** Medium
**Status:** Open

### Problem

When a function is decorated with `@tool(name="...", description="...", when_to_use="...")`, the decorator attaches a `_tool_definition` attribute to the function with all the rich metadata.

However, `ToolRegistry.register()` ignores this and creates a new `ToolDefinition` from scratch using `ToolDefinition.from_function()`, losing the metadata.

### Evidence

```python
from abstractcore.tools import tool, ToolRegistry

@tool(
    name="list_files",
    description="List files and directories in a path.",
    when_to_use="When you need to see what files exist",
)
def list_files(path: str = ".") -> str:
    """List files in directory."""
    ...

# The decorator attaches rich metadata
print(list_files._tool_definition.when_to_use)
# Output: "When you need to see what files exist"

# But register() ignores it
registry = ToolRegistry()
registered = registry.register(list_files)
print(registered.when_to_use)
# Output: None  <-- Lost!
```

### Proposed Fix

In `registry.py`:

```python
def register(self, tool: Union[ToolDefinition, Callable]) -> ToolDefinition:
    if callable(tool):
        # Check if already has tool definition from @tool decorator
        if hasattr(tool, '_tool_definition'):
            tool_def = tool._tool_definition
        else:
            tool_def = ToolDefinition.from_function(tool)
    elif isinstance(tool, ToolDefinition):
        tool_def = tool
    # ...
```

### Impact

This affects any code that:
1. Uses `@tool` decorator with parameters
2. Then registers the function to a `ToolRegistry`

The rich metadata (when_to_use, tags, examples) is lost.

---

## FR-003: Ollama Provider Should Respect execute_tools Parameter

**Date:** 2025-12-14
**Priority:** High
**Status:** Open

### Problem

When calling `provider.generate(tools=[...], execute_tools=True)`, the Ollama provider doesn't execute tools even though `execute_tools=True` is passed.

### Evidence

```python
from abstractcore import create_llm
from abstractcore.tools import tool

@tool(name='get_weather', description='Get weather')
def get_weather(city: str) -> str:
    return f'Weather in {city}: Sunny'

provider = create_llm('ollama', model='qwen3:4b-instruct-2507-q4_K_M')

response = provider.generate(
    prompt='What is the weather in Paris?',
    tools=[get_weather],
    execute_tools=True,  # Should execute, but doesn't
)

print(response.content)
# Output: <|tool_call|>{"name": "get_weather", "arguments": {"city": "Paris"}}</|tool_call|>
# Expected: Tool should have been executed, result included
```

### Root Cause

In `ollama_provider.py` line 303:
```python
if self.execute_tools and tools and ...
```

This checks `self.execute_tools` (instance attribute, default False) instead of the `execute_tools` parameter passed to `generate()`.

### Proposed Fix

The `execute_tools` parameter should be passed through to `_generate_internal` and used there:

```python
# In _generate_internal:
should_execute = execute_tools if execute_tools is not None else self.execute_tools
if should_execute and tools and self.tool_handler.supports_prompted and content:
    return self._handle_tool_execution(generate_response, tools)
```

### Affected Providers

Check all providers for this pattern:
- [x] Ollama - affected
- [ ] OpenAI Compatible
- [ ] vLLM
- [ ] LM Studio
- [ ] MLX
- [ ] HuggingFace

---

## FR-004: Consistent Tool Execution Across All Providers

**Date:** 2025-12-14
**Priority:** Medium
**Status:** Open

### Problem

Different providers handle tool execution differently:
- Some use `_handle_prompted_tool_execution()` from base class
- Some have custom implementations
- Some don't support it at all

### Proposed Solution

1. Standardize on base class method `_handle_prompted_tool_execution()`
2. All providers should call it when `execute_tools=True`
3. Document which providers support tool execution

### Benefits

- Consistent behavior across providers
- Easier to maintain
- Clear expectations for users
