# Bug Report: Tools + Structured Output Incompatibility

## 🐛 Summary

AbstractCore's `tools` and `response_model` parameters are mutually exclusive, preventing users from combining function calling with structured output validation. When both parameters are provided, the LLM generates tool calls but the structured output handler tries to validate them against the Pydantic model, causing validation errors.

## 🎯 Expected Behavior

Users should be able to use both `tools` and `response_model` simultaneously:

```python
response = llm.generate(
    "What time is it? Use tools if needed and respond with structured output.",
    tools=[get_current_time],
    response_model=TaskResponse,
    execute_tools=True
)
```

**Expected**: LLM can call tools AND return a structured response conforming to the Pydantic model.

## 🚫 Actual Behavior

The combination fails with validation errors:

```
ValidationError: 2 validation errors for TaskResponse
answer
  Field required [type=missing, input_value={'name': 'get_current_time', 'arguments': {}}, input_type=dict]
confidence
  Field required [type=missing, input_value={'name': 'get_current_time', 'arguments': {}}, input_type=dict]
```

**Root Cause**: The structured output handler receives tool call JSON instead of the expected response model structure.

## 🔬 Minimal Reproduction

Save this as `reproduce_bug.py`:

```python
#!/usr/bin/env python3
from abstractllm import create_llm
from pydantic import BaseModel
from typing import List, Optional

class TaskResponse(BaseModel):
    answer: str
    tool_calls_needed: Optional[List[str]] = None
    confidence: float

def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def test_bug():
    llm = create_llm("ollama", model="qwen3-coder:30b")  # or any model
    
    print("1. Tools only (works):")
    try:
        response = llm.generate(
            "What time is it?",
            tools=[get_current_time],
            execute_tools=True
        )
        print(f"✅ Success: {response.content[:50]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n2. Structured output only (works):")
    try:
        response = llm.generate(
            "What is 2+2? Respond with confidence.",
            response_model=TaskResponse
        )
        print(f"✅ Success: {response.answer}, confidence: {response.confidence}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n3. Both tools + structured output (BUG):")
    try:
        response = llm.generate(
            "What time is it? Use tools if needed.",
            tools=[get_current_time],
            response_model=TaskResponse,
            execute_tools=True
        )
        print(f"✅ Success: {response.answer}")
    except Exception as e:
        print(f"❌ BUG: {e}")

if __name__ == "__main__":
    test_bug()
```

Run with: `python reproduce_bug.py`

## 🔍 Technical Analysis

**Location**: `abstractllm/structured/handler.py`

**Issue**: The `StructuredOutputHandler.generate_structured()` method calls `provider._generate_internal()` directly, bypassing the normal tool execution flow in `generate_with_telemetry()`.

**Code Path**:
1. User calls `llm.generate(tools=X, response_model=Y)`
2. `BaseProvider.generate_with_telemetry()` detects `response_model` and delegates to `StructuredOutputHandler`
3. `StructuredOutputHandler` calls `provider._generate_internal()` with tools but no tool execution
4. LLM generates tool calls: `{"name": "get_current_time", "arguments": {}}`
5. Structured output handler tries to validate tool call JSON against Pydantic model
6. Validation fails because tool call structure ≠ expected model structure

## 💡 Suggested Solution

Implement hybrid handling in `BaseProvider.generate_with_telemetry()`:

```python
def generate_with_telemetry(self, ..., tools=None, response_model=None, **kwargs):
    # NEW: Handle hybrid case
    if tools and response_model:
        return self._handle_tools_with_structured_output(
            prompt=prompt, tools=tools, response_model=response_model, **kwargs
        )
    
    # Existing single-mode logic...
```

**Strategy Options**:
1. **Sequential**: Generate with tools first, then apply structured output if no tool calls
2. **Unified Schema**: Extend Pydantic models to include optional `tool_calls` field
3. **Prompt Enhancement**: Modify prompt to instruct LLM about available tools within structured format

## 🌟 Business Impact

**High Priority** - This limitation prevents users from building sophisticated LLM applications that need both:
- **Function calling** for external data/actions
- **Structured output** for reliable parsing and validation

**Use Cases Blocked**:
- AI agents with structured reasoning + tool usage
- Chatbots with function calling + response validation
- Workflow automation with tools + structured logging

## 🔧 Environment

- **AbstractCore Version**: [Current]
- **Python**: 3.11+
- **Provider**: Ollama (but affects all providers)
- **Models Tested**: qwen3-coder:30b

## 📋 Acceptance Criteria

- [ ] `llm.generate(tools=X, response_model=Y)` works without errors
- [ ] LLM can call tools AND return structured response
- [ ] Backward compatibility maintained for single-mode usage
- [ ] All existing tests pass
- [ ] Documentation updated with hybrid usage examples

---

**Priority**: High  
**Complexity**: Medium  
**Impact**: Enables advanced LLM application patterns
