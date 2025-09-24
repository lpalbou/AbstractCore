# Proposed Tool Calling Architecture

## Current vs Proposed

### Current (Mixed Responsibilities)
```
User Request → AbstractLLM → Detect Tools → Execute Tools → Return Results
                     ↑           ↑              ↑
                Provider     Provider      Common Logic
                Specific     Specific      (WRONG LAYER)
```

### Proposed (Separation of Concerns)
```
User Request → AbstractLLM → Detect Tools → Emit Events
                     ↑           ↑              ↓
                Provider     Provider      Agent Layer
                Specific     Specific      ↓
                                          Execute Tools → Return Results
                                               ↑
                                          Common Logic
                                          (CORRECT LAYER)
```

## AbstractLLM Responsibilities

### ✅ SHOULD DO (Detection & Abstraction)
- Parse provider-specific tool call formats
- Normalize to common ToolCall format
- Emit TOOL_DETECTED events
- Support streaming tool detection
- Validate tool call structure

### ❌ SHOULD NOT DO (Execution & Business Logic)
- Execute actual tool functions
- Handle tool results
- Orchestrate tool calling sequences
- Apply tool execution policies

## Event-Driven Architecture

```python
# AbstractLLM emits events
if response.has_tool_calls():
    for call in response.tool_calls:
        GlobalEventBus.emit(EventType.TOOL_DETECTED, {
            'tool_name': call.get('name'),
            'arguments': call.get('arguments'),
            'provider': self.provider_name,
            'call_id': call.get('id')
        })

# Agent layer listens and executes
class SimpleAgent:
    def __init__(self):
        GlobalEventBus.on(EventType.TOOL_DETECTED, self.handle_tool_detected)

    def handle_tool_detected(self, event):
        # Execute tool based on business logic
        result = self.execute_tool(event.data['tool_name'], event.data['arguments'])
        # Handle result appropriately for this agent
```

## Benefits

1. **Separation of Concerns**: LLM abstraction vs business logic
2. **Flexibility**: Different agents can handle tools differently
3. **Streaming Support**: Events emitted as tools detected in stream
4. **Testing**: Can test detection without execution
5. **Security**: Tool execution policies at agent level
6. **Provider Independence**: Tool execution same across all providers

## Streaming Implications

In streaming mode, AbstractLLM can emit TOOL_DETECTED events immediately:

```python
# Stream processing
for chunk in stream:
    if chunk.contains_tool_call():
        # Emit immediately, don't wait for full response
        GlobalEventBus.emit(EventType.TOOL_DETECTED, tool_data)
```

## Migration Strategy

1. **Phase 1**: Create event-based tool detection (maintain backward compatibility)
2. **Phase 2**: Create sample agent that uses events
3. **Phase 3**: Update tests to use agent pattern
4. **Phase 4**: Deprecate direct tool execution from AbstractLLM
5. **Phase 5**: Remove execute_tool from AbstractLLM core