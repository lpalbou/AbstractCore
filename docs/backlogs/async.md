Plan 2: Async Implementation for AbstractCore

File: docs/backlog/async.md

# Async Support Implementation Plan for AbstractCore

## Overview
Add comprehensive async support to AbstractCore for non-blocking operations, enabling real-time UIs and concurrent request handling.

## Motivation
- **UI Responsiveness**: Non-blocking generation for responsive interfaces
- **Concurrent Operations**: Handle multiple LLM requests simultaneously  
- **Real-time Events**: Process events while generation continues
- **Server Performance**: Better resource utilization in API servers
- **User Intervention**: Allow cancellation and modification during generation

## Technical Approach

### Phase 1: Core Async Methods
1. **Base Provider Enhancement** (`providers/base.py`):
```python
async def generate_async(self, prompt: str, **kwargs) -> GenerateResponse
async def stream_async(self, prompt: str, **kwargs) -> AsyncIterator[GenerateResponse]

2. Provider Implementations:
- OpenAI: Use AsyncOpenAI client
- Anthropic: Use AsyncAnthropic client
- HTTP-based: Convert to httpx.AsyncClient
- Local models: Wrap in asyncio.run_in_executor()
3. Factory Enhancement (core/factory.py):
async def create_llm_async(provider: str, model: str) -> AsyncLLMInterface

Phase 2: Event System Async

1. Async Event Bus (events/__init__.py):
async def emit_async(event_type: EventType, data: Dict)
async def on_async(event_type: EventType, handler: AsyncCallable)
2. Real-time Streaming:
- WebSocket event streaming
- Server-Sent Events (SSE) support
- Async event handlers for UI updates

Phase 3: Tools & Sessions

1. Async Tool Execution:
async def execute_tool_async(tool_call: ToolCall) -> ToolResult
2. Async Session Management:
class AsyncSession:
    async def generate(self, prompt: str) -> GenerateResponse
    async def stream(self, prompt: str) -> AsyncIterator

Phase 4: Advanced Features

1. Cancellation Support:
- Use asyncio.CancelledError for clean cancellation
- Implement timeout handling
- Graceful cleanup on interruption
2. Concurrent Generation:
- Multiple providers in parallel
- Race conditions (first to complete wins)
- Ensemble generation (combine multiple outputs)
3. Real-time Intervention:
- Modify prompts during generation
- Inject context mid-stream
- Dynamic tool availability

Implementation Strategy

Backward Compatibility

- Keep all existing sync methods unchanged
- Add async versions alongside (_async suffix)
- Auto-detection: sync in sync context, async in async context

Testing Strategy

- Parallel test suite for async methods
- Performance benchmarks (sync vs async)
- Concurrency stress tests
- Event ordering validation

Documentation Updates

- Async examples in all docs
- Migration guide from sync to async
- Performance comparison guide
- Best practices for UI integration

Complexity Assessment

Estimated Effort: 5-7 days
- Core async methods: 2 days
- Provider implementations: 2 days  
- Testing & documentation: 2 days
- Advanced features: 1 day

Risk Areas:
- Local model providers (threading complexity)
- Event ordering in concurrent scenarios
- Backward compatibility maintenance
- Testing coverage for edge cases

Benefits

1. UI Applications: 10-100x better responsiveness
2. Server Applications: 3-5x better throughput
3. Real-time Features: Live progress, intervention, cancellation
4. Modern Stack: Compatible with FastAPI, Streamlit, Gradio

Future Enhancements

1. Async Structured Output with progressive validation
2. Async Embeddings for parallel batch processing
3. Async Retry Logic with concurrent retry attempts
4. WebRTC Integration for P2P streaming

Decision

Status: BACKLOG
Priority: HIGH (after server implementation)
Target: v2.2.1

---

## Implementation Order

### Week 1: SOTA Server
1. **Day 1-2**: Core server with OpenAI-compatible endpoints
2. **Day 3**: Intelligent routing and provider management
3. **Day 4**: WebSocket streaming and real-time events
4. **Day 5**: Testing and documentation

### Future: Async Support
- Implement after server is stable
- Use server as testing ground for async features
- Gradual rollout with backward compatibility

## Key Innovations

### Server Innovations:
1. **Semantic Request Routing** using embeddings
2. **Multi-Provider Ensemble** for best results
3. **Cost-Aware Routing** with budget constraints
4. **Real-time Event Streaming** via WebSocket
5. **Tool Marketplace** for sharing tools

### Async Innovations:
1. **Progressive Generation** with real-time intervention
2. **Concurrent Provider Racing** for lowest latency
3. **Async Tool Chains** with parallel execution
4. **Live Context Injection** during generation

This comprehensive approach transforms AbstractCore from a library into a complete AI infrastructure platform.