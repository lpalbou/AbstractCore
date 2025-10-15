# Multi-Provider/Multi-Model Requests - Architecture Analysis

## Problem Statement

Users often want to send the **same message** to multiple provider/model combinations for various reasons:
- **A/B Testing**: Compare responses from different models
- **Consensus**: Get multiple perspectives and choose the best
- **Speed Racing**: Use the fastest response
- **Cost Optimization**: Try cheap model first, fallback to expensive
- **Quality Assurance**: Cross-validate responses
- **Redundancy**: Ensure availability despite provider outages

## Current Workaround (Manual)

```python
# User has to do this manually today
providers = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("ollama", "llama3:8b")
]

responses = []
for provider, model in providers:
    llm = create_llm(provider, model=model)
    response = llm.generate("What is Python?")
    responses.append(response)
```

## Proposed Feature

Send one message to multiple providers/models with a single call and intelligent handling of responses.

## Architectural Decision: **Core Library vs API-Only**

### Option 1: API-Only Implementation

**Approach**: Add server endpoints that handle multi-provider requests internally.

```python
# API endpoint
POST /v1/multi/chat
{
    "message": "What is Python?",
    "providers": [
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "anthropic", "model": "claude-3-5-haiku-latest"},
        {"provider": "ollama", "model": "llama3:8b"}
    ],
    "strategy": "parallel" // or "cascade", "consensus", "race"
}
```

**Pros:**
- ✅ Simple to implement (just loop over providers in server)
- ✅ No core library complexity
- ✅ Server-specific features (HTTP timeouts, connection pooling)
- ✅ Easy to add advanced UI features

**Cons:**
- ❌ Only available via server, not Python library
- ❌ Limited to basic strategies
- ❌ Can't reuse across different deployment modes

### Option 2: Core Library Implementation

**Approach**: Add multi-provider support directly to AbstractCore.

```python
# Core library usage
from abstractcore import MultiLLM

multi_llm = MultiLLM([
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("ollama", "llama3:8b")
])

# Different strategies
response = multi_llm.generate("What is Python?", strategy="parallel")
response = multi_llm.generate("What is Python?", strategy="cascade")
response = multi_llm.generate("What is Python?", strategy="consensus")
```

**Pros:**
- ✅ Available to all users (Python + server)
- ✅ Consistent across deployment modes
- ✅ Can implement sophisticated strategies
- ✅ Better error handling and retry logic
- ✅ Event system integration
- ✅ Session support for multi-provider conversations

**Cons:**
- ❌ Adds complexity to core library
- ❌ Need to handle async/sync compatibility
- ❌ More testing and edge cases
- ❌ Harder to implement initially

### Option 3: Hybrid Approach (RECOMMENDED)

**Core Library**: Basic multi-provider support with simple strategies
**Server API**: Advanced features and convenient endpoints

```python
# Core library - basic support
from abstractcore import MultiLLM
multi_llm = MultiLLM(providers_config)
response = multi_llm.generate_parallel("message")

# Server API - advanced features
POST /v1/multi/chat  # With UI, monitoring, advanced strategies
```

## Execution Strategies

### 1. Parallel Strategy
**Use Case**: Speed + redundancy
```python
# Send to all providers simultaneously, return first successful response
response = multi_llm.generate("message", strategy="parallel")
```

**Implementation**:
- `asyncio.gather()` or `concurrent.futures`
- Return first successful response
- Cancel remaining requests
- Handle partial failures gracefully

### 2. Cascade Strategy
**Use Case**: Cost optimization
```python
# Try providers in order until one succeeds
response = multi_llm.generate("message", strategy="cascade",
                             order=["ollama", "openai", "anthropic"])
```

**Implementation**:
- Try providers sequentially by priority
- Stop on first success
- Configurable failure conditions
- Cost-aware ordering

### 3. Consensus Strategy
**Use Case**: Quality assurance
```python
# Get responses from all, then choose/combine best
response = multi_llm.generate("message", strategy="consensus",
                             selector="embedding_similarity")  # or "longest", "vote"
```

**Implementation**:
- Wait for all providers to respond
- Use selection algorithm:
  - Embedding similarity (most similar to others)
  - Length-based (longest/shortest)
  - Voting (if structured responses)
  - Custom scoring function

### 4. Race Strategy
**Use Case**: Lowest latency
```python
# Return fastest response, cancel others
response = multi_llm.generate("message", strategy="race")
```

**Implementation**:
- Start all requests simultaneously
- Return first to complete successfully
- Cancel remaining requests immediately
- Track performance metrics

## Implementation Complexity Analysis

### Core Library Implementation

**New Classes Needed**:
```python
class MultiLLM:
    def __init__(self, providers: List[Tuple[str, str]], **config)
    def generate_parallel(self, message: str) -> MultiResponse
    def generate_cascade(self, message: str, order: List[str]) -> GenerateResponse
    def generate_consensus(self, message: str, selector: str) -> GenerateResponse
    def generate_race(self, message: str) -> GenerateResponse

class MultiResponse:
    responses: List[GenerateResponse]
    selected: GenerateResponse  # The chosen response
    metadata: Dict  # Timing, costs, etc.
```

**Key Challenges**:
1. **Async/Sync Compatibility**: Core is currently sync, this needs async
2. **Error Handling**: Partial failures, all-fail scenarios
3. **Resource Management**: Connection pools, rate limiting
4. **Event Integration**: Events for each provider + multi-provider events
5. **Session Support**: Multi-provider conversation state
6. **Cost Tracking**: Aggregate costs across providers

**Estimated Effort**: 2-3 weeks (medium complexity)

### Server API Implementation

**New Endpoints**:
```python
POST /v1/multi/chat           # Multi-provider chat
POST /v1/multi/compare        # A/B comparison UI
GET  /v1/multi/strategies     # Available strategies
POST /v1/multi/benchmark      # Performance testing
```

**Key Features**:
- WebSocket for real-time multi-provider streaming
- UI dashboard for comparing responses
- Performance analytics and cost tracking
- Saved multi-provider configurations

**Estimated Effort**: 1 week (simple to implement on top of existing)

## Recommended Approach

### Phase 1: Server API Only (Low Risk)
Implement basic multi-provider support in the server:
- Parallel requests to multiple providers
- Simple response selection (first, fastest, longest)
- Basic error handling

### Phase 2: Core Library Integration (Medium Risk)
Add `MultiLLM` class to core library:
- Focus on parallel and cascade strategies
- Full event system integration
- Session support

### Phase 3: Advanced Strategies (High Value)
Implement sophisticated selection algorithms:
- Consensus via embedding similarity
- Cost-aware routing and fallbacks
- Learning from user preferences

## API Design Considerations

### Request Format
```json
{
  "message": "What is the capital of France?",
  "providers": [
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "weight": 1.0,
      "priority": 1
    },
    {
      "provider": "anthropic",
      "model": "claude-3-5-haiku-latest",
      "weight": 1.0,
      "priority": 2
    }
  ],
  "strategy": "parallel",
  "selection": "first_success", // or "fastest", "consensus", "best_quality"
  "timeout_ms": 30000,
  "max_cost_usd": 0.01
}
```

### Response Format
```json
{
  "message": "What is the capital of France?",
  "selected_response": {
    "content": "The capital of France is Paris.",
    "provider": "openai",
    "model": "gpt-4o-mini"
  },
  "all_responses": [
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "content": "The capital of France is Paris.",
      "latency_ms": 850,
      "cost_usd": 0.0012,
      "status": "success"
    },
    {
      "provider": "anthropic",
      "model": "claude-3-5-haiku-latest",
      "content": "Paris is the capital city of France.",
      "latency_ms": 1200,
      "cost_usd": 0.0008,
      "status": "success"
    }
  ],
  "metadata": {
    "strategy": "parallel",
    "selection_reason": "first_success",
    "total_cost_usd": 0.0020,
    "fastest_ms": 850,
    "slowest_ms": 1200
  }
}
```

## Integration with Existing Features

### Event System
```python
# New event types needed
EventType.MULTI_PROVIDER_STARTED = "multi_provider_started"
EventType.MULTI_PROVIDER_RESPONSE = "multi_provider_response"  # For each provider
EventType.MULTI_PROVIDER_COMPLETED = "multi_provider_completed"
EventType.MULTI_PROVIDER_SELECTION = "multi_provider_selection"
```

### Session Management
```python
# Multi-provider sessions need special handling
multi_session = MultiSession(providers=provider_configs)
multi_session.generate("Hello")  # Maintains context across ALL providers
```

### Tool Calling
```python
# Tools should work across all providers
@tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

# All providers get the same tools
multi_llm.generate("What's the weather in Paris?", tools=[get_weather])
```

## Use Cases and Examples

### A/B Testing Different Models
```python
# Compare GPT vs Claude responses
response = multi_llm.generate(
    "Write a marketing email for our product",
    strategy="consensus",
    providers=[("openai", "gpt-4"), ("anthropic", "claude-3-5-sonnet-latest")]
)
print(f"Selected: {response.selected.provider}")
print(f"All responses: {len(response.all_responses)}")
```

### Cost-Optimized Cascade
```python
# Try cheap model first, fallback to expensive
response = multi_llm.generate(
    "Simple question: What is 2+2?",
    strategy="cascade",
    providers=[
        ("ollama", "llama3:8b"),      # Free local
        ("openai", "gpt-4o-mini"),    # Cheap API
        ("openai", "gpt-4")           # Expensive API
    ]
)
```

### Speed Racing for User Interfaces
```python
# Get fastest response for snappy UI
response = multi_llm.generate(
    "Quick answer: What is Python?",
    strategy="race",
    providers=[
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-latest"),
        ("ollama", "llama3:8b")
    ]
)
print(f"Winner: {response.selected.provider} ({response.metadata['latency_ms']}ms)")
```

## Risks and Mitigations

### Risk 1: Increased Costs
**Problem**: Users might accidentally run expensive queries across multiple providers
**Mitigation**:
- Built-in cost limits and warnings
- Cost estimation before execution
- Default to cheaper models in multi-provider configs

### Risk 2: Rate Limiting
**Problem**: Hitting rate limits across multiple providers simultaneously
**Mitigation**:
- Intelligent backoff and retry
- Provider-aware rate limiting
- Circuit breaker integration

### Risk 3: Complexity Explosion
**Problem**: Too many strategies and options confuse users
**Mitigation**:
- Start with 2-3 simple strategies
- Good defaults and presets
- Clear documentation and examples

### Risk 4: Latency Issues
**Problem**: Waiting for slowest provider in consensus mode
**Mitigation**:
- Configurable timeouts
- Partial consensus (majority response)
- Async streaming updates

## Competitive Analysis

### OpenRouter
- Multi-provider API service
- Focus on routing and cost optimization
- Limited to API access only

### LiteLLM Proxy
- Basic multi-provider support
- Primarily for compatibility, not multi-requests
- No advanced selection strategies

### AbstractCore Advantage
- **Core Library Integration**: Available in Python directly
- **Event System**: Full observability of multi-provider operations
- **Tool Support**: Universal tools work across all providers
- **Session Management**: Multi-provider conversation state
- **Sophisticated Strategies**: Beyond simple routing

## Decision Matrix

| Factor | API-Only | Core Library | Hybrid |
|--------|----------|--------------|--------|
| **Implementation Speed** | Fast (1 week) | Slow (3 weeks) | Medium (2 weeks) |
| **User Accessibility** | Server users only | All users | All users |
| **Feature Richness** | Basic | Advanced | Advanced |
| **Maintenance Cost** | Low | High | Medium |
| **Differentiation** | Medium | High | High |

**Recommendation**: **Hybrid Approach**
1. Start with server API for immediate value
2. Add core library support in next major version
3. Focus on 3 strategies: parallel, cascade, race

## Implementation Roadmap

### Phase 1 (Server API) - 1 week
- `POST /v1/multi/chat` endpoint
- Parallel and cascade strategies
- Basic error handling and cost tracking
- Documentation and examples

### Phase 2 (Core Library) - 2 weeks
- `MultiLLM` class in core library
- Event system integration
- Session support for multi-provider conversations
- Comprehensive testing

### Phase 3 (Advanced Features) - 1 week
- Consensus strategies with embedding similarity
- WebSocket streaming for real-time updates
- Performance analytics dashboard
- Cost optimization algorithms

**Total Effort**: 4 weeks
**Priority**: Medium (after current server stabilization)
**Dependencies**: None (can build on existing architecture)

## Conclusion

Multi-provider requests represent a significant value-add that would differentiate AbstractCore from competitors. The hybrid approach offers the best balance of implementation speed, user value, and long-term flexibility.

**Next Steps**:
1. Validate demand with users
2. Start with simple server API implementation
3. Gather feedback and usage patterns
4. Evolve towards core library integration

This feature transforms AbstractCore from "one API for all providers" to "intelligent orchestration across all providers" - a compelling evolution of the value proposition.