# Session Management and Serialization

AbstractCore provides comprehensive session management with complete serialization capabilities, preserving every aspect of your conversations including metadata, tool executions, and optional analytics.

## Overview

A **BasicSession** represents a complete conversation with an LLM, including:
- All messages with timestamps and metadata
- Tool calls and their results (inline with conversation flow)
- Session configuration and settings
- Optional analytics: summary, assessment, and extracted facts

## API Design: Two Methods for Different Purposes

The `BasicSession` provides two main methods for managing conversation history:

### `generate()` - For Normal Conversations (Recommended)
Use this for typical chat interactions where you want the LLM to respond:

```python
# Normal conversation flow
response = session.generate("What is Python?", name="alice")
# This automatically:
# 1. Adds your message to history
# 2. Calls the LLM provider
# 3. Adds the assistant's response to history
# 4. Returns a GenerateResponse object with full metadata

# Access the response data
print(f"Response: {response.content}")           # Generated text
print(f"Tokens used: {response.total_tokens}")  # Token count
print(f"Generation time: {response.gen_time}ms") # Performance metrics
```

### `add_message()` - For Manual History Management
Use this when you need fine-grained control over conversation history:

```python
# Add system messages
session.add_message('system', 'You are a helpful assistant.')

# Add messages without triggering LLM generation
session.add_message('user', 'Hello!', name='alice')
session.add_message('assistant', 'Hi there!')

# Add tool messages
session.add_message('tool', '{"result": "success"}')
```

**Key Difference**: `generate()` triggers LLM response generation, `add_message()` only adds to history.

**Parameter Consistency**: Both methods use `name` parameter, which aligns with the `metadata.name` field in the serialization schema.

## Session Serialization

### Why Serialize Sessions?

Session serialization enables:
- **Persistence**: Save and restore conversations across application restarts
- **Portability**: Share conversations between different environments
- **Analytics**: Generate summaries, assessments, and fact extractions
- **Auditing**: Complete conversation history with tool executions
- **Memory Management**: Load partial conversation windows while preserving full history

### Serialization Format

Sessions are serialized as JSON with a versioned schema for future compatibility:

```json
{
  "schema_version": "session-archive/v1",
  "session": {
    "id": "sess_01J8...",
    "created_at": "2025-10-13T14:52:46Z",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "system_prompt": "You are a helpful AI assistant.",
    "settings": { "auto_compact": true },
    
    "summary": { /* optional */ },
    "assessment": { /* optional */ },
    "facts": { /* optional */ }
  },
  "messages": [ /* complete conversation history */ ]
}
```

### Field Descriptions

#### Session Fields

- **`id`**: Unique session identifier for tracking and correlation
- **`created_at`**: ISO timestamp of session creation
- **`provider`**: LLM provider used (openai, anthropic, ollama, etc.)
- **`model`**: Specific model name (gpt-4o-mini, claude-3-5-haiku, etc.)
- **`model_params`**: Model parameters used (temperature, max_tokens, etc.)
- **`system_prompt`**: The system prompt that guides the assistant's behavior
- **`tool_registry`**: Available tools with their schemas (declarative, no executable code)
- **`settings`**: Session configuration (auto_compact, thresholds, etc.)

#### Optional Analytics Fields

- **`summary`** *(optional)*: Compressed representation of the entire conversation
  - `created_at`: When the summary was generated
  - `preserve_recent`: Number of recent messages preserved during compaction
  - `focus`: Summary focus (e.g., "technical decisions", "key outcomes")
  - `text`: The actual summary content
  - `metrics`: Compression statistics (tokens before/after, ratio)

- **`assessment`** *(optional)*: Quality evaluation of the entire conversation
  - `created_at`: When the assessment was generated
  - `criteria`: Evaluation criteria used (clarity, coherence, relevance, etc.)
  - `overall_score`: Numeric score (typically 1-5)
  - `judge_summary`: Brief assessment summary
  - `strengths`: List of conversation strengths
  - `actionable_feedback`: Suggestions for improvement

- **`facts`** *(optional)*: Extracted facts and knowledge from the conversation
  - `extracted_at`: When facts were extracted
  - `simple_triples`: Array of [subject, predicate, object] fact triples
  - `jsonld`: Optional JSON-LD structured data
  - `statistics`: Extraction statistics (entity count, relationship count)

#### Message Structure

Each message preserves the complete conversational flow:

```json
{
  "id": "msg_01J8...",
  "role": "user|assistant|system|tool",
  "timestamp": "2025-10-13T14:55:20.123Z",
  "content": "Message content",
  "metadata": {
    "name": "alice",
    "location": "London, UK",
    "custom_field": "value"
  }
}
```

**Message Fields:**
- **`id`**: Unique message identifier
- **`role`**: Message role (user, assistant, system, tool)
- **`timestamp`**: When the message was created (auto-generated)
- **`content`**: The actual message content
- **`metadata`**: Flexible container for additional context
  - `name`: Username (defaults to "user" for user messages)
  - `location`: Geographic or contextual location
  - Any additional custom fields

#### Tool Execution Flow

Tool calls are preserved inline with the conversation to maintain sequence:

```json
[
  {
    "role": "assistant",
    "content": "Let me read that file for you.",
    "metadata": {
      "requested_tool_calls": [
        {
          "call_id": "tc_01K",
          "name": "read_file",
          "arguments": { "path": "README.md" }
        }
      ]
    }
  },
  {
    "role": "tool",
    "content": "File contents...",
    "metadata": {
      "call_id": "tc_01K",
      "name": "read_file",
      "arguments": { "path": "README.md" },
      "status": "ok",
      "duration_ms": 120,
      "stderr": null
    }
  }
]
```

This approach:
- Preserves exact execution order
- Links tool calls to results via `call_id`
- Captures execution metadata (duration, status, errors)
- Maintains human-readable conversation flow

## Usage Examples

### Basic Session Persistence

```python
from abstractcore import BasicSession, create_llm

# Create and use session
provider = create_llm("openai", model="gpt-4o-mini")
session = BasicSession(provider, system_prompt="You are a helpful assistant.")

session.add_message('user', 'Hello!', name='alice', location='Paris')
response = session.generate('What is Python?')

# Save complete session
session.save('conversation.json')

# Load session later
loaded_session = BasicSession.load('conversation.json', provider=provider)
```

### Session with Analytics

```python
# Generate optional analytics
session.generate_summary(focus="technical discussion")
session.generate_assessment(criteria=["clarity", "completeness"])
session.extract_facts()

# Save with all analytics
session.save('analyzed_conversation.json')
```

### Memory Window Management

```python
# Get recent messages for LLM context
recent_messages = session.get_window(last_n=10)

# Get messages within time range
today_messages = session.get_window(
    since="2025-10-13T00:00:00Z",
    until="2025-10-13T23:59:59Z"
)

# Get messages with token budget
windowed = session.get_window(
    token_budget=4000,
    include_summary=True  # Prepend summary if context trimmed
)
```

## Schema Reference

The complete JSON schema is available at: `abstractcore/assets/session_schema.json`

This schema can be used for:
- Validation of serialized sessions
- Integration with other tools and systems
- Documentation generation
- API contract definition

## CLI Integration

The CLI provides convenient commands for session management:

```bash
# Save session with basic serialization
/save my_conversation

# Save with optional analytics
/save analyzed_session --summary --assessment --facts

# Load session
/load my_conversation

# View session analytics
/summary
/assessment  
/facts
```

## Best Practices

### When to Use Analytics

- **Summary**: For long conversations that need compaction
- **Assessment**: For evaluating conversation quality and outcomes
- **Facts**: For knowledge extraction and structured data needs

### Performance Considerations

- Analytics are optional and computed on-demand
- Large conversations benefit from windowing for active memory
- JSON format balances human readability with performance
- Consider compression (gzip/zstd) for very large sessions

### Security and Privacy

- Sessions contain complete conversation history
- Metadata may include sensitive information (usernames, locations)
- Tool execution results are preserved (may contain file contents, etc.)
- Store sessions securely and consider data retention policies

## Migration and Compatibility

The versioned schema (`session-archive/v1`) ensures:
- Backward compatibility with older session formats
- Forward compatibility through graceful field handling
- Safe evolution of the serialization format

Legacy sessions are automatically migrated on load, preserving all available data while upgrading to the current schema.
