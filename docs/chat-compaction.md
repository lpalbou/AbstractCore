# Chat History Compaction

AbstractCore's BasicSession includes SOTA chat history compaction capabilities that intelligently summarize long conversations while preserving essential context.

## Overview

The `compact()` method implements 2025 state-of-the-art practices for conversation summarization:

- **Preserves system messages** (always maintained)
- **Keeps recent exchanges intact** (configurable sliding window)
- **Summarizes older context** (maintains conversational flow)
- **Focuses on key information** (decisions, solutions, ongoing topics)

## Quick Start

### Auto-Compact Mode (Recommended)
```python
from abstractllm import create_llm, BasicSession

# Create session with auto-compaction enabled
llm = create_llm("ollama", model="gemma3:1b")
session = BasicSession(
    llm,
    system_prompt="You are a helpful assistant",
    auto_compact=True,              # Enable automatic compaction
    auto_compact_threshold=6000     # Compact when exceeding 6000 tokens
)

# Conversation continues indefinitely - compacts automatically
response = session.generate("Let's have a long conversation...")
```

### Manual Compaction
```python
# Create regular session
session = BasicSession(llm, system_prompt="You are a helpful assistant")

# ... have a long conversation ...

# Check if compaction is recommended
if session.should_compact(token_limit=8000):
    # Option 1: In-place compaction (recommended for ongoing conversations)
    session.force_compact(preserve_recent=6, focus="key decisions")

    # Option 2: Create new compacted session
    compacted = session.compact(preserve_recent=6, focus="key decisions")

# Continue with conversation
response = session.generate("Let's continue our discussion...")
```

## SOTA Best Practices (2025)

### 1. System Message Preservation
System messages are **always preserved** to maintain essential context:

```python
session = BasicSession(llm, system_prompt="You are a technical expert")
# System prompt is preserved in compacted session
compacted = session.compact()
```

### 2. Sliding Window Approach
Recent messages stay intact while older ones get summarized:

```python
# Original: 50 messages
# Result: System + Summary + Last 6 messages = ~8-10 messages
compacted = session.compact(preserve_recent=6)
```

### 3. Context-Aware Summarization
The summarization focuses on conversation continuity:

- **Key decisions** and solutions
- **Ongoing topics** and user intent
- **Important technical details**
- **Conversational flow** preservation

### 4. Flexible Focus Control
Specify what aspects to emphasize in the summary:

```python
# Focus on technical aspects
compacted = session.compact(focus="technical solutions and code examples")

# Focus on decisions made
compacted = session.compact(focus="key decisions and next steps")

# Focus on learning objectives
compacted = session.compact(focus="learning progress and concepts covered")
```

## Advanced Usage

### Auto-Compaction Features

```python
# Enable auto-compaction on existing session
session.enable_auto_compact(threshold=8000)

# Disable auto-compaction
session.disable_auto_compact()

# Force immediate compaction (user-requested)
session.force_compact(preserve_recent=8, focus="technical solutions")

# Check current auto-compact status
print(f"Auto-compact: {session.auto_compact}")
print(f"Threshold: {session.auto_compact_threshold} tokens")
```

### Event Monitoring

```python
from abstractllm.events import EventType, on_global

def monitor_compaction(event):
    if event.type == EventType.COMPACTION_STARTED:
        print(f"🗜️ Compaction started: {event.data.get('reason')}")
        print(f"   Messages: {event.data.get('original_message_count')}")
        print(f"   Tokens: ~{event.data.get('original_tokens_estimate')}")
    elif event.type == EventType.COMPACTION_COMPLETED:
        print(f"✅ Compaction completed in {event.data.get('duration_ms'):.0f}ms")
        print(f"   Compression: {event.data.get('compression_ratio', 1):.1f}x")

# Register event handlers
on_global(EventType.COMPACTION_STARTED, monitor_compaction)
on_global(EventType.COMPACTION_COMPLETED, monitor_compaction)

# All compactions will now emit monitored events
session.force_compact(focus="key insights")
```

### Token Management

```python
# Check estimated token count
tokens = session.get_token_estimate()
print(f"Current tokens: {tokens}")

# Check if compaction is recommended
if session.should_compact(token_limit=4000):
    compacted = session.compact()
    print(f"Reduced to: {compacted.get_token_estimate()} tokens")
```

### Different Compaction Providers

```python
# Use a different model for compaction (e.g., faster local model)
fast_llm = create_llm("ollama", model="gemma3:1b")
main_llm = create_llm("openai", model="gpt-4o")

session = BasicSession(main_llm)
# ... conversation ...

# Use fast model for compaction, continue with main model
compacted = session.compact(compact_provider=fast_llm)
```

### Customized Recent Message Count

```python
# For detailed technical discussions - preserve more context
compacted = session.compact(preserve_recent=10)

# For quick summaries - preserve less context
compacted = session.compact(preserve_recent=4)

# For very long conversations - minimal recent context
compacted = session.compact(preserve_recent=2)
```

## Output Structure

The compacted session contains:

1. **System Messages** (preserved exactly)
2. **Conversation Summary** (as a system message for context)
3. **Recent Messages** (preserved exactly)

```python
compacted = session.compact(preserve_recent=4)

# Example structure:
# Message 1: [SYSTEM]: Original system prompt
# Message 2: [SYSTEM]: [CONVERSATION HISTORY]: Summary of older conversation...
# Message 3: [USER]: Recent user message
# Message 4: [ASSISTANT]: Recent assistant response
# Message 5: [USER]: Most recent user message
# Message 6: [ASSISTANT]: Most recent assistant response
```

## Real-World Examples

### Long Technical Discussion

```python
# After a long coding discussion
session = BasicSession(llm, system_prompt="You are a Python expert")

# ... 50+ messages about Python, debugging, optimization ...

compacted = session.compact(
    preserve_recent=6,
    focus="code solutions, debugging insights, and optimization techniques"
)

# Compacted session maintains:
# - System prompt (Python expert)
# - Summary of earlier discussion
# - Last 6 messages of current topic
```

### Customer Support Session

```python
session = BasicSession(llm, system_prompt="You are a helpful customer service agent")

# ... long troubleshooting conversation ...

compacted = session.compact(
    preserve_recent=8,
    focus="customer issue, attempted solutions, and current status"
)

# Perfect for maintaining support context while managing token limits
```

### Educational Conversation

```python
session = BasicSession(llm, system_prompt="You are a patient tutor")

# ... extensive learning conversation ...

compacted = session.compact(
    preserve_recent=6,
    focus="student's learning progress, key concepts covered, and current questions"
)

# Maintains educational context while staying within limits
```

## Performance Considerations

### When to Compact

```python
# Check if compaction is beneficial
if session.should_compact():
    compacted = session.compact()
else:
    # Continue with original session
    pass
```

### Token Estimation

The built-in token estimation uses a simple heuristic:
- **1 token ≈ 4 characters** for English text
- Useful for quick estimates
- For precise counts, use provider-specific tokenizers

### Compression Results

Typical compression ratios:
- **Long conversations**: 3-5x token reduction
- **Technical discussions**: 2-4x reduction
- **Very long sessions**: 10x+ reduction possible

## Error Handling

```python
try:
    compacted = session.compact()
except ImportError:
    print("BasicSummarizer not available - install processing capabilities")
except ValueError:
    print("No provider available for compaction")
except Exception as e:
    print(f"Compaction failed: {e}")
    # Continue with original session
    compacted = session
```

## Integration with Other Features

### With Session Persistence

```python
# Save compacted session
compacted = session.compact()
compacted.save("conversation_compacted.json")

# Load and continue
restored = BasicSession.load("conversation_compacted.json")
restored.provider = llm  # Re-attach provider
```

### With Tools

```python
from abstractllm.tools import tool

@tool
def summarize_conversation():
    """Summarize our current conversation"""
    return "Conversation has been active for X minutes..."

session = BasicSession(llm, tools=[summarize_conversation])
compacted = session.compact()  # Tools are preserved in compacted session
```

### With Events

```python
from abstractllm.events import EventType, on_global

def monitor_compaction(event):
    if event.type == EventType.AFTER_GENERATE:
        print(f"Compaction step completed: {event.duration_ms}ms")

on_global(EventType.AFTER_GENERATE, monitor_compaction)

# Compaction operations emit events for monitoring
compacted = session.compact()
```

## Troubleshooting

### Common Issues

1. **Model Compatibility**: Some local models may struggle with structured output
   - **Solution**: Use a cloud provider for compaction: `compact_provider=cloud_llm`

2. **Import Errors**: BasicSummarizer not available
   - **Solution**: Ensure processing module is properly installed

3. **No Provider**: Session has no provider configured
   - **Solution**: Pass `compact_provider` parameter or set session provider

### Best Practices

1. **Test with your model**: Different models have different structured output capabilities
2. **Monitor compression ratios**: Ensure meaningful reduction without losing context
3. **Use appropriate focus**: Tailor focus to your conversation type
4. **Consider token limits**: Balance between context and efficiency

## Conclusion

Chat history compaction enables unlimited conversation length while maintaining essential context. The implementation follows SOTA 2025 best practices for conversational AI systems, providing intelligent summarization that preserves conversational flow and key information.

Perfect for:
- **Long-running conversations** that exceed model context limits
- **Cost optimization** by reducing token usage
- **Context preservation** while managing memory constraints
- **Production applications** requiring conversation continuity