# AbstractCore Internal CLI

AbstractCore includes a built-in CLI tool for testing, demonstration, and interactive conversations. This is AbstractCore's internal testing CLI, not to be confused with external agentic CLIs like Codex or Gemini CLI.

## Overview

The AbstractCore internal CLI provides advanced chat history management and system control commands for interactive testing of LLM providers.

**Looking for external agentic CLI integration (Codex, Gemini CLI, Crush)?**
→ **See [Server Documentation](server.md)** for complete setup guides.

## Quick Start

```bash
# Start the internal CLI
python -m abstractcore.utils.cli --provider ollama --model qwen3-coder:30b

# Or with any provider
python -m abstractcore.utils.cli --provider openai --model gpt-4o-mini
python -m abstractcore.utils.cli --provider anthropic --model claude-3-5-haiku-latest

# With streaming enabled (--stream flag)
python -m abstractcore.utils.cli --provider ollama --model qwen3-coder:30b --stream
```

## New Commands

### `/compact` - Chat History Compaction

Compacts your chat history using the fast local `gemma3:1b` model to create a summary while preserving recent exchanges.

```bash
# In the CLI
/compact

# Output example:
🗜️  Compacting chat history...
   Before: 15 messages (~450 tokens)
   Using gemma3:1b for compaction...
✅ Compaction completed in 3.2s
   After: 5 messages (~280 tokens)
   Structure:
   1. ⚙️  System prompt
   2. 📚 Conversation summary (1,200 chars)
   3. 👤 How do I handle errors in Python?
   4. 🤖 You can use try/except blocks...
   5. 👤 What about logging?
   💡 Note: Token count may increase initially due to detailed summary
       but will decrease significantly as conversation continues
```

**When to use:**
- Long conversations that slow down responses
- When you want to preserve context but reduce memory usage
- Before switching to a different topic

### `/facts [file]` - Extract Facts from Conversation

Extracts facts from your conversation history as simple triples (subject-predicate-object). Display them in chat or save as structured JSON-LD.

### `/judge` - Evaluate Conversation Quality

**NEW**: Evaluates the quality and interest of the current discussion using LLM-as-a-judge. This is a demonstrator showing objective assessment capabilities with constructive feedback.

```bash
# In the CLI
/judge

# Output example:
⚖️  Evaluating conversation quality...
   Analyzing 450 characters of conversation...
✅ Evaluation completed in 35.2s

📊 Overall Discussion Quality: 4/5

📈 Quality Dimensions:
   Clarity      : 5/5
   Coherence    : 4/5
   Actionability: 3/5
   Relevance    : 5/5
   Completeness : 4/5
   Soundness    : 4/5
   Simplicity   : 5/5

✅ Conversation Strengths:
   • The discussion maintains clear focus and addresses questions directly
   • Explanations are well-structured and easy to follow
   • Technical concepts are explained in accessible language

💡 Suggestions for Better Discussions:
   • Include more specific examples to illustrate key points
   • Add actionable next steps or recommendations where appropriate
   • Consider exploring alternative approaches or edge cases

🤔 Assessment Summary:
   The conversation demonstrates strong clarity and relevance with well-structured exchanges.

📌 Note: This is a demonstrator showing LLM-as-a-judge capabilities for objective assessment.
```

**When to use:**
- Get objective feedback on discussion quality
- Understand strengths and areas for improvement in conversations
- Demonstrate LLM-as-a-judge evaluation capabilities
- Practice critical assessment and constructive feedback

```bash
# In the CLI
/facts

# Output example:
🔍 Extracting facts from conversation history...
   Processing 415 characters of conversation...
✅ Fact extraction completed in 4.5s

📋 Facts extracted from conversation:
==================================================
 1. OpenAI creates GPT-4
 2. Microsoft Copilot uses GPT-4
 3. Google develops TensorFlow
==================================================
📊 Found 3 entities and 2 relationships
```

```bash
# Save to JSON-LD file
/facts myconversation

# Output example:
🔍 Extracting facts from conversation history...
   Processing 415 characters of conversation...
✅ Fact extraction completed in 3.3s
💾 Facts saved to myconversation.jsonld
📊 Saved 3 entities and 2 relationships as JSON-LD
```

**When to use:**
- Extract key information discussed in the conversation
- Create structured knowledge from chat sessions
- Document facts for later reference or analysis
- Generate semantic data for knowledge graphs

### `/stream` - Toggle Streaming Mode

Toggle real-time streaming of responses. You can also start with streaming enabled using the `--stream` flag.

```bash
# Toggle streaming mode in CLI
/stream

# Output examples:
🌊 Stream mode: ON
🌊 Stream mode: OFF
```

**Two ways to enable streaming:**

```bash
# Method 1: Start with streaming enabled
python -m abstractcore.utils.cli --provider ollama --model qwen3-coder:30b --stream

# Method 2: Toggle during conversation
👤 You: /stream
🌊 Stream mode: ON
👤 You: Write a haiku about programming
🤖 Assistant: Code flows like a stream... [appears word by word]
```

**When to use:**
- Real-time response display for better user experience
- Immediate feedback during long responses
- Interactive conversations where responsiveness matters

### `/history [n]` - Show Conversation History

Shows conversation history verbatim without truncation or numbering.

```bash
# Show all conversation history
/history

# Show last 2 interactions (4 messages: Q, A, Q, A)
/history 2

# Show last 1 interaction (2 messages: Q, A)
/history 1
```

**Output example:**
```bash
/history 2

📜 Last 2 interactions:

👤 You:
How do I create a function in Python?

🤖 Assistant:
You create functions in Python using the 'def' keyword:

def my_function(parameter1, parameter2):
    result = parameter1 + parameter2
    return result

Functions help organize code and avoid repetition.

👤 You:
What about error handling?

🤖 Assistant:
Python uses try/except blocks for error handling:

try:
    risky_operation()
except Exception as e:
    print(f"Error occurred: {e}")

This allows you to handle errors gracefully.

📊 Total tokens estimate: ~150
```

**After using /compact, /history shows both summary and recent messages:**
```bash
/history

📜 Conversation History:

📚 Earlier Conversation Summary:
──────────────────────────────────────────────────
The conversation covered Python basics, including variable types,
control structures, and best practices for writing clean code. The
user asked about functions and we discussed function definition,
parameters, and return values in detail.
──────────────────────────────────────────────────

💬 Recent Conversation:

👤 You:
What about error handling?

🤖 Assistant:
Python uses try/except blocks for error handling:

try:
    risky_operation()
except Exception as e:
    print(f"Error occurred: {e}")

This allows you to handle errors gracefully.

📊 Total tokens estimate: ~850
```

**Features:**
- **Verbatim display** - Shows complete messages without truncation
- **Summary visibility** - Displays compacted conversation summaries when available
- **Clean formatting** - No message numbers, just clean conversation flow
- **Full content** - Preserves code blocks, formatting, and long responses
- **Flexible viewing** - Show all history or specific number of interactions
- **Context continuity** - Always shows earlier conversation context after compaction

**When to use:**
- Review recent conversation context in full detail
- Copy/paste previous responses or code examples
- Verify exact wording of previous exchanges

### `/system [prompt]` - System Prompt Management

Control the system prompt that guides the AI's behavior.

```bash
# Show current system prompt
/system

# Change system prompt
/system You are a Python expert focused on data science and machine learning.

# Output example for show:
⚙️  Current System Prompt:
==================================================
📝 Fixed Part:
You are a helpful AI assistant.

🔧 Full Prompt (as seen by LLM):
System Message 1 (Base):
You are a helpful AI assistant.
==================================================

# Output example for change:
✅ System prompt updated!
📝 Old: You are a helpful AI assistant.
📝 New: You are a Python expert focused on data science and machine learning.
```

**Features:**
- **Show current prompt** - Displays both fixed part and full prompt seen by LLM
- **Change prompt** - Updates the system prompt while preserving tool configurations
- **Full visibility** - Shows all system messages including summaries from compaction
- **Safe updates** - Only changes the core prompt, preserves conversation history

**When to use:**
- Adapt AI behavior for specific tasks (coding, writing, analysis)
- Review what instructions the AI is currently following
- Switch between different AI personas during conversation
- Debug unexpected AI behavior by checking the active system prompt

### `/tooltag <opening_tag> <closing_tag>` - Test Tool Call Tag Rewriting

Test tool call tag rewriting with custom tags to verify compatibility with different agentic CLIs.

```bash
# Test LLaMA3 format (Crush CLI)
/tooltag '<function_call>' '</function_call>'

# Test XML format (Gemini CLI)
/tooltag '<tool_call>' '</tool_call>'

# Test Qwen3 format (Codex CLI)
/tooltag '<|tool_call|>' '</|tool_call|>'

# Test Gemma format
/tooltag '```tool_code' '```'

# Test custom format
/tooltag '<my_tool>' '</my_tool>'
```

**Features:**
- Tests custom tool call tag formats
- Compares with default behavior
- Works with both streaming and non-streaming modes
- Verifies tag rewriting functionality
- Shows detailed results and analysis

**Example Output:**
```
🏷️ Testing Tool Call Tag Rewriting
📝 Opening Tag: '<function_call>'
📝 Closing Tag: '</function_call>'
============================================================
🤖 Testing with openai:gpt-4o-mini
🌊 Streaming: OFF
📝 Prompt: Please help me with two tasks:
1. Get the current weather for Paris
2. Calculate what 15 * 23 equals

Use the available tools to help me with these tasks.

============================================================
🔧 Testing with custom tool call tags...
🤖 Assistant (with custom tags): <function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>

🔧 Testing without custom tags (default behavior)...
🤖 Assistant (default): <|tool_call|>{"name": "get_weather", "arguments": {"location": "Paris"}}</|tool_call|>

============================================================
📊 Summary:
   Model: openai:gpt-4o-mini
   Streaming: OFF
   Custom tags: '<function_call>'...'</function_call>'
   Custom tags found: ✅ YES
   Default tags found: ✅ YES
```

**When to use:**
- Test compatibility with specific agentic CLIs (Codex, Crush, Gemini CLI)
- Verify tool call tag rewriting works correctly
- Debug tool call format issues
- Compare different tag formats side by side
- Test both streaming and non-streaming modes

## Usage Examples

### Starting the CLI with Auto-Compact

```bash
# Start CLI with a model that supports auto-compaction
python -m abstractcore.utils.cli --provider ollama --model gemma3:1b

# Have a long conversation...
👤 You: What is Python?
🤖 Assistant: Python is a high-level programming language...

👤 You: What about data types?
🤖 Assistant: Python has several built-in data types...

# Check history
👤 You: /history 2

# Change system prompt for specific task
👤 You: /system You are an expert Python tutor focused on data science.

# Compact when needed
👤 You: /compact

# Extract facts from the conversation
👤 You: /facts

# Continue conversation with preserved context and new system prompt
👤 You: Can you explain more about functions?
🤖 Assistant: [Responds as Python tutor, refers to previous conversation context]
```

### Best Practices

1. **Regular History Checks**: Use `/history n` to review recent context
2. **Strategic Compaction**: Use `/compact` when:
   - Conversation becomes slow
   - You've covered multiple topics
   - Before switching contexts
3. **System Prompt Control**: Use `/system` to:
   - Prevent unwanted tool calls with specific instructions
   - Adapt AI behavior for different tasks
   - Debug unexpected responses by checking active prompts
4. **Context Verification**: After compaction, use `/history` to verify preserved context

### Controlling Tool Usage

If the AI is calling tools unnecessarily (like the example where it called `list_files` just to demonstrate capabilities):

```bash
# Check current system prompt to understand behavior
👤 You: /system

# Update to prevent unnecessary tool calls
👤 You: /system You are a helpful AI assistant. Only use tools when explicitly requested by the user or when necessary to answer a specific question. Do not demonstrate tool capabilities unprompted.

# Now the AI will be less likely to call tools unnecessarily
👤 You: Who are you?
🤖 Assistant: I am an AI assistant. [No tool calls]
```

### Technical Details

- **Compaction Model**: Uses `gemma3:1b` for fast, local processing
- **Preservation**: Keeps last 6 messages (3 exchanges) by default
- **Fallback**: If `gemma3:1b` unavailable, uses current provider
- **Token Estimation**: Provides accurate token count estimates using centralized TokenUtils
- **Message Types**: Distinguishes between system, user, and assistant messages

### Error Handling

The commands gracefully handle various scenarios:
- Not enough history to compact
- Invalid parameters for `/last`
- Missing models or providers
- Empty conversation history

All errors are clearly reported with helpful messages.