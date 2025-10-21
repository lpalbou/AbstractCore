# AbstractCore ReAct Agent System

A clean, efficient implementation of the **ReAct (Reasoning and Acting)** pattern for autonomous agents that can think, act, and observe iteratively to solve complex tasks.

## Overview

The ReAct pattern enables agents to:
- **Think**: Analyze the current situation and plan next actions
- **Act**: Execute planned actions using available tools  
- **Observe**: Analyze results and update understanding
- **Adapt**: Modify strategy based on discoveries and observations

This implementation provides a robust state machine with clear separation of concerns, verbose logging, and seamless integration with AbstractCore's tool system.

## Key Features

- ✅ **Clear State Separation**: Think, Act, Observe phases with defined responsibilities
- ✅ **Scratchpad System**: Tracks thoughts, observations, and discoveries
- ✅ **Adaptive Planning**: Adjusts strategy based on new information
- ✅ **Tool Integration**: Works with all AbstractCore tools
- ✅ **Verbose Logging**: Detailed reasoning process visibility
- ✅ **CLI Integration**: Easy-to-use commands in AbstractCore CLI
- ✅ **Session Persistence**: Save and restore agent sessions
- ✅ **Planning Tools**: Built-in tools for reflection and meta-reasoning

## Quick Start

### Basic Usage

```python
from abstractcore import create_llm
from abstractcore.react import ReActAgent

# Create LLM
llm = create_llm("openai", model="gpt-4o-mini")

# Create ReAct agent
agent = ReActAgent(llm, verbose=True)

# Run autonomous task
result = agent.run("Find all Python files in the current directory and count their lines")

print(f"Result: {result.final_answer}")
print(f"Iterations: {result.iterations}")
```

### CLI Integration

The ReAct agent integrates seamlessly with the AbstractCore CLI:

```bash
# Start CLI with ReAct support
python -m abstractcore.react.cli_integration --provider openai --model gpt-4o-mini

# Run ReAct agent
/react Find all Python files and analyze their structure

# Check agent status
/agent status

# View reasoning scratchpad
/scratchpad

# Save agent session
/agent save my_analysis_session
```

## Architecture

### State Machine

The ReAct agent operates as a state machine with four primary states:

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  THINK  │───▶│   ACT   │───▶│ OBSERVE │
└─────────┘    └─────────┘    └─────────┘
      ▲                             │
      └─────────────────────────────┘
                    │
                    ▼
              ┌─────────┐
              │COMPLETE │
              └─────────┘
```

#### Think State
- Analyzes current situation and progress
- Reviews scratchpad observations
- Plans next action based on goal and discoveries
- Updates strategy when needed

#### Act State  
- Executes planned actions using available tools
- Handles tool execution errors gracefully
- Captures results for observation

#### Observe State
- Processes and interprets action results
- Updates scratchpad with observations
- Determines next steps or completion

### Scratchpad System

The scratchpad maintains a structured history of the agent's reasoning:

```python
# Types of entries
scratchpad.add_thought("Analyzing the current directory structure...")
scratchpad.add_observation("Found 15 Python files in the directory")
scratchpad.add_discovery("The codebase follows a modular architecture")
scratchpad.add_plan("Next: analyze each module's purpose and dependencies")

# Formatted output for LLM consumption
content = scratchpad.get_formatted_content()
summary = scratchpad.get_summary()
```

## Available Tools

### Core Tools (from AbstractCore)
- `list_files`: List directory contents
- `read_file`: Read file contents
- `write_file`: Create/modify files
- `search_files`: Search for patterns in files
- `execute_command`: Run shell commands

### Planning Tools (ReAct-specific)
- `create_plan`: Structure goals into actionable steps
- `reflect_on_progress`: Analyze what's working and adjust strategy
- `assess_completion`: Evaluate if goals are achieved
- `prioritize_actions`: Rank possible next actions by importance

## CLI Commands

### ReAct Execution
```bash
/react <goal>                    # Run ReAct agent with specified goal
/react Find all TODO comments    # Example: search and categorize TODOs
```

### Agent Management
```bash
/agent status                    # Show agent status and configuration
/agent reset                     # Reset agent state and scratchpad
/agent tools                     # List available tools
/agent save <file>               # Save session and scratchpad
/agent load <file>               # Load previous session
```

### Scratchpad
```bash
/scratchpad                      # Show current scratchpad content
```

## Examples

### File Analysis
```python
agent = ReActAgent(llm, verbose=True)
result = agent.run("Analyze the codebase structure and identify the main components")
```

### Code Search
```python
result = agent.run("Find all TODO comments and categorize them by priority")
```

### System Investigation
```python
result = agent.run("Check system resources and identify any performance bottlenecks")
```

## Configuration

### Agent Parameters
```python
agent = ReActAgent(
    llm=llm,
    max_iterations=15,           # Maximum Think-Act-Observe cycles
    verbose=True,                # Show detailed reasoning process
    include_planning_tools=True, # Include meta-reasoning tools
    scratchpad_max_entries=100   # Maximum scratchpad entries
)
```

### Custom Tools
```python
from abstractcore.tools import tool

@tool
def custom_analysis(data: str) -> str:
    """Custom analysis tool"""
    return f"Analysis result: {data}"

# Add to agent
agent.add_tool(custom_analysis)
```

## Session Persistence

Save and restore agent sessions with full context:

```python
# Save session
agent.save_session("analysis_session.json")

# Load session (preserves scratchpad and reasoning history)
agent.load_session("analysis_session.json")
```

## Best Practices

### Goal Formulation
- Be specific and actionable: ✅ "Find all Python files and count their lines"
- Avoid vague requests: ❌ "Analyze the code"

### Iteration Management
- Start with reasonable iteration limits (5-15)
- Monitor progress through scratchpad
- Use verbose mode for debugging

### Tool Selection
- Include domain-specific tools for better results
- Use planning tools for complex multi-step tasks
- Remove unnecessary tools to reduce confusion

### Error Handling
```python
result = agent.run(goal)
if result.success:
    print(f"Success: {result.final_answer}")
else:
    print(f"Failed: {result.error}")
    # Check scratchpad for debugging
    print(agent.scratchpad.get_summary())
```

## Integration with AbstractCore

The ReAct system is fully integrated with AbstractCore:

- **Providers**: Works with all LLM providers (OpenAI, Anthropic, Ollama, etc.)
- **Tools**: Uses AbstractCore's universal tool system
- **Sessions**: Compatible with BasicSession for conversation management
- **Media**: Supports file attachments and media processing
- **Configuration**: Respects AbstractCore configuration settings

## Troubleshooting

### Common Issues

**Agent gets stuck in loops:**
- Reduce max_iterations
- Check if goal is too vague
- Review scratchpad for repeated patterns

**Tool execution failures:**
- Verify tool availability with `/agent tools`
- Check tool permissions and dependencies
- Use debug mode for detailed error info

**Poor reasoning quality:**
- Try different LLM models
- Adjust temperature settings
- Provide more specific goals

### Debug Mode
```python
agent = ReActAgent(llm, verbose=True)  # Enable detailed logging
```

```bash
# CLI debug mode
python -m abstractcore.react.cli_integration --debug --provider openai --model gpt-4o
```

## Performance Considerations

- **Model Selection**: Larger models generally provide better reasoning
- **Iteration Limits**: Balance thoroughness vs. execution time
- **Tool Efficiency**: Optimize tool execution for faster cycles
- **Scratchpad Size**: Limit entries to prevent context overflow

## Future Enhancements

- Multi-agent collaboration
- Hierarchical goal decomposition  
- Learning from previous sessions
- Integration with external knowledge bases
- Advanced planning algorithms

---

For more examples and advanced usage, see the `example.py` file in this directory.
