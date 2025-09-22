#!/usr/bin/env python3
"""
Basic Tool Usage Examples for AbstractLLM Core

This example demonstrates how to use tools with different providers
in both streaming and non-streaming modes.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from abstractllm import AbstractLLM
from abstractllm.events import EventType

# Example tool definitions
EXAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state/country, e.g. 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                    "default": "fahrenheit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]

def print_separator(title: str):
    """Print a nice separator for different sections"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def basic_tool_usage_non_streaming():
    """Demonstrate basic tool usage without streaming"""
    print_separator("Basic Tool Usage (Non-Streaming)")

    # Initialize with different providers to show universal support
    providers_to_test = [
        ("ollama", "llama3.2:3b"),
        ("openai", "gpt-3.5-turbo"),
        ("anthropic", "claude-3-haiku-20240307"),
    ]

    for provider_name, model in providers_to_test:
        print(f"\n--- Testing {provider_name.upper()} Provider ---")

        try:
            # Initialize provider
            llm = AbstractLLM(provider=provider_name, model=model)

            # Simple tool usage
            response = llm.generate(
                prompt="What's the weather like in Paris and what's 15 * 23?",
                tools=EXAMPLE_TOOLS,
                system_prompt="You are a helpful assistant. Use the provided tools when needed."
            )

            print(f"Response: {response.content}")

            if response.tool_calls:
                print(f"Tool calls made: {len(response.tool_calls)}")
                for call in response.tool_calls:
                    print(f"  - {call.get('name')}: {call.get('arguments')}")

        except Exception as e:
            print(f"Error with {provider_name}: {str(e)}")
            print("(This is expected if the provider is not configured)")

def basic_tool_usage_streaming():
    """Demonstrate basic tool usage with streaming"""
    print_separator("Basic Tool Usage (Streaming)")

    try:
        # Use Ollama for streaming example (commonly available)
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        print("Streaming response with tool calls:")
        print("Content: ", end="", flush=True)

        for chunk in llm.generate(
            prompt="Please calculate 42 * 17 and then search for information about Python programming",
            tools=EXAMPLE_TOOLS,
            stream=True,
            system_prompt="You are a helpful assistant. Use tools when appropriate."
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)

            if chunk.tool_calls:
                print(f"\n[Tool calls: {len(chunk.tool_calls)}]")

        print("\n")

    except Exception as e:
        print(f"Error with streaming: {str(e)}")
        print("(This is expected if Ollama is not running)")

def tool_usage_with_event_handling():
    """Demonstrate tool usage with event handling and prevention"""
    print_separator("Tool Usage with Event Handling")

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        # Track tool executions
        tool_executions = []

        def handle_before_tool_execution(event):
            """Handler for before tool execution"""
            print(f"About to execute tools: {[call.name for call in event.data['tool_calls']]}")
            tool_executions.append({
                'type': 'before',
                'tool_calls': event.data['tool_calls'],
                'can_prevent': event.data.get('can_prevent', False)
            })

            # Example: Prevent web search tools for demonstration
            for call in event.data['tool_calls']:
                if call.name == 'search_web':
                    print("🚫 Preventing web search tool execution for security")
                    event.prevent()
                    break

        def handle_after_tool_execution(event):
            """Handler for after tool execution"""
            print(f"Executed tools with results: {len(event.data['results'])} results")
            tool_executions.append({
                'type': 'after',
                'tool_calls': event.data['tool_calls'],
                'results': event.data['results']
            })

        # Register event handlers
        llm.add_event_listener(EventType.BEFORE_TOOL_EXECUTION, handle_before_tool_execution)
        llm.add_event_listener(EventType.AFTER_TOOL_EXECUTION, handle_after_tool_execution)

        # Generate with tools
        response = llm.generate(
            prompt="Calculate 100 / 4 and search for Python tutorials",
            tools=EXAMPLE_TOOLS,
            system_prompt="Use the provided tools to help with the request."
        )

        print(f"Final response: {response.content}")
        print(f"Total tool execution events: {len(tool_executions)}")

    except Exception as e:
        print(f"Error with event handling: {str(e)}")

def multiple_tool_calls_example():
    """Demonstrate multiple tool calls in a single request"""
    print_separator("Multiple Tool Calls Example")

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        response = llm.generate(
            prompt="""Please help me with the following tasks:
            1. Calculate the area of a circle with radius 5 (use π = 3.14159)
            2. Get the weather for Tokyo, Japan
            3. Calculate 25% of 800
            4. Search for information about machine learning""",
            tools=EXAMPLE_TOOLS,
            system_prompt="Complete all the requested tasks using the available tools."
        )

        print(f"Response: {response.content}")

        if response.tool_calls:
            print(f"\nTool calls made: {len(response.tool_calls)}")
            for i, call in enumerate(response.tool_calls, 1):
                print(f"  {i}. {call.get('name')}: {call.get('arguments')}")

    except Exception as e:
        print(f"Error with multiple tools: {str(e)}")

def conversational_tool_usage():
    """Demonstrate tool usage in conversation context"""
    print_separator("Conversational Tool Usage")

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        # First turn
        response1 = llm.generate(
            prompt="What's 15 * 8?",
            tools=EXAMPLE_TOOLS,
            system_prompt="You are a helpful math assistant."
        )

        print(f"Turn 1 - User: What's 15 * 8?")
        print(f"Turn 1 - Assistant: {response1.content}")

        # Continue conversation with history
        messages = [
            {"role": "user", "content": "What's 15 * 8?"},
            {"role": "assistant", "content": response1.content}
        ]

        response2 = llm.generate(
            prompt="Now calculate the square root of that result",
            messages=messages,
            tools=EXAMPLE_TOOLS,
            system_prompt="You are a helpful math assistant."
        )

        print(f"Turn 2 - User: Now calculate the square root of that result")
        print(f"Turn 2 - Assistant: {response2.content}")

    except Exception as e:
        print(f"Error with conversation: {str(e)}")

def main():
    """Run all examples"""
    print("AbstractLLM Core - Tool Usage Examples")
    print("=====================================")
    print("This script demonstrates various ways to use tools with AbstractLLM Core.")
    print("Note: Some examples may fail if providers are not configured.")

    # Run all examples
    basic_tool_usage_non_streaming()
    basic_tool_usage_streaming()
    tool_usage_with_event_handling()
    multiple_tool_calls_example()
    conversational_tool_usage()

    print_separator("Examples Complete")
    print("For more advanced usage, check out the other example scripts!")

if __name__ == "__main__":
    main()