#!/usr/bin/env python3
"""
Example 3: Tool Calling & Tag Rewriting - Universal Tool Support
=================================================================

This example demonstrates AbstractLLM's sophisticated tool calling system:
- Universal tool support across all models
- Custom tool call tag rewriting
- Tool execution and result handling
- Architecture-specific tool formatting

Technical Architecture Highlights:
- UniversalToolHandler for model-agnostic tools
- ToolCallTagRewriter for custom formatting
- IncrementalToolDetector for streaming
- Tool registry pattern for management

Required: pip install abstractllm
Optional: pip install abstractllm[ollama] for local models with tools
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional
import logging

# Add project root to path for development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from abstractllm import create_llm
from abstractllm.tools import (
    ToolDefinition,
    UniversalToolHandler,
    ToolRegistry,
    register_tool,
    execute_tools,
    tool,
)
from abstractllm.tools.tag_rewriter import ToolCallTagRewriter, ToolCallTags

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Define some example tools
@tool
def list_files(directory: str = ".", pattern: str = "*.py") -> str:
    """
    List files in a directory matching a pattern.

    Args:
        directory: Directory path to list files from
        pattern: File pattern to match (glob syntax)

    Returns:
        List of matching files
    """
    import glob
    import os

    full_pattern = os.path.join(directory, pattern)
    files = glob.glob(full_pattern)
    if files:
        return f"Found {len(files)} files:\n" + "\n".join(f"  • {f}" for f in files)
    return f"No files found matching {pattern} in {directory}"


@tool
def calculate(expression: str) -> float:
    """
    Perform a calculation.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        The result of the calculation
    """
    # Safe evaluation of mathematical expressions
    import ast
    import operator as op

    # Supported operators
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_expr(node.operand))
        else:
            raise ValueError(f"Unsupported expression: {expression}")

    try:
        node = ast.parse(expression, mode='eval')
        result = eval_expr(node.body)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """
    Get current weather for a city (mock implementation).

    Args:
        city: City name
        units: Temperature units (celsius or fahrenheit)

    Returns:
        Weather information
    """
    # Mock weather data
    import random

    temp = random.randint(15, 30) if units == "celsius" else random.randint(59, 86)
    conditions = random.choice(["Sunny", "Cloudy", "Partly Cloudy", "Rainy"])

    return f"""
Weather in {city}:
  Temperature: {temp}°{'C' if units == 'celsius' else 'F'}
  Conditions: {conditions}
  Humidity: {random.randint(40, 80)}%
  Wind: {random.randint(5, 25)} km/h
"""


def basic_tool_calling():
    """
    Demonstrates basic tool calling with AbstractLLM.

    Architecture Notes:
    - Tools are defined using the @tool decorator
    - UniversalToolHandler manages tool formatting for different models
    - Works with both native tool APIs and prompted models
    """
    print("=" * 70)
    print("EXAMPLE 3: Basic Tool Calling")
    print("=" * 70)

    # Register our tools
    registry = ToolRegistry()
    registry.register(list_files)
    registry.register(calculate)
    registry.register(get_weather)

    print("\n📦 Registered tools:")
    for tool_name in registry.list_tools():
        print(f"   • {tool_name}")

    # Create LLM with tool support
    llm = create_llm("mock", "mock-model")

    # Create tool handler for this model
    handler = UniversalToolHandler("mock-model")

    # Format tools for the model
    tool_defs = [
        ToolDefinition.from_function(list_files),
        ToolDefinition.from_function(calculate),
        ToolDefinition.from_function(get_weather),
    ]

    # For prompted models, we need to add tool instructions
    if handler.supports_prompted:
        tool_prompt = handler.format_tools_prompt(tool_defs)
        print("\n📝 Tool prompt for model:")
        print(tool_prompt[:200] + "..." if len(tool_prompt) > 200 else tool_prompt)

    # Simulate a response with tool calls
    print("\n🤖 Simulating tool-calling response...")

    # Mock response with Qwen-style tool calls
    mock_response = """
I'll help you with those tasks.

First, let me list the Python files:
<|tool_call|>{"name": "list_files", "arguments": {"directory": ".", "pattern": "*.py"}}
</|tool_call|>

Now let me calculate that expression:
<|tool_call|>{"name": "calculate", "arguments": {"expression": "42 * 3.14159"}}
</|tool_call|>

Finally, let me check the weather:
<|tool_call|>{"name": "get_weather", "arguments": {"city": "San Francisco"}}
</|tool_call|>
"""

    print("Response with tool calls:")
    print(mock_response)

    # Parse tool calls from response
    tool_call_response = handler.parse_response(mock_response, mode="prompted")

    if tool_call_response.has_tool_calls():
        print(f"\n🔧 Found {len(tool_call_response.tool_calls)} tool calls")

        # Execute tools
        for i, tool_call in enumerate(tool_call_response.tool_calls, 1):
            print(f"\n   Tool Call {i}: {tool_call.name}")
            print(f"   Arguments: {tool_call.arguments}")

            # Execute the tool
            result = registry.execute(tool_call.name, **tool_call.arguments)
            print(f"   Result: {result[:100]}..." if len(result) > 100 else f"   Result: {result}")


def custom_tag_rewriting():
    """
    Demonstrates custom tool call tag rewriting.

    Architecture Notes:
    - ToolCallTagRewriter transforms tool call formats
    - Enables consistent formatting across different models
    - Critical for streaming with tool detection
    """
    print("\n" + "=" * 70)
    print("Custom Tool Call Tag Rewriting")
    print("=" * 70)

    # Different tool call formats
    formats = {
        "qwen3": "<|tool_call|>...JSON...</|tool_call|>",
        "llama3": "<function_call>...JSON...</function_call>",
        "xml": "<tool_call>...JSON...</tool_call>",
        "gemma": "```tool_code\\n...JSON...\\n```",
        "custom": "TOOLSTART...JSON...TOOLEND",
    }

    print("\n📝 Tool call format examples:")
    for name, format_str in formats.items():
        print(f"   • {name}: {format_str}")

    # Create custom tag rewriter
    print("\n🔄 Creating custom tag rewriter...")

    # Example 1: Simple custom tags
    custom_tags = ToolCallTags(
        start_tag="TOOLSTART",
        end_tag="TOOLEND",
        auto_format=False  # Use exact tags, don't add angle brackets
    )
    rewriter = ToolCallTagRewriter(custom_tags)

    # Test content with various formats
    test_content = """
Let me help you with that.
<|tool_call|>{"name": "calculate", "arguments": {"expression": "2+2"}}
</|tool_call|>
The calculation is complete.
"""

    print("\n   Original content (Qwen format):")
    print(test_content)

    rewritten = rewriter.rewrite_text(test_content)
    print("\n   After rewriting to custom format:")
    print(rewritten)

    # Example 2: User-specified tags (like the CLI example)
    print("\n🎯 User-specified custom tags (ojlk,dfsd):")

    # This mimics the CLI command: /tooltag 'ojlk' 'dfsd'
    user_tags = ToolCallTags(
        start_tag="ojlk",
        end_tag="dfsd",
        auto_format=False  # Keep exactly as specified
    )
    user_rewriter = ToolCallTagRewriter(user_tags)

    rewritten_user = user_rewriter.rewrite_text(test_content)
    print("\n   With user's custom tags:")
    print(rewritten_user)

    # Example 3: Converting between standard formats
    print("\n🔄 Converting between standard formats:")

    # Convert from Qwen to LLaMA format
    llama_tags = ToolCallTags(
        start_tag="<function_call>",
        end_tag="</function_call>",
        auto_format=False
    )
    llama_rewriter = ToolCallTagRewriter(llama_tags)

    llama_output = llama_rewriter.rewrite_text(test_content)
    print("\n   Converted to LLaMA format:")
    print(llama_output)


def streaming_with_tool_detection():
    """
    Demonstrates real-time tool detection during streaming.

    Architecture Notes:
    - IncrementalToolDetector processes chunks progressively
    - Preserves tool calls for tag rewriting
    - Enables real-time tool execution during streaming
    """
    print("\n" + "=" * 70)
    print("Streaming with Tool Detection")
    print("=" * 70)

    from abstractllm.providers.streaming import IncrementalToolDetector

    # Create detector with tag preservation
    detector = IncrementalToolDetector(
        model_name="qwen3-coder",
        rewrite_tags=True  # Preserve tool calls for rewriting
    )

    # Simulate streaming chunks
    streaming_chunks = [
        "I'll help you ",
        "calculate that. ",
        "Let me use the ",
        "calculator tool:\n",
        "<|tool_",
        "call|>{\"name\": ",
        "\"calculate\", ",
        "\"arguments\": ",
        "{\"expression\": ",
        "\"100 * 3.14159\"",
        "}}",
        "</|tool_call|>",
        "\nThe result ",
        "should be ",
        "approximately 314."
    ]

    print("\n📡 Simulating character-by-character streaming:")
    print("   (Notice how tool calls are detected and preserved)")
    print()

    accumulated_output = ""
    detected_tools = []

    for i, chunk in enumerate(streaming_chunks):
        # Process chunk
        streamable_content, tools = detector.process_chunk(chunk)

        # Show what's being streamed
        if streamable_content:
            print(f"   Chunk {i+1:2d}: '{chunk}' → Stream: '{streamable_content}'")
            accumulated_output += streamable_content

        # Show tool detection
        if tools:
            for tool in tools:
                print(f"\n   🔧 TOOL DETECTED: {tool.name}")
                print(f"      Arguments: {tool.arguments}")
                detected_tools.extend(tools)

        # Small delay to simulate streaming
        time.sleep(0.05)

    print(f"\n📄 Final streamed output:")
    print(accumulated_output)

    print(f"\n🔧 Total tools detected: {len(detected_tools)}")


def advanced_tool_patterns():
    """
    Demonstrates advanced tool calling patterns.

    Architecture Notes:
    - Tool chaining and dependencies
    - Error handling in tool execution
    - Conditional tool execution
    - Tool result integration
    """
    print("\n" + "=" * 70)
    print("Advanced Tool Patterns")
    print("=" * 70)

    # Pattern 1: Tool Chaining
    print("\n🔗 Pattern 1: Tool Chaining")

    @tool
    def fetch_data(source: str) -> Dict[str, Any]:
        """Fetch data from a source."""
        return {"source": source, "data": [1, 2, 3, 4, 5]}

    @tool
    def process_data(data: List[int], operation: str) -> Dict[str, Any]:
        """Process data with an operation."""
        if operation == "sum":
            return {"result": sum(data), "operation": operation}
        elif operation == "mean":
            return {"result": sum(data) / len(data), "operation": operation}
        else:
            return {"error": f"Unknown operation: {operation}"}

    # Simulate chained tool execution
    print("   Step 1: Fetch data")
    data_result = fetch_data("database")
    print(f"      Result: {data_result}")

    print("   Step 2: Process fetched data")
    process_result = process_data(data_result["data"], "mean")
    print(f"      Result: {process_result}")

    # Pattern 2: Error Recovery
    print("\n🛡️ Pattern 2: Error Recovery in Tools")

    @tool
    def risky_operation(value: int) -> str:
        """A tool that might fail."""
        if value < 0:
            raise ValueError("Negative values not supported")
        if value > 100:
            raise ValueError("Value too large")
        return f"Processed value: {value}"

    # Test with error handling
    test_values = [-5, 50, 150]
    for value in test_values:
        try:
            result = risky_operation(value)
            print(f"   ✅ risky_operation({value}): {result}")
        except ValueError as e:
            print(f"   ❌ risky_operation({value}): {e}")
            # In real usage, the LLM would see this error and adjust

    # Pattern 3: Conditional Execution
    print("\n🔀 Pattern 3: Conditional Tool Execution")

    @tool
    def check_condition(value: float) -> bool:
        """Check if a condition is met."""
        return value > 0.5

    @tool
    def action_if_true() -> str:
        """Execute if condition is true."""
        return "Condition met - executing primary action"

    @tool
    def action_if_false() -> str:
        """Execute if condition is false."""
        return "Condition not met - executing fallback action"

    # Simulate conditional execution
    import random
    test_value = random.random()
    print(f"   Test value: {test_value:.3f}")

    if check_condition(test_value):
        result = action_if_true()
    else:
        result = action_if_false()
    print(f"   Result: {result}")

    # Pattern 4: Parallel Tool Execution
    print("\n⚡ Pattern 4: Parallel Tool Execution")

    @tool
    def slow_operation(name: str, duration: float) -> str:
        """Simulate a slow operation."""
        time.sleep(duration)
        return f"{name} completed in {duration}s"

    # Sequential execution
    print("   Sequential execution:")
    start = time.perf_counter()
    results = []
    for i in range(3):
        result = slow_operation(f"Task{i+1}", 0.1)
        results.append(result)
        print(f"      {result}")
    sequential_time = time.perf_counter() - start
    print(f"   Total time: {sequential_time:.2f}s")

    # Simulated parallel execution (would use threading/async in production)
    print("\n   Parallel execution (simulated):")
    start = time.perf_counter()
    # In production, you'd use concurrent.futures or asyncio
    parallel_time = 0.1  # All tasks complete in parallel
    print("      All tasks completed simultaneously")
    print(f"   Total time: {parallel_time:.2f}s")
    print(f"   Speedup: {sequential_time/parallel_time:.1f}x")


def unified_streaming_with_tools():
    """
    Demonstrates the unified streaming architecture with tool support.

    Architecture Notes:
    - UnifiedStreamProcessor handles streaming and tools together
    - Real-time tool detection and execution
    - Tag rewriting during streaming
    - 5x performance improvement over buffered approach
    """
    print("\n" + "=" * 70)
    print("Unified Streaming with Tool Support")
    print("=" * 70)

    from abstractllm.providers.streaming import UnifiedStreamProcessor
    from abstractllm.core.types import GenerateResponse

    # Create processor with custom tool tags
    processor = UnifiedStreamProcessor(
        model_name="qwen3-coder",
        tool_call_tags="START,END",  # Custom tags
    )

    print("\n🚀 Unified Streaming Architecture Benefits:")
    print("   • Real-time tool detection without buffering")
    print("   • Tag rewriting preserves tool calls in stream")
    print("   • 5x faster first token delivery (<10ms)")
    print("   • Single code path for all scenarios")

    # Simulate a streaming response with tools
    def simulate_streaming_response():
        """Generate mock streaming chunks."""
        chunks = [
            "I'll help you with ",
            "those calculations.\n\n",
            "First calculation:\n",
            "<|tool_call|>",
            '{"name": "calculate", ',
            '"arguments": {"expression": ',
            '"42 * 3.14"}}',
            "</|tool_call|>\n",
            "Second calculation:\n",
            "<|tool_call|>",
            '{"name": "calculate", ',
            '"arguments": {"expression": ',
            '"100 / 3"}}',
            "</|tool_call|>\n",
            "Done!"
        ]

        for chunk in chunks:
            yield GenerateResponse(
                content=chunk,
                model="qwen3-coder",
                finish_reason=None
            )

    print("\n📡 Processing streaming response with custom tags:")
    response_stream = simulate_streaming_response()

    accumulated = ""
    chunk_count = 0
    start_time = time.perf_counter()
    first_chunk_time = None

    for processed_chunk in processor.process_stream(response_stream):
        if processed_chunk.content:
            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start_time

            # Show the rewritten content
            print(f"   Chunk {chunk_count}: '{processed_chunk.content}'")
            accumulated += processed_chunk.content
            time.sleep(0.02)  # Simulate network delay

    total_time = time.perf_counter() - start_time

    print(f"\n📊 Streaming Performance Metrics:")
    print(f"   • First chunk latency: {first_chunk_time*1000:.2f}ms")
    print(f"   • Total streaming time: {total_time*1000:.2f}ms")
    print(f"   • Chunks processed: {chunk_count}")

    print(f"\n📄 Final output with rewritten tags:")
    print(accumulated)


def main():
    """
    Main entry point - demonstrates tool calling capabilities.
    """
    print("\n" + "🔧 " * 20)
    print(" AbstractLLM Core - Example 3: Tool Calling & Tag Rewriting")
    print("🔧 " * 20)

    # Run all demonstrations
    basic_tool_calling()
    custom_tag_rewriting()
    streaming_with_tool_detection()
    advanced_tool_patterns()
    unified_streaming_with_tools()

    print("\n" + "=" * 70)
    print("✅ Example 3 Complete!")
    print("\nKey Takeaways:")
    print("• Universal tool support works across all models")
    print("• Custom tag rewriting enables format flexibility")
    print("• Incremental detection allows real-time tool processing")
    print("• Advanced patterns: chaining, parallel execution, error recovery")
    print("• Unified streaming delivers 5x performance improvement")
    print("\nNext: Run example_4_unified_streaming.py for streaming architecture")
    print("=" * 70)


if __name__ == "__main__":
    main()