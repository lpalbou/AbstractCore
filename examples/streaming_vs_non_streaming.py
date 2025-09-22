#!/usr/bin/env python3
"""
Streaming vs Non-Streaming Tool Usage Comparison

This example specifically demonstrates the differences between
streaming and non-streaming tool usage with AbstractLLM Core.
"""

import os
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from abstractllm import AbstractLLM

# Simple tools for demonstration
DEMO_TOOLS = [
    {
        "name": "get_time",
        "description": "Get the current time",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone (e.g., 'UTC', 'EST', 'PST')",
                    "default": "UTC"
                }
            }
        }
    },
    {
        "name": "calculate",
        "description": "Perform a calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_fact",
        "description": "Get an interesting fact about a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to get a fact about"
                }
            },
            "required": ["topic"]
        }
    }
]

def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def demonstrate_non_streaming():
    """Demonstrate non-streaming tool usage"""
    print_section("Non-Streaming Tool Usage")

    try:
        # Initialize LLM
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        print("Request: Calculate 15 * 23 and tell me an interesting fact about mathematics")
        print("\nProcessing request (non-streaming)...")

        start_time = time.time()

        # Non-streaming generation
        response = llm.generate(
            prompt="Please calculate 15 * 23 and then tell me an interesting fact about mathematics",
            tools=DEMO_TOOLS,
            system_prompt="You are a helpful assistant. Use the provided tools when needed."
        )

        end_time = time.time()

        print(f"\n✓ Complete response received in {end_time - start_time:.2f} seconds:")
        print(f"Response: {response.content}")

        if response.tool_calls:
            print(f"\nTool calls made: {len(response.tool_calls)}")
            for i, call in enumerate(response.tool_calls, 1):
                print(f"  {i}. {call.get('name')}: {call.get('arguments')}")

        # Advantages of non-streaming
        print(f"\n📋 Non-Streaming Advantages:")
        print(f"  • Complete response available immediately")
        print(f"  • Easy to process and validate entire response")
        print(f"  • All tool calls visible at once")
        print(f"  • Simpler error handling")
        print(f"  • Better for batch processing")

    except Exception as e:
        print(f"Error in non-streaming: {str(e)}")

def demonstrate_streaming():
    """Demonstrate streaming tool usage"""
    print_section("Streaming Tool Usage")

    try:
        # Initialize LLM
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        print("Request: Calculate 42 * 17 and get the current time")
        print("\nProcessing request (streaming)...")

        start_time = time.time()
        first_content_time = None
        last_content_time = None
        chunk_count = 0
        tool_call_count = 0

        print(f"\n📡 Streaming Response:")
        print(f"Content: ", end="", flush=True)

        # Streaming generation
        for chunk in llm.generate(
            prompt="Please calculate 42 * 17 and then get the current time in UTC",
            tools=DEMO_TOOLS,
            stream=True,
            system_prompt="You are a helpful assistant. Use the provided tools when needed."
        ):
            chunk_count += 1

            if chunk.content:
                if first_content_time is None:
                    first_content_time = time.time()
                last_content_time = time.time()
                print(chunk.content, end="", flush=True)

            if chunk.tool_calls:
                tool_call_count += len(chunk.tool_calls)
                print(f"\n[🔧 Tool execution: {len(chunk.tool_calls)} tool(s)]", end="", flush=True)

        end_time = time.time()

        print(f"\n\n✓ Streaming completed in {end_time - start_time:.2f} seconds")
        print(f"  • Total chunks: {chunk_count}")
        print(f"  • Tool calls: {tool_call_count}")

        if first_content_time:
            print(f"  • Time to first content: {first_content_time - start_time:.2f} seconds")

        # Advantages of streaming
        print(f"\n📋 Streaming Advantages:")
        print(f"  • Real-time response display")
        print(f"  • Lower perceived latency")
        print(f"  • Progressive rendering possible")
        print(f"  • Better user experience for long responses")
        print(f"  • Can handle very large responses efficiently")

    except Exception as e:
        print(f"Error in streaming: {str(e)}")

def side_by_side_comparison():
    """Run both streaming and non-streaming for direct comparison"""
    print_section("Side-by-Side Performance Comparison")

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        # Same prompt for both
        prompt = "Calculate 100 / 4, get an interesting fact about space, and tell me the current time"
        system_prompt = "You are a helpful assistant that uses tools efficiently."

        # Test non-streaming
        print("🔄 Testing Non-Streaming...")
        start_time_ns = time.time()

        response_ns = llm.generate(
            prompt=prompt,
            tools=DEMO_TOOLS,
            system_prompt=system_prompt
        )

        end_time_ns = time.time()
        non_streaming_time = end_time_ns - start_time_ns

        print(f"✓ Non-streaming completed in {non_streaming_time:.2f} seconds")
        print(f"Response length: {len(response_ns.content)} characters")

        # Test streaming
        print(f"\n🔄 Testing Streaming...")
        start_time_s = time.time()
        first_chunk_time = None
        streaming_content = ""
        chunk_count = 0

        for chunk in llm.generate(
            prompt=prompt,
            tools=DEMO_TOOLS,
            stream=True,
            system_prompt=system_prompt
        ):
            chunk_count += 1
            if chunk.content:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                streaming_content += chunk.content

        end_time_s = time.time()
        streaming_time = end_time_s - start_time_s

        print(f"✓ Streaming completed in {streaming_time:.2f} seconds")
        print(f"Response length: {len(streaming_content)} characters")
        print(f"Total chunks: {chunk_count}")

        if first_chunk_time:
            print(f"Time to first chunk: {first_chunk_time - start_time_s:.2f} seconds")

        # Comparison summary
        print(f"\n📊 Performance Summary:")
        print(f"  Non-streaming total time: {non_streaming_time:.2f}s")
        print(f"  Streaming total time:     {streaming_time:.2f}s")

        if first_chunk_time:
            print(f"  Time to first content:    {first_chunk_time - start_time_s:.2f}s")

        time_diff = abs(non_streaming_time - streaming_time)
        faster_method = "Non-streaming" if non_streaming_time < streaming_time else "Streaming"
        print(f"  Faster method: {faster_method} by {time_diff:.2f}s")

    except Exception as e:
        print(f"Error in comparison: {str(e)}")

def when_to_use_which():
    """Guidelines on when to use streaming vs non-streaming"""
    print_section("When to Use Which Approach")

    print("""
🔧 Use NON-STREAMING when:
  • You need the complete response before processing
  • Implementing batch operations
  • Building APIs that return complete data
  • Working with structured data that needs validation
  • Response size is predictable and small
  • You need to analyze the entire response content
  • Building automated systems/pipelines

📡 Use STREAMING when:
  • Building interactive user interfaces
  • Responses might be very long
  • User experience is priority
  • Building chat applications
  • Real-time display is important
  • Handling potentially large responses
  • Want to show progress to users

⚖️ HYBRID APPROACH:
  • Use streaming for display + collect for processing
  • Best of both worlds for many applications
  • Example: Stream to UI while building complete response
  """)

def demonstrate_hybrid_approach():
    """Demonstrate collecting streaming response for post-processing"""
    print_section("Hybrid Approach: Stream + Collect")

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        print("Demonstrating streaming with collection for post-processing...")
        print(f"\n📡 Real-time display: ", end="", flush=True)

        # Collect streaming response
        complete_response = ""
        tool_calls_made = []
        chunk_count = 0

        for chunk in llm.generate(
            prompt="Calculate the area of a circle with radius 7 and give me a fact about circles",
            tools=DEMO_TOOLS,
            stream=True,
            system_prompt="Use tools to provide accurate information."
        ):
            chunk_count += 1

            # Real-time display
            if chunk.content:
                print(chunk.content, end="", flush=True)
                complete_response += chunk.content

            # Collect tool calls
            if chunk.tool_calls:
                tool_calls_made.extend(chunk.tool_calls)
                print(f"[🔧]", end="", flush=True)

        # Post-processing
        print(f"\n\n📊 Post-Processing Analysis:")
        print(f"  • Complete response: {len(complete_response)} characters")
        print(f"  • Total chunks processed: {chunk_count}")
        print(f"  • Tool calls made: {len(tool_calls_made)}")
        print(f"  • Words in response: {len(complete_response.split())}")

        if "area" in complete_response.lower():
            print(f"  ✓ Response contains area calculation")
        if "circle" in complete_response.lower():
            print(f"  ✓ Response mentions circles")

        print(f"\n💡 This hybrid approach gives you:")
        print(f"  • Real-time user feedback (streaming)")
        print(f"  • Complete data for analysis (collection)")
        print(f"  • Best user experience with full functionality")

    except Exception as e:
        print(f"Error in hybrid approach: {str(e)}")

def main():
    """Run all comparison examples"""
    print("AbstractLLM Core - Streaming vs Non-Streaming Tool Usage")
    print("=" * 60)
    print("This script compares streaming and non-streaming approaches")
    print("for tool usage with AbstractLLM Core.")
    print("\nNote: Examples require Ollama to be running locally.")

    # Run all comparisons
    demonstrate_non_streaming()
    demonstrate_streaming()
    side_by_side_comparison()
    when_to_use_which()
    demonstrate_hybrid_approach()

    print(f"\n{'='*60}")
    print(" Comparison Complete")
    print(f"{'='*60}")
    print("Choose the approach that best fits your use case!")

if __name__ == "__main__":
    main()