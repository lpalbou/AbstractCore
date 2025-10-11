#!/usr/bin/env python3
"""
Comprehensive test for streaming + tool execution
Tests the actual user scenario reported in the issue
"""
import sys
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from abstractllm.providers.streaming import IncrementalToolDetector, UnifiedStreamProcessor
from abstractllm.core.types import GenerateResponse
from abstractllm.tools.core import ToolCall, ToolResult

print("="  * 80)
print("TEST: Streaming + Tool Execution for qwen/qwen3-next-80b")
print("="  * 80)

# Simulate the exact scenario from the user's CLI output
test_input = "read README.md"
model_output = '<function_call>\n{"name": "read_file", "arguments": {"file_path": "README.md"}}\n</function_call>'

print(f"\n1. User Input: {test_input}")
print(f"2. Model Output: {repr(model_output)}")

# Create detector for qwen model
detector = IncrementalToolDetector(model_name="qwen/qwen3-next-80b")
print(f"\n3. Detector Patterns: {[p['start'] for p in detector.active_patterns]}")

# Simulate streaming in chunks
chunks = []
chunk_size = 15
for i in range(0, len(model_output), chunk_size):
    chunks.append(model_output[i:i+chunk_size])

print(f"\n4. Processing {len(chunks)} streaming chunks...")

all_streamable = []
all_completed_tools = []

for i, chunk in enumerate(chunks):
    print(f"\n  Chunk {i+1}/{len(chunks)}: {repr(chunk)}")
    streamable, completed = detector.process_chunk(chunk)

    print(f"    State: {detector.state.value}")
    print(f"    Streamable: {repr(streamable)}")
    print(f"    Completed tools: {len(completed)}")

    if streamable:
        all_streamable.append(streamable)
    all_completed_tools.extend(completed)

# Finalize
final_tools = detector.finalize()
all_completed_tools.extend(final_tools)

print(f"\n5. Final Results:")
print(f"   Streamable content (user sees): {repr(''.join(all_streamable))}")
print(f"   Completed tools: {len(all_completed_tools)}")

if all_completed_tools:
    for tool in all_completed_tools:
        print(f"     - Tool: {tool.name}")
        print(f"       Args: {tool.arguments}")
else:
    print("   ❌ NO TOOLS DETECTED!")
    sys.exit(1)

# Now test with UnifiedStreamProcessor + mock stream
print("\n" + "=" * 80)
print("TEST: UnifiedStreamProcessor with tool execution")
print("=" * 80)

def mock_response_stream():
    """Simulate the response stream from provider"""
    for chunk in chunks:
        yield GenerateResponse(
            content=chunk,
            model="qwen/qwen3-next-80b",
            finish_reason=None
        )

# Mock tools - simulate the CLI tools
mock_tools = [
    {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"}
            },
            "required": ["file_path"]
        }
    }
]

processor = UnifiedStreamProcessor(
    model_name="qwen/qwen3-next-80b",
    execute_tools=True,
    tool_call_tags=None
)

print("\nProcessing stream with tool execution...")
output_chunks = []
tool_result_chunks = []

for chunk in processor.process_stream(mock_response_stream(), mock_tools):
    content = chunk.content or ""
    print(f"  Yielded: {repr(content[:80])}")

    if "Tool Results" in content or "🔧" in content:
        tool_result_chunks.append(content)
    else:
        output_chunks.append(content)

print(f"\nFinal output:")
print(f"  User-visible content: {repr(''.join(output_chunks))}")
print(f"  Tool results: {len(tool_result_chunks)} chunks")

if tool_result_chunks:
    print(f"  Tool output preview: {tool_result_chunks[0][:200]}")
    print("\n✅ SUCCESS: Tool execution working with streaming!")
else:
    print("\n❌ FAILURE: Tools detected but not executed!")
    sys.exit(1)

# Validate the expected behavior:
# 1. Tool tags should NOT be in user-visible content
# 2. Tool results SHOULD be present
streamable_text = ''.join(output_chunks)
if '<function_call>' in streamable_text:
    print("\n❌ FAILURE: Tool tags leaked to user-visible output!")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("="  * 80)
print("\nSummary:")
print("  ✅ Streaming works (real-time chunks)")
print("  ✅ Tool detection works (<function_call> format)")
print("  ✅ Tool execution works (results appear)")
print("  ✅ Tool tags hidden from user (not streamed)")
