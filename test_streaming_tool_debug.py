#!/usr/bin/env python3
"""
Debug script to test streaming + tool execution
"""
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from abstractllm.providers.streaming import IncrementalToolDetector, UnifiedStreamProcessor
from abstractllm.core.types import GenerateResponse

# Test 1: Check if IncrementalToolDetector can detect <function_call> format
print("=" * 80)
print("TEST 1: IncrementalToolDetector with <function_call> format")
print("=" * 80)

detector = IncrementalToolDetector(model_name="qwen/qwen3-next-80b")
print(f"Active patterns for qwen/qwen3-next-80b: {[p['start'] for p in detector.active_patterns]}")

# Simulate streaming chunks
test_content = '<function_call>\n{"name": "read_file", "arguments": {"file_path": "README.md"}}\n</function_call>'
chunks = [test_content[i:i+10] for i in range(0, len(test_content), 10)]

print(f"\nProcessing {len(chunks)} chunks...")
all_completed_tools = []
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}: {repr(chunk)}")
    streamable, completed_tools = detector.process_chunk(chunk)
    print(f"  Streamable: {repr(streamable)}")
    print(f"  Completed tools: {completed_tools}")
    print(f"  Detector state: {detector.state.value}")
    all_completed_tools.extend(completed_tools)

# Finalize any remaining
final_tools = detector.finalize()
all_completed_tools.extend(final_tools)

print(f"\nFinal result:")
print(f"  Total completed tools: {len(all_completed_tools)}")
for tool in all_completed_tools:
    print(f"    - {tool.name}: {tool.arguments}")

# Test 2: Check UnifiedStreamProcessor
print("\n" + "=" * 80)
print("TEST 2: UnifiedStreamProcessor with mock stream")
print("=" * 80)

def mock_stream():
    """Mock stream that yields chunks"""
    content = '<function_call>\n{"name": "read_file", "arguments": {"file_path": "README.md"}}\n</function_call>'
    chunks = [content[i:i+10] for i in range(0, len(content), 10)]
    for chunk in chunks:
        yield GenerateResponse(content=chunk, model="qwen/qwen3-next-80b", finish_reason=None)

processor = UnifiedStreamProcessor(
    model_name="qwen/qwen3-next-80b",
    execute_tools=True,
    tool_call_tags=None
)

# Mock tools
mock_tools = [
    {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"}
            },
            "required": ["file_path"]
        }
    }
]

print("\nProcessing stream...")
outputs = []
for chunk in processor.process_stream(mock_stream(), mock_tools):
    print(f"Yielded chunk: {repr(chunk.content[:100] if chunk.content else '')}")
    outputs.append(chunk.content or "")

print(f"\nFinal output: {''.join(outputs)}")
print("\nDone!")
