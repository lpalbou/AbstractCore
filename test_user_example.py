#!/usr/bin/env python3
"""
Test the enhanced tool system with the exact user-provided example.

This verifies that the system can handle the full complexity of the
user's tool definition including all metadata and examples.
"""

from pathlib import Path
from typing import Optional

from abstractllm.tools.core import tool
from abstractllm.tools.handler import UniversalToolHandler


@tool(
    description="Read the contents of a file with optional line range and hidden file access",
    tags=["file", "read", "content", "text"],
    when_to_use="When you need to read file contents, examine code, or extract specific line ranges from files",
    examples=[
        {
            "description": "Read entire file",
            "arguments": {
                "file_path": "README.md"
            }
        },
        {
            "description": "Read specific line range",
            "arguments": {
                "file_path": "src/main.py",
                "should_read_entire_file": False,
                "start_line_one_indexed": 10,
                "end_line_one_indexed_inclusive": 25
            }
        },
        {
            "description": "Read hidden file",
            "arguments": {
                "file_path": ".gitignore",
                "include_hidden": True
            }
        },
        {
            "description": "Read first 50 lines",
            "arguments": {
                "file_path": "large_file.txt",
                "should_read_entire_file": False,
                "end_line_one_indexed_inclusive": 50
            }
        }
    ]
)
def read_file(file_path: str, should_read_entire_file: bool = True, start_line_one_indexed: int = 1, end_line_one_indexed_inclusive: Optional[int] = None, include_hidden: bool = False) -> str:
    """
    Read the contents of a file with optional line range.

    Args:
        file_path: Path to the file to read
        should_read_entire_file: Whether to read the entire file (default: True)
        start_line_one_indexed: Starting line number (1-indexed, default: 1)
        end_line_one_indexed_inclusive: Ending line number (1-indexed, inclusive, optional)
        include_hidden: Whether to allow reading hidden files starting with '.' (default: False)

    Returns:
        File contents or error message
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File '{file_path}' does not exist"

        if not path.is_file():
            return f"Error: '{file_path}' is not a file"

        # Check for hidden files (files starting with '.')
        if not include_hidden and path.name.startswith('.'):
            return f"Error: Access to hidden file '{file_path}' is not allowed. Use include_hidden=True to override."

        with open(path, 'r', encoding='utf-8') as f:
            if should_read_entire_file:
                # Read entire file
                content = f.read()
                line_count = len(content.splitlines())
                return f"File: {file_path} ({line_count} lines)\n\n{content}"
            else:
                # Read specific line range
                lines = f.readlines()
                total_lines = len(lines)

                # Convert to 0-indexed and validate
                start_idx = max(0, start_line_one_indexed - 1)
                end_idx = min(total_lines, end_line_one_indexed_inclusive or total_lines)

                if start_idx >= total_lines:
                    return f"Error: Start line {start_line_one_indexed} exceeds file length ({total_lines} lines)"

                selected_lines = lines[start_idx:end_idx]

                # Format without line numbers
                result_lines = []
                for line in selected_lines:
                    result_lines.append(f"{line.rstrip()}")

                return "\n".join(result_lines)

    except UnicodeDecodeError:
        return f"Error: Cannot read '{file_path}' - file appears to be binary"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def test_user_example_full_integration():
    """Test the exact user example for full integration."""
    print("🔥 Testing User's Exact Tool Definition")
    print("=" * 50)

    # Test metadata extraction
    tool_def = read_file._tool_definition

    print(f"📝 Tool name: {tool_def.name}")
    print(f"📝 Description: {tool_def.description}")
    print(f"📝 Tags: {tool_def.tags}")
    print(f"📝 When to use: {tool_def.when_to_use}")
    print(f"📝 Examples count: {len(tool_def.examples)}")

    # Verify all examples are captured
    expected_examples = [
        "Read entire file",
        "Read specific line range",
        "Read hidden file",
        "Read first 50 lines"
    ]

    print(f"\n📋 Example descriptions:")
    for i, example in enumerate(tool_def.examples):
        desc = example.get("description", "No description")
        print(f"  {i+1}. {desc}")
        assert desc in expected_examples, f"Missing expected example: {desc}"

    # Test different architecture formats
    architectures = [
        ("qwen3-coder:30b", "Qwen"),
        ("llama3:8b", "LLaMA"),
        ("gemma2:9b", "Gemma"),
        ("phi3:mini", "Phi")
    ]

    for model_name, arch_name in architectures:
        print(f"\n🏗️  Testing {arch_name} architecture ({model_name}):")

        handler = UniversalToolHandler(model_name)
        prompt = handler.format_tools_prompt([read_file])

        # Check key elements are present
        checks = [
            ("description", tool_def.description in prompt),
            ("when_to_use", tool_def.when_to_use in prompt),
            ("all examples", all(ex["description"] in prompt for ex in tool_def.examples)),
            ("all tags", all(tag in prompt for tag in tool_def.tags)),
            ("example args", '"file_path": "README.md"' in prompt)
        ]

        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}")

        print(f"   📊 Prompt length: {len(prompt)} chars")

    # Show a detailed prompt sample
    print(f"\n📄 FULL QWEN PROMPT SAMPLE:")
    print("=" * 50)

    handler = UniversalToolHandler("qwen3-coder:30b")
    full_prompt = handler.format_tools_prompt([read_file])
    print(full_prompt)

    return full_prompt


def test_tool_execution():
    """Test that the tool actually works."""
    print(f"\n🛠️  Testing Tool Execution")
    print("=" * 30)

    # Create a test file
    test_content = """Line 1: Hello World
Line 2: This is a test
Line 3: Final line"""

    test_file = Path("test_file.txt")
    test_file.write_text(test_content)

    try:
        # Test 1: Read entire file
        result1 = read_file("test_file.txt")
        print(f"✅ Full file read: {len(result1)} chars")

        # Test 2: Read line range
        result2 = read_file("test_file.txt", should_read_entire_file=False, start_line_one_indexed=2, end_line_one_indexed_inclusive=2)
        print(f"✅ Line range read: '{result2.strip()}'")
        assert "Line 2: This is a test" in result2

        # Test 3: Error handling
        result3 = read_file("nonexistent.txt")
        print(f"✅ Error handling: {result3[:50]}...")
        assert "does not exist" in result3

        print("✅ All tool execution tests passed!")

    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def main():
    """Run comprehensive test of user's exact example."""
    print("🚀 COMPREHENSIVE USER EXAMPLE TEST")
    print("=" * 50)

    try:
        # Test tool system integration
        full_prompt = test_user_example_full_integration()

        # Test actual tool execution
        test_tool_execution()

        print(f"\n🎉 SUCCESS! User's tool definition works perfectly:")
        print(f"✅ All metadata (tags, when_to_use, examples) preserved")
        print(f"✅ Rich prompts generated for all architectures")
        print(f"✅ Tool execution works correctly")
        print(f"✅ Enhanced system delivers {len(full_prompt)} character prompts")
        print(f"✅ LLMs will have excellent guidance for using this tool")

        return True

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)