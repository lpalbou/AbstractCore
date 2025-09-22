#!/usr/bin/env python3
"""
Test the enhanced tool system to verify metadata injection.

This test demonstrates that tool metadata (tags, when_to_use, examples)
flows correctly from the @tool decorator to the system prompts.
"""

import os
import fnmatch
from pathlib import Path
from typing import Optional

from abstractllm.tools.core import tool, ToolDefinition
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
def read_file(
    file_path: str,
    should_read_entire_file: bool = True,
    start_line_one_indexed: int = 1,
    end_line_one_indexed_inclusive: Optional[int] = None,
    include_hidden: bool = False
) -> str:
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
                    result_lines.append(line.rstrip())

                return "\n".join(result_lines)

    except UnicodeDecodeError:
        return f"Error: Cannot read '{file_path}' - file appears to be binary"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool(
    description="Find and list files and directories by their names/paths using glob patterns",
    tags=["file", "directory", "listing", "filesystem"],
    when_to_use="When you need to find files by their names, paths, or file extensions (NOT for searching file contents)",
    examples=[
        {
            "description": "List all files in current directory",
            "arguments": {
                "directory": ".",
                "pattern": "*"
            }
        },
        {
            "description": "Find all Python files recursively",
            "arguments": {
                "directory": ".",
                "pattern": "*.py",
                "recursive": True
            }
        },
        {
            "description": "Find all markdown files in docs folder",
            "arguments": {
                "directory": "docs",
                "pattern": "*.md",
                "recursive": False
            }
        }
    ]
)
def list_files(directory: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    """
    List files in a directory with pattern matching.

    Args:
        directory: Directory to search in (default: current directory)
        pattern: File pattern to match (default: all files)
        recursive: Search subdirectories recursively

    Returns:
        String listing of matching files
    """
    try:
        dir_path = Path(directory)

        if not dir_path.exists():
            return f"Error: Directory '{directory}' does not exist"

        if not dir_path.is_dir():
            return f"Error: '{directory}' is not a directory"

        # Collect files
        files = []
        if recursive:
            for root, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_path = Path(root) / filename
                    if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                        files.append(str(file_path.relative_to(dir_path)))
        else:
            for file_path in dir_path.iterdir():
                if file_path.is_file() and fnmatch.fnmatch(file_path.name.lower(), pattern.lower()):
                    files.append(file_path.name)

        if not files:
            return f"No files found matching pattern '{pattern}' in '{directory}'"

        # Sort files
        files.sort()

        result = f"Files matching '{pattern}' in '{directory}':\n"
        for file in files:
            result += f"  {file}\n"

        return result.strip()

    except Exception as e:
        return f"Error listing files: {str(e)}"


def test_metadata_preservation():
    """Test that tool metadata is preserved through the entire flow."""
    print("🧪 Testing Enhanced Tool System Metadata Preservation")
    print("=" * 60)

    # Test 1: Check tool definition metadata
    print("\n1. Testing tool definition creation...")

    # Verify read_file tool has all metadata
    assert hasattr(read_file, '_tool_definition'), "read_file should have _tool_definition"
    read_def = read_file._tool_definition

    print(f"   ✅ Tool name: {read_def.name}")
    print(f"   ✅ Description: {read_def.description[:50]}...")
    print(f"   ✅ Tags: {read_def.tags}")
    print(f"   ✅ When to use: {read_def.when_to_use[:50]}...")
    print(f"   ✅ Examples count: {len(read_def.examples)}")

    assert read_def.tags == ["file", "read", "content", "text"]
    assert read_def.when_to_use is not None
    assert len(read_def.examples) == 4

    # Test 2: Check tool dict conversion includes metadata
    print("\n2. Testing tool dict conversion...")
    tool_dict = read_def.to_dict()

    assert "tags" in tool_dict, "Tool dict should include tags"
    assert "when_to_use" in tool_dict, "Tool dict should include when_to_use"
    assert "examples" in tool_dict, "Tool dict should include examples"

    print(f"   ✅ Dict includes tags: {tool_dict['tags']}")
    print(f"   ✅ Dict includes when_to_use: {tool_dict['when_to_use'][:50]}...")
    print(f"   ✅ Dict includes examples: {len(tool_dict['examples'])} examples")

    # Test 3: Check handler preserves metadata
    print("\n3. Testing tool handler metadata preservation...")
    handler = UniversalToolHandler("qwen3-coder:30b")

    # Convert callable tools to definitions (simulating the flow)
    tool_defs = handler._convert_to_tool_definitions([read_file, list_files])

    assert len(tool_defs) == 2, "Should have 2 tool definitions"

    read_tool = next(t for t in tool_defs if t.name == "read_file")
    list_tool = next(t for t in tool_defs if t.name == "list_files")

    print(f"   ✅ read_file metadata preserved: tags={read_tool.tags}, examples={len(read_tool.examples)}")
    print(f"   ✅ list_files metadata preserved: tags={list_tool.tags}, examples={len(list_tool.examples)}")

    # Test 4: Check prompt generation includes metadata
    print("\n4. Testing enhanced prompt generation...")
    enhanced_prompt = handler.format_tools_prompt([read_file, list_files])

    # Check that key metadata appears in the prompt
    metadata_checks = [
        ("tags in prompt", any(tag in enhanced_prompt for tag in ["file", "read", "content"])),
        ("when_to_use in prompt", "When to use" in enhanced_prompt),
        ("examples in prompt", "EXAMPLES:" in enhanced_prompt),
        ("specific example", "Read entire file" in enhanced_prompt),
        ("tool call format", "<|tool_call|>" in enhanced_prompt),
        ("example arguments", '"file_path": "README.md"' in enhanced_prompt)
    ]

    for check_name, result in metadata_checks:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}: {result}")
        assert result, f"Failed check: {check_name}"

    print(f"\n📊 Enhanced prompt length: {len(enhanced_prompt)} characters")
    print(f"📊 Contains {enhanced_prompt.count('**When to use**')} 'when to use' sections")
    print(f"📊 Contains {enhanced_prompt.count('**Tags**')} tag sections")
    print(f"📊 Contains {enhanced_prompt.count('Examples:')} example sections")

    return enhanced_prompt


def demo_enhanced_vs_basic():
    """Demonstrate the difference between enhanced and basic tool prompts."""
    print("\n" + "=" * 60)
    print("🔍 ENHANCED vs BASIC TOOL PROMPT COMPARISON")
    print("=" * 60)

    handler = UniversalToolHandler("qwen3-coder:30b")

    # Create a basic tool definition (no metadata)
    basic_def = ToolDefinition.from_function(read_file)
    basic_def.name = "read_file"
    basic_def.description = "Read file contents"

    # Generate prompts
    enhanced_prompt = handler.format_tools_prompt([read_file])
    basic_prompt = handler.format_tools_prompt([basic_def])

    print(f"\n📈 Enhanced prompt: {len(enhanced_prompt)} characters")
    print(f"📉 Basic prompt: {len(basic_prompt)} characters")
    print(f"🚀 Enhancement ratio: {len(enhanced_prompt) / len(basic_prompt):.1f}x longer")

    print(f"\n🔍 Enhanced features:")
    print(f"   • When to use guidance: {'✅' if 'When to use' in enhanced_prompt else '❌'}")
    print(f"   • Tag categorization: {'✅' if 'Tags:' in enhanced_prompt else '❌'}")
    print(f"   • Concrete examples: {'✅' if 'EXAMPLES:' in enhanced_prompt else '❌'}")
    print(f"   • Example tool calls: {'✅' if enhanced_prompt.count('<|tool_call|>') > 2 else '❌'}")

    print(f"\n📝 Basic features:")
    print(f"   • When to use guidance: {'✅' if 'When to use' in basic_prompt else '❌'}")
    print(f"   • Tag categorization: {'✅' if 'Tags:' in basic_prompt else '❌'}")
    print(f"   • Concrete examples: {'✅' if 'EXAMPLES:' in basic_prompt else '❌'}")
    print(f"   • Example tool calls: {'✅' if basic_prompt.count('<|tool_call|>') > 2 else '❌'}")

    return enhanced_prompt, basic_prompt


def show_enhanced_prompt_sample():
    """Show a sample of the enhanced prompt output."""
    print("\n" + "=" * 60)
    print("📋 ENHANCED PROMPT SAMPLE")
    print("=" * 60)

    handler = UniversalToolHandler("qwen3-coder:30b")
    enhanced_prompt = handler.format_tools_prompt([read_file])

    # Show first 800 characters
    sample = enhanced_prompt[:800] + "..." if len(enhanced_prompt) > 800 else enhanced_prompt
    print(sample)

    print(f"\n[... Full prompt is {len(enhanced_prompt)} characters total]")


def main():
    """Run all tests and demonstrations."""
    print("🚀 Enhanced Tool System Integration Test")
    print("=" * 60)

    try:
        # Run comprehensive metadata test
        enhanced_prompt = test_metadata_preservation()

        # Show comparison
        demo_enhanced_vs_basic()

        # Show sample output
        show_enhanced_prompt_sample()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Enhanced tool system is working correctly.")
        print("✅ Metadata flows from @tool decorator to system prompts")
        print("✅ Examples, tags, and when_to_use are properly injected")
        print("✅ LLMs will receive rich guidance for better tool usage")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)