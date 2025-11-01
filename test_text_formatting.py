#!/usr/bin/env python3
"""
Test script for the new text formatting functionality.

This script tests the TextFormatter class with various markdown-like content
to verify that formatting is applied correctly.
"""

from abstractcore.compression.text_formatter import TextFormatter, FormattingConfig


def test_newline_processing():
    """Test newline processing functionality."""
    print("=== Testing Newline Processing ===")
    
    formatter = TextFormatter()
    
    # Test consecutive newlines
    text1 = "Line 1\n\nLine 2\n\n\nLine 3"
    result1 = formatter.format_text(text1)
    print(f"Input:  '{text1}'")
    print(f"Output: '{result1}'")
    print()
    
    # Test single newlines
    text2 = "Word1\nWord2\nWord3"
    result2 = formatter.format_text(text2)
    print(f"Input:  '{text2}'")
    print(f"Output: '{result2}'")
    print()


def test_markdown_formatting():
    """Test markdown formatting functionality."""
    print("=== Testing Markdown Formatting ===")
    
    formatter = TextFormatter()
    
    # Test bold formatting
    text1 = "This is **bold text** and this is normal."
    result1 = formatter.format_text(text1)
    print(f"Bold test:")
    print(f"Input:  '{text1}'")
    print(f"Output: '{result1}'")
    print()
    
    # Test italic formatting
    text2 = "This is *italic text* and this is normal."
    result2 = formatter.format_text(text2)
    print(f"Italic test:")
    print(f"Input:  '{text2}'")
    print(f"Output: '{result2}'")
    print()
    
    # Test mixed formatting
    text3 = "This has **bold** and *italic* text."
    result3 = formatter.format_text(text3)
    print(f"Mixed test:")
    print(f"Input:  '{text3}'")
    print(f"Output: '{result3}'")
    print()


def test_header_formatting():
    """Test header formatting functionality."""
    print("=== Testing Header Formatting ===")
    
    formatter = TextFormatter()
    
    # Test different header levels
    text = """# First Header
## Second Header
### Third Header
# Another First Header
## Another Second Header
### Another Third Header
### Yet Another Third Header"""
    
    result = formatter.format_text(text)
    print(f"Headers test:")
    print(f"Input:")
    print(text)
    print(f"\nOutput:")
    print(result)
    print()


def test_complex_document():
    """Test formatting on a complex document with mixed content."""
    print("=== Testing Complex Document ===")
    
    formatter = TextFormatter()
    
    # Complex document with various formatting
    text = """# Introduction

This is a **complex document** with various formatting elements.

## Key Features

The system supports:
- **Bold text** for emphasis
- *Italic text* for subtle emphasis
- Multiple newlines that should be collapsed

### Implementation Details

Here are the implementation details:

1. First detail with\nsingle newline
2. Second detail with\n\nmultiple newlines
3. **Important detail** in bold

## Conclusion

This demonstrates the **formatting capabilities** of the system."""
    
    result = formatter.format_text(text)
    print(f"Complex document test:")
    print(f"Input:")
    print(text)
    print(f"\nOutput:")
    print(result)
    print()


def test_disabled_formatting():
    """Test with formatting disabled."""
    print("=== Testing Disabled Formatting ===")
    
    # Create formatter with all formatting disabled
    config = FormattingConfig()
    config.bold_formatting = False
    config.italic_formatting = False
    config.header_formatting = False
    config.consecutive_newlines_to_break = False
    config.single_newline_to_spaces = False
    
    formatter = TextFormatter(config)
    
    text = """# Header
This is **bold** and *italic* text.
Line 1\n\nLine 2"""
    
    result = formatter.format_text(text)
    print(f"Disabled formatting test:")
    print(f"Input:  '{text}'")
    print(f"Output: '{result}'")
    print("(Should be identical)")
    print()


def main():
    """Run all formatting tests."""
    print("Text Formatter Test Suite")
    print("=" * 50)
    print()
    
    test_newline_processing()
    test_markdown_formatting()
    test_header_formatting()
    test_complex_document()
    test_disabled_formatting()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()
