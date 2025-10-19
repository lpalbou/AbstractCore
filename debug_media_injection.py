#!/usr/bin/env python3
"""
Debug script to check what message is actually being sent to Ollama.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from abstractcore.media.handlers import LocalMediaHandler
from abstractcore.media.types import MediaContent, MediaType, ContentFormat

def test_media_handler_directly():
    """Test the LocalMediaHandler directly to see what it generates."""
    print("🔍 Testing LocalMediaHandler Directly")
    print("=" * 60)

    # Create mock media content like what would be processed from CSV
    csv_content = """Date,Product,Sales
2024-01-01,Product A,10000
2024-01-02,Product B,15000
2024-01-03,Product C,25000"""

    media_content = MediaContent(
        content=csv_content,
        media_type=MediaType.TEXT,
        content_format=ContentFormat.TEXT,
        mime_type="text/csv",
        metadata={"file_name": "data.csv", "file_path": "/tmp/data.csv"}
    )

    # Create LocalMediaHandler for Ollama
    handler = LocalMediaHandler("ollama", None, model_name="qwen3:4b-instruct")

    # Test what create_multimodal_message produces
    prompt = "What is the total sales amount in this CSV file?"
    result = handler.create_multimodal_message(prompt, [media_content])

    print(f"📄 Input prompt: {prompt}")
    print(f"📊 Media content: {len(csv_content)} chars")
    print(f"🎯 Result type: {type(result)}")
    print(f"📝 Result content:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Check if the CSV data is embedded
    if isinstance(result, str) and csv_content in result:
        print("✅ CSV content is properly embedded in the message!")
    else:
        print("❌ CSV content is NOT embedded in the message")

if __name__ == "__main__":
    test_media_handler_directly()