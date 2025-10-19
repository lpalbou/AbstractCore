#!/usr/bin/env python3
"""
Test CLI media handling to compare with server implementation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from abstractcore import BasicSession

def test_cli_media_handling():
    """Test how CLI handles media attachments."""
    print("🧪 Testing CLI Media Handling")
    print("=" * 50)

    # Create session like CLI does
    session = BasicSession()

    # Test with CSV file
    csv_path = "tests/media_examples/data.csv"

    try:
        print(f"📄 Testing with: {csv_path}")

        # This is how CLI calls it: session.generate() with media parameter
        response = session.generate(
            "What is the total sales amount in this CSV file?",
            media=[csv_path],
            model="ollama/qwen3:4b-instruct",
            max_tokens=200
        )

        print(f"✅ CLI Response: {response.content[:200]}...")

        # Check if response shows evidence of reading the CSV
        if any(keyword in response.content.lower() for keyword in ['total', 'sales', '15000', '10000', '25000']):
            print("✅ CLI appears to be reading the CSV content!")
        else:
            print("❌ CLI doesn't seem to be reading the CSV content")

    except Exception as e:
        print(f"❌ CLI test failed: {e}")

if __name__ == "__main__":
    test_cli_media_handling()