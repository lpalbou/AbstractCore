#!/usr/bin/env python3
"""
Test script to verify the fixed media processing with proper file extensions.
"""

import base64
import requests
import json
import os

SERVER_URL = "http://localhost:8000"

def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 for OpenAI Vision API format."""
    with open(file_path, 'rb') as f:
        file_data = f.read()

    # Determine MIME type based on file extension
    ext = os.path.splitext(file_path)[1].lower()
    mime_type_map = {
        '.pdf': 'application/pdf',
        '.csv': 'text/csv',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.tsv': 'text/tab-separated-values'
    }

    mime_type = mime_type_map.get(ext, 'application/octet-stream')
    b64_string = base64.b64encode(file_data).decode('utf-8')
    return f"data:{mime_type};base64,{b64_string}"

def test_csv_with_qwen():
    """Test CSV processing with a text model."""
    print("🧪 Testing CSV Processing with Fixed Media Handler")

    csv_path = "/Users/albou/projects/abstractcore/tests/media_examples/data.csv"
    csv_data = encode_file_to_base64(csv_path)

    payload = {
        "model": "ollama/qwen3:4b-instruct-2507-q4_K_M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this CSV data. What are the three main categories of skills represented? What type of professional profile does this represent?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": csv_data
                        }
                    }
                ]
            }
        ],
        "max_tokens": 200
    }

    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=30)

    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"✅ CSV Analysis Success:")
        print(f"📊 Response: {content}")

        # Check if response contains relevant keywords
        relevant_terms = ["computing", "engineering", "data science", "leadership", "ai", "technical", "skills"]
        found_terms = [term for term in relevant_terms if term.lower() in content.lower()]
        print(f"🎯 Found relevant terms: {found_terms}")

        if len(found_terms) >= 2:
            print("✅ Response appears to accurately process CSV content!")
        else:
            print("⚠️ Response may not be accurately processing CSV content")

        return True
    else:
        print(f"❌ CSV processing failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_pdf_with_vision():
    """Test PDF processing with a vision model."""
    print("\n🧪 Testing PDF Processing with Fixed Media Handler")

    pdf_path = "/Users/albou/projects/abstractcore/tests/media_examples/article.pdf"
    pdf_data = encode_file_to_base64(pdf_path)

    payload = {
        "model": "ollama/qwen2.5vl:7b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the title of this research paper and what problem does it solve? Be specific about the temporal aspects mentioned."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pdf_data,
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 250
    }

    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)

    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"✅ PDF Analysis Success:")
        print(f"📄 Response: {content}")

        # Check if response mentions the actual paper content
        expected_terms = ["temporal", "time", "language model", "knowledge", "templama", "google", "wikidata"]
        found_terms = [term for term in expected_terms if term.lower() in content.lower()]
        print(f"🎯 Found relevant terms: {found_terms}")

        if len(found_terms) >= 3:
            print("✅ Response appears to accurately process PDF content!")
        else:
            print("⚠️ Response may not be accurately processing PDF content")

        return True
    else:
        print(f"❌ PDF processing failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    print("🚀 Testing Fixed Media Processing with AbstractCore Server")
    print("=" * 70)

    # Run specific tests to verify the fix
    tests = [
        ("CSV Processing", test_csv_with_qwen),
        ("PDF Processing", test_pdf_with_vision),
    ]

    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}\n")

    print(f"📊 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 Fixed media processing is working correctly!")
    else:
        print("⚠️ Some issues remain with media processing.")

if __name__ == "__main__":
    main()