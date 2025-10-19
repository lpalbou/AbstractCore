#!/usr/bin/env python3
"""
Manual verification of media content processing.

Tests specific questions that can only be answered if the models
are actually reading and processing the file content.
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

    ext = os.path.splitext(file_path)[1].lower()
    mime_type_map = {
        '.pdf': 'application/pdf',
        '.csv': 'text/csv',
        '.tsv': 'text/tab-separated-values'
    }

    mime_type = mime_type_map.get(ext, 'application/octet-stream')
    b64_string = base64.b64encode(file_data).decode('utf-8')
    return f"data:{mime_type};base64,{b64_string}"

def test_csv_specific_content():
    """Test CSV with specific question about the actual content."""
    print("🧪 Testing CSV with Specific Content Question")
    print("Expected: The CSV contains skill categories: Scientific Computing & Platform Engineering, AI/ML & Data Science, Leadership & Strategy")

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
                        "text": "Looking at this CSV data, what are the exact column headers in the first row? List them exactly as they appear."
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
        "max_tokens": 150
    }

    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=30)

    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"📊 Response: {content}")

        # Check if the response contains the actual CSV headers
        expected_headers = ["Scientific Computing & Platform Engineering", "AI/ML & Data Science", "Leadership & Strategy"]
        found_any = any(header in content for header in expected_headers)

        if found_any:
            print("✅ SUCCESS: Model correctly read CSV content!")
            return True
        else:
            print("❌ FAILED: Model did not read actual CSV content")
            return False
    else:
        print(f"❌ Request failed: {response.status_code}")
        return False

def test_pdf_specific_content():
    """Test PDF with specific question about the research paper title."""
    print("\n🧪 Testing PDF with Specific Content Question")
    print("Expected: The paper title is 'Time-Aware Language Models as Temporal Knowledge Bases'")

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
                        "text": "What is the exact title of this research paper? Just give me the title as it appears in the document."
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
        "max_tokens": 100
    }

    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=60)

    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"📄 Response: {content}")

        # Check if the response contains the actual paper title
        expected_title_words = ["Time-Aware", "Language Models", "Temporal Knowledge", "Bases"]
        found_words = sum(1 for word in expected_title_words if word.lower() in content.lower())

        if found_words >= 3:
            print("✅ SUCCESS: Model correctly read PDF content!")
            return True
        else:
            print("❌ FAILED: Model did not read actual PDF content")
            return False
    else:
        print(f"❌ Request failed: {response.status_code}")
        return False

def test_tsv_specific_content():
    """Test TSV with specific question about the content."""
    print("\n🧪 Testing TSV with Specific Content Question")

    tsv_path = "/Users/albou/projects/abstractcore/tests/media_examples/data.tsv"
    tsv_data = encode_file_to_base64(tsv_path)

    payload = {
        "model": "ollama/qwen3:4b-instruct-2507-q4_K_M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a TSV (tab-separated values) file. How many columns does it have and what programming languages are mentioned in the data?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": tsv_data
                        }
                    }
                ]
            }
        ],
        "max_tokens": 150
    }

    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=30)

    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"📋 Response: {content}")

        # Check if programming languages are mentioned
        expected_languages = ["Python", "R", "Java", "C++", "SQL", "SPARQL"]
        found_languages = [lang for lang in expected_languages if lang in content]

        if len(found_languages) >= 2:
            print(f"✅ SUCCESS: Model found programming languages: {found_languages}")
            return True
        else:
            print("❌ FAILED: Model did not identify specific programming languages from TSV")
            return False
    else:
        print(f"❌ Request failed: {response.status_code}")
        return False

def main():
    print("🔍 Manual Verification of Media Content Processing")
    print("=" * 60)
    print("Testing if models actually read file content vs just responding generically")
    print()

    tests = [
        ("CSV Content Verification", test_csv_specific_content),
        ("PDF Content Verification", test_pdf_specific_content),
        ("TSV Content Verification", test_tsv_specific_content),
    ]

    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}\n")

    print(f"📊 Final Results: {passed}/{len(tests)} tests successfully read actual file content")

    if passed == len(tests):
        print("🎉 EXCELLENT: All models are correctly processing media content!")
    elif passed > 0:
        print("⚠️ PARTIAL: Some models are processing content, others are not")
    else:
        print("❌ CRITICAL: No models are actually reading the file content")
        print("💡 This suggests the media content is not being properly injected into model context")

if __name__ == "__main__":
    main()