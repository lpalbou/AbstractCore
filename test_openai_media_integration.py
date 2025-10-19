#!/usr/bin/env python3
"""
Test script for OpenAI Vision API compatible media processing.

This script tests the AbstractCore server's OpenAI-compatible endpoints
with various media attachments using proper OpenAI Vision API format.
"""

import base64
import requests
import json
import os
from pathlib import Path

SERVER_URL = "http://localhost:8000"

def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 for OpenAI Vision API format."""
    with open(file_path, 'rb') as f:
        file_data = f.read()

    # Determine MIME type based on file extension
    ext = Path(file_path).suffix.lower()
    mime_type_map = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.tsv': 'text/tab-separated-values'
    }

    mime_type = mime_type_map.get(ext, 'application/octet-stream')
    b64_string = base64.b64encode(file_data).decode('utf-8')
    return f"data:{mime_type};base64,{b64_string}"

def test_server_health():
    """Test if server is healthy."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy: {data}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_pdf_processing():
    """Test PDF processing using OpenAI Vision API format."""
    print("\n🧪 Testing PDF Processing with OpenAI Vision API Format")

    pdf_path = "/Users/albou/projects/abstractcore/tests/media_examples/article.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False

    try:
        # Encode PDF to base64
        pdf_data = encode_file_to_base64(pdf_path)

        # Create OpenAI Vision API compatible request
        payload = {
            "model": "ollama/qwen2.5vl:7b",  # Vision model for document processing
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this research paper and provide a concise summary of its main contribution, methodology, and key findings. What problem does it solve?"
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
            "max_tokens": 300
        }

        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ PDF Analysis Success:")
            print(f"📄 Response: {content}")

            # Analyze response quality
            analyze_pdf_response(content)
            return True
        else:
            print(f"❌ PDF processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return False

def analyze_pdf_response(content: str):
    """Analyze the quality of PDF processing response."""
    print("\n🔍 Analyzing PDF Response Quality:")

    # The PDF is about "Time-Aware Language Models as Temporal Knowledge Bases"
    expected_keywords = [
        "temporal", "time", "language model", "knowledge", "fact",
        "templama", "google", "research", "wikidata", "training"
    ]

    found_keywords = []
    for keyword in expected_keywords:
        if keyword.lower() in content.lower():
            found_keywords.append(keyword)

    print(f"📊 Found relevant keywords: {found_keywords}")
    print(f"📈 Relevance score: {len(found_keywords)}/{len(expected_keywords)}")

    if len(found_keywords) >= 3:
        print("✅ Response appears relevant to the PDF content")
    else:
        print("⚠️ Response may not be accurately processing the PDF content")

def test_csv_processing():
    """Test CSV processing using OpenAI format."""
    print("\n🧪 Testing CSV Processing with OpenAI Vision API Format")

    csv_path = "/Users/albou/projects/abstractcore/tests/media_examples/data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False

    try:
        # For CSV, we can use either text content or base64
        csv_data = encode_file_to_base64(csv_path)

        payload = {
            "model": "ollama/qwen3:4b-instruct-2507-q4_K_M",  # Text model for data analysis
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this CSV data and describe what type of information it contains. What are the main categories or themes?"
                        },
                        {
                            "type": "image_url",  # Even for non-images, using image_url structure
                            "image_url": {
                                "url": csv_data
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ CSV Analysis Success:")
            print(f"📊 Response: {content}")

            # Analyze response quality
            analyze_csv_response(content)
            return True
        else:
            print(f"❌ CSV processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ CSV processing error: {e}")
        return False

def analyze_csv_response(content: str):
    """Analyze the quality of CSV processing response."""
    print("\n🔍 Analyzing CSV Response Quality:")

    # The CSV contains professional skills in categories
    expected_keywords = [
        "computing", "engineering", "data science", "leadership",
        "cloud", "ai", "ml", "python", "technical", "skills"
    ]

    found_keywords = []
    for keyword in expected_keywords:
        if keyword.lower() in content.lower():
            found_keywords.append(keyword)

    print(f"📊 Found relevant keywords: {found_keywords}")
    print(f"📈 Relevance score: {len(found_keywords)}/{len(expected_keywords)}")

    if len(found_keywords) >= 3:
        print("✅ Response appears relevant to the CSV content")
    else:
        print("⚠️ Response may not be accurately processing the CSV content")

def test_excel_processing():
    """Test Excel file processing."""
    print("\n🧪 Testing Excel Processing with OpenAI Vision API Format")

    xlsx_path = "/Users/albou/projects/abstractcore/tests/media_examples/data.xlsx"
    if not os.path.exists(xlsx_path):
        print(f"❌ Excel file not found: {xlsx_path}")
        return False

    try:
        xlsx_data = encode_file_to_base64(xlsx_path)

        payload = {
            "model": "ollama/qwen3:4b-instruct-2507-q4_K_M",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this Excel spreadsheet and summarize what data it contains. What insights can you extract?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": xlsx_data
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Excel Analysis Success:")
            print(f"📈 Response: {content}")
            return True
        else:
            print(f"❌ Excel processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Excel processing error: {e}")
        return False

def test_docx_processing():
    """Test Word document processing."""
    print("\n🧪 Testing Word Document Processing with OpenAI Vision API Format")

    docx_path = "/Users/albou/projects/abstractcore/tests/media_examples/false-report.docx"
    if not os.path.exists(docx_path):
        print(f"❌ Word document not found: {docx_path}")
        return False

    try:
        docx_data = encode_file_to_base64(docx_path)

        payload = {
            "model": "ollama/qwen2.5vl:7b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read and summarize the content of this Word document. What is it about?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": docx_data
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Word Document Analysis Success:")
            print(f"📝 Response: {content}")
            return True
        else:
            print(f"❌ Word document processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Word document processing error: {e}")
        return False

def test_streaming_with_media():
    """Test streaming responses with media attachments."""
    print("\n🧪 Testing Streaming with Media")

    csv_path = "/Users/albou/projects/abstractcore/tests/media_examples/data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False

    try:
        csv_data = encode_file_to_base64(csv_path)

        payload = {
            "model": "ollama/qwen3:4b-instruct-2507-q4_K_M",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this CSV data step by step and explain what you find."
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
            "max_tokens": 150,
            "stream": True
        }

        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=30
        )

        if response.status_code == 200:
            print("✅ Streaming Response:")
            chunks = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: ') and not line_str.endswith('[DONE]'):
                        try:
                            data = json.loads(line_str[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    print(delta['content'], end='', flush=True)
                                    chunks.append(delta['content'])
                        except:
                            pass

            full_content = ''.join(chunks)
            print(f"\n🏁 Complete streaming response received: {len(full_content)} characters")
            return True
        else:
            print(f"❌ Streaming failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Streaming error: {e}")
        return False

def main():
    print("🚀 Testing AbstractCore OpenAI Vision API Compatible Media Processing")
    print("=" * 80)

    # Test server health first
    if not test_server_health():
        print("💡 Make sure the server is running: uvicorn abstractcore.server.app:app --port 8000")
        return

    # Run tests
    tests = [
        ("PDF Processing", test_pdf_processing),
        ("CSV Processing", test_csv_processing),
        ("Excel Processing", test_excel_processing),
        ("Word Document Processing", test_docx_processing),
        ("Streaming with Media", test_streaming_with_media)
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")

    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! OpenAI Vision API media integration is working!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()