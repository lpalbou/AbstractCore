#!/usr/bin/env python3
"""
Simple test to trigger debug logging.
"""
import base64
import requests
import json

def test_csv_with_debug():
    """Test CSV processing with debug logging enabled."""

    # Test multiple file types

    # CSV content test
    csv_content = """Name,Age,Salary
Alice,30,75000
Bob,25,65000"""
    csv_b64 = base64.b64encode(csv_content.encode()).decode()

    tests = [
        {
            "name": "CSV Analysis",
            "question": "What is the average salary in this CSV?",
            "data_url": f"data:text/csv;base64,{csv_b64}",
            "expected_keywords": ["70000", "average", "salary"]
        }
    ]

    for test in tests:
        print(f"\n🧪 Testing: {test['name']}")
        payload = {
            "model": "lmstudio/qwen/qwen3-next-80b",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": test["question"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": test["data_url"]
                        }
                    }
                ]
            }],
            "max_tokens": 200
        }

        response = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Response: {content}")

            # Check if response shows evidence of reading the content
            content_lower = content.lower()
            found_keywords = [kw for kw in test["expected_keywords"] if kw in content_lower]
            if found_keywords:
                print(f"✅ Model correctly processed content! Found: {found_keywords}")
            else:
                print(f"⚠️  Expected keywords {test['expected_keywords']} not found in response")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_csv_with_debug()