#!/usr/bin/env python3
"""
Test script for the new server parameters: tool_call_tags and execute_tools
"""

import requests
import json
import time

def test_server_parameters():
    """Test the new server parameters."""
    print("🧪 Testing Server Tool Parameters")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test data
    test_data = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "What's 2+2? Use a calculator tool."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate mathematical expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression"}
                    },
                    "required": ["expression"]
                }
            }
        }],
        "max_tokens": 100
    }
    
    print("1. Testing tool_call_tags parameter...")
    
    # Test different tool call tag formats
    tag_formats = ["qwen3", "llama3", "xml", "gemma"]
    
    for tag_format in tag_formats:
        print(f"   Testing {tag_format} format...")
        test_data["tool_call_tags"] = tag_format
        
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"   ✅ {tag_format}: {content[:100]}...")
            else:
                print(f"   ❌ {tag_format}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {tag_format}: {e}")
    
    print("\n2. Testing execute_tools parameter...")
    
    # Test with execute_tools=True (default)
    print("   Testing execute_tools=True...")
    test_data["execute_tools"] = True
    test_data["tool_call_tags"] = "qwen3"
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"   ✅ execute_tools=True: {content[:100]}...")
        else:
            print(f"   ❌ execute_tools=True: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ execute_tools=True: {e}")
    
    # Test with execute_tools=False
    print("   Testing execute_tools=False...")
    test_data["execute_tools"] = False
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"   ✅ execute_tools=False: {content[:100]}...")
        else:
            print(f"   ❌ execute_tools=False: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ execute_tools=False: {e}")
    
    print("\n3. Testing Anthropic Messages API...")
    
    # Test Anthropic Messages API with new parameters
    anthropic_data = {
        "model": "openai/gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "What's 2+2? Use a calculator tool."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate mathematical expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression"}
                    },
                    "required": ["expression"]
                }
            }
        }],
        "tool_call_tags": "llama3",
        "execute_tools": False
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/messages",
            json=anthropic_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["content"][0]["text"]
            print(f"   ✅ Anthropic API: {content[:100]}...")
        else:
            print(f"   ❌ Anthropic API: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Anthropic API: {e}")
    
    print("\n4. Testing Responses API...")
    
    # Test Responses API with new parameters
    responses_data = {
        "model": "openai/gpt-4o-mini",
        "input": [{"type": "message", "role": "user", "content": "What's 2+2? Use a calculator tool."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate mathematical expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression"}
                    },
                    "required": ["expression"]
                }
            }
        }],
        "tool_call_tags": "xml",
        "execute_tools": True,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/responses",
            json=responses_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["output"][0]["content"]
            print(f"   ✅ Responses API: {content[:100]}...")
        else:
            print(f"   ❌ Responses API: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Responses API: {e}")
    
    print("\n✅ Server parameter testing completed!")

if __name__ == "__main__":
    test_server_parameters()