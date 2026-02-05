#!/usr/bin/env python3
"""
Debug test to identify the issue with LMStudio connection
"""

import sys
from pathlib import Path

# Add the abstractcore directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abstractcore import create_llm

def test_lmstudio_connection():
    """Test basic LMStudio connection"""
    print("🔍 Testing LMStudio connection...")
    
    try:
        # Create LLM
        llm = create_llm(
            "lmstudio",
            model="qwen/qwen3-next-80b",
            base_url="http://localhost:1234/v1"
        )
        print("✅ LLM created successfully")
        
        # Test simple generation
        print("🔄 Testing simple text generation...")
        response = llm.generate("Hello, how are you?")
        
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        if response is not None:
            print(f"Content: {response.content}")
            print(f"Model: {response.model}")
            print(f"Usage: {response.usage}")
            print(f"Metadata: {response.metadata}")
        else:
            print("❌ Response is None!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_pdf_processing():
    """Test PDF processing without questions"""
    print("\n🔍 Testing PDF processing...")
    
    try:
        # Create LLM
        llm = create_llm(
            "lmstudio",
            model="qwen/qwen3-next-80b",
            base_url="http://localhost:1234/v1"
        )
        
        # Test with PDF
        print("🔄 Testing PDF processing...")
        response = llm.generate(
            "What is this document about?",
            media=["preserving_privacy.pdf"]
        )
        
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        if response is not None:
            print(f"Content length: {len(response.content) if response.content else 0}")
            print(f"Model: {response.model}")
            print(f"Usage: {response.usage}")
        else:
            print("❌ Response is None!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lmstudio_connection()
    test_pdf_processing()

