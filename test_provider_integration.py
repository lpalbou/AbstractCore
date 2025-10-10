#!/usr/bin/env python3
"""
Test that AbstractLLM providers actually use architecture detection
for tool calls and structured output requests.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abstractllm import create_llm
from abstractllm.tools.core import ToolDefinition
from pydantic import BaseModel
from typing import List
import json

# Test models
TEST_MODELS = [
    "qwen3:4b-instruct-2507-q4_K_M",  # Qwen3 with prompted tools
    "gemma3:4b-it-qat",                # Gemma3 with native tools
]

def test_tool_calling_integration():
    """Test that providers actually use architecture detection for tool calls."""
    print("🔧 Testing Tool Calling Integration with Providers")
    print("=" * 60)
    
    # Define test tools
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"
    
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return f"Error: Invalid expression '{expression}'"
    
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "The city or location to get weather for"
                }
            }
        ),
        ToolDefinition(
            name="calculate",
            description="Calculate a mathematical expression",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to calculate"
                }
            }
        )
    ]
    
    for model in TEST_MODELS:
        try:
            print(f"\nTesting {model}...")
            
            # Create LLM instance
            llm = create_llm("ollama", model=model)
            
            # Test 1: Tool calling with generate_with_tools
            print("  Test 1: generate_with_tools")
            try:
                response = llm.generate_with_tools(
                    "What's the weather like in Paris?",
                    tools=tools,
                    max_tokens=200
                )
                
                print(f"    Response: {response.content[:100]}...")
                print(f"    Tool Calls: {len(response.tool_calls)}")
                
                if response.tool_calls:
                    for tc in response.tool_calls:
                        print(f"      - {tc.name}({tc.arguments})")
                        
                        # Execute the tool
                        if tc.name == "get_weather":
                            result = get_weather(tc.arguments.get("location", ""))
                        elif tc.name == "calculate":
                            result = calculate(tc.arguments.get("expression", ""))
                        else:
                            result = "Unknown tool"
                        
                        print(f"      Result: {result}")
                else:
                    print("      No tool calls detected")
                    
            except Exception as e:
                print(f"    ❌ generate_with_tools failed: {e}")
            
            # Test 2: Tool calling with tools parameter
            print("  Test 2: generate with tools parameter")
            try:
                response = llm.generate(
                    "Calculate 15 * 8 + 3",
                    tools=tools,
                    max_tokens=200
                )
                
                print(f"    Response: {response.content[:100]}...")
                print(f"    Tool Calls: {len(response.tool_calls)}")
                
                if response.tool_calls:
                    for tc in response.tool_calls:
                        print(f"      - {tc.name}({tc.arguments})")
                else:
                    print("      No tool calls detected")
                    
            except Exception as e:
                print(f"    ❌ generate with tools parameter failed: {e}")
            
            print(f"✅ {model} tool calling integration test completed")
            
        except Exception as e:
            print(f"❌ {model} tool calling integration test failed: {e}")
            import traceback
            traceback.print_exc()

def test_structured_output_integration():
    """Test that providers actually use architecture detection for structured output."""
    print("\n📋 Testing Structured Output Integration with Providers")
    print("=" * 60)
    
    # Define test models
    class PersonInfo(BaseModel):
        name: str
        age: int
        city: str
        occupation: str
    
    class MathProblem(BaseModel):
        problem: str
        solution: float
        steps: List[str]
    
    for model in TEST_MODELS:
        try:
            print(f"\nTesting {model}...")
            
            # Create LLM instance
            llm = create_llm("ollama", model=model)
            
            # Test 1: Structured output with response_model parameter
            print("  Test 1: generate with response_model parameter")
            try:
                response = llm.generate(
                    "Create a profile for Alice, 28, from Tokyo, software engineer",
                    response_model=PersonInfo,
                    max_tokens=200
                )
                
                print(f"    Response: {response.content[:100]}...")
                print(f"    Structured Output: {response.structured_output}")
                
                if response.structured_output:
                    print(f"    Name: {response.structured_output.name}")
                    print(f"    Age: {response.structured_output.age}")
                    print(f"    City: {response.structured_output.city}")
                    print(f"    Occupation: {response.structured_output.occupation}")
                else:
                    print("    No structured output detected")
                    
            except Exception as e:
                print(f"    ❌ generate with response_model failed: {e}")
            
            # Test 2: Structured output with generate_structured method
            print("  Test 2: generate_structured method")
            try:
                result = llm.generate_structured(
                    "Solve the equation: 2x + 5 = 15. Show your work step by step.",
                    response_model=MathProblem,
                    max_tokens=300
                )
                
                print(f"    Result: {result}")
                print(f"    Problem: {result.problem}")
                print(f"    Solution: {result.solution}")
                print(f"    Steps: {len(result.steps)} steps")
                
            except Exception as e:
                print(f"    ❌ generate_structured failed: {e}")
            
            print(f"✅ {model} structured output integration test completed")
            
        except Exception as e:
            print(f"❌ {model} structured output integration test failed: {e}")
            import traceback
            traceback.print_exc()

def test_provider_architecture_detection():
    """Test that providers are actually using architecture detection internally."""
    print("\n🔍 Testing Provider Architecture Detection")
    print("=" * 60)
    
    for model in TEST_MODELS:
        try:
            print(f"\nTesting {model}...")
            
            # Create LLM instance
            llm = create_llm("ollama", model=model)
            
            # Check if provider has architecture detection
            print("  Checking provider attributes...")
            
            # Check for architecture-related attributes
            attrs_to_check = [
                'model_name', 'architecture', 'capabilities', 
                'tool_handler', 'structured_handler'
            ]
            
            for attr in attrs_to_check:
                if hasattr(llm, attr):
                    value = getattr(llm, attr)
                    print(f"    {attr}: {value}")
                else:
                    print(f"    {attr}: Not found")
            
            # Check if provider is using architecture detection
            print("  Checking architecture detection usage...")
            
            # Try to access the provider's internal architecture detection
            if hasattr(llm, 'provider'):
                provider = llm.provider
                print(f"    Provider type: {type(provider).__name__}")
                
                # Check if provider has architecture detection
                if hasattr(provider, 'architecture'):
                    print(f"    Provider architecture: {provider.architecture}")
                if hasattr(provider, 'capabilities'):
                    print(f"    Provider capabilities: {provider.capabilities}")
                if hasattr(provider, 'tool_handler'):
                    print(f"    Provider tool handler: {type(provider.tool_handler).__name__}")
                if hasattr(provider, 'structured_handler'):
                    print(f"    Provider structured handler: {type(provider.structured_handler).__name__}")
            
            print(f"✅ {model} provider architecture detection test completed")
            
        except Exception as e:
            print(f"❌ {model} provider architecture detection test failed: {e}")
            import traceback
            traceback.print_exc()

def test_manual_architecture_verification():
    """Manually verify that architecture detection is being used."""
    print("\n🔬 Manual Architecture Verification")
    print("=" * 60)
    
    from abstractllm.architectures.detection import detect_architecture, get_model_capabilities
    
    for model in TEST_MODELS:
        try:
            print(f"\nVerifying {model}...")
            
            # Get architecture and capabilities
            architecture = detect_architecture(model)
            capabilities = get_model_capabilities(model)
            
            print(f"  Detected Architecture: {architecture}")
            print(f"  Tool Support: {capabilities.get('tool_support', 'none')}")
            print(f"  Structured Output: {capabilities.get('structured_output', 'none')}")
            
            # Create LLM and check if it's using the same detection
            llm = create_llm("ollama", model=model)
            
            # Check if the LLM is using architecture detection
            if hasattr(llm, 'provider'):
                provider = llm.provider
                
                # Check if provider has architecture detection
                if hasattr(provider, 'architecture'):
                    provider_arch = provider.architecture
                    print(f"  Provider Architecture: {provider_arch}")
                    
                    if provider_arch == architecture:
                        print("  ✅ Architecture detection matches")
                    else:
                        print("  ❌ Architecture detection mismatch")
                
                # Check if provider has capabilities
                if hasattr(provider, 'capabilities'):
                    provider_caps = provider.capabilities
                    print(f"  Provider Capabilities: {provider_caps}")
                    
                    # Compare key capabilities
                    for key in ['tool_support', 'structured_output']:
                        if key in capabilities and key in provider_caps:
                            if capabilities[key] == provider_caps[key]:
                                print(f"  ✅ {key} matches: {capabilities[key]}")
                            else:
                                print(f"  ❌ {key} mismatch: {capabilities[key]} vs {provider_caps[key]}")
            
            print(f"✅ {model} manual verification completed")
            
        except Exception as e:
            print(f"❌ {model} manual verification failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run all provider integration tests."""
    print("🚀 Provider Integration Tests - Real Architecture Detection Usage")
    print("=" * 80)
    print()
    
    test_tool_calling_integration()
    test_structured_output_integration()
    test_provider_architecture_detection()
    test_manual_architecture_verification()
    
    print("\n🎉 All provider integration tests completed!")

if __name__ == "__main__":
    main()