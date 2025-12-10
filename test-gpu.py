#!/usr/bin/env python3
"""
GPU Test Script for vLLM Provider with AbstractCore Server

This script:
1. Tests the vLLM provider directly (basic functionality)
2. Starts an AbstractCore server with vLLM provider
3. Provides curl examples to test the OpenAI-compatible endpoint

Prerequisites:
- vLLM server running: vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000
- AbstractCore installed: pip install -e .

Usage:
    python test-gpu.py

The script will:
- Test vLLM provider connectivity
- Start AbstractCore server on port 8080
- Keep server running for manual testing
- Press Ctrl+C to stop
"""

import os
import sys
import asyncio
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_vllm_connectivity():
    """Test basic vLLM provider functionality."""
    print_section("STEP 1: Testing vLLM Provider Connectivity")

    try:
        from abstractcore import create_llm

        print("Creating vLLM provider instance...")
        llm = create_llm('vllm', model='Qwen/Qwen3-Coder-30B-A3B-Instruct')

        print(f"✅ Provider created successfully")
        print(f"   Provider: {llm.provider}")
        print(f"   Model: {llm.model}")
        print(f"   Base URL: {llm.base_url}")

        # Test model listing
        print("\nListing available models from vLLM server...")
        models = llm.list_available_models()
        print(f"✅ Found {len(models)} model(s):")
        for model in models:
            print(f"   - {model}")

        # Test basic generation
        print("\nTesting basic generation...")
        response = llm.generate("Say 'Hello from vLLM!' and nothing else.", temperature=0)
        print(f"✅ Response: {response.content}")

        # Test streaming
        print("\nTesting streaming generation...")
        print("Response: ", end="", flush=True)
        for chunk in llm.generate("Count from 1 to 5", stream=True, temperature=0):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n✅ Streaming works")

        # Test guided JSON (vLLM-specific feature)
        print("\nTesting guided JSON (vLLM-specific)...")
        response = llm.generate(
            "List 3 colors",
            guided_json={
                "type": "object",
                "properties": {
                    "colors": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["colors"]
            },
            temperature=0
        )
        print(f"✅ Guided JSON response: {response.content}")

        # Test capabilities
        print("\nTesting provider capabilities...")
        capabilities = llm.get_capabilities()
        print(f"✅ Capabilities: {', '.join(capabilities)}")

        print("\n" + "🎉 All vLLM provider tests passed!" + "\n")
        return True

    except Exception as e:
        print(f"\n❌ Error testing vLLM provider: {e}")
        print("\nMake sure vLLM server is running:")
        print("  vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000")
        return False


def print_curl_examples(port=8080):
    """Print curl command examples for testing the server."""
    print_section("CURL EXAMPLES - Test the OpenAI-Compatible Endpoint")

    print("The AbstractCore server is now running and provides an OpenAI-compatible")
    print("endpoint at http://localhost:{port}/v1/chat/completions".replace("{port}", str(port)))
    print("\nYou can test it with these curl commands:\n")

    # Example 1: Basic chat completion
    print("# 1. Basic Chat Completion")
    print("# ---------------------------")
    print(f"""curl -X POST http://localhost:{port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {{"role": "system", "content": "You are a helpful AI assistant."}},
      {{"role": "user", "content": "What is vLLM?"}}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }}'
""")

    # Example 2: Streaming response
    print("\n# 2. Streaming Response")
    print("# ----------------------")
    print(f"""curl -X POST http://localhost:{port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {{"role": "user", "content": "Count from 1 to 10 slowly"}}
    ],
    "stream": true,
    "temperature": 0
  }}'
""")

    # Example 3: Code generation with temperature 0
    print("\n# 3. Code Generation (Deterministic)")
    print("# -----------------------------------")
    print(f"""curl -X POST http://localhost:{port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {{"role": "user", "content": "Write a Python function to calculate factorial"}}
    ],
    "temperature": 0,
    "max_tokens": 300
  }}'
""")

    # Example 4: List available models
    print("\n# 4. List Available Models")
    print("# -------------------------")
    print(f"curl http://localhost:{port}/v1/models")

    # Example 5: List providers
    print("\n# 5. List All Providers")
    print("# ---------------------")
    print(f"curl http://localhost:{port}/providers")

    # Example 6: List only vLLM models
    print("\n# 6. List Only vLLM Models")
    print("# -------------------------")
    print(f"curl http://localhost:{port}/v1/models?provider=vllm")

    # Example 7: Multi-turn conversation
    print("\n# 7. Multi-Turn Conversation")
    print("# ---------------------------")
    print(f"""curl -X POST http://localhost:{port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {{"role": "user", "content": "What is machine learning?"}},
      {{"role": "assistant", "content": "Machine learning is a subset of AI..."}},
      {{"role": "user", "content": "Can you give an example?"}}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }}'
""")

    # Example 8: With seed for reproducibility
    print("\n# 8. Reproducible Generation (with seed)")
    print("# ---------------------------------------")
    print(f"""curl -X POST http://localhost:{port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {{"role": "user", "content": "Generate a random number between 1 and 100"}}
    ],
    "temperature": 0,
    "seed": 42,
    "max_tokens": 50
  }}'
""")

    # Python examples
    print("\n" + "=" * 80)
    print("  PYTHON EXAMPLES - Using OpenAI SDK")
    print("=" * 80 + "\n")

    print("""# Install OpenAI SDK: pip install openai

from openai import OpenAI

# Point to AbstractCore server
client = OpenAI(
    api_key="EMPTY",  # Not needed for local server
    base_url="http://localhost:{port}/v1"
)

# Basic completion
response = client.chat.completions.create(
    model="vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    messages=[
        {{"role": "user", "content": "Explain quantum computing"}}
    ],
    temperature=0.7,
    max_tokens=150
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    messages=[{{"role": "user", "content": "Count to 10"}}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
""".replace("{port}", str(port)))

    print("\n" + "=" * 80 + "\n")


async def start_abstractcore_server(port=8080):
    """Start the AbstractCore server with vLLM provider."""
    print_section("STEP 2: Starting AbstractCore Server")

    print(f"Starting server on port {port}...")
    print("This provides an OpenAI-compatible endpoint for vLLM")
    print("\nServer URL: http://localhost:{port}".replace("{port}", str(port)))
    print("OpenAI endpoint: http://localhost:{port}/v1/chat/completions".replace("{port}", str(port)))

    try:
        # Import server components
        from abstractcore.server.app import app
        import uvicorn

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)

        print(f"\n✅ Server starting on http://0.0.0.0:{port}")
        print("\nPress Ctrl+C to stop the server\n")

        # Print curl examples before starting
        print_curl_examples(port)

        print("\n" + "=" * 80)
        print("  SERVER LOGS")
        print("=" * 80 + "\n")

        # Start server
        await server.serve()

    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        raise


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("  vLLM Provider GPU Test Script")
    print("  AbstractCore + vLLM OpenAI-Compatible Server")
    print("=" * 80)

    # Check if vLLM base URL is set
    vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    print(f"\nvLLM Server URL: {vllm_url}")
    print("(Set VLLM_BASE_URL environment variable to change)\n")

    # Step 1: Test vLLM connectivity
    if not test_vllm_connectivity():
        print("\n⚠️  vLLM connectivity test failed. Starting server anyway...")
        print("The server will be available, but requests will fail until vLLM is running.\n")

    # Step 2: Start server
    try:
        asyncio.run(start_abstractcore_server(port=8080))
    except KeyboardInterrupt:
        print("\n\n✅ Test completed successfully!")
        print("\nTo test again, run: python test-gpu.py\n")


if __name__ == "__main__":
    main()
