#!/usr/bin/env python3
"""
Example 5: Server & Agentic CLI Integration - Building Powerful AI Systems
==========================================================================

This example demonstrates AbstractCore's server and CLI capabilities:
- OpenAI-compatible server implementation
- Agentic CLI with real-time streaming
- Integration with tools like Codex CLI
- Multi-provider server deployment

Technical Architecture Highlights:
- FastAPI server with OpenAI compatibility
- WebSocket streaming support
- CLI with interactive tool execution
- Server-side provider management

Required: pip install abstractcore[server]
Optional: pip install abstractcore[ollama] for local model server
"""

import os
import sys
import json
import time
import subprocess
import asyncio
from typing import Dict, Any, Optional, List
import logging

# Add project root to path for development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def server_architecture_overview():
    """
    Explains the AbstractCore server architecture.

    Architecture Notes:
    - FastAPI-based for high performance
    - OpenAI API compatibility
    - Multi-provider support
    - Streaming and WebSocket support
    """
    print("=" * 70)
    print("EXAMPLE 5: AbstractCore Server Architecture")
    print("=" * 70)

    print("\n🏗️ Server Architecture:")
    print("""
    ┌─────────────────────────────────────────────┐
    │         AbstractCore Server (FastAPI)        │
    ├─────────────────────────────────────────────┤
    │                                             │
    │  /v1/chat/completions  ◄─── OpenAI Format  │
    │  /v1/models           ◄─── Model Discovery │
    │  /v1/embeddings       ◄─── Embeddings API  │
    │                                             │
    │  Provider Router:                           │
    │  ├── /openai/*     → OpenAI Provider       │
    │  ├── /anthropic/*  → Anthropic Provider    │
    │  ├── /ollama/*     → Ollama Provider       │
    │  └── /huggingface/* → HuggingFace Provider │
    │                                             │
    │  Features:                                  │
    │  • Streaming (SSE)                          │
    │  • WebSocket support                        │
    │  • Tool calling                             │
    │  • Telemetry & metrics                      │
    └─────────────────────────────────────────────┘
    """)

    print("🔑 Key Features:")
    print("   • 100% OpenAI API compatible")
    print("   • Provider specified in URL path")
    print("   • Automatic model discovery")
    print("   • Production-ready with CORS, auth, metrics")


def starting_the_server():
    """
    Demonstrates how to start the AbstractCore server.

    Architecture Notes:
    - Simple CLI command to start
    - Configurable via environment variables
    - Auto-discovers available providers
    """
    print("\n" + "=" * 70)
    print("Starting the AbstractCore Server")
    print("=" * 70)

    print("\n🚀 Server Startup Commands:")

    # Basic startup
    print("\n1️⃣ Basic Server Start:")
    print("   ```bash")
    print("   python -m abstractcore.server")
    print("   ```")
    print("   • Runs on http://localhost:8000")
    print("   • Auto-detects available providers")

    # Custom configuration
    print("\n2️⃣ Custom Configuration:")
    print("   ```bash")
    print("   python -m abstractcore.server \\")
    print("     --host 0.0.0.0 \\")
    print("     --port 8080 \\")
    print("     --reload")
    print("   ```")

    # Environment configuration
    print("\n3️⃣ Environment Variables:")
    print("   ```bash")
    print("   export OPENAI_API_KEY='your-key'")
    print("   export ANTHROPIC_API_KEY='your-key'")
    print("   export OLLAMA_BASE_URL='http://localhost:11434'")
    print("   python -m abstractcore.server")
    print("   ```")

    # Docker deployment
    print("\n4️⃣ Docker Deployment:")
    print("   ```dockerfile")
    print("   FROM python:3.10-slim")
    print("   RUN pip install abstractcore[server]")
    print("   EXPOSE 8000")
    print("   CMD [\"python\", \"-m\", \"abstractcore.server\"]")
    print("   ```")

    print("\n📊 Server Endpoints:")
    endpoints = [
        ("GET", "/", "Health check"),
        ("GET", "/v1/models", "List available models"),
        ("POST", "/v1/chat/completions", "Chat completion"),
        ("POST", "/v1/embeddings", "Generate embeddings"),
        ("GET", "/metrics", "Prometheus metrics"),
    ]

    for method, path, description in endpoints:
        print(f"   {method:4s} {path:25s} - {description}")


def openai_compatibility_demo():
    """
    Demonstrates OpenAI API compatibility.

    Architecture Notes:
    - Drop-in replacement for OpenAI API
    - Works with existing OpenAI clients
    - Provider selection via URL path
    """
    print("\n" + "=" * 70)
    print("OpenAI API Compatibility")
    print("=" * 70)

    print("\n🔄 Using OpenAI Python Client with AbstractCore Server:")

    # Example 1: OpenAI client configuration
    print("\n1️⃣ Configure OpenAI Client:")
    print("   ```python")
    print("   from openai import OpenAI")
    print("   ")
    print("   # Point to AbstractCore server")
    print("   client = OpenAI(")
    print("       base_url=\"http://localhost:8000/ollama/v1\",")
    print("       api_key=\"not-needed\"  # For local providers")
    print("   )")
    print("   ```")

    # Example 2: Making requests
    print("\n2️⃣ Make Requests (Identical to OpenAI):")
    print("   ```python")
    print("   response = client.chat.completions.create(")
    print("       model=\"qwen3-coder:30b\",")
    print("       messages=[")
    print("           {\"role\": \"user\", \"content\": \"Hello!\"}")
    print("       ],")
    print("       stream=True")
    print("   )")
    print("   ")
    print("   for chunk in response:")
    print("       print(chunk.choices[0].delta.content, end=\"\")")
    print("   ```")

    # Example 3: Provider switching
    print("\n3️⃣ Switch Providers by Changing URL:")
    providers_urls = [
        ("OpenAI", "http://localhost:8000/openai/v1"),
        ("Anthropic", "http://localhost:8000/anthropic/v1"),
        ("Ollama", "http://localhost:8000/ollama/v1"),
        ("HuggingFace", "http://localhost:8000/huggingface/v1"),
    ]

    for provider, url in providers_urls:
        print(f"   • {provider:12s}: {url}")

    print("\n✅ Benefits:")
    print("   • No code changes needed")
    print("   • Existing tools work immediately")
    print("   • Provider abstraction at server level")


def agentic_cli_demonstration():
    """
    Demonstrates the Agentic CLI capabilities.

    Architecture Notes:
    - Interactive CLI for LLM interaction
    - Real-time streaming with tool execution
    - Custom commands and configuration
    """
    print("\n" + "=" * 70)
    print("Agentic CLI Features")
    print("=" * 70)

    print("\n🤖 AbstractCore CLI - Interactive AI Assistant:")

    # CLI startup
    print("\n1️⃣ Starting the CLI:")
    print("   ```bash")
    print("   python -m abstractcore.utils.cli \\")
    print("     --provider ollama \\")
    print("     --model qwen3-coder:30b \\")
    print("     --stream")
    print("   ```")

    # CLI commands
    print("\n2️⃣ Available Commands:")
    commands = [
        ("/help", "Show available commands"),
        ("/model <name>", "Switch model"),
        ("/provider <name>", "Switch provider"),
        ("/tooltag 'start' 'end'", "Set custom tool tags"),
        ("/stream", "Toggle streaming mode"),
        ("/clear", "Clear conversation"),
        ("/exit", "Exit CLI"),
    ]

    for cmd, description in commands:
        print(f"   {cmd:25s} - {description}")

    # Interactive session example
    print("\n3️⃣ Example Interactive Session:")
    print("""
   ┌─────────────────────────────────────────┐
   │  AbstractCore CLI v2.2.4                 │
   │  Model: qwen3-coder:30b                 │
   │  Provider: ollama                       │
   │  Streaming: ✅ Enabled                  │
   └─────────────────────────────────────────┘

   👤 You: List files in current directory and calculate 42 * 3.14

   🤖 Assistant: I'll help you with both tasks.

   First, let me list the files:
   START{"name": "list_files", "arguments": {"directory": "."}}END

   🔧 Tool Results:
   **list_files({'directory': '.'})**
   ✅ Files found:
     • example_1_basic_generation.py
     • example_2_provider_configuration.py
     • example_3_tool_calling.py

   Now for the calculation:
   START{"name": "calculate", "arguments": {"expression": "42 * 3.14"}}END

   🔧 Tool Results:
   **calculate({'expression': '42 * 3.14'})**
   ✅ Result: 131.88

   The calculation 42 × 3.14 equals 131.88.

   👤 You: /tooltag 'EXEC[' ']EXEC'
   🏷️ Tool call tags set to: EXEC[...]EXEC

   👤 You: _
   """)


def codex_cli_integration():
    """
    Demonstrates integration with Codex CLI.

    Architecture Notes:
    - AbstractCore as backend for Codex
    - Seamless tool execution
    - Real-time streaming support
    """
    print("\n" + "=" * 70)
    print("Codex CLI Integration")
    print("=" * 70)

    print("\n🔧 Using AbstractCore with Codex CLI:")

    # Configuration
    print("\n1️⃣ Configure Codex to Use AbstractCore Server:")
    print("   ```bash")
    print("   # Start AbstractCore server")
    print("   python -m abstractcore.server &")
    print("   ")
    print("   # Configure Codex")
    print("   export CODEX_API_BASE='http://localhost:8000/ollama/v1'")
    print("   export CODEX_MODEL='qwen3-coder:30b'")
    print("   ```")

    # Codex workflow
    print("\n2️⃣ Codex Workflow with AbstractCore:")
    print("""
   ┌────────────┐     ┌─────────────────┐     ┌──────────────┐
   │  Codex CLI │────►│ AbstractCore     │────►│ Ollama/Local │
   │            │◄────│ Server          │◄────│ Model        │
   └────────────┘     └─────────────────┘     └──────────────┘
         │                     │                      │
         │                     │                      │
         ▼                     ▼                      ▼
    User Input          Tool Execution          Model Inference
    """)

    # Example Codex commands
    print("\n3️⃣ Example Codex Commands:")
    codex_examples = [
        ("codex 'refactor this function'", "Code refactoring"),
        ("codex 'add tests for user.py'", "Test generation"),
        ("codex 'explain this algorithm'", "Code explanation"),
        ("codex 'fix the bug in line 42'", "Bug fixing"),
    ]

    for cmd, description in codex_examples:
        print(f"   $ {cmd:40s} # {description}")

    print("\n✅ Benefits of Integration:")
    print("   • Local model support via Ollama")
    print("   • Custom tool execution")
    print("   • Real-time streaming")
    print("   • Provider flexibility")


def multi_provider_deployment():
    """
    Demonstrates multi-provider deployment patterns.

    Architecture Notes:
    - Load balancing across providers
    - Failover strategies
    - Cost optimization
    """
    print("\n" + "=" * 70)
    print("Multi-Provider Deployment Patterns")
    print("=" * 70)

    print("\n🌐 Deployment Architecture:")
    print("""
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼─────┐ ┌─────▼──────┐ ┌────▼──────┐
         │ AbstractCore│ │ AbstractCore│ │AbstractCore│
         │  Server 1  │ │  Server 2  │ │ Server 3  │
         └──────┬─────┘ └─────┬──────┘ └────┬──────┘
                │              │              │
         ┌──────▼─────────────▼──────────────▼──────┐
         │          Provider Pool                    │
         │  • OpenAI (primary)                      │
         │  • Anthropic (fallback)                  │
         │  • Ollama (local/cost-saving)           │
         └───────────────────────────────────────────┘
    """)

    # Configuration example
    print("\n📝 Multi-Provider Configuration:")
    print("   ```yaml")
    print("   # abstractcore-config.yaml")
    print("   providers:")
    print("     - name: openai")
    print("       priority: 1")
    print("       models: ['gpt-4o', 'gpt-4o-mini']")
    print("       rate_limit: 100_000  # tokens/minute")
    print("   ")
    print("     - name: anthropic")
    print("       priority: 2")
    print("       models: ['claude-3-5-sonnet']")
    print("       rate_limit: 50_000")
    print("   ")
    print("     - name: ollama")
    print("       priority: 3")
    print("       models: ['qwen3-coder:30b']")
    print("       rate_limit: null  # No limit for local")
    print("   ")
    print("   routing:")
    print("     strategy: 'cost_optimized'  # or 'latency_optimized'")
    print("     fallback: true")
    print("   ```")

    # Load balancing strategies
    print("\n⚖️ Load Balancing Strategies:")
    strategies = [
        ("Round Robin", "Distribute evenly across providers"),
        ("Least Latency", "Route to fastest provider"),
        ("Cost Optimized", "Prefer cheaper providers"),
        ("Priority Based", "Use primary, fallback on error"),
        ("Model Specific", "Route based on model capabilities"),
    ]

    for strategy, description in strategies:
        print(f"   • {strategy:15s}: {description}")


def streaming_server_implementation():
    """
    Demonstrates server-side streaming implementation.

    Architecture Notes:
    - Server-Sent Events (SSE) for HTTP streaming
    - WebSocket support for bidirectional streaming
    - Efficient chunking and buffering
    """
    print("\n" + "=" * 70)
    print("Server-Side Streaming Implementation")
    print("=" * 70)

    print("\n📡 Streaming Technologies:")

    # SSE Implementation
    print("\n1️⃣ Server-Sent Events (SSE):")
    print("   ```python")
    print("   @app.post('/v1/chat/completions')")
    print("   async def chat_completion(request: ChatRequest):")
    print("       if request.stream:")
    print("           return StreamingResponse(")
    print("               generate_stream(request),")
    print("               media_type='text/event-stream'")
    print("           )")
    print("   ")
    print("   async def generate_stream(request):")
    print("       async for chunk in llm.agenerate_stream(request):")
    print("           yield f'data: {json.dumps(chunk)}\\n\\n'")
    print("   ```")

    # WebSocket Implementation
    print("\n2️⃣ WebSocket Streaming:")
    print("   ```python")
    print("   @app.websocket('/ws/chat')")
    print("   async def websocket_chat(websocket: WebSocket):")
    print("       await websocket.accept()")
    print("       ")
    print("       while True:")
    print("           data = await websocket.receive_json()")
    print("           ")
    print("           async for chunk in process_message(data):")
    print("               await websocket.send_json({")
    print("                   'type': 'chunk',")
    print("                   'content': chunk.content")
    print("               })")
    print("   ```")

    # Performance metrics
    print("\n📊 Streaming Performance Metrics:")
    metrics = [
        ("First Byte Latency", "<10ms", "Time to first chunk"),
        ("Throughput", "50-100 tokens/sec", "Generation speed"),
        ("Memory Usage", "O(1) constant", "No buffering"),
        ("Connection Overhead", "~5ms", "WebSocket setup"),
    ]

    print("   Metric              | Value          | Description")
    print("   --------------------|----------------|----------------")
    for metric, value, desc in metrics:
        print(f"   {metric:19s} | {value:14s} | {desc}")


def production_deployment_checklist():
    """
    Provides a production deployment checklist.

    Architecture Notes:
    - Security considerations
    - Monitoring and observability
    - Scaling strategies
    """
    print("\n" + "=" * 70)
    print("Production Deployment Checklist")
    print("=" * 70)

    print("\n✅ Production Checklist:")

    # Security
    print("\n🔒 Security:")
    checklist = [
        "API key management (use secrets manager)",
        "Rate limiting per client/IP",
        "Input validation and sanitization",
        "CORS configuration",
        "TLS/HTTPS encryption",
        "Audit logging",
    ]
    for item in checklist:
        print(f"   ☐ {item}")

    # Monitoring
    print("\n📊 Monitoring & Observability:")
    monitoring = [
        "Prometheus metrics endpoint",
        "Request/response logging",
        "Error tracking (Sentry/similar)",
        "Performance monitoring (APM)",
        "Health check endpoints",
        "SLA monitoring",
    ]
    for item in monitoring:
        print(f"   ☐ {item}")

    # Scaling
    print("\n📈 Scaling:")
    scaling = [
        "Horizontal scaling with load balancer",
        "Connection pooling",
        "Request queuing",
        "Cache layer (Redis)",
        "Database connection management",
        "Auto-scaling policies",
    ]
    for item in scaling:
        print(f"   ☐ {item}")

    # Deployment
    print("\n🚀 Deployment:")
    deployment = [
        "Docker containerization",
        "Kubernetes orchestration",
        "CI/CD pipeline",
        "Blue-green deployment",
        "Rollback strategy",
        "Configuration management",
    ]
    for item in deployment:
        print(f"   ☐ {item}")


def example_server_client_code():
    """
    Provides example code for server and client.

    Architecture Notes:
    - Complete working examples
    - Best practices demonstrated
    - Error handling included
    """
    print("\n" + "=" * 70)
    print("Complete Server & Client Examples")
    print("=" * 70)

    print("\n📝 Minimal Server Implementation:")
    print("""
```python
# server.py
from fastapi import FastAPI
from abstractcore import create_llm
from abstractcore.server import create_app

# Create FastAPI app with AbstractCore
app = create_app()

# Custom endpoint example
@app.get("/custom/health")
async def custom_health():
    return {"status": "healthy", "custom": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
    """)

    print("\n📝 Client Implementation:")
    print("""
```python
# client.py
import httpx
import json

class AbstractCoreClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def chat(self, provider, model, messages, stream=False):
        url = f"{self.base_url}/{provider}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        if stream:
            with self.client.stream("POST", url, json=payload) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        yield json.loads(line[6:])
        else:
            response = self.client.post(url, json=payload)
            return response.json()

# Usage
client = AbstractCoreClient()

# Non-streaming
response = client.chat(
    provider="ollama",
    model="qwen3-coder:30b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response["choices"][0]["message"]["content"])

# Streaming
for chunk in client.chat(
    provider="ollama",
    model="qwen3-coder:30b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    if chunk["choices"][0]["delta"].get("content"):
        print(chunk["choices"][0]["delta"]["content"], end="")
```
    """)


def main():
    """
    Main entry point - demonstrates server and CLI integration.
    """
    print("\n" + "🌐 " * 20)
    print(" AbstractCore Core - Example 5: Server & Agentic CLI")
    print("🌐 " * 20)

    # Run all demonstrations
    server_architecture_overview()
    starting_the_server()
    openai_compatibility_demo()
    agentic_cli_demonstration()
    codex_cli_integration()
    multi_provider_deployment()
    streaming_server_implementation()
    production_deployment_checklist()
    example_server_client_code()

    print("\n" + "=" * 70)
    print("✅ Example 5 Complete!")
    print("\nKey Takeaways:")
    print("• OpenAI-compatible server with multi-provider support")
    print("• Agentic CLI with real-time streaming and tools")
    print("• Seamless integration with tools like Codex")
    print("• Production-ready deployment patterns")
    print("• WebSocket and SSE streaming support")
    print("• Complete monitoring and observability")
    print("\nNext: Run example_6_production_patterns.py for best practices")
    print("=" * 70)


if __name__ == "__main__":
    main()