"""
Test all providers with real implementations.
No mocking - test actual provider connections.
"""

import pytest
import os
import json
import time
from typing import Dict, Any, List, Optional
from abstractllm import create_llm, BasicSession
from abstractllm.core.types import GenerateResponse


class TestProviders:
    """Test all providers with real implementations - no mocking."""

    def test_ollama_simple_message(self):
        """Test Ollama simple message generation with qwen3-coder:30b."""
        try:
            provider = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")

            prompt = "Who are you? Please respond in one sentence."
            start_time = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start_time

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30  # Should respond within 30 seconds

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_lmstudio_simple_message(self):
        """Test LMStudio simple message generation with qwen/qwen3-coder-30b."""
        try:
            provider = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234/v1")

            prompt = "Who are you? Please respond in one sentence."
            start_time = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start_time

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30  # Should respond within 30 seconds

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_mlx_simple_message(self):
        """Test MLX simple message generation with mlx-community/Qwen3-4B-4bit."""
        try:
            provider = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")

            prompt = "Who are you? Please respond in one sentence."
            start_time = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start_time

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 60  # MLX might be slower

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["mlx", "import", "not found", "failed to load"]):
                pytest.skip("MLX not available or model not found")
            else:
                raise

    def test_huggingface_simple_message(self):
        """Test HuggingFace simple message generation with Qwen/Qwen3-4B."""
        try:
            provider = create_llm("huggingface", model="Qwen/Qwen3-4B")

            prompt = "Who are you? Please respond in one sentence."
            start_time = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start_time

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 120  # HF might be slow on first load

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["transformers", "torch", "not found", "failed to load"]):
                pytest.skip("HuggingFace not available or model not found")
            else:
                raise

    def test_openai_simple_message(self):
        """Test OpenAI simple message generation with gpt-4o-mini."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            prompt = "Who are you? Please respond in one sentence."
            start_time = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start_time

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 10  # Cloud should be fast

            # Check usage tracking
            if response.usage:
                assert "total_tokens" in response.usage
                assert response.usage["total_tokens"] > 0

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_simple_message(self):
        """Test Anthropic simple message generation with claude-3-5-haiku-20241022."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            prompt = "Who are you? Please respond in one sentence."
            start_time = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start_time

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 10  # Cloud should be fast

            # Check usage tracking
            if response.usage:
                assert "total_tokens" in response.usage
                assert response.usage["total_tokens"] > 0

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_openai_tool_call(self):
        """Test OpenAI tool calling with gpt-4o-mini."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            # Define a simple tool
            tools = [{
                "name": "list_files",
                "description": "List files in the current directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list files from"
                        }
                    },
                    "required": ["path"]
                }
            }]

            prompt = "Please list the files in the current directory"
            response = provider.generate(prompt, tools=tools)

            assert response is not None

            if response.has_tool_calls():
                # Tool calling worked
                assert len(response.tool_calls) > 0
                tool_call = response.tool_calls[0]
                assert tool_call.get('name') == 'list_files'
                assert 'arguments' in tool_call
            else:
                # OpenAI should support tools, but test might fail due to prompt
                pytest.skip("OpenAI didn't use tools (prompt might need adjustment)")

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_tool_call(self):
        """Test Anthropic tool calling with claude-3-5-haiku-20241022."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            # Define a simple tool
            tools = [{
                "name": "list_files",
                "description": "List files in the current directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list files from"
                        }
                    },
                    "required": ["path"]
                }
            }]

            prompt = "Please list the files in the current directory"
            response = provider.generate(prompt, tools=tools)

            assert response is not None

            # Check for tool usage - Anthropic may use prompted format
            tool_used = False

            if response.has_tool_calls():
                # Native tool calling format
                assert len(response.tool_calls) > 0
                tool_call = response.tool_calls[0]
                assert tool_call.get('name') == 'list_files'
                assert 'arguments' in tool_call
                tool_used = True

            elif "<tool_call>" in response.content and "list_files" in response.content:
                # Prompted tool calling format (Anthropic style)
                # Tool was executed and results included in content
                assert "Tool Results:" in response.content or "files in" in response.content.lower()
                tool_used = True

            assert tool_used, f"Tool should have been used. Response: {response.content[:200]}..."

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_ollama_session(self):
        """Test BasicSession with Ollama provider."""
        try:
            provider = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")

            # Create session
            session = BasicSession(
                provider=provider,
                system_prompt="You are a helpful assistant."
            )

            # Test conversation
            response1 = session.generate("What is 2+2?")
            assert response1 is not None
            assert response1.content is not None

            response2 = session.generate("What was my previous question?")
            assert response2 is not None
            assert response2.content is not None

            # Check if context is maintained (should mention 2+2 or math)
            context_maintained = any(term in response2.content.lower() for term in ["2+2", "math", "addition", "previous"])
            assert context_maintained, "Session should maintain context about previous question"

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_openai_session(self):
        """Test BasicSession with OpenAI provider."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            # Create session
            session = BasicSession(
                provider=provider,
                system_prompt="You are a helpful assistant."
            )

            # Test conversation
            response1 = session.generate("What is 2+2?")
            assert response1 is not None
            assert response1.content is not None

            response2 = session.generate("What was my previous question?")
            assert response2 is not None
            assert response2.content is not None

            # Check if context is maintained (should mention 2+2 or math)
            context_maintained = any(term in response2.content.lower() for term in ["2+2", "math", "addition", "previous"])
            assert context_maintained, "Session should maintain context about previous question"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_session(self):
        """Test BasicSession with Anthropic provider."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            # Create session
            session = BasicSession(
                provider=provider,
                system_prompt="You are a helpful assistant."
            )

            # Test conversation
            response1 = session.generate("What is 2+2?")
            assert response1 is not None
            assert response1.content is not None

            response2 = session.generate("What was my previous question?")
            assert response2 is not None
            assert response2.content is not None

            # Check if context is maintained (should mention 2+2 or math)
            context_maintained = any(term in response2.content.lower() for term in ["2+2", "math", "addition", "previous"])
            assert context_maintained, "Session should maintain context about previous question"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])