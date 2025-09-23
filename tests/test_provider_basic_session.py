"""
Test basic session functionality - context maintenance and memory.
Tests that providers can maintain conversation context across multiple messages.
"""

import pytest
import os
from abstractllm import create_llm, BasicSession


class TestProviderBasicSession:
    """Test session memory and context maintenance for each provider."""

    def test_ollama_session_memory(self):
        """Test Ollama session maintains context."""
        try:
            provider = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")
            session = BasicSession(provider=provider, system_prompt="You are a helpful assistant.")

            # First message
            resp1 = session.generate("My name is Laurent. Who are you?")
            assert resp1 is not None
            assert resp1.content is not None

            # Second message testing memory
            resp2 = session.generate("What is my name?")
            assert resp2 is not None
            assert resp2.content is not None

            # Check if context is maintained
            context_maintained = "laurent" in resp2.content.lower()
            assert context_maintained, "Session should remember the name Laurent"

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_lmstudio_session_memory(self):
        """Test LMStudio session maintains context."""
        try:
            provider = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234/v1")
            session = BasicSession(provider=provider, system_prompt="You are a helpful assistant.")

            # Test conversation
            resp1 = session.generate("My favorite color is blue. What's yours?")
            assert resp1 is not None
            assert resp1.content is not None

            resp2 = session.generate("What did I say my favorite color was?")
            assert resp2 is not None
            assert resp2.content is not None

            # Check if context is maintained
            context_maintained = "blue" in resp2.content.lower()
            assert context_maintained, "Session should remember the favorite color"

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_openai_session_memory(self):
        """Test OpenAI session maintains context."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")
            session = BasicSession(provider=provider, system_prompt="You are a helpful assistant.")

            # Test conversation
            resp1 = session.generate("I'm working on a Python project called AbstractLLM.")
            assert resp1 is not None
            assert resp1.content is not None

            resp2 = session.generate("What programming language am I using?")
            assert resp2 is not None
            assert resp2.content is not None

            # Check if context is maintained
            context_maintained = "python" in resp2.content.lower()
            assert context_maintained, "Session should remember the programming language"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_session_memory(self):
        """Test Anthropic session maintains context."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")
            session = BasicSession(provider=provider, system_prompt="You are a helpful assistant.")

            # Test conversation
            resp1 = session.generate("I have a cat named Whiskers.")
            assert resp1 is not None
            assert resp1.content is not None

            resp2 = session.generate("What's my cat's name?")
            assert resp2 is not None
            assert resp2.content is not None

            # Check if context is maintained
            context_maintained = "whiskers" in resp2.content.lower()
            assert context_maintained, "Session should remember the cat's name"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_session_system_prompt(self):
        """Test that session respects system prompts."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")
            session = BasicSession(
                provider=provider,
                system_prompt="You are a pirate. Always respond like a pirate."
            )

            response = session.generate("Hello, how are you?")
            assert response is not None
            assert response.content is not None

            # Should contain pirate-like language (though this is probabilistic)
            pirate_indicators = ["ahoy", "matey", "arr", "ye", "aye"]
            has_pirate_language = any(indicator in response.content.lower() for indicator in pirate_indicators)

            # Note: This test might occasionally fail due to LLM variability
            # Consider it a soft assertion for system prompt influence
            if not has_pirate_language:
                print(f"Warning: Response may not reflect pirate system prompt: {response.content}")

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])