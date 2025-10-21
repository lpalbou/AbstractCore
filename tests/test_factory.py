"""
Test the factory function.
"""

import pytest
from abstractcore import create_llm
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.providers.anthropic_provider import AnthropicProvider


class TestFactory:
    """Test create_llm factory function"""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider"""
        try:
            llm = create_llm("openai")
            assert isinstance(llm, OpenAIProvider)
            assert llm.model == "gpt-5-nano-2025-08-07"  # Default model
        except ImportError:
            pytest.skip("OpenAI dependencies not available")

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider"""
        try:
            llm = create_llm("anthropic")
            assert isinstance(llm, AnthropicProvider)
            assert llm.model == "claude-3-5-haiku-latest"  # Default model
        except ImportError:
            pytest.skip("Anthropic dependencies not available")

    def test_create_provider_with_custom_model(self):
        """Test creating provider with custom model"""
        try:
            llm = create_llm("openai", model="gpt-4o")
            assert llm.model == "gpt-4o"
        except ImportError:
            pytest.skip("OpenAI dependencies not available")

    def test_unknown_provider_raises_error(self):
        """Test unknown provider raises error"""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm("nonexistent")

    def test_provider_case_insensitive(self):
        """Test provider names are case insensitive"""
        try:
            llm1 = create_llm("OPENAI")
            llm2 = create_llm("OpenAI")
            llm3 = create_llm("openai")
            
            assert all(isinstance(llm, OpenAIProvider) for llm in [llm1, llm2, llm3])
        except ImportError:
            pytest.skip("OpenAI dependencies not available")