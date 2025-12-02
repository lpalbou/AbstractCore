"""
Tests for custom base_url support in API providers.

Tests the ability to configure custom base URLs for OpenAI and Anthropic providers,
enabling use of OpenAI-compatible proxies and enterprise gateways.

All tests use REAL implementations (no mocking) per project requirements.
"""
import pytest
import os
from abstractcore import create_llm


class TestOpenAIBaseUrl:
    """Test base_url support for OpenAI provider."""

    def test_base_url_programmatic(self):
        """Test base_url parameter is accepted programmatically."""
        # Test with minimal valid key format (don't call API)
        llm = create_llm(
            "openai",
            model="gpt-4o-mini",
            api_key="sk-test-key-for-base-url-testing",
            base_url="https://custom.example.com/v1"
        )
        assert llm.base_url == "https://custom.example.com/v1"
        assert llm.provider == "openai"

    def test_base_url_environment_variable(self):
        """Test OPENAI_BASE_URL environment variable is respected."""
        # Set environment variable
        os.environ["OPENAI_BASE_URL"] = "https://env.example.com/v1"

        try:
            llm = create_llm(
                "openai",
                model="gpt-4o-mini",
                api_key="sk-test-key"
            )
            assert llm.base_url == "https://env.example.com/v1"
        finally:
            # Clean up
            del os.environ["OPENAI_BASE_URL"]

    def test_base_url_parameter_overrides_environment(self):
        """Test that parameter takes precedence over environment variable."""
        os.environ["OPENAI_BASE_URL"] = "https://env.example.com/v1"

        try:
            llm = create_llm(
                "openai",
                model="gpt-4o-mini",
                api_key="sk-test-key",
                base_url="https://param.example.com/v1"
            )
            assert llm.base_url == "https://param.example.com/v1"
        finally:
            del os.environ["OPENAI_BASE_URL"]

    def test_base_url_none_by_default(self):
        """Test that base_url is None when not specified."""
        try:
            llm = create_llm(
                "openai",
                model="gpt-4o-mini",
                api_key="sk-test-key"
            )
            assert llm.base_url is None
        except Exception as e:
            # Provider validates model exists during init, skip if auth fails
            if "authentication" in str(e).lower() or "401" in str(e):
                pytest.skip("Skipping - test key fails model validation")
            raise


class TestAnthropicBaseUrl:
    """Test base_url support for Anthropic provider."""

    def test_base_url_programmatic(self):
        """Test base_url parameter is accepted programmatically."""
        llm = create_llm(
            "anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key="sk-ant-test-key-for-base-url-testing",
            base_url="https://custom.example.com"
        )
        assert llm.base_url == "https://custom.example.com"
        assert llm.provider == "anthropic"

    def test_base_url_environment_variable(self):
        """Test ANTHROPIC_BASE_URL environment variable is respected."""
        os.environ["ANTHROPIC_BASE_URL"] = "https://env.example.com"

        try:
            llm = create_llm(
                "anthropic",
                model="claude-sonnet-4-5-20250929",
                api_key="sk-ant-test-key"
            )
            assert llm.base_url == "https://env.example.com"
        finally:
            del os.environ["ANTHROPIC_BASE_URL"]

    def test_base_url_parameter_overrides_environment(self):
        """Test that parameter takes precedence over environment variable."""
        os.environ["ANTHROPIC_BASE_URL"] = "https://env.example.com"

        try:
            llm = create_llm(
                "anthropic",
                model="claude-sonnet-4-5-20250929",
                api_key="sk-ant-test-key",
                base_url="https://param.example.com"
            )
            assert llm.base_url == "https://param.example.com"
        finally:
            del os.environ["ANTHROPIC_BASE_URL"]

    def test_base_url_none_by_default(self):
        """Test that base_url is None when not specified."""
        llm = create_llm(
            "anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key="sk-ant-test-key"
        )
        assert llm.base_url is None


class TestBackwardCompatibility:
    """Test that base_url addition doesn't break existing code."""

    def test_openai_without_base_url_still_works(self):
        """Verify OpenAI provider works without base_url (backward compatible)."""
        # Should work exactly as before
        try:
            llm = create_llm(
                "openai",
                model="gpt-4o-mini",
                api_key="sk-test-key"
            )
            assert llm.provider == "openai"
            assert llm.model == "gpt-4o-mini"
            assert hasattr(llm, 'base_url')  # Attribute exists
            assert llm.base_url is None  # But defaults to None
        except Exception as e:
            # Provider validates model exists during init, skip if auth fails
            if "authentication" in str(e).lower() or "401" in str(e):
                pytest.skip("Skipping - test key fails model validation")
            raise

    def test_anthropic_without_base_url_still_works(self):
        """Verify Anthropic provider works without base_url (backward compatible)."""
        llm = create_llm(
            "anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key="sk-ant-test-key"
        )
        assert llm.provider == "anthropic"
        assert llm.model == "claude-sonnet-4-5-20250929"
        assert hasattr(llm, 'base_url')
        assert llm.base_url is None


if __name__ == "__main__":
    # Allow running directly for quick validation
    pytest.main([__file__, "-v", "--tb=short"])
