"""
Test enhanced error messages with actionable guidance.

Tests the format_auth_error() and format_provider_error() helper functions,
as well as their integration with providers.

All tests use REAL implementations (no mocking) per project requirements.
"""
import os
import pytest
from abstractcore import create_llm
from abstractcore.exceptions import (
    format_auth_error,
    format_provider_error,
    format_model_error,
    AuthenticationError
)


class TestErrorMessageHelpers:
    """Test helper functions for formatting error messages."""

    def test_format_auth_error_openai(self):
        """Test format_auth_error() for OpenAI."""
        msg = format_auth_error("openai", "Invalid API key")

        # Verify SOTA format (3-5 lines)
        lines = msg.split('\n')
        assert len(lines) <= 5, "Error message should be ≤ 5 lines (SOTA format)"

        # Verify content
        assert "OPENAI authentication failed" in msg
        assert "Invalid API key" in msg
        assert "abstractcore --set-api-key openai" in msg
        assert "https://platform.openai.com/api-keys" in msg

    def test_format_auth_error_anthropic(self):
        """Test format_auth_error() for Anthropic."""
        msg = format_auth_error("anthropic", "API key not found")

        # Verify SOTA format
        lines = msg.split('\n')
        assert len(lines) <= 5

        # Verify content
        assert "ANTHROPIC authentication failed" in msg
        assert "API key not found" in msg
        assert "abstractcore --set-api-key anthropic" in msg
        assert "https://console.anthropic.com/settings/keys" in msg

    def test_format_auth_error_without_reason(self):
        """Test format_auth_error() without optional reason parameter."""
        msg = format_auth_error("openai")

        assert "OPENAI authentication failed" in msg
        assert "abstractcore --set-api-key openai" in msg
        # Should still be concise without reason
        assert len(msg.split('\n')) <= 5

    def test_format_provider_error_ollama(self):
        """Test format_provider_error() for Ollama."""
        msg = format_provider_error("ollama", "Connection refused")

        # Verify SOTA format
        lines = msg.split('\n')
        assert len(lines) <= 5

        # Verify content
        assert "Provider 'ollama' unavailable" in msg
        assert "Connection refused" in msg
        assert "https://ollama.com/download" in msg
        assert "ollama serve" in msg

    def test_format_provider_error_lmstudio(self):
        """Test format_provider_error() for LMStudio."""
        msg = format_provider_error("lmstudio", "Server not responding")

        # Verify SOTA format
        lines = msg.split('\n')
        assert len(lines) <= 5

        # Verify content
        assert "Provider 'lmstudio' unavailable" in msg
        assert "Server not responding" in msg
        assert "https://lmstudio.ai" in msg
        assert "Enable API in settings" in msg

    def test_format_model_error_still_works(self):
        """Test that existing format_model_error() still works correctly."""
        available = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        msg = format_model_error("OpenAI", "gpt-5-mini", available)

        # Verify content
        assert "gpt-5-mini" in msg
        assert "OpenAI" in msg
        assert "gpt-4o" in msg
        assert "Available models" in msg


class TestProviderIntegration:
    """Test integration of error message helpers with providers."""

    @pytest.mark.slow
    def test_openai_auth_error_format(self):
        """Test OpenAI provider uses format_auth_error() with REAL invalid API key."""
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Requires live OpenAI network call; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run")
        with pytest.raises(AuthenticationError) as exc_info:
            # Use obviously invalid API key to trigger auth error
            llm = create_llm(
                "openai",
                model="gpt-4o-mini",
                api_key="sk-invalid-key-for-testing-12345"
            )
            # Try to generate - should fail with auth error
            llm.generate("test")

        error_msg = str(exc_info.value)

        # Verify new format is used
        assert "abstractcore --set-api-key openai" in error_msg, \
            "OpenAI provider should use format_auth_error()"
        assert "https://platform.openai.com/api-keys" in error_msg, \
            "Error should include API key URL"

        # Verify SOTA format (concise)
        lines = error_msg.split('\n')
        assert len(lines) <= 5, "Error should be ≤ 5 lines (SOTA format)"

    @pytest.mark.slow
    def test_anthropic_auth_error_format(self):
        """Test Anthropic provider uses format_auth_error() with REAL invalid API key."""
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Requires live Anthropic network call; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run")
        with pytest.raises(AuthenticationError) as exc_info:
            # Use obviously invalid API key
            llm = create_llm(
                "anthropic",
                model="claude-haiku-4-5",
                api_key="sk-ant-invalid-key-for-testing-12345"
            )
            # Try to generate
            llm.generate("test")

        error_msg = str(exc_info.value)

        # Verify new format is used
        assert "abstractcore --set-api-key anthropic" in error_msg, \
            "Anthropic provider should use format_auth_error()"
        assert "https://console.anthropic.com/settings/keys" in error_msg, \
            "Error should include API key URL"

        # Verify SOTA format
        lines = error_msg.split('\n')
        assert len(lines) <= 5, "Error should be ≤ 5 lines (SOTA format)"


class TestBackwardCompatibility:
    """Test that error message enhancements don't break existing functionality."""

    def test_exception_classes_unchanged(self):
        """Verify exception class signatures are unchanged (backward compatibility)."""
        # Should be able to raise exceptions with original signatures
        try:
            raise AuthenticationError("Test error")
        except AuthenticationError as e:
            assert str(e) == "Test error"

        # Helper functions are optional - not required
        assert hasattr(AuthenticationError, '__init__')

    def test_all_helpers_exported(self):
        """Verify all helper functions are exported in __all__."""
        from abstractcore.exceptions import __all__

        assert 'format_auth_error' in __all__
        assert 'format_provider_error' in __all__
        assert 'format_model_error' in __all__


if __name__ == "__main__":
    # Allow running directly for quick validation
    pytest.main([__file__, "-v", "--tb=short"])
