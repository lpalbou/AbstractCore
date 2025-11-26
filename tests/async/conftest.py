"""
Async test fixtures and configuration.
"""
import pytest
import asyncio
import os


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Provider availability helpers
def is_provider_available(provider: str, model: str = None) -> tuple[bool, str]:
    """Check if a provider is available."""
    if provider == "ollama":
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
            return response.status_code == 200, ""
        except Exception:
            return False, "Ollama not running"

    elif provider == "lmstudio":
        try:
            import httpx
            response = httpx.get("http://localhost:1234/v1/models", timeout=1.0)
            return response.status_code == 200, ""
        except Exception:
            return False, "LMStudio not running"

    elif provider == "mlx":
        try:
            import mlx_lm
            return True, ""
        except ImportError:
            return False, "MLX not installed"

    elif provider == "huggingface":
        return True, ""  # HuggingFace always available

    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY not set"
        return True, ""

    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY not set"
        return True, ""

    return False, f"Unknown provider: {provider}"


@pytest.fixture
def skip_if_provider_unavailable():
    """Fixture to skip tests if provider is unavailable."""
    def _skip(provider: str, model: str = None):
        available, reason = is_provider_available(provider, model)
        if not available:
            pytest.skip(reason)
    return _skip
