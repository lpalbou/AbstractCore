"""
Async test fixtures and configuration.
"""

import asyncio
import os
from typing import Any

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Provider availability helpers
def _ollama_has_model(model: str) -> tuple[bool, str]:
    """Return (ok, reason) for whether a given Ollama model exists locally.

    We intentionally use the public Ollama tags endpoint so tests can skip cleanly
    when the required model isn't installed on the machine running the suite.
    """
    model = str(model or "").strip()
    if not model:
        return False, "No model specified"

    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        if resp.status_code != 200:
            return False, f"Ollama tags endpoint returned {resp.status_code}"
        data: dict[str, Any] = resp.json() if isinstance(resp.json(), dict) else {}
        models = data.get("models", [])
        if not isinstance(models, list):
            models = []
        names = {str(m.get("name") or "").strip() for m in models if isinstance(m, dict)}
        return (model in names), ("" if model in names else f"Model '{model}' not installed")
    except Exception as e:
        return False, f"Could not query Ollama models: {e}"


def is_provider_available(provider: str, model: str | None = None) -> tuple[bool, str]:
    """Check if a provider is available."""
    if provider == "ollama":
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            return (
                False,
                "Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)",
            )
        try:
            import httpx

            response = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
            if response.status_code != 200:
                return False, f"Ollama returned {response.status_code}"

            if model:
                ok, reason = _ollama_has_model(model)
                if not ok:
                    return False, reason

            return True, ""
        except Exception:
            return False, "Ollama not running"

    elif provider == "lmstudio":
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            return (
                False,
                "Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)",
            )
        try:
            import httpx

            response = httpx.get("http://localhost:1234/v1/models", timeout=1.0)
            if response.status_code != 200:
                return False, f"LMStudio returned {response.status_code}"

            if model:
                model = str(model or "").strip()
                try:
                    data = response.json()
                except Exception:
                    data = {}
                items = data.get("data") if isinstance(data, dict) else None
                if not isinstance(items, list):
                    items = []
                model_ids = {str(m.get("id") or "").strip() for m in items if isinstance(m, dict)}
                if model not in model_ids:
                    return False, f"Model '{model}' not present in LMStudio (/v1/models)"

            return True, ""
        except Exception:
            return False, "LMStudio not running"

    elif provider == "mlx":
        try:
            import mlx_lm  # noqa: F401

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

    def _skip(provider: str, model: str | None = None):
        available, reason = is_provider_available(provider, model)
        if not available:
            pytest.skip(reason)

    return _skip
