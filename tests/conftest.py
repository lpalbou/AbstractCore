"""
PyTest configuration for AbstractCore tests
Provides fixtures and utilities for vision testing
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractcore import create_llm
from abstractcore.media.capabilities import is_vision_model


@pytest.fixture(scope="session")
def vision_examples_dir():
    """Path to vision examples directory."""
    return Path(__file__).parent / "vision_examples"


@pytest.fixture(scope="session")
def vision_test_images(vision_examples_dir):
    """List of all available vision test images."""
    if not vision_examples_dir.exists():
        pytest.skip("Vision examples directory not found")

    images = list(vision_examples_dir.glob("*.jpg"))
    if not images:
        pytest.skip("No test images found in vision_examples")

    return [str(img) for img in images]


@pytest.fixture(scope="session")
def vision_reference_files(vision_examples_dir):
    """Dictionary mapping image names to reference JSON files."""
    if not vision_examples_dir.exists():
        pytest.skip("Vision examples directory not found")

    references = {}
    for json_file in vision_examples_dir.glob("*.json"):
        image_name = json_file.name.replace('.json', '.jpg')
        references[image_name] = str(json_file)

    return references


def check_provider_availability(provider: str, model: str = None) -> tuple[bool, str]:
    """
    Check if a provider and optionally a specific model is available.
    Returns (is_available, skip_reason)
    """
    try:
        if provider == "ollama":
            if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
                return False, "Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)"
            # Check if Ollama is running
            llm = create_llm(provider, model=model or "qwen2.5vl:7b", timeout=5.0)
            # Try to get model info to verify it's actually available
            if model and hasattr(llm, '_client'):
                # This will fail if model isn't installed
                pass
            return True, ""

        elif provider == "lmstudio":
            if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
                return False, "Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)"
            # Check if LMStudio is running
            llm = create_llm(provider, model=model or "qwen/qwen2.5-vl-7b", timeout=5.0)
            return True, ""

        elif provider == "openai":
            if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
                return False, "Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)"
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                return False, "OPENAI_API_KEY not set"
            llm = create_llm(provider, model=model or "gpt-4o", timeout=5.0)
            return True, ""

        elif provider == "anthropic":
            if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
                return False, "Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)"
            # Check for API key
            if not os.getenv("ANTHROPIC_API_KEY"):
                return False, "ANTHROPIC_API_KEY not set"
            llm = create_llm(provider, model=model or "claude-haiku-4-5", timeout=5.0)
            return True, ""

        elif provider == "huggingface":
            # HuggingFace model loading can be heavy and may require large downloads / native backends.
            # Default to "unavailable" unless explicitly enabled.
            if os.getenv("ABSTRACTCORE_RUN_HUGGINGFACE_TESTS") != "1" and os.getenv("ABSTRACTCORE_RUN_GGUF_TESTS") != "1":
                return False, "HuggingFace tests disabled (set ABSTRACTCORE_RUN_HUGGINGFACE_TESTS=1)"

            # Best-effort lightweight check: if a specific model is requested, only claim availability
            # when it appears to be cached locally.
            if model:
                model_dir = (Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model.replace('/', '--')}")
                if not model_dir.exists():
                    return False, f"HuggingFace model not found in cache: {model}"
            return True, ""

        elif provider == "mlx":
            if os.getenv("ABSTRACTCORE_RUN_MLX_TESTS") != "1":
                return False, "MLX tests disabled (set ABSTRACTCORE_RUN_MLX_TESTS=1)"
            # Avoid loading a model here; tests that opt-in can create the provider directly.
            return True, ""

        else:
            return False, f"Unknown provider: {provider}"

    except ImportError as e:
        return False, f"Missing dependency for {provider}: {e}"
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["connection", "refused", "timeout", "operation not permitted", "not found", "not running"]):
            return False, f"{provider} not available: {e}"
        else:
            # Re-raise unexpected errors
            raise


def check_vision_capability(provider: str, model: str) -> tuple[bool, str]:
    """
    Check if a model actually supports vision.
    Returns (supports_vision, skip_reason)
    """
    try:
        if is_vision_model(model):
            return True, ""
        else:
            return False, f"Model {model} does not support vision"
    except Exception as e:
        return False, f"Could not determine vision capability: {e}"


@pytest.fixture(scope="session")
def available_vision_providers():
    """Dictionary of available vision providers and their models."""
    provider_models = {
        "ollama": [
            "qwen2.5vl:7b",
            "llama3.2-vision:11b",
            "gemma3:4b"
        ],
        "lmstudio": [
            "qwen/qwen2.5-vl-7b",
            "qwen/qwen3-vl-4b",
            "google/gemma-3n-e4b"
        ],
        "openai": [
            "gpt-4o",
            "gpt-4-turbo"
        ],
        "anthropic": [
            "claude-haiku-4-5",
        ],
        "huggingface": [
            "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
        ]
    }

    available = {}

    for provider, models in provider_models.items():
        available_models = []

        for model in models:
            # Check provider availability first
            provider_available, provider_reason = check_provider_availability(provider, model)
            if not provider_available:
                continue

            # Check vision capability
            vision_available, vision_reason = check_vision_capability(provider, model)
            if vision_available:
                available_models.append(model)

        if available_models:
            available[provider] = available_models

    return available


@pytest.fixture
def skip_if_provider_unavailable():
    """Decorator function to skip tests if provider is unavailable."""
    def skipper(provider: str, model: str = None):
        available, reason = check_provider_availability(provider, model)
        if not available:
            pytest.skip(f"Provider {provider} not available: {reason}")

        if model:
            vision_ok, vision_reason = check_vision_capability(provider, model)
            if not vision_ok:
                pytest.skip(f"Vision not supported: {vision_reason}")

    return skipper


@pytest.fixture
def create_vision_llm():
    """Factory function to create vision LLMs with proper error handling."""
    def factory(provider: str, model: str):
        try:
            llm = create_llm(provider, model=model)

            # Verify vision capability
            if not is_vision_model(model):
                pytest.skip(f"Model {model} does not support vision")

            return llm
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "refused", "timeout", "not found"]):
                pytest.skip(f"Provider {provider} not available: {e}")
            else:
                raise

    return factory


# Test markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "vision: mark test as requiring vision capabilities"
    )
    config.addinivalue_line(
        "markers", "provider_required: mark test as requiring specific provider"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "comprehensive: mark test as comprehensive vision test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add vision marker to tests in vision-related files
        if "vision" in str(item.fspath).lower():
            item.add_marker(pytest.mark.vision)

        # Add slow marker to comprehensive tests
        if "comprehensive" in item.name.lower():
            item.add_marker(pytest.mark.slow)
