"""
Simple model discovery - minimalist approach for graceful fallback.
"""

import httpx
from typing import List
from pathlib import Path


def get_available_models(provider: str, **kwargs) -> List[str]:
    """Get available models for a provider using API calls."""

    if provider.lower() == "anthropic":
        return get_anthropic_models(kwargs.get("api_key"))
    elif provider.lower() == "openai":
        return get_openai_models(kwargs.get("api_key"))
    elif provider.lower() == "ollama":
        return get_ollama_models(kwargs.get("base_url", "http://localhost:11434"))
    elif provider.lower() == "lmstudio":
        return get_lmstudio_models(kwargs.get("base_url", "http://localhost:1234"))
    elif provider.lower() == "mlx":
        return get_mlx_local_models()
    elif provider.lower() == "huggingface":
        return get_hf_local_models()
    else:
        return []


def get_anthropic_models(api_key: str) -> List[str]:
    """Get available Anthropic models via API."""
    try:
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        response = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers=headers,
            timeout=10.0
        )

        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return sorted(models)
        else:
            # Fallback if API fails
            return []
    except Exception:
        return []


def get_openai_models(api_key: str) -> List[str]:
    """Get available OpenAI models via API."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = httpx.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=10.0
        )

        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            # Filter to chat models only
            chat_models = [m for m in models if any(x in m for x in ["gpt-3.5", "gpt-4", "gpt-o1"])]
            return sorted(chat_models)
        else:
            return []
    except Exception:
        return []


def get_ollama_models(base_url: str) -> List[str]:
    """Get available Ollama models."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return sorted(models)
    except:
        pass
    return []


def get_lmstudio_models(base_url: str) -> List[str]:
    """Get available LMStudio models."""
    try:
        response = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return sorted(models)
    except:
        pass
    return []


def get_mlx_local_models() -> List[str]:
    """Get locally cached MLX models."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists():
        return []

    models = []
    for item in hf_cache.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            # Convert models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit to mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
            model_name = item.name.replace("models--", "").replace("--", "/")
            if "mlx" in model_name.lower():
                models.append(model_name)

    return sorted(models)


def get_hf_local_models() -> List[str]:
    """Get locally cached HuggingFace models."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists():
        return []

    models = []
    for item in hf_cache.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            # Convert models--microsoft--DialoGPT-medium to microsoft/DialoGPT-medium
            model_name = item.name.replace("models--", "").replace("--", "/")
            models.append(model_name)

    return sorted(models)


def format_model_error(provider: str, invalid_model: str, available_models: List[str]) -> str:
    """Simple error message with available models."""

    message = f"❌ Model '{invalid_model}' not found for {provider} provider.\n"

    if available_models:
        message += f"\n✅ Available models ({len(available_models)}):\n"
        for model in available_models[:30]:  # Show max 10
            message += f"  • {model}\n"
        if len(available_models) > 30:
            message += f"  ... and {len(available_models) - 30} more\n"
    else:
        # Show provider documentation when we can't fetch models
        doc_links = {
            "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models",
            "openai": "https://platform.openai.com/docs/models",
            "ollama": "https://ollama.com/library",
            "huggingface": "https://huggingface.co/models",
            "mlx": "https://huggingface.co/mlx-community"
        }

        provider_lower = provider.lower()
        if provider_lower in doc_links:
            message += f"\n📚 See available models: {doc_links[provider_lower]}\n"
        else:
            message += f"\n⚠️  Could not fetch available models for {provider}.\n"

    return message.rstrip()