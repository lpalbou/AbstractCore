"""
Simple model discovery - minimalist approach for graceful fallback.
"""

import httpx
from typing import List, Optional
from pathlib import Path
import os


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


def get_anthropic_models(api_key: Optional[str]) -> List[str]:
    """Get current Anthropic models."""
    # Updated list for 2024
    return [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]


def get_openai_models(api_key: Optional[str]) -> List[str]:
    """Get available OpenAI models via API."""
    if not api_key:
        # Return current models if no API key
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4",
            "gpt-3.5-turbo"
        ]

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = httpx.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=5.0
        )

        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            # Filter to chat models only
            chat_models = [m for m in models if any(x in m for x in ["gpt-3.5", "gpt-4", "gpt-o1"])]
            return sorted(chat_models)
    except:
        pass

    # Fallback to current models
    return [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo"
    ]


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
        for model in available_models[:10]:  # Show max 10
            message += f"  • {model}\n"
        if len(available_models) > 10:
            message += f"  ... and {len(available_models) - 10} more\n"
    else:
        message += f"\n⚠️  Could not fetch available models for {provider}.\n"

    return message.rstrip()