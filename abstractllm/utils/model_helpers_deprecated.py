"""
Utilities for model discovery and validation.
"""

import httpx
from typing import List, Dict, Any, Optional
import json


class ModelDiscovery:
    """Helper class to discover available models for different providers"""

    @staticmethod
    def get_anthropic_models(api_key: str) -> List[str]:
        """
        Get available Anthropic models.
        Note: Anthropic doesn't have a public models endpoint like OpenAI.
        We'll try a workaround or return empty list to force documentation lookup.
        """
        try:
            # Anthropic doesn't provide a models endpoint
            # We could try to make test calls to discover valid models,
            # but this would be expensive and unreliable
            # Better to return empty and direct user to documentation
            return []
        except Exception:
            return []

    @staticmethod
    def get_openai_models(api_key: str) -> List[str]:
        """Get available OpenAI models via API"""
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
                # Filter to common chat models
                chat_models = [m for m in models if any(x in m for x in ["gpt-3.5", "gpt-4", "gpt-o1"])]
                return sorted(chat_models)
            else:
                # Return known models if API fails
                return [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-4",
                    "gpt-3.5-turbo"
                ]
        except Exception:
            # Fallback to known models
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ]

    @staticmethod
    def get_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
        """Get available Ollama models"""
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return sorted(models)
            else:
                return []
        except Exception:
            return []

    @staticmethod
    def get_huggingface_models(query: str = "") -> List[str]:
        """Get suggested HuggingFace models"""
        # Return popular models that work well
        popular_models = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "bigscience/bloom-560m",
            "bigscience/bloom-1b7",
            "Salesforce/codegen-350M-mono",
            "Salesforce/codegen-2B-mono",
        ]
        return popular_models

    @staticmethod
    def create_model_error_message(provider: str, invalid_model: str,
                                 available_models: Optional[List[str]] = None) -> str:
        """
        Create a helpful error message for invalid model names.

        Args:
            provider: Provider name
            invalid_model: The invalid model name that was requested
            available_models: List of available models (if known)

        Returns:
            Formatted error message with helpful information
        """

        base_msg = f"Model '{invalid_model}' not found for {provider} provider."

        # Provider-specific documentation links
        doc_links = {
            "openai": "https://platform.openai.com/docs/models",
            "anthropic": "https://docs.claude.com/en/docs/about-claude/models/overview",
            "ollama": "https://ollama.com/library",
            "huggingface": "https://huggingface.co/models",
            "mlx": "https://huggingface.co/mlx-community"
        }

        provider_lower = provider.lower()

        # Build helpful message
        message_parts = [base_msg, ""]

        # Add available models if we have them
        if available_models and len(available_models) > 0:
            if len(available_models) <= 10:
                message_parts.append("Available models:")
                for model in available_models:
                    message_parts.append(f"  • {model}")
            else:
                message_parts.append(f"Found {len(available_models)} available models:")
                for model in available_models[:8]:
                    message_parts.append(f"  • {model}")
                message_parts.append(f"  ... and {len(available_models) - 8} more")
            message_parts.append("")

        # Add documentation link
        if provider_lower in doc_links:
            message_parts.append(f"📚 For complete model list, see: {doc_links[provider_lower]}")

        # Add provider-specific tips
        if provider_lower == "anthropic":
            message_parts.append("💡 Tip: Anthropic model names include dates (e.g., claude-3-haiku-20240307)")
        elif provider_lower == "openai":
            message_parts.append("💡 Tip: Use 'gpt-4o' or 'gpt-3.5-turbo' for latest models")
        elif provider_lower == "ollama":
            message_parts.append("💡 Tip: Pull models first with 'ollama pull <model>'")

        return "\n".join(message_parts)