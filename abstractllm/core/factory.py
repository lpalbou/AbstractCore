"""
Factory for creating LLM providers.
"""

from typing import Optional
from .interface import AbstractLLMInterface
from ..exceptions import ModelNotFoundError, AuthenticationError, ProviderAPIError


def create_llm(provider: str, model: Optional[str] = None, **kwargs) -> AbstractLLMInterface:
    """
    Create an LLM provider instance.

    Args:
        provider: Provider name (openai, anthropic, ollama, mock, etc.)
        model: Model name (optional, will use provider default)
        **kwargs: Additional configuration

    Returns:
        Configured LLM provider instance
    """

    # Mock provider for testing
    if provider.lower() == "mock":
        from ..providers.mock_provider import MockProvider
        return MockProvider(model=model or "mock-model", **kwargs)

    # Import providers dynamically to avoid hard dependencies
    elif provider.lower() == "openai":
        try:
            from ..providers.openai_provider import OpenAIProvider
            return OpenAIProvider(model=model or "gpt-5-nano-2025-08-07", **kwargs)
        except ImportError:
            raise ImportError("OpenAI dependencies not installed. Install with: pip install abstractllm[openai]")
        except (ModelNotFoundError, AuthenticationError, ProviderAPIError) as e:
            # Re-raise provider exceptions cleanly
            raise e

    elif provider.lower() == "anthropic":
        try:
            from ..providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(model=model or "claude-3-5-haiku-latest", **kwargs)
        except ImportError:
            raise ImportError("Anthropic dependencies not installed. Install with: pip install abstractllm[anthropic]")
        except (ModelNotFoundError, AuthenticationError, ProviderAPIError) as e:
            # Re-raise provider exceptions cleanly
            raise e

    elif provider.lower() == "ollama":
        try:
            from ..providers.ollama_provider import OllamaProvider
            return OllamaProvider(model=model or "qwen3-coder:30b", **kwargs)
        except ImportError:
            raise ImportError("Ollama dependencies not installed. Install with: pip install abstractllm[ollama]")

    elif provider.lower() == "huggingface":
        try:
            from ..providers.huggingface_provider import HuggingFaceProvider
            return HuggingFaceProvider(model=model or "Qwen/Qwen3-4B/", **kwargs)
        except ImportError:
            raise ImportError("HuggingFace dependencies not installed. Install with: pip install abstractllm[huggingface]")

    elif provider.lower() == "mlx":
        try:
            from ..providers.mlx_provider import MLXProvider
            return MLXProvider(model=model or "mlx-community/Qwen3-4B", **kwargs)
        except ImportError:
            raise ImportError("MLX dependencies not installed. Install with: pip install abstractllm[mlx]")

    elif provider.lower() == "lmstudio":
        try:
            from ..providers.lmstudio_provider import LMStudioProvider
            return LMStudioProvider(model=model or "qwen/qwen3-4b-2507", **kwargs)
        except ImportError:
            raise ImportError("LM Studio provider not available")

    else:
        raise ValueError(f"Unknown provider: {provider}. Available providers: openai, anthropic, ollama, huggingface, mlx, lmstudio, mock")