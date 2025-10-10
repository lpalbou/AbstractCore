"""
Factory for creating LLM providers.
"""

from typing import Optional
from .interface import AbstractLLMInterface
from ..exceptions import ModelNotFoundError, AuthenticationError, ProviderAPIError


def create_llm(provider: str, model: Optional[str] = None, **kwargs) -> AbstractLLMInterface:
    """
    Create an LLM provider instance with unified token parameter support.

    Args:
        provider: Provider name (openai, anthropic, ollama, huggingface, mlx, lmstudio, mock)
        model: Model name (optional, will use provider default)
        **kwargs: Additional configuration including token parameters

    Token Parameters (AbstractLLM Unified Standard):
        max_tokens: Total context window budget (input + output combined)
        max_output_tokens: Maximum tokens reserved for generation (default: 2048)
        max_input_tokens: Maximum tokens for input (auto-calculated if not specified)

    Examples:
        # Strategy 1: Budget + Output Reserve (Recommended)
        llm = create_llm(
            provider="openai",
            model="gpt-4o",
            max_tokens=8000,        # Total budget
            max_output_tokens=2000  # Reserve for output
        )

        # Strategy 2: Explicit Input + Output (Advanced)
        llm = create_llm(
            provider="anthropic",
            model="claude-3.5-sonnet",
            max_input_tokens=6000,   # Explicit input limit
            max_output_tokens=2000   # Explicit output limit
        )

        # Quick setup with defaults
        llm = create_llm("ollama", "qwen3-coder:30b")

        # Get configuration help
        print(llm.get_token_configuration_summary())
        warnings = llm.validate_token_constraints()

    Returns:
        Configured LLM provider instance with unified token management

    Raises:
        ImportError: If provider dependencies are not installed
        ValueError: If provider is not supported
        ModelNotFoundError: If specified model is not available
        AuthenticationError: If API credentials are invalid
    """

    # Auto-detect provider from model name if needed
    if model:
        # MLX models should use MLX provider
        if "mlx-community" in model.lower() and provider.lower() == "huggingface":
            provider = "mlx"
        # GGUF models should use HuggingFace GGUF backend
        elif (".gguf" in model.lower() or "-gguf" in model.lower()) and provider.lower() == "mlx":
            provider = "huggingface"

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