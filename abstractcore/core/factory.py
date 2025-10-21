"""
Factory for creating LLM providers.
"""

from typing import Optional
from .interface import AbstractCoreInterface
from ..exceptions import ModelNotFoundError, AuthenticationError, ProviderAPIError


def create_llm(provider: str, model: Optional[str] = None, **kwargs) -> AbstractCoreInterface:
    """
    Create an LLM provider instance with unified token parameter support.

    Args:
        provider: Provider name (openai, anthropic, ollama, huggingface, mlx, lmstudio)
        model: Model name (optional, will use provider default)
        **kwargs: Additional configuration including token parameters

    Token Parameters (AbstractCore Unified Standard):
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

    # Use centralized provider registry for all provider creation
    try:
        from ..providers.registry import create_provider
        return create_provider(provider, model, **kwargs)
    except (ModelNotFoundError, AuthenticationError, ProviderAPIError) as e:
        # Re-raise provider exceptions cleanly
        raise e