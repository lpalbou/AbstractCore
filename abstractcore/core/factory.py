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

    # Auto-detect provider from model name if needed.
    #
    # ADR 0009: this rewrite is a re-routing, not a substitution — the model the
    # caller named is the model that loads. But it changes the ENGINE (mlx_lm vs
    # llama.cpp vs transformers), and with it the speed, memory profile and cache
    # mechanism, while `provider=` is the field callers and harnesses record as the
    # lane label. It happens before HuggingFaceProvider.__init__, so the artifact
    # guard installed there cannot see it. Silent was the wrong default: say it.
    if model:
        rerouted_from = None
        # MLX models should use MLX provider
        if "mlx-community" in model.lower() and provider.lower() == "huggingface":
            rerouted_from, provider = provider, "mlx"
        # GGUF models should use HuggingFace GGUF backend
        elif (".gguf" in model.lower() or "-gguf" in model.lower()) and provider.lower() == "mlx":
            rerouted_from, provider = provider, "huggingface"

        if rerouted_from is not None:
            import logging
            import warnings

            detail = (
                f"#FALLBACK provider re-routed: you asked for provider={rerouted_from!r}, "
                f"but the handle {model!r} names a "
                f"{'MLX' if provider == 'mlx' else 'GGUF'} artifact, so it will run on "
                f"provider={provider!r}. The model is unchanged; the execution engine "
                f"is not — speed, memory profile and the prompt-cache mechanism all "
                f"differ. Pass provider={provider!r} explicitly to silence this."
            )
            # `logger.warning` ALONE DOES NOT REACH ANYONE. Importing abstractcore
            # configures the root logger to ERROR with a single ERROR-level handler,
            # and every `abstractcore.*` logger is NOTSET, so its effective level is
            # ERROR. The first version of this announcement was logger-only and was
            # therefore discarded in every default process — the re-route stayed
            # exactly as silent as before the "fix". `warnings.warn` is on by default
            # and is what actually informs the caller; the logger line is kept for
            # structured capture by hosts that raise the level.
            logging.getLogger(__name__).warning(detail)
            warnings.warn(detail, RuntimeWarning, stacklevel=2)

    # Use centralized provider registry for all provider creation
    try:
        from ..providers.registry import create_provider
        return create_provider(provider, model, **kwargs)
    except (ModelNotFoundError, AuthenticationError, ProviderAPIError) as e:
        # Re-raise provider exceptions cleanly
        raise e