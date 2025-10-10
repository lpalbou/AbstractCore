"""
Core interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator
from .types import GenerateResponse, Message


class AbstractLLMInterface(ABC):
    """
    Abstract base class for all LLM providers.

    AbstractLLM Token Parameter Vocabulary (Unified Standard):
    =========================================================

    • max_tokens: Total context window budget (input + output combined) - YOUR BUDGET
    • max_output_tokens: Maximum tokens reserved for generation - RESERVE FOR OUTPUT
    • max_input_tokens: Maximum tokens allowed for input/prompt - AUTO-CALCULATED OR EXPLICIT

    Key Constraint: max_input_tokens + max_output_tokens ≤ max_tokens

    Configuration Strategies:
    ========================

    Strategy 1: Budget + Output Reserve (Recommended)
    -------------------------------------------------
    Specify total budget and how much to reserve for output:

        llm = create_llm(
            provider="openai",
            model="gpt-4o",
            max_tokens=8000,           # Total budget
            max_output_tokens=2000     # Reserve for output
            # max_input_tokens=6000    # Auto-calculated
        )

    Strategy 2: Explicit Input + Output (Advanced)
    ----------------------------------------------
    Explicitly specify input and output limits:

        llm = create_llm(
            provider="anthropic",
            model="claude-3.5-sonnet",
            max_input_tokens=6000,     # Explicit input limit
            max_output_tokens=2000     # Explicit output limit
            # max_tokens=8000          # Auto-calculated
        )

    Helper Methods:
    ==============

    • llm.get_token_configuration_summary() - View current setup
    • llm.validate_token_constraints() - Get warnings/suggestions
    • llm.calculate_token_budget(text, desired_output) - Estimate needed max_tokens
    • llm.estimate_tokens(text) - Rough token count estimation

    Provider Abstraction:
    ===================
    AbstractLLM handles provider-specific parameter mapping internally:
    • OpenAI: max_tokens → max_completion_tokens (o1 models) or max_tokens (others)
    • Anthropic: max_output_tokens → max_tokens (output-focused API)
    • Google: max_output_tokens → max_output_tokens (direct mapping)
    • HuggingFace: max_tokens → n_ctx (context), max_output_tokens → max_tokens (output)

    This abstraction ensures your code works across all providers with consistent naming.
    """

    def __init__(self, model: str,
                 max_tokens: Optional[int] = None,
                 max_input_tokens: Optional[int] = None,
                 max_output_tokens: int = 2048,
                 debug: bool = False,
                 **kwargs):
        self.model = model
        self.config = kwargs

        # Unified token parameters
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.debug = debug

        # Validate token parameters
        self._validate_token_parameters()

    @abstractmethod
    def generate(self,
                prompt: str,
                messages: Optional[List[Dict[str, str]]] = None,
                system_prompt: Optional[str] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                stream: bool = False,
                **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Generate response from the LLM.

        Args:
            prompt: The input prompt
            messages: Optional conversation history
            system_prompt: Optional system prompt
            tools: Optional list of available tools
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            GenerateResponse or iterator of GenerateResponse for streaming
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        pass

    def validate_config(self) -> bool:
        """Validate provider configuration"""
        return True


    def _validate_token_parameters(self):
        """Validate token parameter constraints"""
        if self.max_tokens is not None:
            if self.max_input_tokens is not None and self.max_output_tokens is not None:
                if self.max_input_tokens + self.max_output_tokens > self.max_tokens:
                    raise ValueError(
                        f"Token constraint violation: max_input_tokens ({self.max_input_tokens}) + "
                        f"max_output_tokens ({self.max_output_tokens}) = "
                        f"{self.max_input_tokens + self.max_output_tokens} > "
                        f"max_tokens ({self.max_tokens})"
                    )

    def _calculate_effective_token_limits(self) -> tuple[Optional[int], int, Optional[int]]:
        """
        Calculate effective token limits based on provided parameters.

        Returns:
            Tuple of (max_tokens, max_output_tokens, max_input_tokens)
        """
        effective_max_tokens = self.max_tokens
        effective_max_output_tokens = self.max_output_tokens
        effective_max_input_tokens = self.max_input_tokens

        # If max_tokens is set but max_input_tokens is not, calculate it
        if effective_max_tokens and effective_max_input_tokens is None:
            effective_max_input_tokens = effective_max_tokens - effective_max_output_tokens

        return effective_max_tokens, effective_max_output_tokens, effective_max_input_tokens

    def validate_token_usage(self, input_tokens: int, requested_output_tokens: int) -> bool:
        """
        Validate if the requested token usage is within limits.

        Args:
            input_tokens: Number of tokens in the input
            requested_output_tokens: Number of tokens requested for output

        Returns:
            True if within limits, False otherwise

        Raises:
            ValueError: If token limits would be exceeded
        """
        _, _, max_in = self._calculate_effective_token_limits()

        # Check input token limit
        if max_in is not None and input_tokens > max_in:
            raise ValueError(
                f"Input tokens ({input_tokens}) exceed max_input_tokens ({max_in})"
            )

        # Check total token limit
        if self.max_tokens is not None:
            total_tokens = input_tokens + requested_output_tokens
            if total_tokens > self.max_tokens:
                raise ValueError(
                    f"Total tokens ({total_tokens}) would exceed max_tokens ({self.max_tokens}). "
                    f"Input: {input_tokens}, Requested output: {requested_output_tokens}"
                )

        return True

    def calculate_token_budget(self, input_text: str, desired_output_tokens: int,
                              safety_margin: float = 0.1) -> tuple[int, List[str]]:
        """
        Helper to estimate required max_tokens given input and desired output.

        Args:
            input_text: The input text to estimate tokens for
            desired_output_tokens: Desired maximum output tokens
            safety_margin: Safety margin as percentage (0.1 = 10% buffer)

        Returns:
            Tuple of (recommended_max_tokens, warnings)
        """
        warnings = []

        # Rough token estimation (approximately 4 characters per token for English)
        estimated_input_tokens = len(input_text) // 4

        # Add safety margin
        input_with_margin = int(estimated_input_tokens * (1 + safety_margin))
        recommended_max_tokens = input_with_margin + desired_output_tokens

        # Check against current configuration
        if self.max_tokens is not None and recommended_max_tokens > self.max_tokens:
            warnings.append(
                f"Recommended max_tokens ({recommended_max_tokens}) exceeds current "
                f"max_tokens ({self.max_tokens}). Consider increasing max_tokens."
            )

        # Check if input would exceed calculated max_input_tokens
        _, _, max_input = self._calculate_effective_token_limits()
        if max_input is not None and input_with_margin > max_input:
            warnings.append(
                f"Estimated input tokens ({input_with_margin}) may exceed "
                f"max_input_tokens ({max_input}). Consider increasing max_tokens "
                f"or reducing input size."
            )

        # Warn if desired output is very large relative to typical context windows
        if desired_output_tokens > 8192:
            warnings.append(
                f"Desired output tokens ({desired_output_tokens}) is quite large. "
                f"Some models may not support such long outputs."
            )

        return recommended_max_tokens, warnings

    def validate_token_constraints(self) -> List[str]:
        """
        Validate token configuration and return warnings/suggestions.

        Returns:
            List of warning/suggestion messages
        """
        warnings = []
        max_tokens, max_output, max_input = self._calculate_effective_token_limits()

        # Check if max_tokens is not set
        if max_tokens is None:
            warnings.append(
                "max_tokens is not set. This may cause issues with long inputs. "
                "Consider setting an appropriate context window limit."
            )

        # Check if max_output_tokens is very small
        if max_output < 100:
            warnings.append(
                f"max_output_tokens ({max_output}) is very small. "
                "This may truncate responses unexpectedly."
            )

        # Check if max_output_tokens is very large relative to max_tokens
        if max_tokens is not None and max_output > max_tokens * 0.8:
            warnings.append(
                f"max_output_tokens ({max_output}) uses {max_output/max_tokens*100:.1f}% "
                f"of total context window. This leaves little room for input. "
                f"Consider adjusting the ratio."
            )

        # Check for common inefficient configurations
        if max_tokens is not None and max_input is not None:
            if max_input < 1000:
                warnings.append(
                    f"max_input_tokens ({max_input}) is quite small. "
                    "This may limit the usefulness of the model for complex prompts."
                )

            # Check if the split is reasonable
            input_ratio = max_input / max_tokens
            if input_ratio < 0.2:
                warnings.append(
                    f"Input tokens use only {input_ratio*100:.1f}% of context window. "
                    "Consider reducing max_output_tokens to allow more input."
                )

        # Provider-specific suggestions
        model_lower = self.model.lower()

        if "gpt-4" in model_lower and max_tokens is not None and max_tokens > 128000:
            warnings.append(
                "GPT-4 models support up to 128k tokens. Your max_tokens setting "
                "may exceed model capabilities."
            )
        elif "claude" in model_lower and max_tokens is not None and max_tokens > 200000:
            warnings.append(
                "Claude models typically support up to 200k tokens. Your max_tokens "
                "setting may exceed model capabilities."
            )
        elif "gemini" in model_lower and max_tokens is not None and max_tokens > 1000000:
            warnings.append(
                "Gemini models support up to 1M tokens but may have output limits. "
                "Check your max_output_tokens setting."
            )

        return warnings

    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count for given text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count

        Note:
            This is a rough approximation (~4 chars per token for English).
            For precise counts, use provider-specific tokenizers.
        """
        if not text:
            return 0

        # Basic estimation: ~4 characters per token for English
        # This varies by language and tokenizer, but gives a reasonable baseline
        base_estimate = len(text) // 4

        # Adjust for whitespace and punctuation density
        # More spaces/punctuation = fewer tokens per character
        whitespace_ratio = len([c for c in text if c.isspace()]) / len(text) if text else 0
        punct_ratio = len([c for c in text if not c.isalnum() and not c.isspace()]) / len(text) if text else 0

        # Adjust estimate based on content type
        adjustment_factor = 1.0
        if whitespace_ratio > 0.2:  # Very spaced out text
            adjustment_factor = 0.8
        elif punct_ratio > 0.1:  # Heavy punctuation
            adjustment_factor = 0.9

        return max(1, int(base_estimate * adjustment_factor))

    def get_token_configuration_summary(self) -> str:
        """
        Get a human-readable summary of current token configuration.

        Returns:
            Formatted string describing the token setup
        """
        max_tokens, max_output, max_input = self._calculate_effective_token_limits()

        lines = [
            f"Token Configuration for {self.model}:",
            f"  • Total context window (max_tokens): {max_tokens or 'Not set'}",
            f"  • Maximum output tokens: {max_output}",
            f"  • Maximum input tokens: {max_input or 'Auto-calculated'}",
        ]

        if max_tokens and max_input:
            output_pct = (max_output / max_tokens) * 100
            input_pct = (max_input / max_tokens) * 100
            lines.extend([
                f"  • Output allocation: {output_pct:.1f}% of context",
                f"  • Input allocation: {input_pct:.1f}% of context"
            ])

        # Add warnings if any
        warnings = self.validate_token_constraints()
        if warnings:
            lines.append("\n⚠️  Warnings:")
            for warning in warnings:
                lines.append(f"  • {warning}")

        return "\n".join(lines)