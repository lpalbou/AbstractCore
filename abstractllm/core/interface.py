"""
Core interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator
from .types import GenerateResponse, Message


class AbstractLLMInterface(ABC):
    """
    Abstract base class for all LLM providers.

    Token Parameter Vocabulary:
    - max_tokens: Total context window (input + output combined)
    - max_input_tokens: Maximum allowed input/prompt tokens
    - max_output_tokens: Maximum allowed output/generation tokens

    Constraint: max_input_tokens + max_output_tokens ≤ max_tokens
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