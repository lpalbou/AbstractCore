"""
Retry strategies for structured output validation.
"""

import json
from abc import ABC, abstractmethod
from typing import Type, Optional
from pydantic import BaseModel, ValidationError

from ..events import EventType, emit_global, create_structured_output_event


class Retry(ABC):
    """Abstract base class for retry strategies."""

    @abstractmethod
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """
        Determine if we should retry based on the attempt number and error.

        Args:
            attempt: Current attempt number (1-based)
            error: The exception that occurred

        Returns:
            True if we should retry, False otherwise
        """
        pass

    @abstractmethod
    def prepare_retry_prompt(self, original_prompt: str, error: Exception, attempt: int) -> str:
        """
        Prepare the prompt for retry, potentially including error feedback.

        Args:
            original_prompt: The original prompt
            error: The exception that occurred
            attempt: Current attempt number (1-based)

        Returns:
            Modified prompt for retry
        """
        pass


class FeedbackRetry(Retry):
    """
    Retry strategy that feeds Pydantic validation errors back to the LLM.

    This strategy allows the LLM to self-correct by providing detailed
    feedback about what went wrong with the previous attempt.
    """

    def __init__(self, max_attempts: int = 3):
        """
        Initialize FeedbackRetry.

        Args:
            max_attempts: Maximum number of attempts (including initial)
        """
        self.max_attempts = max_attempts

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """
        Retry only for Pydantic ValidationErrors and within attempt limit.
        """
        return isinstance(error, ValidationError) and attempt < self.max_attempts

    def prepare_retry_prompt(self, original_prompt: str, error: ValidationError, attempt: int) -> str:
        """
        Create a retry prompt with validation error feedback.
        """
        error_feedback = self._format_validation_errors(error)

        retry_prompt = f"""{original_prompt}

IMPORTANT: Your previous response (attempt {attempt}) had validation errors:

{error_feedback}

Please correct these errors and provide a valid JSON response that matches the required schema exactly."""

        return retry_prompt

    def _format_validation_errors(self, error: ValidationError) -> str:
        """
        Format Pydantic validation errors into clear, actionable feedback.

        Args:
            error: The ValidationError from Pydantic

        Returns:
            Formatted error message for the LLM
        """
        error_lines = []

        for err in error.errors():
            field_path = " -> ".join(str(loc) for loc in err["loc"])
            error_type = err["type"]
            error_msg = err["msg"]

            # Create user-friendly error messages
            if error_type == "missing":
                error_lines.append(f"• Missing required field: '{field_path}'")
            elif error_type == "int_parsing":
                error_lines.append(f"• Field '{field_path}': Expected an integer, got text that can't be converted")
            elif error_type == "string_type":
                error_lines.append(f"• Field '{field_path}': Expected a string")
            elif error_type == "value_error":
                error_lines.append(f"• Field '{field_path}': {error_msg}")
            elif error_type == "json_invalid":
                error_lines.append(f"• Invalid JSON format: {error_msg}")
            else:
                error_lines.append(f"• Field '{field_path}': {error_msg}")

        return "\n".join(error_lines)