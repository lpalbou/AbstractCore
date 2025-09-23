"""
Structured output handler for managing schema-based LLM responses.
"""

import json
import re
import time
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

from .retry import FeedbackRetry
from ..utils.structured_logging import get_logger
from ..events import EventType, emit_global, create_structured_output_event


class StructuredOutputHandler:
    """
    Handles structured output generation using two strategies:
    1. Native support (when provider supports it)
    2. Prompted generation with validation and retry
    """

    def __init__(self, retry_strategy: Optional[FeedbackRetry] = None):
        """
        Initialize the handler.

        Args:
            retry_strategy: Strategy for handling validation failures
        """
        self.retry_strategy = retry_strategy or FeedbackRetry()
        self.logger = get_logger(__name__)

    def generate_structured(
        self,
        provider,
        prompt: str,
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Generate structured output using the best available strategy.

        Args:
            provider: The LLM provider instance
            prompt: Input prompt
            response_model: Pydantic model to validate against
            **kwargs: Additional parameters passed to provider

        Returns:
            Validated instance of response_model

        Raises:
            ValidationError: If validation fails after all retries
        """
        start_time = time.time()
        provider_name = getattr(provider, '__class__', {}).__name__ or 'unknown'
        model_name = getattr(provider, 'model', 'unknown')

        # Note: STRUCTURED_OUTPUT_REQUESTED event removed in simplification
        # Structured output is just a special type of generation

        # Log structured output request
        self.logger.info("Starting structured output generation",
                        provider=provider_name,
                        model=model_name,
                        response_model=response_model.__name__,
                        prompt_length=len(prompt),
                        max_retries=self.retry_strategy.max_attempts)

        try:
            # Strategy 1: Use native support if available
            if self._has_native_support(provider):
                self.logger.debug("Using native structured output support",
                                provider=provider_name)
                result = self._generate_native(provider, prompt, response_model, **kwargs)
                strategy = "native"
            else:
                self.logger.debug("Using prompted structured output with retry",
                                provider=provider_name)
                result = self._generate_prompted(provider, prompt, response_model, **kwargs)
                strategy = "prompted"

            # Note: Removed STRUCTURED_OUTPUT_GENERATED - using VALIDATION_SUCCEEDED instead
            duration_ms = (time.time() - start_time) * 1000

            # Log successful completion
            self.logger.info("Structured output generation completed",
                           provider=provider_name,
                           model=model_name,
                           response_model=response_model.__name__,
                           strategy=strategy,
                           duration_ms=duration_ms,
                           success=True)

            return result

        except Exception as e:
            # Emit failure event
            duration_ms = (time.time() - start_time) * 1000
            error_data = {
                "response_model": response_model.__name__,
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False
            }
            emit_global(EventType.ERROR, error_data,
                       source="StructuredOutputHandler",
                       model_name=model_name,
                       provider_name=provider_name,
                       duration_ms=duration_ms)

            # Log failure
            self.logger.error("Structured output generation failed",
                            provider=provider_name,
                            model=model_name,
                            response_model=response_model.__name__,
                            duration_ms=duration_ms,
                            error=str(e),
                            error_type=type(e).__name__,
                            success=False)
            raise

    def _has_native_support(self, provider) -> bool:
        """
        Check if provider has native structured output support.

        Args:
            provider: The LLM provider instance

        Returns:
            True if provider supports native structured outputs
        """
        capabilities = getattr(provider, 'model_capabilities', {})
        return capabilities.get("structured_output") == "native"

    def _generate_native(
        self,
        provider,
        prompt: str,
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Generate using provider's native structured output support.

        Args:
            provider: The LLM provider instance
            prompt: Input prompt
            response_model: Pydantic model to validate against
            **kwargs: Additional parameters

        Returns:
            Validated instance of response_model
        """
        # The provider will handle structured output natively
        # This is implemented in each provider's _generate_internal method
        response = provider._generate_internal(
            prompt=prompt,
            response_model=response_model,
            **kwargs
        )

        # For native support, the response content should already be structured
        if isinstance(response.content, dict):
            return response_model.model_validate(response.content)
        else:
            # Parse JSON string
            try:
                data = json.loads(response.content)
                return response_model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                # Even native support can fail, fallback to prompted
                return self._generate_prompted(provider, prompt, response_model, **kwargs)

    def _generate_prompted(
        self,
        provider,
        prompt: str,
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Generate using prompted approach with validation and retry.

        Args:
            provider: The LLM provider instance
            prompt: Input prompt
            response_model: Pydantic model to validate against
            **kwargs: Additional parameters

        Returns:
            Validated instance of response_model

        Raises:
            ValidationError: If all retry attempts fail
        """
        # Create schema-enhanced prompt
        enhanced_prompt = self._create_schema_prompt(prompt, response_model)
        provider_name = getattr(provider, '__class__', {}).__name__ or 'unknown'
        model_name = getattr(provider, 'model', 'unknown')

        last_error = None
        current_prompt = enhanced_prompt

        for attempt in range(1, self.retry_strategy.max_attempts + 1):
            attempt_start_time = time.time()

            # Note: Removed VALIDATION_STARTED - redundant, only tracking results

            self.logger.debug("Starting validation attempt",
                            provider=provider_name,
                            attempt=attempt,
                            max_attempts=self.retry_strategy.max_attempts,
                            response_model=response_model.__name__)

            try:
                # Generate response
                response = provider._generate_internal(
                    prompt=current_prompt,
                    **kwargs
                )

                # Extract and validate JSON
                json_content = self._extract_json(response.content)
                data = json.loads(json_content)
                result = response_model.model_validate(data)

                # Emit validation success event
                attempt_duration_ms = (time.time() - attempt_start_time) * 1000
                # Note: VALIDATION_SUCCEEDED event removed in simplification
                # Success is indicated by successfully parsing the response

                # Log successful validation
                self.logger.info("Validation attempt succeeded",
                               provider=provider_name,
                               attempt=attempt,
                               max_attempts=self.retry_strategy.max_attempts,
                               response_model=response_model.__name__,
                               attempt_duration_ms=attempt_duration_ms,
                               response_length=len(response.content),
                               validation_success=True)

                return result

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                attempt_duration_ms = (time.time() - attempt_start_time) * 1000

                # Emit validation failure event
                failure_data = create_structured_output_event(
                    response_model=response_model.__name__,
                    validation_attempt=attempt,
                    validation_error=str(e),
                    error_type=type(e).__name__,
                    response_length=len(getattr(response, 'content', ''))
                )
                emit_global(EventType.VALIDATION_FAILED, failure_data,
                           source="StructuredOutputHandler",
                           model_name=model_name,
                           provider_name=provider_name,
                           duration_ms=attempt_duration_ms)

                # Log validation failure
                self.logger.warning("Validation attempt failed",
                                  provider=provider_name,
                                  attempt=attempt,
                                  max_attempts=self.retry_strategy.max_attempts,
                                  response_model=response_model.__name__,
                                  attempt_duration_ms=attempt_duration_ms,
                                  error_type=type(e).__name__,
                                  error_message=str(e),
                                  response_length=len(getattr(response, 'content', '')),
                                  validation_success=False)

                # Check if we should retry
                if self.retry_strategy.should_retry(attempt, e):
                    # Note: RETRY_ATTEMPTED event removed in simplification
                    # Retry logic tracked through VALIDATION_FAILED event with attempt number

                    self.logger.info("Preparing retry with validation feedback",
                                   provider=provider_name,
                                   attempt=attempt + 1,
                                   max_attempts=self.retry_strategy.max_attempts,
                                   retry_reason="validation_error")

                    # Prepare retry prompt with feedback
                    current_prompt = self.retry_strategy.prepare_retry_prompt(
                        enhanced_prompt, e, attempt
                    )
                else:
                    # No more retries
                    self.logger.error("All validation attempts exhausted",
                                    provider=provider_name,
                                    total_attempts=attempt,
                                    max_attempts=self.retry_strategy.max_attempts,
                                    response_model=response_model.__name__,
                                    final_error=str(e),
                                    validation_success=False)
                    break

        # All retries exhausted
        raise last_error

    def _create_schema_prompt(self, prompt: str, response_model: Type[BaseModel]) -> str:
        """
        Create a prompt that includes the JSON schema.

        Args:
            prompt: Original prompt
            response_model: Pydantic model

        Returns:
            Enhanced prompt with schema information
        """
        schema = response_model.model_json_schema()
        model_name = response_model.__name__

        # Create example from schema
        example = self._create_example_from_schema(schema)

        enhanced_prompt = f"""{prompt}

Please respond with valid JSON that matches this exact schema for {model_name}:

{json.dumps(schema, indent=2)}

Example format:
{json.dumps(example, indent=2)}

Important: Return ONLY the JSON object, no additional text or formatting."""

        return enhanced_prompt

    def _create_example_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a simple example from JSON schema.

        Args:
            schema: JSON schema dictionary

        Returns:
            Example data structure
        """
        properties = schema.get("properties", {})
        example = {}

        for field, field_schema in properties.items():
            field_type = field_schema.get("type")

            if field_type == "string":
                example[field] = "example_string"
            elif field_type == "integer":
                example[field] = 42
            elif field_type == "number":
                example[field] = 3.14
            elif field_type == "boolean":
                example[field] = True
            elif field_type == "array":
                example[field] = ["example_item"]
            elif field_type == "object":
                example[field] = {"key": "value"}
            else:
                example[field] = None

        return example

    def _extract_json(self, content: str) -> str:
        """
        Extract JSON from response content that might contain additional text.

        Args:
            content: Response content

        Returns:
            Extracted JSON string

        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # First try to parse the content directly
        content = content.strip()
        if content.startswith('{') and content.endswith('}'):
            return content

        # Look for JSON within code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            return match.group(1)

        # Look for JSON object in the content
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            return match.group(0)

        # If nothing found, try the original content
        return content