"""
Structured output handler for managing schema-based LLM responses.
"""

import json
import re
import time
from typing import Type, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, ValidationError

from .retry import FeedbackRetry
from ..utils.structured_logging import get_logger
from ..utils.self_fixes import fix_json
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
            # Store provider for schema generation
            self.current_provider = provider
            
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

        Checks both provider type (Ollama, LMStudio, HuggingFace, MLX with Outlines)
        and model capabilities configuration as fallback.

        Args:
            provider: The LLM provider instance

        Returns:
            True if provider supports native structured outputs
        """
        # Ollama and LMStudio always support native structured outputs
        # via the format and response_format parameters respectively
        provider_name = provider.__class__.__name__
        if provider_name in ['OllamaProvider', 'LMStudioProvider']:
            return True

        # HuggingFaceProvider supports native via GGUF or Transformers+Outlines
        if provider_name == 'HuggingFaceProvider':
            # Check if it's a GGUF model - these use llama-cpp-python which supports native structured outputs
            if hasattr(provider, 'model_type') and provider.model_type == 'gguf':
                return True

            # Check if it's a Transformers model with Outlines available
            if hasattr(provider, 'model_type') and provider.model_type == 'transformers':
                try:
                    import outlines
                    return True
                except ImportError:
                    return False

        # MLXProvider supports native via Outlines
        if provider_name == 'MLXProvider':
            try:
                import outlines
                return True
            except ImportError:
                return False

        # For other providers, check model capabilities
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
                # Try to fix the JSON before falling back
                self.logger.debug("Native JSON parsing failed, attempting self-fix", 
                                error=str(e), 
                                response_length=len(response.content))
                
                fixed_json = fix_json(response.content)
                if fixed_json:
                    try:
                        data = json.loads(fixed_json)
                        result = response_model.model_validate(data)
                        self.logger.info("JSON self-fix successful for native response")
                        return result
                    except (json.JSONDecodeError, ValidationError) as fix_error:
                        self.logger.debug("Self-fix failed", error=str(fix_error))
                
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
                
                # Try parsing the extracted JSON
                try:
                    data = json.loads(json_content)
                    # Preprocess enum responses if we have mappings
                    if hasattr(self, '_enum_mappings') and self._enum_mappings:
                        data = self._preprocess_enum_response(data, self._enum_mappings)
                    result = response_model.model_validate(data)
                except (json.JSONDecodeError, ValidationError) as parse_error:
                    # Try to fix the JSON
                    self.logger.debug("JSON parsing failed, attempting self-fix", 
                                    error=str(parse_error), 
                                    json_length=len(json_content),
                                    attempt=attempt + 1)
                    
                    fixed_json = fix_json(json_content)
                    if fixed_json:
                        try:
                            data = json.loads(fixed_json)
                            # Preprocess enum responses if we have mappings
                            if hasattr(self, '_enum_mappings') and self._enum_mappings:
                                data = self._preprocess_enum_response(data, self._enum_mappings)
                            result = response_model.model_validate(data)
                            self.logger.info("JSON self-fix successful", attempt=attempt + 1)
                        except (json.JSONDecodeError, ValidationError) as fix_error:
                            self.logger.debug("Self-fix failed", error=str(fix_error), attempt=attempt + 1)
                            raise parse_error  # Re-raise original error for retry logic
                    else:
                        raise parse_error  # Re-raise original error for retry logic

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
        
        # For prompted providers, simplify enum schemas to avoid LLM confusion
        # Store original enum mappings for response preprocessing
        if hasattr(self, 'current_provider') and not self._has_native_support(self.current_provider):
            schema, self._enum_mappings = self._simplify_enum_schemas(schema)
        else:
            self._enum_mappings = {}
        
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

    def _simplify_enum_schemas(self, schema: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
        """
        Simplify enum schemas for prompted providers while preserving enum mappings.
        
        Args:
            schema: Original JSON schema
            
        Returns:
            Tuple of (simplified_schema, enum_mappings)
            enum_mappings maps field_paths to {enum_notation: enum_value}
        """
        if '$defs' not in schema:
            return schema, {}
        
        # Find enum definitions and build mappings
        enum_mappings = {}
        enum_refs_to_simplify = {}
        
        for def_name, def_schema in schema['$defs'].items():
            if def_schema.get('type') == 'string' and 'enum' in def_schema:
                ref_key = f"#/$defs/{def_name}"
                enum_values = def_schema['enum']
                
                # Build mapping from Python enum notation to actual values
                enum_class_name = def_name
                field_mappings = {}
                for value in enum_values:
                    # Map both "EnumClass.VALUE_NAME" and "<EnumClass.VALUE_NAME: 'value'>" patterns
                    enum_notation = f"{enum_class_name}.{value.upper().replace(' ', '_')}"
                    field_mappings[enum_notation] = value
                    # Also handle the repr format
                    repr_notation = f"<{enum_class_name}.{value.upper().replace(' ', '_')}: '{value}'>"
                    field_mappings[repr_notation] = value
                
                enum_refs_to_simplify[ref_key] = {
                    'type': 'string',
                    'description': f"Use one of: {', '.join(enum_values)}. IMPORTANT: Use the exact string values, not Python enum notation.",
                    'enum': enum_values
                }
                
                # Store mappings by reference for later use
                enum_mappings[ref_key] = field_mappings
        
        # Create simplified schema by replacing enum references
        def replace_enum_refs(obj, path=""):
            if isinstance(obj, dict):
                if '$ref' in obj and obj['$ref'] in enum_refs_to_simplify:
                    # Store the field path for this enum reference
                    if path:
                        enum_mappings[path] = enum_mappings[obj['$ref']]
                    return enum_refs_to_simplify[obj['$ref']]
                return {k: replace_enum_refs(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_enum_refs(item, path) for item in obj]
            return obj
        
        simplified_schema = replace_enum_refs(schema)
        
        # Remove the $defs section since we've inlined the enum definitions
        if '$defs' in simplified_schema:
            # Only remove enum definitions, keep other definitions
            remaining_defs = {k: v for k, v in simplified_schema['$defs'].items() 
                            if not (v.get('type') == 'string' and 'enum' in v)}
            if remaining_defs:
                simplified_schema['$defs'] = remaining_defs
            else:
                del simplified_schema['$defs']
        
        return simplified_schema, enum_mappings

    def _preprocess_enum_response(self, data: Dict[str, Any], enum_mappings: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """
        Preprocess LLM response to convert Python enum notation back to valid enum values.
        
        Args:
            data: Parsed JSON data from LLM
            enum_mappings: Mappings from field paths to enum notation conversions
            
        Returns:
            Preprocessed data with enum notations converted to valid values
        """
        if not enum_mappings:
            return data
        
        def convert_enum_values(obj, path=""):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    field_path = f"{path}.{key}" if path else key
                    
                    # Check if this field has enum mappings
                    field_mappings = None
                    for enum_path, mappings in enum_mappings.items():
                        if field_path in enum_path or enum_path in field_path:
                            field_mappings = mappings
                            break
                    
                    if field_mappings and isinstance(value, str):
                        # Try to convert enum notation to actual value
                        converted_value = field_mappings.get(value, value)
                        result[key] = converted_value
                    else:
                        result[key] = convert_enum_values(value, field_path)
                return result
            elif isinstance(obj, list):
                return [convert_enum_values(item, path) for item in obj]
            return obj
        
        return convert_enum_values(data)