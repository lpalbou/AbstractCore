"""
Base provider with integrated telemetry, events, and exception handling.
"""

import time
from typing import List, Dict, Any, Optional, Union, Iterator, Type
from abc import ABC

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

from ..core.interface import AbstractLLMInterface
from ..core.types import GenerateResponse
from ..events import EventType, Event
from datetime import datetime
from ..utils.structured_logging import get_logger
from ..exceptions import (
    ProviderAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError
)
from ..architectures import detect_architecture, get_architecture_format, get_model_capabilities
from ..tools import execute_tools
from ..core.retry import RetryManager, RetryConfig


class BaseProvider(AbstractLLMInterface, ABC):
    """
    Base provider class with integrated telemetry and events.
    All providers should inherit from this class.
    """

    def __init__(self, model: str, **kwargs):
        AbstractLLMInterface.__init__(self, model, **kwargs)

        # Setup structured logging
        self.logger = get_logger(self.__class__.__name__)

        # Detect architecture and get model capabilities
        self.architecture = detect_architecture(model)
        self.architecture_config = get_architecture_format(self.architecture)
        self.model_capabilities = get_model_capabilities(model)

        # Setup retry manager with optional configuration
        retry_config = kwargs.get('retry_config', None)
        if retry_config is None:
            # Use default retry configuration
            retry_config = RetryConfig()
        self.retry_manager = RetryManager(retry_config)

        # Create provider key for circuit breaker tracking
        self.provider_key = f"{self.__class__.__name__}:{self.model}"

        # Provider created successfully - no event emission needed
        # (The simplified event system focuses on generation and tool events only)

        # Set default token limits if not provided
        self._initialize_token_limits()

    def _track_generation(self, prompt: str, response: Optional[GenerateResponse],
                         start_time: float, success: bool = True,
                         error: Optional[Exception] = None, stream: bool = False):
        """
        Track generation with telemetry and events.

        Args:
            prompt: Input prompt
            response: Generated response
            start_time: Generation start time
            success: Whether generation succeeded
            error: Error if failed
            stream: Whether this was a streaming generation
        """
        latency_ms = (time.time() - start_time) * 1000

        # Extract token and cost information from response
        tokens_input = None
        tokens_output = None
        cost_usd = None

        if response and response.usage:
            tokens_input = response.usage.get('prompt_tokens') or response.usage.get('input_tokens')
            tokens_output = response.usage.get('completion_tokens') or response.usage.get('output_tokens')
            # Calculate cost if possible (simplified - could be enhanced)
            total_tokens = response.usage.get('total_tokens', 0)
            if total_tokens > 0:
                # Very rough cost estimation - should be provider-specific
                cost_usd = total_tokens * 0.00002  # ~$0.02 per 1K tokens average

        # Emit comprehensive event with all data in one dict
        event_data = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "success": success,
            "error": str(error) if error else None,
            "response_length": len(response.content) if response and response.content else 0,
            "stream": stream,
            "model": self.model,
            "provider": self.__class__.__name__,
            "duration_ms": latency_ms,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cost_usd": cost_usd
        }

        from ..events import emit_global
        emit_global(EventType.GENERATION_COMPLETED, event_data, source=self.__class__.__name__)

        # Track with structured logging (using formatted strings)
        if error:
            # Only log debug info for model not found errors to avoid duplication
            if isinstance(error, ModelNotFoundError):
                self.logger.debug(f"Model not found: {self.model}")
            else:
                self.logger.error(f"Generation failed for {self.model}: {error} (latency: {latency_ms:.2f}ms)")
        else:
            tokens_info = ""
            if response and response.usage:
                tokens_info = f" (tokens: {response.usage.get('total_tokens', 0)})"

            self.logger.info(f"Generation completed for {self.model}: {latency_ms:.2f}ms{tokens_info}")

    def _track_tool_call(self, tool_name: str, arguments: Dict[str, Any],
                        result: Optional[Any] = None, success: bool = True,
                        error: Optional[Exception] = None):
        """
        Track tool call with telemetry and events.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            success: Whether call succeeded
            error: Error if failed
        """
        # Emit comprehensive tool event
        event_type = EventType.TOOL_COMPLETED if success else EventType.ERROR
        event_data = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": str(result)[:100] if result else None,
            "error": str(error) if error else None,
            "success": success
        }

        # Add model and provider to event data
        event_data["model"] = self.model
        event_data["provider"] = self.__class__.__name__

        from ..events import emit_global
        emit_global(event_type, event_data, source=self.__class__.__name__)

        # Track with structured logging (using formatted strings)
        if error:
            self.logger.warning(f"Tool call failed: {tool_name} - {error}")
        else:
            result_info = f" (result length: {len(str(result))})" if result else ""
            self.logger.info(f"Tool call completed: {tool_name}{result_info}")


    def _handle_api_error(self, error: Exception) -> Exception:
        """
        Convert API errors to custom exceptions.

        Args:
            error: Original exception

        Returns:
            Custom exception
        """
        # Don't re-wrap our custom exceptions
        if isinstance(error, (ModelNotFoundError, AuthenticationError, RateLimitError,
                            InvalidRequestError, ProviderAPIError)):
            return error

        error_str = str(error).lower()

        if "rate" in error_str and "limit" in error_str:
            return RateLimitError(f"Rate limit exceeded: {error}")
        elif "auth" in error_str or "api key" in error_str or "unauthorized" in error_str:
            return AuthenticationError(f"Authentication failed: {error}")
        elif "invalid" in error_str or "bad request" in error_str:
            return InvalidRequestError(f"Invalid request: {error}")
        else:
            return ProviderAPIError(f"API error: {error}")

    def generate_with_telemetry(self,
                               prompt: str,
                               messages: Optional[List[Dict[str, str]]] = None,
                               system_prompt: Optional[str] = None,
                               tools: Optional[List] = None,  # Accept both ToolDefinition and Dict
                               stream: bool = False,
                               response_model: Optional[Type[BaseModel]] = None,
                               retry_strategy=None,  # Custom retry strategy for structured output
                               **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse], BaseModel]:
        """
        Generate with integrated telemetry and error handling.
        Providers should override _generate_internal instead of generate.

        Args:
            response_model: Optional Pydantic model for structured output
            retry_strategy: Optional retry strategy for structured output validation
        """
        # Handle structured output request
        if response_model is not None:
            if not PYDANTIC_AVAILABLE:
                raise ImportError(
                    "Pydantic is required for structured outputs. "
                    "Install with: pip install pydantic>=2.0.0"
                )

            from ..structured import StructuredOutputHandler
            handler = StructuredOutputHandler(retry_strategy=retry_strategy)
            return handler.generate_structured(
                provider=self,
                prompt=prompt,
                response_model=response_model,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                stream=stream,
                **kwargs
            )

        # Convert tools to ToolDefinition objects first (outside retry loop)
        converted_tools = None
        if tools:
            converted_tools = []
            for tool in tools:
                if hasattr(tool, 'to_dict'):  # ToolDefinition object
                    converted_tools.append(tool.to_dict())
                elif callable(tool):  # Function - check for enhanced metadata
                    if hasattr(tool, '_tool_definition'):
                        # Use the enhanced tool definition from @tool decorator
                        converted_tools.append(tool._tool_definition.to_dict())
                    else:
                        # Fall back to basic conversion
                        from ..tools.core import ToolDefinition
                        tool_def = ToolDefinition.from_function(tool)
                        converted_tools.append(tool_def.to_dict())
                elif isinstance(tool, dict):  # Already a dict
                    converted_tools.append(tool)
                else:
                    # Handle other types gracefully
                    self.logger.warning(f"Unknown tool type: {type(tool)}, skipping")

        # Define generation function for retry wrapper
        def _execute_generation():
            start_time = time.time()

            # Emit generation started event (covers request received)
            event_data = {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "has_tools": bool(tools),
                "stream": stream,
                "model": self.model,
                "provider": self.__class__.__name__
            }
            from ..events import emit_global
            emit_global(EventType.GENERATION_STARTED, event_data, source=self.__class__.__name__)

            try:
                # Call the actual generation (implemented by subclass)
                response = self._generate_internal(
                    prompt=prompt,
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=converted_tools,
                    stream=stream,
                    **kwargs
                )

                return response, start_time

            except Exception as e:
                # Convert to custom exception and re-raise for retry handling
                custom_error = self._handle_api_error(e)
                raise custom_error

        # Execute with retry
        try:
            response, start_time = self.retry_manager.execute_with_retry(
                _execute_generation,
                provider_key=self.provider_key
            )

            # Handle streaming vs non-streaming differently
            if stream:
                # Wrap the stream to emit events
                def stream_with_events():
                    try:
                        # Yield all chunks from the original stream
                        for chunk in response:
                            yield chunk

                        # Track generation after streaming completes
                        self._track_generation(prompt, None, start_time, success=True, stream=True)

                    except Exception as e:
                        # Track error
                        self._track_generation(prompt, None, start_time, success=False, error=e, stream=True)
                        raise

                return stream_with_events()
            else:
                # Non-streaming: track after completion
                self._track_generation(prompt, response, start_time, success=True, stream=False)
                return response

        except Exception as e:
            # This exception comes from the retry manager after all attempts failed
            # Track final error (start_time may not be available, use current time)
            current_time = time.time()
            self._track_generation(prompt, None, current_time, success=False, error=e, stream=stream)

            # Emit error event
            from ..events import emit_global
            emit_global(EventType.ERROR, {
                "error": str(e),
                "error_type": type(e).__name__,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "model": self.model,
                "provider": self.__class__.__name__
            }, source=self.__class__.__name__)

            # Re-raise the exception
            raise e

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Internal generation method to be implemented by subclasses.
        This is called by generate_with_telemetry.

        Args:
            response_model: Optional Pydantic model for structured output
        """
        raise NotImplementedError("Subclasses must implement _generate_internal")

    def _initialize_token_limits(self):
        """Initialize default token limits based on model capabilities"""
        # Set default max_tokens if not provided
        if self.max_tokens is None:
            self.max_tokens = self._get_default_context_window()

        # Set default max_output_tokens if not provided
        if self.max_output_tokens == 2048:  # Only if still default value
            default_max_output = self._get_default_max_output_tokens()
            if default_max_output != 2048:  # If we found a different default
                self.max_output_tokens = default_max_output

        # Validate parameters after setting defaults
        self._validate_token_parameters()

    def _get_default_max_output_tokens(self) -> int:
        """Get default max_output_tokens using JSON capabilities as single source of truth"""
        from ..architectures import get_model_capabilities

        # Fallback chain: Exact model → Model family → Provider defaults → Global defaults
        capabilities = get_model_capabilities(self.model)

        if capabilities:
            max_output_tokens = capabilities.get('max_output_tokens')
            if max_output_tokens:
                self.logger.debug(f"Using max_output_tokens {max_output_tokens} from model capabilities for {self.model}")
                return max_output_tokens

        # If no exact match, try model family/generation fallback
        model_lower = self.model.lower()

        # Family-based fallback patterns (same as context window)
        family_patterns = {
            'gpt-4': ['gpt-4', 'gpt4'],
            'gpt-3.5': ['gpt-3.5', 'gpt3.5'],
            'claude-3': ['claude-3'],
            'claude-3.5': ['claude-3.5'],
            'llama': ['llama'],
            'qwen': ['qwen'],
            'mistral': ['mistral']
        }

        for family, patterns in family_patterns.items():
            if any(pattern in model_lower for pattern in patterns):
                family_caps = get_model_capabilities(family)
                if family_caps and family_caps.get('max_output_tokens'):
                    max_output_tokens = family_caps['max_output_tokens']
                    self.logger.debug(f"Using max_output_tokens {max_output_tokens} from family {family} for {self.model}")
                    return max_output_tokens

        # Provider-specific defaults as final fallback
        provider_defaults = {
            'OpenAIProvider': 4096,
            'AnthropicProvider': 8192,
            'OllamaProvider': 2048,
            'HuggingFaceProvider': 2048,
            'MLXProvider': 2048,
            'LMStudioProvider': 2048
        }

        provider_default = provider_defaults.get(self.__class__.__name__, 2048)
        self.logger.debug(f"Using provider default max_output_tokens {provider_default} for {self.model}")
        return provider_default

    def _get_default_context_window(self) -> int:
        """Get default context window using JSON capabilities as single source of truth"""
        from ..architectures import get_model_capabilities

        # Fallback chain: Exact model → Model family → Provider defaults → Global defaults
        capabilities = get_model_capabilities(self.model)

        if capabilities:
            context_length = capabilities.get('context_length')
            if context_length:
                self.logger.debug(f"Using context_length {context_length} from model capabilities for {self.model}")
                return context_length

        # If no exact match, try model family/generation fallback
        model_lower = self.model.lower()

        # Family-based fallback patterns
        family_patterns = {
            'gpt-4': ['gpt-4', 'gpt4'],
            'gpt-3.5': ['gpt-3.5', 'gpt3.5'],
            'claude-3': ['claude-3'],
            'claude-3.5': ['claude-3.5'],
            'llama': ['llama'],
            'qwen': ['qwen'],
            'mistral': ['mistral']
        }

        for family, patterns in family_patterns.items():
            if any(pattern in model_lower for pattern in patterns):
                family_caps = get_model_capabilities(family)
                if family_caps and family_caps.get('context_length'):
                    context_length = family_caps['context_length']
                    self.logger.debug(f"Using context_length {context_length} from family {family} for {self.model}")
                    return context_length

        # Provider-specific defaults as final fallback
        provider_defaults = {
            'OpenAIProvider': 8192,
            'AnthropicProvider': 200000,
            'OllamaProvider': 4096,
            'HuggingFaceProvider': 8192,
            'MLXProvider': 8192,
            'LMStudioProvider': 8192
        }

        provider_default = provider_defaults.get(self.__class__.__name__, 8192)
        self.logger.warning(f"No model capabilities found for {self.model}, using provider default {provider_default}")
        return provider_default

    def _prepare_generation_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare generation kwargs by translating unified token parameters
        to provider-specific ones.

        Args:
            **kwargs: Generation parameters including unified token params

        Returns:
            Dictionary with provider-specific parameters
        """
        # Get effective token limits
        max_tokens, max_output_tokens, max_input_tokens = self._calculate_effective_token_limits()

        # Override max_output_tokens if provided in kwargs
        effective_max_output = kwargs.get("max_output_tokens", max_output_tokens)

        # Return base kwargs with unified parameter
        result_kwargs = kwargs.copy()
        result_kwargs["max_output_tokens"] = effective_max_output

        return result_kwargs

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """
        Extract the appropriate max tokens parameter for this provider.
        This should be overridden by subclasses to return the provider-specific
        parameter name and value.

        Args:
            kwargs: Generation parameters

        Returns:
            Max output tokens for the provider's API
        """
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _handle_prompted_tool_execution(self, response: GenerateResponse, tools: List[Dict[str, Any]]) -> GenerateResponse:
        """Handle tool execution for prompted responses (shared implementation)"""
        if not response.content:
            return response

        # Parse tool calls from response content using UniversalToolHandler
        tool_call_response = self.tool_handler.parse_response(response.content, mode="prompted")
        tool_calls = tool_call_response.tool_calls

        if not tool_calls:
            return response

        # Execute with events and return result
        return self._execute_tools_with_events(response, tool_calls)

    def _execute_tools_with_events(self, response: GenerateResponse, tool_calls: List) -> GenerateResponse:
        """Core tool execution with event emission (shared implementation)"""
        # Emit tool started event
        event_data = {
            "tool_calls": [{
                "name": call.name,
                "arguments": call.arguments
            } for call in tool_calls],
            "tool_count": len(tool_calls),
            "model": self.model,
            "provider": self.__class__.__name__
        }

        from ..events import emit_global
        emit_global(EventType.TOOL_STARTED, event_data, source=self.__class__.__name__)

        # Execute tools
        tool_results = execute_tools(tool_calls)

        # Emit tool completed event
        after_event_data = {
            "tool_results": [{
                "name": call.name,
                "success": result.success,
                "error": str(result.error) if result.error else None
            } for call, result in zip(tool_calls, tool_results)],
            "successful_count": sum(1 for r in tool_results if r.success),
            "failed_count": sum(1 for r in tool_results if not r.success),
            "model": self.model,
            "provider": self.__class__.__name__
        }

        emit_global(EventType.TOOL_COMPLETED, after_event_data, source=self.__class__.__name__)

        # Track tool calls
        for call, result in zip(tool_calls, tool_results):
            self._track_tool_call(
                tool_name=call.name,
                arguments=call.arguments,
                success=result.success,
                error=result.error if not result.success else None
            )

        # Format tool results and append to response
        results_text = self._format_tool_results(tool_results)

        # Return updated response with tool results
        # Handle case where original content might be None (e.g., OpenAI tool calls)
        original_content = response.content or ""
        return GenerateResponse(
            content=original_content + results_text,
            model=response.model,
            finish_reason=response.finish_reason,
            raw_response=response.raw_response,
            usage=response.usage,
            tool_calls=response.tool_calls  # Keep original format
        )

    def _format_tool_results(self, tool_results: List) -> str:
        """Format tool results as text (shared implementation)"""
        results_text = "\n\nTool Results:\n"
        for result in tool_results:
            if result.success:
                results_text += f"- {result.output}\n"
            else:
                results_text += f"- Error: {result.error}\n"
        return results_text

    def _convert_native_tool_calls_to_standard(self, native_tool_calls: List[Dict[str, Any]]) -> List:
        """Convert native API tool calls to standard ToolCall objects (shared implementation)"""
        from ..tools.core import ToolCall
        import json

        tool_calls = []
        for call in native_tool_calls:
            arguments = call.get('arguments', {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tool_calls.append(ToolCall(
                name=call.get('name', ''),
                arguments=arguments,
                call_id=call.get('id')
            ))
        return tool_calls