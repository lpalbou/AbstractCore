"""
Base provider with integrated telemetry, events, and exception handling.
"""

import time
from typing import List, Dict, Any, Optional, Union, Iterator, Type
from abc import ABC, abstractmethod

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

        # Setup timeout configuration
        self._timeout = kwargs.get('timeout', 300.0)  # Default 300 seconds (5 minutes) for HTTP requests
        self._tool_timeout = kwargs.get('tool_timeout', 300.0)  # Default 300 seconds for tool execution

        # Setup tool execution mode
        # execute_tools: True = AbstractCore executes tools (legacy mode)
        #                False = Pass-through mode (default - for API server / agentic CLI)
        self.execute_tools = kwargs.get('execute_tools', False)

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
                               tool_call_tags: Optional[str] = None,  # Tool call tag rewriting
                               execute_tools: Optional[bool] = None,  # Tool execution control
                               **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse], BaseModel]:
        """
        Generate with integrated telemetry and error handling.
        Providers should override _generate_internal instead of generate.

        Args:
            response_model: Optional Pydantic model for structured output
            retry_strategy: Optional retry strategy for structured output validation
            tool_call_tags: Optional tool call tag format for rewriting
            execute_tools: Whether to execute tools automatically (True) or let agent handle execution (False)
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
        
        # Handle tool execution control
        should_execute_tools = execute_tools if execute_tools is not None else self.execute_tools
        if not should_execute_tools and converted_tools:
            # If tools are provided but execution is disabled,
            # we still pass them to the provider for generation but won't execute them
            self.logger.info("Tool execution disabled - tools will be generated but not executed")

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
                    execute_tools=should_execute_tools,
                    tool_call_tags=tool_call_tags,
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

            # Handle streaming with unified processor
            if stream:
                def unified_stream():
                    try:
                        # Import and create unified stream processor
                        from .streaming import UnifiedStreamProcessor

                        # Use the should_execute_tools value (defaults to False)
                        actual_execute_tools = should_execute_tools

                        processor = UnifiedStreamProcessor(
                            model_name=self.model,
                            execute_tools=actual_execute_tools,  # Default: False (pass-through mode)
                            tool_call_tags=tool_call_tags,
                            default_target_format="qwen3"  # Always rewrite to qwen3 format
                        )

                        # Process stream with incremental tool detection and execution
                        for processed_chunk in processor.process_stream(response, converted_tools):
                            yield processed_chunk

                        # Track generation after streaming completes
                        self._track_generation(prompt, None, start_time, success=True, stream=True)

                    except Exception as e:
                        # Track error
                        self._track_generation(prompt, None, start_time, success=False, error=e, stream=True)
                        raise

                return unified_stream()
            else:
                # Non-streaming: apply tag rewriting if needed
                if response and response.content and converted_tools:
                    # Apply default qwen3 rewriting for non-streaming responses
                    response = self._apply_non_streaming_tag_rewriting(response, tool_call_tags)

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
                          execute_tools: Optional[bool] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Internal generation method to be implemented by subclasses.
        This is called by generate_with_telemetry.

        Args:
            response_model: Optional Pydantic model for structured output
            execute_tools: Whether to execute tools automatically (True) or let agent handle execution (False)
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

        # Run enhanced validation and warn about potential issues
        self._check_token_configuration_warnings()

    def _check_token_configuration_warnings(self):
        """Check token configuration and emit warnings for potential issues"""
        import warnings

        try:
            warnings_list = self.validate_token_constraints()
            for warning in warnings_list[:3]:  # Limit to first 3 warnings to avoid spam
                warnings.warn(f"Token configuration warning for {self.model}: {warning}",
                             UserWarning, stacklevel=4)

            # Also log warnings for debugging
            if warnings_list and hasattr(self, 'logger'):
                self.logger.debug(f"Token configuration warnings for {self.model}: {'; '.join(warnings_list)}")

        except Exception as e:
            # Don't fail provider initialization due to validation warnings
            if hasattr(self, 'logger'):
                self.logger.debug(f"Error checking token configuration warnings: {e}")

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

        # Use JSON capabilities as single source of truth for defaults
        from ..architectures import get_context_limits
        limits = get_context_limits(self.model)
        max_output_tokens = limits['max_output_tokens']
        self.logger.debug(f"Using default max_output_tokens {max_output_tokens} from model_capabilities.json for {self.model}")
        return max_output_tokens

    def _get_default_context_window(self) -> int:
        """Get default context window using JSON capabilities as single source of truth"""
        from ..architectures import get_model_capabilities

        # Fallback chain: Exact model → Model family → Provider defaults → Global defaults
        capabilities = get_model_capabilities(self.model)

        if capabilities:
            max_tokens = capabilities.get('max_tokens')
            if max_tokens:
                self.logger.debug(f"Using max_tokens {max_tokens} from model capabilities for {self.model}")
                return max_tokens

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
                if family_caps and family_caps.get('max_tokens'):
                    max_tokens = family_caps['max_tokens']
                    self.logger.debug(f"Using max_tokens {max_tokens} from family {family} for {self.model}")
                    return max_tokens

        # Use JSON capabilities as single source of truth for defaults
        from ..architectures import get_context_limits
        limits = get_context_limits(self.model)
        max_tokens = limits['max_tokens']
        self.logger.debug(f"Using default max_tokens {max_tokens} from model_capabilities.json for {self.model}")
        return max_tokens

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

    def _handle_prompted_tool_execution(self, response: GenerateResponse, tools: List[Dict[str, Any]], execute_tools_param: bool = None) -> GenerateResponse:
        """Handle tool execution for prompted responses (shared implementation)"""
        if not response.content:
            return response

        # Parse tool calls from response content using UniversalToolHandler
        tool_call_response = self.tool_handler.parse_response(response.content, mode="prompted")
        tool_calls = tool_call_response.tool_calls

        if not tool_calls:
            return response

        # Execute with events and return result
        return self._execute_tools_with_events(response, tool_calls, execute_tools_param)

    def _execute_tools_with_events(self, response: GenerateResponse, tool_calls: List, execute_tools_param: bool = None) -> GenerateResponse:
        """Core tool execution with event emission (shared implementation)"""
        # Check if tool execution is enabled
        should_execute = execute_tools_param if execute_tools_param is not None else self.execute_tools
        
        if not should_execute:
            # Tool execution disabled - return response with tool calls but don't execute
            self.logger.info("Tool execution disabled - returning response with tool calls")
            return response
        
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
        results_text = self._format_tool_results(tool_calls, tool_results)

        # Return updated response with tool results
        # Use the cleaned content from tool parsing
        tool_call_response = self.tool_handler.parse_response(response.content, mode="prompted")
        cleaned_content = tool_call_response.content or ""
        return GenerateResponse(
            content=cleaned_content + results_text,
            model=response.model,
            finish_reason=response.finish_reason,
            raw_response=response.raw_response,
            usage=response.usage,
            tool_calls=response.tool_calls  # Keep original format
        )

    def _format_tool_results(self, tool_calls: List, tool_results: List) -> str:
        """Format tool results with tool transparency (shared implementation)"""
        results_text = "\n\nTool Results:\n"
        for call, result in zip(tool_calls, tool_results):
            # Format parameters for display (limit size)
            params_str = str(call.arguments) if call.arguments else "{}"
            if len(params_str) > 100:
                params_str = params_str[:97] + "..."

            # Show tool name and parameters for transparency
            results_text += f"🔧 Tool: {call.name}({params_str})\n"

            # Show result
            if result.success:
                results_text += f"- {result.output}\n"
            else:
                results_text += f"- Error: {result.error}\n"
            results_text += "\n"  # Add spacing between tool calls

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

    # Timeout management methods
    def get_timeout(self) -> float:
        """Get the current HTTP request timeout in seconds."""
        return self._timeout

    def set_timeout(self, timeout: float) -> None:
        """Set the HTTP request timeout in seconds."""
        self._timeout = timeout
        # Update HTTP clients if they exist
        self._update_http_client_timeout()

    def get_recovery_timeout(self) -> float:
        """Get the current circuit breaker recovery timeout in seconds."""
        return self.retry_manager.config.recovery_timeout

    def set_recovery_timeout(self, timeout: float) -> None:
        """Set the circuit breaker recovery timeout in seconds."""
        self.retry_manager.config.recovery_timeout = timeout

    def get_tool_timeout(self) -> float:
        """Get the current tool execution timeout in seconds."""
        return self._tool_timeout

    def set_tool_timeout(self, timeout: float) -> None:
        """Set the tool execution timeout in seconds."""
        self._tool_timeout = timeout

    def _update_http_client_timeout(self) -> None:
        """Update HTTP client timeout if the provider has one. Override in subclasses."""
        pass

    # Memory management methods
    def unload(self) -> None:
        """
        Unload the model from memory.

        For local providers (MLX, HuggingFace), this explicitly frees model memory.
        For server-based providers (Ollama, LMStudio), this requests server unload.
        For API providers (OpenAI, Anthropic), this is a no-op.

        After calling unload(), the provider instance should not be used for generation.
        Create a new provider instance if you need to generate again.

        Usage:
            provider = create_llm("mlx", model="...")
            response = provider.generate("Hello")
            provider.unload()  # Free memory
            del provider  # Remove reference
        """
        # Default implementation does nothing (suitable for API providers)
        pass

    # Token configuration helpers - expose interface methods for user convenience
    def get_token_configuration_summary(self) -> str:
        """Get a human-readable summary of current token configuration"""
        return super().get_token_configuration_summary()

    def validate_token_constraints(self) -> List[str]:
        """Validate token configuration and return warnings/suggestions"""
        return super().validate_token_constraints()

    def calculate_token_budget(self, input_text: str, desired_output_tokens: int,
                              safety_margin: float = 0.1) -> tuple[int, List[str]]:
        """Helper to estimate required max_tokens given input and desired output"""
        return super().calculate_token_budget(input_text, desired_output_tokens, safety_margin)

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count for given text"""
        return super().estimate_tokens(text)

    @abstractmethod
    def list_available_models(self, **kwargs) -> List[str]:
        """
        List available models for this provider.

        Each provider must implement this method to return a list of available models.
        The server will use this method to aggregate models across all providers.

        Args:
            **kwargs: Provider-specific parameters (e.g., api_key, base_url)

        Returns:
            List of model names available for this provider

        Note:
            This is an abstract method that MUST be implemented by all provider subclasses.
            Each provider should implement its own discovery logic (API calls, local scanning, etc.).
        """
        pass

    def _needs_tag_rewriting(self, tool_call_tags) -> bool:
        """Check if tag rewriting is needed (tags are non-standard)"""
        try:
            from ..tools.tag_rewriter import ToolCallTags

            if isinstance(tool_call_tags, str):
                # String format - handle comma-separated format
                if ',' in tool_call_tags:
                    # Comma-separated format like '<function_call>,</function_call>'
                    parts = tool_call_tags.split(',')
                    if len(parts) == 2:
                        opening_tag = parts[0].strip()
                        closing_tag = parts[1].strip()
                        if opening_tag == "<function_call>" and closing_tag == "</function_call>":
                            return False
                else:
                    # Single tag format
                    if tool_call_tags in ["<function_call>", "</function_call>"]:
                        return False
            elif isinstance(tool_call_tags, ToolCallTags):
                # ToolCallTags object - check if it contains standard tags
                if (hasattr(tool_call_tags, 'start_tag') and hasattr(tool_call_tags, 'end_tag')):
                    # Only standard if exactly matches the standard format
                    if (tool_call_tags.start_tag == "<function_call>" and tool_call_tags.end_tag == "</function_call>"):
                        return False

            # Any other format or non-standard tags need rewriting
            return True

        except Exception:
            # If we can't determine, err on the side of applying rewriting
            return True

    def _apply_non_streaming_tag_rewriting(self, response: GenerateResponse, tool_call_tags: Optional[str] = None) -> GenerateResponse:
        """Apply tag rewriting to non-streaming response content."""
        try:
            from .streaming import UnifiedStreamProcessor

            # Create a temporary processor for tag rewriting
            processor = UnifiedStreamProcessor(
                model_name=self.model,
                execute_tools=False,  # No execution, just rewriting
                tool_call_tags=tool_call_tags,
                default_target_format="qwen3"  # Always rewrite to qwen3 format
            )

            # Apply tag rewriting to the content
            if processor.tag_rewriter and response.content:
                rewritten_content = processor._apply_tag_rewriting_direct(response.content)

                # Return new response with rewritten content
                return GenerateResponse(
                    content=rewritten_content,
                    model=response.model,
                    finish_reason=response.finish_reason,
                    raw_response=response.raw_response,
                    usage=response.usage,
                    tool_calls=response.tool_calls
                )

        except Exception as e:
            self.logger.debug(f"Non-streaming tag rewriting failed: {e}")

        # Return original response if rewriting fails
        return response