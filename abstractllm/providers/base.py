"""
Base provider with integrated telemetry, events, and exception handling.
"""

import time
from typing import List, Dict, Any, Optional, Union, Iterator
from abc import ABC

from ..core.interface import AbstractLLMInterface
from ..core.types import GenerateResponse
from ..events import EventType, EventEmitter
from ..utils.telemetry import get_telemetry
from ..utils.logging import get_logger
from ..exceptions import (
    ProviderAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError
)
from ..architectures import detect_architecture, get_architecture_format, get_model_capabilities
from ..tools import execute_tools


class BaseProvider(AbstractLLMInterface, EventEmitter, ABC):
    """
    Base provider class with integrated telemetry and events.
    All providers should inherit from this class.
    """

    def __init__(self, model: str, **kwargs):
        AbstractLLMInterface.__init__(self, model, **kwargs)
        EventEmitter.__init__(self)

        # Setup logging and telemetry
        self.logger = get_logger(self.__class__.__name__)
        self.telemetry = get_telemetry()

        # Detect architecture and get model capabilities
        self.architecture = detect_architecture(model)
        self.architecture_config = get_architecture_format(self.architecture)
        self.model_capabilities = get_model_capabilities(model)

        # Emit provider created event to both local and global bus
        event_data = {
            "provider": self.__class__.__name__,
            "model": model,
            "architecture": self.architecture
        }
        self.emit(EventType.PROVIDER_CREATED, event_data)

        # Also emit to global bus for system-wide listeners
        from ..events import emit_global
        emit_global(EventType.PROVIDER_CREATED, event_data, source=self.__class__.__name__)

        # Set default token limits if not provided
        self._initialize_token_limits()

    def _track_generation(self, prompt: str, response: Optional[GenerateResponse],
                         start_time: float, success: bool = True,
                         error: Optional[Exception] = None):
        """
        Track generation with telemetry and events.

        Args:
            prompt: Input prompt
            response: Generated response
            start_time: Generation start time
            success: Whether generation succeeded
            error: Error if failed
        """
        latency_ms = (time.time() - start_time) * 1000

        # Emit event to both local and global bus
        event_data = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "success": success,
            "latency_ms": latency_ms,
            "error": str(error) if error else None
        }
        self.emit(EventType.AFTER_GENERATE, event_data)

        from ..events import emit_global
        emit_global(EventType.AFTER_GENERATE, event_data, source=self.__class__.__name__)

        # Track telemetry
        self.telemetry.track_generation(
            provider=self.__class__.__name__,
            model=self.model,
            prompt=prompt,
            response=response.content if response else None,
            tokens=response.usage if response else None,
            latency_ms=latency_ms,
            success=success,
            error=str(error) if error else None
        )

        # Log
        if success:
            self.logger.debug(f"Generation completed in {latency_ms:.2f}ms")
            if response and response.usage:
                self.logger.debug(f"Tokens: {response.usage}")
        else:
            # Only log debug info for model not found errors to avoid duplication
            if isinstance(error, ModelNotFoundError):
                self.logger.debug(f"Model not found: {self.model}")
            else:
                self.logger.error(f"Generation failed: {error}")

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
        # Emit event
        self.emit(EventType.TOOL_CALLED if success else EventType.ERROR_OCCURRED, {
            "tool": tool_name,
            "arguments": arguments,
            "result": str(result)[:100] if result else None,
            "error": str(error) if error else None
        })

        # Track telemetry
        self.telemetry.track_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            error=str(error) if error else None
        )

        # Log
        if success:
            self.logger.info(f"Tool called: {tool_name}")
            self.logger.debug(f"Arguments: {arguments}")
        else:
            self.logger.error(f"Tool call failed: {tool_name} - {error}")

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
                               tools: Optional[List[Dict[str, Any]]] = None,
                               stream: bool = False,
                               **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Generate with integrated telemetry and error handling.
        Providers should override _generate_internal instead of generate.
        """
        start_time = time.time()

        # Emit before event to both local and global bus
        event_data = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "has_tools": bool(tools),
            "stream": stream
        }
        self.emit(EventType.BEFORE_GENERATE, event_data)

        from ..events import emit_global
        emit_global(EventType.BEFORE_GENERATE, event_data, source=self.__class__.__name__)

        try:
            # Call the actual generation (implemented by subclass)
            response = self._generate_internal(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                stream=stream,
                **kwargs
            )

            # Track if not streaming (streaming tracked separately)
            if not stream:
                self._track_generation(prompt, response, start_time, success=True)

            return response

        except Exception as e:
            # Convert to custom exception
            custom_error = self._handle_api_error(e)

            # Track error
            self._track_generation(prompt, None, start_time, success=False, error=custom_error)

            # Emit error event
            self.emit_error(custom_error, {"prompt": prompt})

            # Re-raise custom exception
            raise custom_error

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          stream: bool = False,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Internal generation method to be implemented by subclasses.
        This is called by generate_with_telemetry.
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
        """Get default max_output_tokens from model capabilities"""
        # First try to get from model capabilities
        if hasattr(self, 'model_capabilities') and self.model_capabilities:
            max_output_tokens = self.model_capabilities.get('max_output_tokens')
            if max_output_tokens:
                return max_output_tokens

        # Fallback to default
        return 2048

    def _get_default_context_window(self) -> int:
        """Get default context window for this provider/model from capabilities"""
        # First try to get from model capabilities
        if hasattr(self, 'model_capabilities') and self.model_capabilities:
            context_length = self.model_capabilities.get('context_length')
            if context_length:
                return context_length

        # Fallback to conservative default
        return 8192

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
        # Emit before tool execution event with prevention capability
        event_data = {
            "tool_calls": tool_calls,
            "model": self.model,
            "can_prevent": True
        }
        event = self.emit(EventType.BEFORE_TOOL_EXECUTION, event_data)

        # Check if execution was prevented
        if event.prevented:
            return response  # Return original response without tool execution

        # Execute tools
        tool_results = execute_tools(tool_calls)

        # Emit after tool execution event
        self.emit(EventType.AFTER_TOOL_EXECUTION, {
            "tool_calls": tool_calls,
            "results": tool_results,
            "model": self.model
        })

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
        return GenerateResponse(
            content=response.content + results_text,
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