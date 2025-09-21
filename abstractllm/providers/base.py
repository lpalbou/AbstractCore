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
from ..architectures import detect_architecture, get_architecture_config


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

        # Detect architecture
        self.architecture = detect_architecture(model)
        self.architecture_config = get_architecture_config(self.architecture)

        # Emit provider created event to both local and global bus
        event_data = {
            "provider": self.__class__.__name__,
            "model": model,
            "architecture": self.architecture.value
        }
        self.emit(EventType.PROVIDER_CREATED, event_data)

        # Also emit to global bus for system-wide listeners
        from ..events import emit_global
        emit_global(EventType.PROVIDER_CREATED, event_data, source=self.__class__.__name__)

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