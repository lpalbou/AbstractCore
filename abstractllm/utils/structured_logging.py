"""
Structured logging with verbatim capture for AbstractLLM.

This module provides a comprehensive logging system with:
- Structured logging using structlog
- Verbatim capture of prompts and responses
- Separate console and file logging levels
- Context binding for request/session tracking
- JSON output for machine readability
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from contextlib import contextmanager

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# Global configuration
class LogConfig:
    """Global logging configuration singleton."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Default configuration
        self.console_level = logging.WARNING
        self.file_level = logging.DEBUG
        self.log_dir = None
        self.verbatim_enabled = True
        self.console_json = False
        self.file_json = True
        self._initialized = True

    def configure(self,
                  console_level: Optional[int] = None,
                  file_level: Optional[int] = None,
                  log_dir: Optional[str] = None,
                  verbatim_enabled: bool = True,
                  console_json: bool = False,
                  file_json: bool = True):
        """
        Configure global logging settings.

        Args:
            console_level: Logging level for console (None to disable)
            file_level: Logging level for files (None to disable)
            log_dir: Directory for log files (None to disable file logging)
            verbatim_enabled: Whether to capture full prompts/responses
            console_json: Use JSON format for console output
            file_json: Use JSON format for file output
        """
        if console_level is not None:
            self.console_level = console_level
        if file_level is not None:
            self.file_level = file_level
        if log_dir is not None:
            self.log_dir = log_dir
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.verbatim_enabled = verbatim_enabled
        self.console_json = console_json
        self.file_json = file_json

        # Reinitialize structlog with new config
        self._setup_structlog()

    def _setup_structlog(self):
        """Setup structlog with current configuration."""
        # Setup standard logging handlers first (always, regardless of structlog availability)
        self._setup_logging_handlers()

        if not STRUCTLOG_AVAILABLE:
            return

        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        # Add JSON processor for structured output
        if self.console_json or self.file_json:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def _setup_logging_handlers(self):
        """Setup standard logging handlers for file and console output."""
        # Get root logger
        root_logger = logging.getLogger()

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        if self.console_level is not None:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level)

            if self.console_json:
                console_formatter = logging.Formatter('%(message)s')
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%H:%M:%S'
                )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if self.log_dir and self.file_level is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(self.log_dir) / f"abstractllm_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.file_level)

            if self.file_json:
                file_formatter = logging.Formatter('%(message)s')
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Set root logger level to the most verbose level
        if self.console_level is not None and self.file_level is not None:
            root_logger.setLevel(min(self.console_level, self.file_level))
        elif self.console_level is not None:
            root_logger.setLevel(self.console_level)
        elif self.file_level is not None:
            root_logger.setLevel(self.file_level)


# Global config instance
_config = LogConfig()


class VerbatimCapture:
    """Captures verbatim prompt/response pairs for analysis."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.session_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_file = self.log_dir / f"verbatim_{timestamp}.jsonl"

    def capture_interaction(self,
                           provider: str,
                           model: str,
                           prompt: str,
                           response: str,
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Capture a complete prompt/response interaction.

        Args:
            provider: Provider name (e.g., 'openai', 'ollama')
            model: Model name
            prompt: Full prompt sent to model
            response: Full response from model
            metadata: Additional metadata (tokens, timing, etc.)
        """
        if not _config.verbatim_enabled or not self.session_file:
            return

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }

        try:
            with open(self.session_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
        except Exception as e:
            # Fail silently to not disrupt main application
            pass


class StructuredLogger:
    """
    Enhanced logger with structured logging and verbatim capture.
    """

    def __init__(self, name: str):
        self.name = name
        self._setup_logger()

        # Verbatim capture
        self.verbatim = VerbatimCapture(_config.log_dir)

        # Context for binding additional fields
        self._context = {}

    def _setup_logger(self):
        """Setup the underlying logger."""
        if STRUCTLOG_AVAILABLE:
            self.logger = structlog.get_logger(self.name)
        else:
            # Fallback to standard logging - use the configured root logger system
            self.logger = logging.getLogger(self.name)
            # Don't add handlers or set level - let it inherit from root logger
            # which was properly configured by _setup_logging_handlers()

    def bind(self, **kwargs) -> 'StructuredLogger':
        """
        Bind additional context to all log messages.

        Args:
            **kwargs: Key-value pairs to add to log context

        Returns:
            New logger instance with bound context
        """
        new_logger = StructuredLogger(self.name)
        new_logger._context = {**self._context, **kwargs}
        if STRUCTLOG_AVAILABLE:
            new_logger.logger = self.logger.bind(**kwargs)
        return new_logger

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("critical", message, **kwargs)

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""
        # Merge context
        log_data = {**self._context, **kwargs}

        if STRUCTLOG_AVAILABLE:
            # For structlog, pass as keyword arguments
            log_method = getattr(self.logger, level)
            log_method(message, **log_data)
        else:
            # Fallback logging
            extra_info = ", ".join([f"{k}={v}" for k, v in log_data.items()])
            full_message = f"{message}" + (f" | {extra_info}" if extra_info else "")
            getattr(self.logger, level)(full_message)

    def log_generation(self,
                      provider: str,
                      model: str,
                      prompt: str,
                      response: str,
                      tokens: Optional[Dict[str, Any]] = None,
                      latency_ms: Optional[float] = None,
                      success: bool = True,
                      error: Optional[str] = None):
        """
        Log a complete generation with verbatim capture.

        Args:
            provider: Provider name
            model: Model name
            prompt: Full prompt
            response: Full response
            tokens: Token usage information
            latency_ms: Response latency in milliseconds
            success: Whether generation succeeded
            error: Error message if failed
        """
        # Structured log entry
        log_data = {
            "event_type": "generation",
            "provider": provider,
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "success": success,
            "error": error
        }

        if success:
            self.info("Generation completed", **log_data)
        else:
            self.error("Generation failed", **log_data)

        # Verbatim capture
        self.verbatim.capture_interaction(
            provider=provider,
            model=model,
            prompt=prompt,
            response=response,
            metadata={
                "tokens": tokens,
                "latency_ms": latency_ms,
                "success": success,
                "error": error
            }
        )

    def log_tool_call(self,
                     tool_name: str,
                     arguments: Dict[str, Any],
                     result: Optional[str] = None,
                     success: bool = True,
                     error: Optional[str] = None,
                     execution_time_ms: Optional[float] = None):
        """
        Log tool call execution.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool execution result
            success: Whether tool call succeeded
            error: Error message if failed
            execution_time_ms: Execution time in milliseconds
        """
        log_data = {
            "event_type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
            "result_length": len(str(result)) if result else 0,
            "success": success,
            "error": error,
            "execution_time_ms": execution_time_ms
        }

        if success:
            self.info(f"Tool executed: {tool_name}", **log_data)
        else:
            self.error(f"Tool failed: {tool_name}", **log_data)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


def configure_logging(
    console_level: Optional[int] = logging.WARNING,
    file_level: Optional[int] = logging.DEBUG,
    log_dir: Optional[str] = None,
    verbatim_enabled: bool = True,
    console_json: bool = False,
    file_json: bool = True
):
    """
    Configure the global logging system.

    Args:
        console_level: Logging level for console output (None to disable)
        file_level: Logging level for file output (None to disable)
        log_dir: Directory for log files (None to disable file logging)
        verbatim_enabled: Whether to capture full prompts/responses
        console_json: Use JSON format for console output
        file_json: Use JSON format for file output

    Example:
        >>> # Development: Debug to console, everything to file
        >>> configure_logging(
        ...     console_level=logging.DEBUG,
        ...     file_level=logging.DEBUG,
        ...     log_dir="logs",
        ...     verbatim_enabled=True
        ... )
        >>>
        >>> # Production: Warnings to console, debug to file
        >>> configure_logging(
        ...     console_level=logging.WARNING,
        ...     file_level=logging.DEBUG,
        ...     log_dir="/var/log/abstractllm",
        ...     verbatim_enabled=False
        ... )
    """
    _config.configure(
        console_level=console_level,
        file_level=file_level,
        log_dir=log_dir,
        verbatim_enabled=verbatim_enabled,
        console_json=console_json,
        file_json=file_json
    )


@contextmanager
def capture_session(session_id: str):
    """
    Context manager for capturing a complete session.

    Args:
        session_id: Unique session identifier

    Example:
        >>> with capture_session("user_123_session"):
        ...     response = llm.generate("Hello")
    """
    logger = get_logger("session").bind(session_id=session_id)
    start_time = time.time()

    logger.info("Session started")
    try:
        yield logger
    finally:
        duration = time.time() - start_time
        logger.info("Session ended", duration_seconds=duration)


# Context manager for suppressing output (from original)
@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr output.
    Useful for silencing verbose library initialization messages.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr