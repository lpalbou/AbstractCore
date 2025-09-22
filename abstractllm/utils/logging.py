"""
Logging utilities for AbstractLLM.

This module provides both legacy compatibility and modern structured logging.
For new code, prefer the structured logging from .structured_logging module.
"""

import logging
import sys
import warnings
from typing import Optional
from pathlib import Path

# Import new structured logging system
from .structured_logging import (
    get_logger as get_structured_logger,
    configure_logging as configure_structured_logging,
    capture_session,
    suppress_stdout_stderr,
    STRUCTLOG_AVAILABLE
)


def setup_logging(level: str = "INFO",
                  log_file: Optional[Path] = None,
                  format_string: Optional[str] = None):
    """
    Setup logging configuration (legacy).

    DEPRECATED: Use configure_structured_logging() instead.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs
        format_string: Custom format string
    """
    warnings.warn(
        "setup_logging() is deprecated. Use configure_structured_logging() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    logging.getLogger().addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str):
    """
    Get a logger instance.

    This function now returns a StructuredLogger if available,
    otherwise falls back to standard logging.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance (StructuredLogger if available, else standard Logger)
    """
    if STRUCTLOG_AVAILABLE:
        return get_structured_logger(name)
    else:
        return logging.getLogger(name)


# Legacy convenience functions (deprecated)
def log_request(logger, provider: str, prompt: str, params: Optional[dict] = None):
    """
    Log an API request (legacy).

    DEPRECATED: Use logger.log_generation() instead.
    """
    warnings.warn(
        "log_request() is deprecated. Use logger.log_generation() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if hasattr(logger, 'debug'):
        logger.debug(f"[{provider}] Request - Prompt: {prompt[:100]}...")
        if params:
            logger.debug(f"[{provider}] Parameters: {params}")


def log_response(logger, provider: str, response: str, tokens: Optional[dict] = None):
    """
    Log an API response (legacy).

    DEPRECATED: Use logger.log_generation() instead.
    """
    warnings.warn(
        "log_response() is deprecated. Use logger.log_generation() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if hasattr(logger, 'debug'):
        logger.debug(f"[{provider}] Response: {response[:100]}...")
        if tokens:
            logger.debug(f"[{provider}] Tokens: {tokens}")


def log_error(logger, provider: str, error: Exception):
    """
    Log an error (legacy).

    DEPRECATED: Use logger.error() with structured data instead.
    """
    warnings.warn(
        "log_error() is deprecated. Use logger.error() with structured data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if hasattr(logger, 'error'):
        logger.error(f"[{provider}] Error: {type(error).__name__}: {str(error)}")


def log_tool_call(logger, tool_name: str, arguments: Optional[dict] = None):
    """
    Log a tool call (legacy).

    DEPRECATED: Use logger.log_tool_call() instead.
    """
    warnings.warn(
        "log_tool_call() is deprecated. Use logger.log_tool_call() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if hasattr(logger, 'info'):
        logger.info(f"Tool called: {tool_name}")
        if arguments and hasattr(logger, 'debug'):
            logger.debug(f"Tool arguments: {arguments}")


# Re-export key functions for convenience
configure_logging = configure_structured_logging


# Legacy alias
def get_telemetry():
    """
    Legacy telemetry function.

    DEPRECATED: Use structured logging instead.
    """
    warnings.warn(
        "get_telemetry() is deprecated. Use structured logging instead.",
        DeprecationWarning,
        stacklevel=2
    )

    class LegacyTelemetry:
        def track_generation(self, **kwargs):
            pass
        def track_tool_call(self, **kwargs):
            pass

    return LegacyTelemetry()