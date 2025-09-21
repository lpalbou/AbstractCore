"""
Logging utilities for AbstractLLM.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logging(level: str = "INFO",
                  log_file: Optional[Path] = None,
                  format_string: Optional[str] = None):
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs
        format_string: Custom format string
    """
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


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Convenience functions for logging API calls
def log_request(logger: logging.Logger, provider: str, prompt: str,
               params: Optional[dict] = None):
    """Log an API request"""
    logger.debug(f"[{provider}] Request - Prompt: {prompt[:100]}...")
    if params:
        logger.debug(f"[{provider}] Parameters: {params}")


def log_response(logger: logging.Logger, provider: str,
                response: str, tokens: Optional[dict] = None):
    """Log an API response"""
    logger.debug(f"[{provider}] Response: {response[:100]}...")
    if tokens:
        logger.debug(f"[{provider}] Tokens: {tokens}")


def log_error(logger: logging.Logger, provider: str, error: Exception):
    """Log an error"""
    logger.error(f"[{provider}] Error: {type(error).__name__}: {str(error)}")


def log_tool_call(logger: logging.Logger, tool_name: str,
                 arguments: Optional[dict] = None):
    """Log a tool call"""
    logger.info(f"Tool called: {tool_name}")
    if arguments:
        logger.debug(f"Tool arguments: {arguments}")