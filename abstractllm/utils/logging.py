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


# Context managers for suppressing output
import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr output.
    Useful for silencing verbose library initialization messages.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@contextmanager
def suppress_stderr():
    """
    Context manager to suppress stderr output only.
    Useful for silencing warnings while keeping stdout intact.
    """
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class StderrToLogger:
    """
    Context manager to redirect stderr to a logger.
    This allows us to capture and control library warnings/errors through our logging system.
    """

    def __init__(self, logger: logging.Logger, debug: bool = False):
        self.logger = logger
        self.debug = debug
        self.old_stderr = None
        self.buffer = []

    def write(self, msg):
        """Write method for file-like interface"""
        if msg and msg.strip():
            # Choose log level based on debug setting
            if self.debug:
                # In debug mode, show all stderr output at INFO level
                self.logger.info(f"[stderr] {msg.strip()}")
            else:
                # In non-debug mode, log at DEBUG level (won't show unless logger debug is enabled)
                self.logger.debug(f"[stderr] {msg.strip()}")

    def flush(self):
        """Flush method for file-like interface"""
        pass

    def __enter__(self):
        """Enter context manager"""
        self.old_stderr = sys.stderr
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        sys.stderr = self.old_stderr


@contextmanager
def redirect_stderr_to_logger(logger: logging.Logger, debug: bool = False):
    """
    Context manager to redirect stderr to a logger.

    Args:
        logger: Logger instance to receive stderr output
        debug: If True, logs at INFO level; if False, logs at DEBUG level
    """
    with StderrToLogger(logger, debug) as stderr_logger:
        yield stderr_logger