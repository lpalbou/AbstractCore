"""
Simple logging configuration for AbstractLLM Core.

Provides dual output system with separate console and file levels.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def configure_logging(
    console_level: str = "WARNING",
    file_level: str = "DEBUG",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True
):
    """
    Configure dual output logging system.

    Args:
        console_level: Logging level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: Logging level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to ~/.abstractllm/logs)
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
    """
    # Convert string levels to logging constants
    console_level_const = getattr(logging, console_level.upper())
    file_level_const = getattr(logging, file_level.upper())

    # Set up log directory
    if log_dir is None:
        log_dir = Path.home() / ".abstractllm" / "logs"
    else:
        log_dir = Path(log_dir)

    # Ensure log directory exists
    if enable_file:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger to lowest level so handlers can filter
    root_logger.setLevel(min(console_level_const, file_level_const) if enable_file else console_level_const)

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level_const)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if enable_file:
        log_file = log_dir / "abstractllm.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level_const)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: console={console_level if enable_console else 'disabled'}, "
                f"file={file_level if enable_file else 'disabled'}")
    if enable_file:
        logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Configure with sensible defaults on import
if not logging.getLogger().handlers:
    configure_logging()