"""
AbstractCore CLI Module

Provides command-line interface for AbstractCore configuration and tools.
"""

from .vision_config import handle_vision_commands, add_vision_arguments

__all__ = ['handle_vision_commands', 'add_vision_arguments']