"""
AbstractCore Configuration Module

Provides configuration management and command-line interface for AbstractCore.
"""

from .vision_config import handle_vision_commands, add_vision_arguments
from .manager import get_config_manager

__all__ = ['handle_vision_commands', 'add_vision_arguments', 'get_config_manager']