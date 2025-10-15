"""
JSON-based architecture detection for message formatting and communication patterns.

Architecture = HOW to communicate with a model family (templates, tokens, formats).
Capabilities = WHAT a specific model can do (tools, context, vision).
"""

from .detection import (
    detect_architecture,
    detect_model_type,
    get_architecture_format,
    get_model_capabilities,
    format_messages,
    # Convenience functions
    supports_tools,
    supports_vision,
    supports_audio,
    supports_embeddings,
    get_context_limits,
    is_instruct_model,
)

from .enums import (
    ToolCallFormat,
    ModelType,
    ArchitectureFamily,
)

__all__ = [
    "detect_architecture",
    "detect_model_type",
    "get_architecture_format",
    "get_model_capabilities",
    "format_messages",
    # Convenience functions
    "supports_tools",
    "supports_vision",
    "supports_audio",
    "supports_embeddings",
    "get_context_limits",
    "is_instruct_model",
    # Enums
    "ToolCallFormat",
    "ModelType",
    "ArchitectureFamily",
]