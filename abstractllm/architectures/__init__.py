"""
Model architecture detection and configuration.
"""

from typing import Dict, Any, Optional
from enum import Enum


class Architecture(Enum):
    """Known model architectures"""
    GPT = "gpt"
    CLAUDE = "claude"
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"
    STARCODER = "starcoder"
    CODEGEN = "codegen"
    BLOOM = "bloom"
    FALCON = "falcon"
    VICUNA = "vicuna"
    ALPACA = "alpaca"
    UNKNOWN = "unknown"


def detect_architecture(model_name: str) -> Architecture:
    """
    Detect model architecture from model name.

    Args:
        model_name: Name of the model

    Returns:
        Detected architecture
    """
    model_lower = model_name.lower()

    # GPT models
    if "gpt" in model_lower:
        return Architecture.GPT

    # Claude models
    if "claude" in model_lower:
        return Architecture.CLAUDE

    # Llama models
    if "llama" in model_lower:
        return Architecture.LLAMA

    # Mistral models
    if "mistral" in model_lower or "mixtral" in model_lower:
        return Architecture.MISTRAL

    # Qwen models
    if "qwen" in model_lower:
        return Architecture.QWEN

    # Gemma models
    if "gemma" in model_lower:
        return Architecture.GEMMA

    # Phi models
    if "phi" in model_lower:
        return Architecture.PHI

    # StarCoder models
    if "starcoder" in model_lower:
        return Architecture.STARCODER

    # CodeGen models
    if "codegen" in model_lower:
        return Architecture.CODEGEN

    # BLOOM models
    if "bloom" in model_lower:
        return Architecture.BLOOM

    # Falcon models
    if "falcon" in model_lower:
        return Architecture.FALCON

    # Vicuna models
    if "vicuna" in model_lower:
        return Architecture.VICUNA

    # Alpaca models
    if "alpaca" in model_lower:
        return Architecture.ALPACA

    return Architecture.UNKNOWN


def get_architecture_config(architecture: Architecture) -> Dict[str, Any]:
    """
    Get configuration for a specific architecture.

    Args:
        architecture: Model architecture

    Returns:
        Configuration dictionary
    """
    configs = {
        Architecture.GPT: {
            "supports_tools": True,
            "tool_format": "json",
            "context_window": 8192,
            "supports_system_prompt": True,
        },
        Architecture.CLAUDE: {
            "supports_tools": True,
            "tool_format": "xml",
            "context_window": 200000,
            "supports_system_prompt": True,
        },
        Architecture.LLAMA: {
            "supports_tools": False,
            "tool_format": None,
            "context_window": 4096,
            "supports_system_prompt": True,
        },
        Architecture.QWEN: {
            "supports_tools": True,
            "tool_format": "special_tokens",
            "tool_start_token": "<|tool_call|>",
            "tool_end_token": "<|tool_call_end|>",
            "context_window": 32768,
            "supports_system_prompt": True,
        },
        Architecture.MISTRAL: {
            "supports_tools": False,
            "tool_format": None,
            "context_window": 8192,
            "supports_system_prompt": True,
        },
        Architecture.GEMMA: {
            "supports_tools": False,
            "tool_format": None,
            "context_window": 8192,
            "supports_system_prompt": True,
        },
        Architecture.PHI: {
            "supports_tools": False,
            "tool_format": None,
            "context_window": 2048,
            "supports_system_prompt": True,
        },
        Architecture.UNKNOWN: {
            "supports_tools": False,
            "tool_format": None,
            "context_window": 2048,
            "supports_system_prompt": False,
        }
    }

    return configs.get(architecture, configs[Architecture.UNKNOWN])