"""
Enums for AbstractLLM.
"""

from enum import Enum


class MessageRole(Enum):
    """Message roles in conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ModelParameter(Enum):
    """Standard model parameters"""
    MODEL = "model"
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    TOP_P = "top_p"
    TOP_K = "top_k"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    SEED = "seed"


class ModelCapability(Enum):
    """Model capabilities"""
    CHAT = "chat"
    TOOLS = "tools"
    VISION = "vision"
    STREAMING = "streaming"
    ASYNC = "async"
    JSON_MODE = "json_mode"