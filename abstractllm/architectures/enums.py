"""
Enums for architecture detection and model capabilities.
"""

from enum import Enum


class ToolCallFormat(Enum):
    """Tool call formats supported by different architectures."""

    # Native API formats
    NATIVE = "native"                   # OpenAI/Anthropic native API

    # Prompted formats
    JSON = "json"                       # JSON object format
    XML = "xml"                         # XML wrapped format
    SPECIAL_TOKEN = "special_token"     # <|tool_call|> format
    PYTHONIC = "pythonic"               # Python function syntax

    # Legacy/unsupported
    NONE = "none"                       # No tool support


class ModelType(Enum):
    """High-level model types."""

    CHAT = "chat"                       # Chat/conversation models
    INSTRUCT = "instruct"               # Instruction-following models
    BASE = "base"                       # Base/foundation models
    CODE = "code"                       # Code-specialized models
    VISION = "vision"                   # Vision/multimodal models
    EMBEDDING = "embedding"             # Embedding models
    UNKNOWN = "unknown"                 # Unknown type


class ArchitectureFamily(Enum):
    """Known architecture families."""

    # Commercial APIs
    OPENAI = "openai"                   # GPT family
    ANTHROPIC = "anthropic"             # Claude family

    # Open source families
    LLAMA = "llama"                     # Meta LLaMA family
    QWEN = "qwen"                       # Alibaba Qwen family
    MISTRAL = "mistral"                 # Mistral AI family
    GEMMA = "gemma"                     # Google Gemma family
    PHI = "phi"                         # Microsoft Phi family
    GLM = "glm"                         # Zhipu GLM family
    DEEPSEEK = "deepseek"               # DeepSeek family
    GRANITE = "granite"                 # IBM Granite family

    # Other
    GENERIC = "generic"                 # Generic/unknown

    @classmethod
    def from_string(cls, name: str) -> "ArchitectureFamily":
        """Convert string to ArchitectureFamily enum."""
        try:
            return cls(name.lower())
        except ValueError:
            return cls.GENERIC