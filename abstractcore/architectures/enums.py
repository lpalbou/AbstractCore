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

    # Open source families - LLaMA variants
    LLAMA2 = "llama2"                   # Meta LLaMA 2 family
    LLAMA3 = "llama3"                   # Meta LLaMA 3 family
    LLAMA3_1 = "llama3_1"               # Meta LLaMA 3.1 family
    LLAMA3_2 = "llama3_2"               # Meta LLaMA 3.2 family
    LLAMA3_3 = "llama3_3"               # Meta LLaMA 3.3 family
    LLAMA4 = "llama4"                   # Meta LLaMA 4 family

    # Qwen variants
    QWEN2 = "qwen2"                     # Alibaba Qwen2 family
    QWEN2_5 = "qwen2_5"                 # Alibaba Qwen2.5 family
    QWEN2_VL = "qwen2_vl"               # Alibaba Qwen2-VL family
    QWEN3 = "qwen3"                     # Alibaba Qwen3 family
    QWEN3_MOE = "qwen3_moe"             # Alibaba Qwen3 MoE family
    QWEN3_NEXT = "qwen3_next"           # Alibaba Qwen3-Next family
    QWEN3_VL = "qwen3_vl"               # Alibaba Qwen3-VL family

    # Mistral variants
    MISTRAL = "mistral"                 # Mistral AI base family
    MIXTRAL = "mixtral"                 # Mistral AI MoE family
    MISTRAL_LARGE = "mistral_large"     # Mistral AI Large models
    CODESTRAL = "codestral"             # Mistral AI Code models

    # Gemma variants
    GEMMA = "gemma"                     # Google Gemma v1 family
    GEMMA2 = "gemma2"                   # Google Gemma 2 family
    GEMMA3 = "gemma3"                   # Google Gemma 3 family
    GEMMA3N = "gemma3n"                 # Google Gemma 3n family
    CODEGEMMA = "codegemma"             # Google CodeGemma family
    PALIGEMMA = "paligemma"             # Google PaliGemma family

    # GLM variants
    GLM4 = "glm4"                       # Zhipu GLM-4 family
    GLM4_MOE = "glm4_moe"               # Zhipu GLM-4.5+ MoE family

    # Other families
    PHI = "phi"                         # Microsoft Phi family
    DEEPSEEK = "deepseek"               # DeepSeek family
    GRANITE = "granite"                 # IBM Granite family
    SEED_OSS = "seed_oss"               # ByteDance Seed-OSS family
    YI = "yi"                           # 01.AI Yi family
    CLAUDE = "claude"                   # Anthropic Claude (compatibility)

    # Generic
    GENERIC = "generic"                 # Generic/unknown

    @classmethod
    def from_string(cls, name: str) -> "ArchitectureFamily":
        """Convert string to ArchitectureFamily enum."""
        try:
            return cls(name.lower())
        except ValueError:
            return cls.GENERIC