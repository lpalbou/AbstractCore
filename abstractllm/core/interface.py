"""
Core interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator
from .types import GenerateResponse, Message


class AbstractLLMInterface(ABC):
    """
    Abstract base class for all LLM providers.
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def generate(self,
                prompt: str,
                messages: Optional[List[Dict[str, str]]] = None,
                system_prompt: Optional[str] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                stream: bool = False,
                **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Generate response from the LLM.

        Args:
            prompt: The input prompt
            messages: Optional conversation history
            system_prompt: Optional system prompt
            tools: Optional list of available tools
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            GenerateResponse or iterator of GenerateResponse for streaming
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        pass

    def validate_config(self) -> bool:
        """Validate provider configuration"""
        return True

    def get_token_limit(self) -> Optional[int]:
        """Get maximum token limit for this model"""
        return None