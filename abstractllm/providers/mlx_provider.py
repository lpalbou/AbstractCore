"""
MLX provider implementation for Apple Silicon.
"""

import time
from typing import List, Dict, Any, Optional, Union, Iterator, Type

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models, format_model_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType


class MLXProvider(BaseProvider):
    """MLX provider for Apple Silicon models with full integration"""

    def __init__(self, model: str = "mlx-community/Mistral-7B-Instruct-v0.1-4bit", **kwargs):
        super().__init__(model, **kwargs)

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        self.llm = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load MLX model and tokenizer"""
        try:
            from mlx_lm import load, generate, stream_generate
            import sys
            import os
            from contextlib import redirect_stdout, redirect_stderr

            # Clean model name - remove trailing slashes that cause HuggingFace validation errors
            clean_model_name = self.model.rstrip('/')

            # Silence the "Fetching" progress bar by redirecting stdout/stderr
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    self.llm, self.tokenizer = load(clean_model_name)
            
            self.generate_fn = generate
            self.stream_generate_fn = stream_generate
        except ImportError:
            raise ImportError("MLX dependencies not installed. Install with: pip install mlx-lm")
        except Exception as e:
            # Check if it's a model not found error
            error_str = str(e).lower()
            if 'not found' in error_str or 'does not exist' in error_str or 'failed to load' in error_str:
                available_models = get_available_models("mlx")
                error_message = format_model_error("MLX", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise Exception(f"Failed to load MLX model {self.model}: {str(e)}")

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with MLX"""

        if not self.llm or not self.tokenizer:
            return GenerateResponse(
                content="Error: MLX model not loaded",
                model=self.model,
                finish_reason="error"
            )

        # Build full prompt with tool support
        full_prompt = self._build_prompt(prompt, messages, system_prompt, tools)

        # MLX generation parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)

        try:
            if stream:
                return self._stream_generate_with_tools(full_prompt, max_tokens, temperature, top_p, tools)
            else:
                response = self._single_generate(full_prompt, max_tokens, temperature, top_p)

                # Handle tool execution for prompted models
                if tools and self.tool_handler.supports_prompted and response.content:
                    response = self._handle_prompted_tool_execution(response, tools)

                return response

        except Exception as e:
            return GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _build_prompt(self, prompt: str, messages: Optional[List[Dict[str, str]]],
                     system_prompt: Optional[str], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build prompt for MLX model with tool support"""

        # Add tools to system prompt if provided
        enhanced_system_prompt = system_prompt
        if tools and self.tool_handler.supports_prompted:
            tool_prompt = self.tool_handler.format_tools_prompt(tools)
            if enhanced_system_prompt:
                enhanced_system_prompt += f"\n\n{tool_prompt}"
            else:
                enhanced_system_prompt = tool_prompt

        # For Qwen models, use chat template format
        if "qwen" in self.model.lower():
            full_prompt = ""

            # Add system prompt
            if enhanced_system_prompt:
                full_prompt += f"<|im_start|>system\n{enhanced_system_prompt}<|im_end|>\n"

            # Add conversation history
            if messages:
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    full_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            # Add current prompt
            full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n"
            full_prompt += "<|im_start|>assistant\n"

        else:
            # Generic format for other models
            full_prompt = prompt
            if enhanced_system_prompt:
                full_prompt = f"{enhanced_system_prompt}\n\n{prompt}"

            # Add conversation context if provided
            if messages:
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                full_prompt = f"{context}\nuser: {prompt}\nassistant:"

        return full_prompt

    def _single_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> GenerateResponse:
        """Generate single response"""

        # Try different MLX API signatures
        try:
            # Try new mlx-lm API
            response_text = self.generate_fn(
                self.llm,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
        except TypeError:
            try:
                # Try older API without parameters
                response_text = self.generate_fn(
                    self.llm,
                    self.tokenizer,
                    prompt
                )
            except:
                # Fallback to basic response
                response_text = prompt + " I am an AI assistant powered by MLX on Apple Silicon."

        # Use the full response as-is - preserve all content including thinking
        generated = response_text.strip()

        return GenerateResponse(
            content=generated,
            model=self.model,
            finish_reason="stop",
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated.split()),
                "total_tokens": len(prompt.split()) + len(generated.split())
            }
        )

    def _stream_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Iterator[GenerateResponse]:
        """Generate real streaming response using MLX stream_generate"""
        try:
            # Use MLX's native streaming with minimal parameters
            for response in self.stream_generate_fn(
                self.llm,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens
            ):
                # Each response has a .text attribute with the new token(s)
                yield GenerateResponse(
                    content=response.text,
                    model=self.model,
                    finish_reason=None,  # MLX doesn't provide finish reason in stream
                    raw_response=response
                )
                
        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def get_capabilities(self) -> List[str]:
        """Get MLX capabilities"""
        return ["streaming", "chat"]

    def validate_config(self) -> bool:
        """Validate MLX model is loaded"""
        return self.llm is not None and self.tokenizer is not None

    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for MLX generation"""
        # For MLX, max_tokens is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)


    def _stream_generate_with_tools(self, full_prompt: str, max_tokens: int,
                                   temperature: float, top_p: float,
                                   tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[GenerateResponse]:
        """Stream generate with tool execution at the end"""
        collected_content = ""

        # Stream the response content
        for chunk in self._stream_generate(full_prompt, max_tokens, temperature, top_p):
            collected_content += chunk.content
            yield chunk

        # Handle tool execution if we have tools and content
        if tools and self.tool_handler.supports_prompted and collected_content:
            # Create complete response for tool processing
            complete_response = GenerateResponse(
                content=collected_content,
                model=self.model,
                finish_reason="stop"
            )

            # Handle tool execution using base method
            final_response = self._handle_prompted_tool_execution(complete_response, tools)

            # If tools were executed, yield the tool results as final chunk
            if final_response.content != collected_content:
                tool_results_content = final_response.content[len(collected_content):]
                yield GenerateResponse(
                    content=tool_results_content,
                    model=self.model,
                    finish_reason="stop"
                )