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
from ..exceptions import ProviderAPIError, ModelNotFoundError, format_model_error
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
                available_models = self.list_available_models()
                error_message = format_model_error("MLX", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise Exception(f"Failed to load MLX model {self.model}: {str(e)}")

    def unload(self) -> None:
        """
        Unload the MLX model from memory.

        Clears model and tokenizer references and forces garbage collection
        to free GPU/CPU memory immediately.
        """
        import gc
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                # Clear MLX model
                del self.llm
                self.llm = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # Clear tokenizer
                del self.tokenizer
                self.tokenizer = None

            if hasattr(self, 'generate_fn'):
                self.generate_fn = None

            if hasattr(self, 'stream_generate_fn'):
                self.stream_generate_fn = None

            # Force garbage collection to free memory immediately
            gc.collect()
        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

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
                return self._stream_generate_with_tools(full_prompt, max_tokens, temperature, top_p, tools, kwargs.get('tool_call_tags'))
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
            usage=self._calculate_usage(prompt, generated)
        )

    def _calculate_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Calculate token usage using centralized token utilities."""
        from ..utils.token_utils import TokenUtils
        
        prompt_tokens = TokenUtils.estimate_tokens(prompt, self.model)
        completion_tokens = TokenUtils.estimate_tokens(response, self.model)
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

    def _stream_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float, tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Generate real streaming response using MLX stream_generate with tool tag rewriting support"""
        try:
            # Initialize tool tag rewriter if needed
            rewriter = None
            buffer = ""
            if tool_call_tags:
                try:
                    from ..tools.tag_rewriter import create_tag_rewriter
                    rewriter = create_tag_rewriter(tool_call_tags)
                except ImportError:
                    pass

            # Use MLX's native streaming with minimal parameters
            for response in self.stream_generate_fn(
                self.llm,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens
            ):
                # Each response has a .text attribute with the new token(s)
                content = response.text
                
                # Apply tool tag rewriting if enabled
                if rewriter and content:
                    rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                    content = rewritten_content
                
                yield GenerateResponse(
                    content=content,
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
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Stream generate with tool execution at the end"""
        collected_content = ""

        # Stream the response content
        for chunk in self._stream_generate(full_prompt, max_tokens, temperature, top_p, tool_call_tags):
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

    @classmethod
    def list_available_models(cls, **kwargs) -> List[str]:
        """List available MLX models from HuggingFace cache."""
        from pathlib import Path

        try:
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if not hf_cache.exists():
                return []

            models = []
            for item in hf_cache.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Convert models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit to mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
                    model_name = item.name.replace("models--", "").replace("--", "/")

                    # Include ANY model with "mlx" in the name (case-insensitive)
                    # This captures: mlx-community/*, */mlx-*, *-mlx-*, etc.
                    if "mlx" in model_name.lower():
                        models.append(model_name)

            return sorted(models)

        except Exception:
            return []