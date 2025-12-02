"""
Model download API with async progress reporting.

Provides a provider-agnostic interface for downloading models from Ollama,
HuggingFace Hub, and MLX with streaming progress updates.
"""

import json
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

import httpx


class DownloadStatus(Enum):
    """Download progress status."""

    STARTING = "starting"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class DownloadProgress:
    """Progress information for model download."""

    status: DownloadStatus
    message: str
    percent: Optional[float] = None  # 0-100
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None


async def download_model(
    provider: str,
    model: str,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> AsyncIterator[DownloadProgress]:
    """
    Download a model with async progress reporting.

    This function provides a unified interface for downloading models across
    different providers. Progress updates are yielded as DownloadProgress
    dataclasses that include status, message, and optional progress percentage.

    Args:
        provider: Provider name ("ollama", "huggingface", "mlx")
        model: Model identifier:
            - Ollama: "llama3:8b", "gemma3:1b", etc.
            - HuggingFace/MLX: "meta-llama/Llama-2-7b", "mlx-community/Qwen3-4B-4bit", etc.
        token: Optional auth token (for HuggingFace gated models)
        base_url: Optional custom base URL (for Ollama, default: http://localhost:11434)

    Yields:
        DownloadProgress: Progress updates with status, message, and optional metrics

    Raises:
        ValueError: If provider doesn't support downloads (OpenAI, Anthropic, LMStudio)
        httpx.HTTPStatusError: If Ollama server returns error
        Exception: Various exceptions from HuggingFace Hub (RepositoryNotFoundError, etc.)

    Examples:
        Download Ollama model:
            >>> async for progress in download_model("ollama", "gemma3:1b"):
            ...     print(f"{progress.status.value}: {progress.message}")
            ...     if progress.percent:
            ...         print(f"  Progress: {progress.percent:.1f}%")

        Download HuggingFace model with token:
            >>> async for progress in download_model(
            ...     "huggingface",
            ...     "meta-llama/Llama-2-7b",
            ...     token="hf_..."
            ... ):
            ...     print(f"{progress.message}")
    """
    provider_lower = provider.lower()

    if provider_lower == "ollama":
        async for progress in _download_ollama(model, base_url):
            yield progress
    elif provider_lower in ("huggingface", "mlx"):
        async for progress in _download_huggingface(model, token):
            yield progress
    else:
        raise ValueError(
            f"Provider '{provider}' does not support model downloads. "
            f"Supported providers: ollama, huggingface, mlx. "
            f"Note: OpenAI and Anthropic are cloud-only; LMStudio has no download API."
        )


async def _download_ollama(
    model: str,
    base_url: Optional[str] = None,
) -> AsyncIterator[DownloadProgress]:
    """
    Download model from Ollama using /api/pull endpoint.

    Args:
        model: Ollama model name (e.g., "llama3:8b", "gemma3:1b")
        base_url: Ollama server URL (default: http://localhost:11434)

    Yields:
        DownloadProgress with status updates from Ollama streaming response
    """
    url = (base_url or "http://localhost:11434").rstrip("/")

    yield DownloadProgress(
        status=DownloadStatus.STARTING, message=f"Pulling {model} from Ollama..."
    )

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{url}/api/pull",
                json={"name": model, "stream": True},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    status_msg = data.get("status", "")

                    # Parse progress from Ollama response
                    # Format: {"status": "downloading...", "total": 123, "completed": 45}
                    if "total" in data and "completed" in data:
                        total = data["total"]
                        completed = data["completed"]
                        percent = (completed / total * 100) if total > 0 else 0

                        yield DownloadProgress(
                            status=DownloadStatus.DOWNLOADING,
                            message=status_msg,
                            percent=percent,
                            downloaded_bytes=completed,
                            total_bytes=total,
                        )
                    elif status_msg == "success":
                        yield DownloadProgress(
                            status=DownloadStatus.COMPLETE,
                            message=f"Successfully pulled {model}",
                            percent=100.0,
                        )
                    elif "verifying" in status_msg.lower():
                        yield DownloadProgress(
                            status=DownloadStatus.VERIFYING,
                            message=status_msg,
                        )
                    else:
                        # Other status messages (pulling manifest, etc.)
                        yield DownloadProgress(
                            status=DownloadStatus.DOWNLOADING,
                            message=status_msg,
                        )

    except httpx.HTTPStatusError as e:
        yield DownloadProgress(
            status=DownloadStatus.ERROR,
            message=f"Ollama server error: {e.response.status_code} - {e.response.text}",
        )
    except httpx.ConnectError:
        yield DownloadProgress(
            status=DownloadStatus.ERROR,
            message=f"Cannot connect to Ollama server at {url}. Is Ollama running?",
        )
    except Exception as e:
        yield DownloadProgress(
            status=DownloadStatus.ERROR,
            message=f"Download failed: {str(e)}",
        )


async def _download_huggingface(
    model: str,
    token: Optional[str] = None,
) -> AsyncIterator[DownloadProgress]:
    """
    Download model from HuggingFace Hub.

    Args:
        model: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b")
        token: Optional HuggingFace token (required for gated models)

    Yields:
        DownloadProgress with status updates
    """
    yield DownloadProgress(
        status=DownloadStatus.STARTING,
        message=f"Downloading {model} from HuggingFace Hub...",
    )

    try:
        # Import here to make huggingface_hub optional
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        yield DownloadProgress(
            status=DownloadStatus.ERROR,
            message=(
                "huggingface_hub is not installed. "
                "Install with: pip install abstractcore[huggingface]"
            ),
        )
        return

    try:
        # Run blocking download in thread
        # Note: snapshot_download doesn't have built-in async progress callbacks
        # We provide start and completion messages
        await asyncio.to_thread(
            snapshot_download,
            repo_id=model,
            token=token,
        )

        yield DownloadProgress(
            status=DownloadStatus.COMPLETE,
            message=f"Successfully downloaded {model}",
            percent=100.0,
        )

    except RepositoryNotFoundError:
        yield DownloadProgress(
            status=DownloadStatus.ERROR,
            message=f"Model '{model}' not found on HuggingFace Hub",
        )
    except GatedRepoError:
        yield DownloadProgress(
            status=DownloadStatus.ERROR,
            message=(
                f"Model '{model}' requires authentication. "
                f"Provide a HuggingFace token via the 'token' parameter."
            ),
        )
    except Exception as e:
        yield DownloadProgress(
            status=DownloadStatus.ERROR,
            message=f"Download failed: {str(e)}",
        )
