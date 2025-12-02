"""
Tests for model download API.

All tests use REAL implementations (no mocking) per project requirements.
Tests require actual services to be running:
- Ollama: Local server at http://localhost:11434
- HuggingFace: Internet connection to HuggingFace Hub
"""
import pytest
from abstractcore import download_model, DownloadProgress, DownloadStatus


class TestOllamaDownload:
    """Test Ollama model downloads via /api/pull endpoint."""

    @pytest.mark.asyncio
    async def test_download_small_model(self):
        """Test downloading a small Ollama model with real server."""
        progress_updates = []

        async for progress in download_model("ollama", "gemma3:1b"):
            progress_updates.append(progress)
            assert isinstance(progress, DownloadProgress)
            assert isinstance(progress.status, DownloadStatus)
            assert isinstance(progress.message, str)

            # Print progress for visibility
            print(f"{progress.status.value}: {progress.message}", end="")
            if progress.percent:
                print(f" ({progress.percent:.1f}%)")
            else:
                print()

        # Verify we got at least start and complete
        assert len(progress_updates) >= 2, "Should have at least starting and complete messages"
        assert progress_updates[0].status == DownloadStatus.STARTING
        assert progress_updates[-1].status == DownloadStatus.COMPLETE
        assert progress_updates[-1].percent == 100.0

    @pytest.mark.asyncio
    async def test_download_with_custom_base_url(self):
        """Test download with custom Ollama base URL."""
        progress_updates = []

        async for progress in download_model(
            "ollama",
            "gemma3:1b",
            base_url="http://localhost:11434"
        ):
            progress_updates.append(progress)

        assert progress_updates[-1].status == DownloadStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_download_nonexistent_model(self):
        """Test downloading non-existent model returns error."""
        progress_updates = []

        async for progress in download_model("ollama", "nonexistent-model-xyz:1b"):
            progress_updates.append(progress)

        # Should get error status (model not found or server error)
        # Note: Ollama may return different error formats
        has_error = any(p.status == DownloadStatus.ERROR for p in progress_updates)
        assert has_error or progress_updates[-1].status != DownloadStatus.COMPLETE


class TestHuggingFaceDownload:
    """Test HuggingFace/MLX model downloads via snapshot_download."""

    @pytest.mark.asyncio
    async def test_download_small_model(self):
        """Test downloading a small HuggingFace model."""
        progress_updates = []

        async for progress in download_model(
            "huggingface",
            "hf-internal-testing/tiny-random-gpt2"
        ):
            progress_updates.append(progress)
            print(f"{progress.status.value}: {progress.message}")

        # Verify completion
        assert len(progress_updates) >= 2
        assert progress_updates[0].status == DownloadStatus.STARTING
        assert progress_updates[-1].status == DownloadStatus.COMPLETE
        assert progress_updates[-1].percent == 100.0

    @pytest.mark.asyncio
    async def test_download_with_mlx_provider_name(self):
        """Test that 'mlx' provider works (same as HuggingFace)."""
        progress_updates = []

        async for progress in download_model(
            "mlx",
            "hf-internal-testing/tiny-random-gpt2"
        ):
            progress_updates.append(progress)

        assert progress_updates[-1].status == DownloadStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_download_nonexistent_model(self):
        """Test downloading non-existent HuggingFace model returns error."""
        progress_updates = []

        async for progress in download_model(
            "huggingface",
            "nonexistent-org/nonexistent-model-xyz"
        ):
            progress_updates.append(progress)

        # Should get ERROR status with RepositoryNotFoundError
        assert progress_updates[-1].status == DownloadStatus.ERROR
        assert "not found" in progress_updates[-1].message.lower()


class TestUnsupportedProvider:
    """Test error handling for unsupported providers."""

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        """Test that unsupported providers raise ValueError."""
        with pytest.raises(ValueError, match="does not support"):
            async for _ in download_model("openai", "gpt-4"):
                pass

    @pytest.mark.asyncio
    async def test_lmstudio_unsupported(self):
        """Test that LMStudio raises ValueError (no download API)."""
        with pytest.raises(ValueError, match="does not support"):
            async for _ in download_model("lmstudio", "model"):
                pass

    @pytest.mark.asyncio
    async def test_anthropic_unsupported(self):
        """Test that Anthropic raises ValueError (cloud-only)."""
        with pytest.raises(ValueError, match="does not support"):
            async for _ in download_model("anthropic", "claude-sonnet-4-5-20250929"):
                pass


class TestDownloadProgress:
    """Test DownloadProgress dataclass."""

    def test_download_progress_creation(self):
        """Test creating DownloadProgress instances."""
        progress = DownloadProgress(
            status=DownloadStatus.DOWNLOADING,
            message="Downloading...",
            percent=50.0,
            downloaded_bytes=1024,
            total_bytes=2048,
        )

        assert progress.status == DownloadStatus.DOWNLOADING
        assert progress.message == "Downloading..."
        assert progress.percent == 50.0
        assert progress.downloaded_bytes == 1024
        assert progress.total_bytes == 2048

    def test_download_progress_minimal(self):
        """Test creating DownloadProgress with minimal fields."""
        progress = DownloadProgress(
            status=DownloadStatus.STARTING,
            message="Starting download..."
        )

        assert progress.status == DownloadStatus.STARTING
        assert progress.message == "Starting download..."
        assert progress.percent is None
        assert progress.downloaded_bytes is None
        assert progress.total_bytes is None


if __name__ == "__main__":
    # Allow running directly for quick validation
    pytest.main([__file__, "-v", "--tb=short"])
