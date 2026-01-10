"""
Test model unloading functionality for memory management.

These tests verify that the unload() method works correctly for all local providers,
enabling explicit memory management in long-running tests and applications.
"""

import pytest
import gc
import os
from pathlib import Path
from abstractcore import create_llm


class TestModelUnloading:
    """Test unload() method for local providers"""

    def test_huggingface_unload(self):
        """Test HuggingFace GGUF model unloading"""
        if os.getenv("ABSTRACTCORE_RUN_GGUF_TESTS") != "1":
            pytest.skip("GGUF unload test is heavy; set ABSTRACTCORE_RUN_GGUF_TESTS=1 to run")
        # Skip if model not available
        if not Path.home().joinpath(".cache/huggingface/hub/models--unsloth/Qwen3-4B-Instruct-2507-GGUF").exists():
            pytest.skip("Test GGUF model not found in cache")

        llm = create_llm("huggingface", model="unsloth/Qwen3-4B-Instruct-2507-GGUF")

        # Verify model is loaded
        assert llm.llm is not None

        # Unload the model
        llm.unload()

        # Verify model is unloaded
        assert llm.llm is None

        # Cleanup
        del llm
        gc.collect()

    def test_ollama_unload(self):
        """Test Ollama model unloading"""
        try:
            llm = create_llm("ollama", model="qwen3:4b-instruct", base_url="http://localhost:11434", timeout=10.0)

            # Unload the model (sends keep_alive=0 to server)
            llm.unload()

            # Verify HTTP client is closed
            # (Model is unloaded on server side)

            # Cleanup
            del llm

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "operation not permitted"]):
                pytest.skip("Ollama not running")
            raise

    def test_mlx_unload(self):
        """Test MLX model unloading"""
        if os.getenv("ABSTRACTCORE_RUN_MLX_TESTS") != "1":
            pytest.skip("MLX unload test is heavy; set ABSTRACTCORE_RUN_MLX_TESTS=1 to run")
        try:
            llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit", timeout=5.0)

            # Verify model is loaded
            assert llm.llm is not None
            assert llm.tokenizer is not None

            # Unload the model
            llm.unload()

            # Verify model is unloaded
            assert llm.llm is None
            assert llm.tokenizer is None

            # Cleanup
            del llm
            gc.collect()

        except ImportError:
            pytest.skip("MLX not available")
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                pytest.skip("MLX model not found")
            raise

    def test_lmstudio_unload(self):
        """Test LMStudio model unloading"""
        try:
            llm = create_llm("lmstudio", model="qwen/qwen3-4b-2507", base_url="http://localhost:1234/v1", timeout=10.0)

            # Unload (closes HTTP client)
            llm.unload()

            # Cleanup
            del llm

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "operation not permitted"]):
                pytest.skip("LMStudio not running")
            raise

    def test_openai_unload_noop(self):
        """Test that OpenAI unload is a no-op (doesn't raise error)"""
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API test; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run")
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        llm = create_llm("openai", model="gpt-4o-mini", timeout=5.0)

        # Should not raise error (is a no-op for API providers)
        llm.unload()

        # Cleanup
        del llm

    def test_sequential_model_loading(self):
        """Test loading multiple models sequentially with unload between them"""
        if os.getenv("ABSTRACTCORE_RUN_GGUF_TESTS") != "1":
            pytest.skip("Sequential GGUF unload test is heavy; set ABSTRACTCORE_RUN_GGUF_TESTS=1 to run")
        # Skip if models not available
        if not Path.home().joinpath(".cache/huggingface/hub/models--unsloth/Qwen3-4B-Instruct-2507-GGUF").exists():
            pytest.skip("Test GGUF model not found in cache")

        # Load first model
        llm1 = create_llm("huggingface", model="unsloth/Qwen3-4B-Instruct-2507-GGUF")
        response1 = llm1.generate("Hi", max_tokens=5)
        assert response1.content is not None

        # Unload first model before loading second
        llm1.unload()
        del llm1
        gc.collect()

        # Load second model (should succeed without OOM)
        try:
            llm2 = create_llm("ollama", model="qwen3:4b-instruct", timeout=10.0)
            llm2.unload()
            del llm2
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "operation not permitted"]):
                pytest.skip("Ollama not running")
            raise
