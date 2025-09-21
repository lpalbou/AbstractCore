"""
Integration tests for GGUF models in the AbstractLLM ecosystem
"""
import pytest
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abstractllm import create_llm


TEST_GGUF_MODEL = "unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF"


class TestGGUFIntegration:
    """Integration tests for GGUF models"""

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_gguf_provider_in_factory(self):
        """Test GGUF models work through the factory function"""
        llm = create_llm("huggingface", model=TEST_GGUF_MODEL, debug=False)

        assert llm is not None
        assert hasattr(llm, 'model_type')
        assert llm.model_type == "gguf"

        # Test basic functionality
        response = llm.generate("Hello", max_output_tokens=10)
        assert response.content is not None
        assert len(response.content) > 0

        # Cleanup
        del llm

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_gguf_unified_token_system_integration(self):
        """Test GGUF models work with the unified token system"""
        llm = create_llm(
            "huggingface",
            model=TEST_GGUF_MODEL,
            max_tokens=2048,
            max_output_tokens=50,
            debug=False
        )

        # Verify unified parameters are set
        assert llm.max_tokens == 2048
        assert llm.max_output_tokens == 50

        # Test token validation works
        assert llm.validate_token_usage(100, 30) == True

        # Test generation respects limits
        response = llm.generate("Tell me about Python")
        assert response.content is not None

        # Cleanup
        del llm

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_gguf_capabilities_integration(self):
        """Test GGUF model capabilities are properly reported"""
        llm = create_llm("huggingface", model=TEST_GGUF_MODEL, debug=False)

        capabilities = llm.get_capabilities()
        expected_capabilities = ["chat", "streaming", "gguf", "tools"]

        for cap in expected_capabilities:
            assert cap in capabilities, f"Missing capability: {cap}"

        # Cleanup
        del llm

    def test_gguf_with_non_cached_model(self):
        """Test error handling for non-cached GGUF models"""
        with pytest.raises(Exception) as exc_info:
            create_llm("huggingface", model="fake/model-GGUF", debug=False)

        # Should get a proper error message
        assert "GGUF model" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])