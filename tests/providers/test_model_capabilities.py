"""
Test suite for the new model capability filtering system.

This test suite verifies that the new ModelInputCapability and ModelOutputCapability
system works correctly across all providers and scenarios.
"""

import pytest
from unittest.mock import Mock, patch
from abstractcore.providers.model_capabilities import (
    ModelInputCapability,
    ModelOutputCapability,
    get_model_input_capabilities,
    get_model_output_capabilities,
    model_matches_input_capabilities,
    model_matches_output_capabilities,
    filter_models_by_capabilities,
    get_capability_summary
)
from abstractcore.architectures.detection import get_model_capabilities, supports_embeddings


class TestModelCapabilityEnums:
    """Test the capability enum definitions."""
    
    def test_input_capability_values(self):
        """Test that input capability enum has correct values."""
        assert ModelInputCapability.TEXT.value == "text"
        assert ModelInputCapability.IMAGE.value == "image"
        assert ModelInputCapability.AUDIO.value == "audio"
        assert ModelInputCapability.VIDEO.value == "video"
    
    def test_output_capability_values(self):
        """Test that output capability enum has correct values."""
        assert ModelOutputCapability.TEXT.value == "text"
        assert ModelOutputCapability.EMBEDDINGS.value == "embeddings"


class TestCapabilityDetection:
    """Test capability detection functions."""
    
    @patch('abstractcore.providers.model_capabilities.get_model_capabilities')
    def test_get_model_input_capabilities_text_only(self, mock_get_caps):
        """Test detection of text-only model capabilities."""
        mock_get_caps.return_value = {
            "vision_support": False,
            "audio_support": False,
            "video_support": False
        }
        
        caps = get_model_input_capabilities("gpt-4")
        assert caps == [ModelInputCapability.TEXT]
    
    @patch('abstractcore.providers.model_capabilities.get_model_capabilities')
    def test_get_model_input_capabilities_vision(self, mock_get_caps):
        """Test detection of vision model capabilities."""
        mock_get_caps.return_value = {
            "vision_support": True,
            "audio_support": False,
            "video_support": False
        }
        
        caps = get_model_input_capabilities("gpt-4-vision-preview")
        assert ModelInputCapability.TEXT in caps
        assert ModelInputCapability.IMAGE in caps
        assert len(caps) == 2
    
    @patch('abstractcore.providers.model_capabilities.get_model_capabilities')
    def test_get_model_input_capabilities_multimodal(self, mock_get_caps):
        """Test detection of multimodal model capabilities."""
        mock_get_caps.return_value = {
            "vision_support": True,
            "audio_support": True,
            "video_support": False
        }
        
        caps = get_model_input_capabilities("gpt-4o")
        assert ModelInputCapability.TEXT in caps
        assert ModelInputCapability.IMAGE in caps
        assert ModelInputCapability.AUDIO in caps
        assert len(caps) == 3
    
    @patch('abstractcore.providers.model_capabilities.get_model_capabilities')
    def test_get_model_input_capabilities_error_fallback(self, mock_get_caps):
        """Test fallback to text-only when capabilities can't be determined."""
        mock_get_caps.side_effect = Exception("Model not found")
        
        caps = get_model_input_capabilities("unknown-model")
        assert caps == [ModelInputCapability.TEXT]
    
    @patch('abstractcore.providers.model_capabilities.get_model_capabilities')
    def test_get_model_output_capabilities_text(self, mock_get_caps):
        """Test detection of text generation model output capabilities."""
        mock_get_caps.return_value = {
            "model_type": "text"
        }
        
        caps = get_model_output_capabilities("gpt-4")
        assert caps == [ModelOutputCapability.TEXT]
    
    @patch('abstractcore.providers.model_capabilities.get_model_capabilities')
    def test_get_model_output_capabilities_embedding(self, mock_get_caps):
        """Test detection of embedding model output capabilities."""
        mock_get_caps.return_value = {
            "model_type": "embedding"
        }
        
        caps = get_model_output_capabilities("text-embedding-3-small")
        assert caps == [ModelOutputCapability.EMBEDDINGS]
    
    @patch('abstractcore.providers.model_capabilities.get_model_capabilities')
    def test_get_model_output_capabilities_error_fallback(self, mock_get_caps):
        """Test fallback to text generation when capabilities can't be determined."""
        mock_get_caps.side_effect = Exception("Model not found")
        
        caps = get_model_output_capabilities("unknown-model")
        assert caps == [ModelOutputCapability.TEXT]


class TestCapabilityMatching:
    """Test capability matching functions."""
    
    @patch('abstractcore.providers.model_capabilities.get_model_input_capabilities')
    def test_model_matches_input_capabilities_single(self, mock_get_input_caps):
        """Test matching single input capability."""
        mock_get_input_caps.return_value = [ModelInputCapability.TEXT, ModelInputCapability.IMAGE]
        
        # Should match - model supports image
        assert model_matches_input_capabilities("gpt-4-vision", [ModelInputCapability.IMAGE])
        
        # Should match - model supports text
        assert model_matches_input_capabilities("gpt-4-vision", [ModelInputCapability.TEXT])
        
        # Should not match - model doesn't support audio
        assert not model_matches_input_capabilities("gpt-4-vision", [ModelInputCapability.AUDIO])
    
    @patch('abstractcore.providers.model_capabilities.get_model_input_capabilities')
    def test_model_matches_input_capabilities_multiple(self, mock_get_input_caps):
        """Test matching multiple input capabilities."""
        mock_get_input_caps.return_value = [ModelInputCapability.TEXT, ModelInputCapability.IMAGE]
        
        # Should match - model supports both text and image
        assert model_matches_input_capabilities(
            "gpt-4-vision", 
            [ModelInputCapability.TEXT, ModelInputCapability.IMAGE]
        )
        
        # Should not match - model doesn't support audio
        assert not model_matches_input_capabilities(
            "gpt-4-vision", 
            [ModelInputCapability.TEXT, ModelInputCapability.AUDIO]
        )
    
    def test_model_matches_input_capabilities_empty(self):
        """Test matching with empty requirements."""
        # Empty requirements should always match
        assert model_matches_input_capabilities("any-model", [])
    
    @patch('abstractcore.providers.model_capabilities.get_model_output_capabilities')
    def test_model_matches_output_capabilities(self, mock_get_output_caps):
        """Test matching output capabilities."""
        mock_get_output_caps.return_value = [ModelOutputCapability.TEXT]
        
        # Should match - model generates text
        assert model_matches_output_capabilities("gpt-4", [ModelOutputCapability.TEXT])
        
        # Should not match - model doesn't generate embeddings
        assert not model_matches_output_capabilities("gpt-4", [ModelOutputCapability.EMBEDDINGS])


class TestModelFiltering:
    """Test model filtering functionality."""
    
    @patch('abstractcore.providers.model_capabilities.get_model_input_capabilities')
    @patch('abstractcore.providers.model_capabilities.get_model_output_capabilities')
    def test_filter_models_by_capabilities_input_only(self, mock_get_output_caps, mock_get_input_caps):
        """Test filtering models by input capabilities only."""
        # Setup mock responses
        def mock_input_caps(model_name):
            if "vision" in model_name:
                return [ModelInputCapability.TEXT, ModelInputCapability.IMAGE]
            else:
                return [ModelInputCapability.TEXT]
        
        mock_get_input_caps.side_effect = mock_input_caps
        mock_get_output_caps.return_value = [ModelOutputCapability.TEXT]
        
        models = ["gpt-4", "gpt-4-vision-preview", "claude-3-sonnet"]
        
        # Filter for vision models
        vision_models = filter_models_by_capabilities(
            models, 
            input_capabilities=[ModelInputCapability.IMAGE]
        )
        
        assert vision_models == ["gpt-4-vision-preview"]
    
    @patch('abstractcore.providers.model_capabilities.get_model_input_capabilities')
    @patch('abstractcore.providers.model_capabilities.get_model_output_capabilities')
    def test_filter_models_by_capabilities_output_only(self, mock_get_output_caps, mock_get_input_caps):
        """Test filtering models by output capabilities only."""
        # Setup mock responses
        def mock_output_caps(model_name):
            if "embedding" in model_name:
                return [ModelOutputCapability.EMBEDDINGS]
            else:
                return [ModelOutputCapability.TEXT]
        
        mock_get_input_caps.return_value = [ModelInputCapability.TEXT]
        mock_get_output_caps.side_effect = mock_output_caps
        
        models = ["gpt-4", "text-embedding-3-small", "claude-3-sonnet"]
        
        # Filter for embedding models
        embedding_models = filter_models_by_capabilities(
            models, 
            output_capabilities=[ModelOutputCapability.EMBEDDINGS]
        )
        
        assert embedding_models == ["text-embedding-3-small"]
    
    @patch('abstractcore.providers.model_capabilities.get_model_input_capabilities')
    @patch('abstractcore.providers.model_capabilities.get_model_output_capabilities')
    def test_filter_models_by_capabilities_both(self, mock_get_output_caps, mock_get_input_caps):
        """Test filtering models by both input and output capabilities."""
        # Setup mock responses
        def mock_input_caps(model_name):
            if "vision" in model_name:
                return [ModelInputCapability.TEXT, ModelInputCapability.IMAGE]
            else:
                return [ModelInputCapability.TEXT]
        
        def mock_output_caps(model_name):
            if "embedding" in model_name:
                return [ModelOutputCapability.EMBEDDINGS]
            else:
                return [ModelOutputCapability.TEXT]
        
        mock_get_input_caps.side_effect = mock_input_caps
        mock_get_output_caps.side_effect = mock_output_caps
        
        models = ["gpt-4", "gpt-4-vision-preview", "text-embedding-3-small", "claude-3-sonnet"]
        
        # Filter for vision models that generate text
        vision_text_models = filter_models_by_capabilities(
            models, 
            input_capabilities=[ModelInputCapability.IMAGE],
            output_capabilities=[ModelOutputCapability.TEXT]
        )
        
        assert vision_text_models == ["gpt-4-vision-preview"]
    
    def test_filter_models_by_capabilities_no_filters(self):
        """Test filtering with no filters returns all models."""
        models = ["gpt-4", "gpt-4-vision-preview", "text-embedding-3-small"]
        
        filtered = filter_models_by_capabilities(models)
        
        assert filtered == models
    
    @patch('abstractcore.providers.model_capabilities.get_model_capabilities')
    def test_filter_models_by_capabilities_error_handling(self, mock_get_caps):
        """Test that models with capability detection errors fall back to text-only."""
        def mock_caps(model_name):
            if model_name == "error-model":
                raise Exception("Capability detection failed")
            return {
                "vision_support": False,
                "audio_support": False,
                "video_support": False
            }
        
        mock_get_caps.side_effect = mock_caps
        
        models = ["gpt-4", "error-model", "claude-3-sonnet"]
        
        # Filter for text models - error-model should be included (fallback to text-only)
        text_models = filter_models_by_capabilities(
            models, 
            input_capabilities=[ModelInputCapability.TEXT]
        )
        
        # All models should be included since they all support text (error-model falls back to text-only)
        assert "error-model" in text_models
        assert "gpt-4" in text_models
        assert "claude-3-sonnet" in text_models
        
        # Filter for image models - error-model should be excluded (no image support in fallback)
        image_models = filter_models_by_capabilities(
            models, 
            input_capabilities=[ModelInputCapability.IMAGE]
        )
        
        # error-model should be excluded since fallback doesn't include image support
        assert "error-model" not in image_models


class TestCapabilitySummary:
    """Test capability summary functionality."""
    
    @patch('abstractcore.providers.model_capabilities.get_model_input_capabilities')
    @patch('abstractcore.providers.model_capabilities.get_model_output_capabilities')
    def test_get_capability_summary_text_only(self, mock_get_output_caps, mock_get_input_caps):
        """Test capability summary for text-only model."""
        mock_get_input_caps.return_value = [ModelInputCapability.TEXT]
        mock_get_output_caps.return_value = [ModelOutputCapability.TEXT]
        
        summary = get_capability_summary("gpt-4")
        
        expected = {
            'model_name': 'gpt-4',
            'input_capabilities': ['text'],
            'output_capabilities': ['text'],
            'is_multimodal': False,
            'is_embedding_model': False
        }
        
        assert summary == expected
    
    @patch('abstractcore.providers.model_capabilities.get_model_input_capabilities')
    @patch('abstractcore.providers.model_capabilities.get_model_output_capabilities')
    def test_get_capability_summary_vision(self, mock_get_output_caps, mock_get_input_caps):
        """Test capability summary for vision model."""
        mock_get_input_caps.return_value = [ModelInputCapability.TEXT, ModelInputCapability.IMAGE]
        mock_get_output_caps.return_value = [ModelOutputCapability.TEXT]
        
        summary = get_capability_summary("gpt-4-vision-preview")
        
        expected = {
            'model_name': 'gpt-4-vision-preview',
            'input_capabilities': ['text', 'image'],
            'output_capabilities': ['text'],
            'is_multimodal': True,
            'is_embedding_model': False
        }
        
        assert summary == expected
    
    @patch('abstractcore.providers.model_capabilities.get_model_input_capabilities')
    @patch('abstractcore.providers.model_capabilities.get_model_output_capabilities')
    def test_get_capability_summary_embedding(self, mock_get_output_caps, mock_get_input_caps):
        """Test capability summary for embedding model."""
        mock_get_input_caps.return_value = [ModelInputCapability.TEXT]
        mock_get_output_caps.return_value = [ModelOutputCapability.EMBEDDINGS]
        
        summary = get_capability_summary("text-embedding-3-small")
        
        expected = {
            'model_name': 'text-embedding-3-small',
            'input_capabilities': ['text'],
            'output_capabilities': ['embeddings'],
            'is_multimodal': False,
            'is_embedding_model': True
        }
        
        assert summary == expected


class TestBackwardCompatibility:
    """Test backward compatibility with old system."""
    
    def test_old_filters_module_removed(self):
        """Test that the old filters module has been completely removed."""
        # The old filters module should no longer exist
        try:
            from abstractcore.providers.filters import filter_models_by_capability
            # This should fail since we deleted the old module
            assert False, "Old filters module should have been removed"
        except ImportError:
            # This is expected - the old system is completely gone
            pass
    
    def test_new_imports_work(self):
        """Test that new imports work correctly."""
        from abstractcore.providers.model_capabilities import (
            ModelInputCapability,
            ModelOutputCapability,
            filter_models_by_capabilities
        )
        
        # Basic smoke test
        assert ModelInputCapability.TEXT.value == "text"
        assert ModelOutputCapability.TEXT.value == "text"
        assert callable(filter_models_by_capabilities)


class TestCapabilitiesJsonIntegration:
    """
    Integration checks against the real `assets/model_capabilities.json`.

    These tests ensure we recognize known model IDs (including provider-specific aliases)
    without relying on mocks.
    """

    def test_lmstudio_nomic_embed_v1_5_alias_is_embedding_model(self):
        model_name = "text-embedding-nomic-embed-text-v1.5@q6_k"

        caps = get_model_capabilities(model_name)
        assert caps.get("model_type") == "embedding"
        assert caps.get("max_tokens") == 8192
        assert caps.get("max_output_tokens") == 0

        # Helper should respect `model_type: "embedding"` (and not rely only on legacy flags)
        assert supports_embeddings(model_name) is True

        # Provider-side capability utilities should also treat it as an embedding model.
        assert get_model_output_capabilities(model_name) == [ModelOutputCapability.EMBEDDINGS]


if __name__ == "__main__":
    pytest.main([__file__])
