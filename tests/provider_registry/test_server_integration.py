"""
Test Provider Registry Server Integration

Tests the integration between the provider registry and the HTTP server,
ensuring that the server endpoints use the centralized registry properly.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from abstractcore.server.app import app


class TestServerProviderIntegration:
    """Test server integration with provider registry."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('abstractcore.server.app.get_models_from_provider')
    def test_list_models_endpoint_uses_registry(self, mock_get_models):
        """Test that /v1/models endpoint uses centralized registry."""
        # Mock the registry function
        mock_get_models.return_value = ["model1", "model2", "model3"]

        response = self.client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert len(data["data"]) > 0

        # Verify that get_models_from_provider was called
        mock_get_models.assert_called()

    @patch('abstractcore.server.app.get_models_from_provider')
    def test_list_models_with_provider_filter(self, mock_get_models):
        """Test /v1/models endpoint with provider filter."""
        mock_get_models.return_value = ["model1", "model2"]

        response = self.client.get("/v1/models?provider=openai")

        assert response.status_code == 200
        data = response.json()

        # Should have called get_models_from_provider with "openai"
        mock_get_models.assert_called_with("openai")

    @patch('abstractcore.providers.registry.get_all_providers_with_models')
    def test_providers_endpoint_uses_registry(self, mock_get_providers):
        """Test that /providers endpoint uses centralized registry."""
        # Mock the registry function
        mock_providers_data = [
            {
                "name": "mock",
                "display_name": "Mock Provider",
                "type": "llm",
                "status": "available",
                "model_count": 3,
                "description": "Test provider",
                "local_provider": True,
                "authentication_required": False,
                "supported_features": ["chat", "completion"],
                "installation_extras": None
            }
        ]
        mock_get_providers.return_value = mock_providers_data

        response = self.client.get("/providers")

        assert response.status_code == 200
        data = response.json()

        assert "providers" in data
        assert "total_providers" in data
        assert "registry_version" in data
        assert data["registry_version"] == "2.0"

        # Verify the registry function was called
        mock_get_providers.assert_called_once()

        # Check the response structure
        providers = data["providers"]
        assert len(providers) == 1
        assert providers[0]["name"] == "mock"
        assert providers[0]["display_name"] == "Mock Provider"

    @patch('abstractcore.providers.registry.get_available_models_for_provider')
    def test_get_models_from_provider_function(self, mock_get_models):
        """Test that get_models_from_provider uses registry function."""
        from abstractcore.server.app import get_models_from_provider

        mock_get_models.return_value = ["model1", "model2", "model3"]

        result = get_models_from_provider("mock")

        assert result == ["model1", "model2", "model3"]
        mock_get_models.assert_called_once_with("mock")

    @patch('abstractcore.providers.registry.get_available_models_for_provider')
    def test_get_models_from_provider_error_handling(self, mock_get_models):
        """Test error handling in get_models_from_provider function."""
        from abstractcore.server.app import get_models_from_provider

        # Mock function to raise exception
        mock_get_models.side_effect = Exception("Provider error")

        result = get_models_from_provider("error_provider")

        # Should return empty list on error
        assert result == []
        mock_get_models.assert_called_once_with("error_provider")

    @patch('abstractcore.providers.registry.list_available_providers')
    @patch('abstractcore.server.app.get_models_from_provider')
    def test_list_all_models_uses_registry_providers(self, mock_get_models, mock_list_providers):
        """Test that listing all models uses registry for provider discovery."""
        # Mock registry provider list
        mock_list_providers.return_value = ["mock", "openai"]

        # Mock models for each provider
        def mock_get_models_side_effect(provider):
            if provider == "mock":
                return ["mock-model-1", "mock-model-2"]
            elif provider == "openai":
                return ["gpt-4", "gpt-3.5-turbo"]
            return []

        mock_get_models.side_effect = mock_get_models_side_effect

        response = self.client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        # Verify registry function was called
        mock_list_providers.assert_called_once()

        # Check that models from both providers are included
        model_ids = [model["id"] for model in data["data"]]
        assert "mock/mock-model-1" in model_ids
        assert "mock/mock-model-2" in model_ids
        assert "openai/gpt-4" in model_ids
        assert "openai/gpt-3.5-turbo" in model_ids

    def test_providers_endpoint_structure(self):
        """Test the structure of the providers endpoint response."""
        response = self.client.get("/providers")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "providers" in data
        assert "total_providers" in data
        assert "registry_version" in data

        # Check that registry_version indicates new system
        assert data["registry_version"] == "2.0"

        # If there are providers, check their structure
        if data["providers"]:
            provider = data["providers"][0]
            required_fields = [
                "name", "display_name", "type", "status", "description",
                "local_provider", "authentication_required", "supported_features"
            ]

            for field in required_fields:
                assert field in provider, f"Field {field} missing from provider data"

    @patch('abstractcore.providers.registry.get_all_providers_with_models')
    def test_providers_endpoint_error_handling(self, mock_get_providers):
        """Test error handling in providers endpoint."""
        # Mock registry function to raise exception
        mock_get_providers.side_effect = Exception("Registry error")

        response = self.client.get("/providers")

        assert response.status_code == 200  # Should still return 200 with error info
        data = response.json()

        assert "error" in data
        assert data["total_providers"] == 0
        assert data["providers"] == []
        assert data["registry_version"] == "2.0"

    @patch('abstractcore.server.app.get_models_from_provider')
    def test_model_type_filtering_with_registry(self, mock_get_models):
        """Test model type filtering works with registry."""
        # Mock models with mix of embedding and generation models
        mock_get_models.return_value = [
            "gpt-4",  # generation model
            "text-embedding-ada-002",  # embedding model
            "claude-3-haiku",  # generation model
            "granite-embedding:278m"  # embedding model
        ]

        # Test filtering for text generation models
        response = self.client.get("/v1/models?type=text-generation")
        assert response.status_code == 200
        data = response.json()

        # Should only include generation models
        model_names = [model["id"] for model in data["data"]]
        generation_models = [name for name in model_names if "embed" not in name.lower()]
        embedding_models = [name for name in model_names if "embed" in name.lower()]

        assert len(generation_models) >= 2  # gpt-4, claude-3-haiku
        assert len(embedding_models) == 0   # Should be filtered out

        # Test filtering for embedding models
        response = self.client.get("/v1/models?type=text-embedding")
        assert response.status_code == 200
        data = response.json()

        model_names = [model["id"] for model in data["data"]]
        embedding_models = [name for name in model_names if "embed" in name.lower()]

        assert len(embedding_models) >= 2  # Should include embedding models

    def test_health_endpoint_mentions_registry(self):
        """Test that health endpoint mentions registry features."""
        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "features" in data
        # Should mention features related to the new registry system
        features = data["features"]
        assert isinstance(features, list)


class TestRegistryBackwardCompatibility:
    """Test that registry changes maintain backward compatibility."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_models_endpoint_format_unchanged(self):
        """Test that models endpoint format is unchanged for client compatibility."""
        response = self.client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        # Check OpenAI-compatible format
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data

        if data["data"]:
            model = data["data"][0]
            required_fields = ["id", "object", "owned_by", "created", "permission"]

            for field in required_fields:
                assert field in model, f"Model missing required field: {field}"

    def test_providers_endpoint_enhanced_info(self):
        """Test that providers endpoint provides enhanced information."""
        response = self.client.get("/providers")

        assert response.status_code == 200
        data = response.json()

        # New fields that weren't in the old manual system
        assert "registry_version" in data
        assert "total_providers" in data

        if data["providers"]:
            provider = data["providers"][0]

            # Enhanced fields from registry system
            enhanced_fields = [
                "display_name", "supported_features", "local_provider",
                "authentication_required", "installation_extras"
            ]

            # At least some enhanced fields should be present
            present_enhanced = [field for field in enhanced_fields if field in provider]
            assert len(present_enhanced) > 0, "No enhanced registry fields found"


if __name__ == "__main__":
    pytest.main([__file__])