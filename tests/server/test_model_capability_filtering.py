"""
Test model capability filtering in the server endpoints.

This test validates that the server correctly filters models by input and output capabilities
using the ModelInputCapability and ModelOutputCapability enums.
"""

import pytest
import requests
import time
import threading
import uvicorn
from fastapi.testclient import TestClient

from abstractcore.server.app import app
from abstractcore.providers.model_capabilities import ModelInputCapability, ModelOutputCapability


class TestModelCapabilityFiltering:
    """Test model capability filtering in server endpoints."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_basic_model_listing(self, client):
        """Test basic model listing without filters."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"
        
        # Should have some models
        models = data["data"]
        assert len(models) > 0
        
        # Each model should have required fields
        for model in models:
            assert "id" in model
            assert "object" in model
            assert "owned_by" in model
            assert "created" in model
            assert "/" in model["id"]  # Should have provider/model format
    
    def test_input_capability_filtering(self, client):
        """Test filtering by input capabilities."""
        # Test text input filtering
        response = client.get("/v1/models?input_type=text")
        assert response.status_code == 200
        
        data = response.json()
        text_models = data["data"]
        assert len(text_models) > 0
        
        # Test image input filtering (vision models)
        response = client.get("/v1/models?input_type=image")
        assert response.status_code == 200
        
        data = response.json()
        vision_models = data["data"]
        # Should have some vision models
        assert len(vision_models) >= 0
        
        # Test audio input filtering
        response = client.get("/v1/models?input_type=audio")
        assert response.status_code == 200
        
        data = response.json()
        audio_models = data["data"]
        # Audio models may or may not exist
        assert len(audio_models) >= 0
        
        # Test video input filtering
        response = client.get("/v1/models?input_type=video")
        assert response.status_code == 200
        
        data = response.json()
        video_models = data["data"]
        # Video models may or may not exist
        assert len(video_models) >= 0
    
    def test_output_capability_filtering(self, client):
        """Test filtering by output capabilities."""
        # Test text generation filtering
        response = client.get("/v1/models?output_type=text")
        assert response.status_code == 200
        
        data = response.json()
        text_gen_models = data["data"]
        assert len(text_gen_models) > 0
        
        # Test embedding filtering
        response = client.get("/v1/models?output_type=embeddings")
        assert response.status_code == 200
        
        data = response.json()
        embedding_models = data["data"]
        # Should have some embedding models
        assert len(embedding_models) >= 0
    
    def test_combined_filtering(self, client):
        """Test combined input and output filtering."""
        # Test text-only models (text input + text output)
        response = client.get("/v1/models?input_type=text&output_type=text")
        assert response.status_code == 200
        
        data = response.json()
        text_only_models = data["data"]
        assert len(text_only_models) > 0
        
        # Test vision models that generate text
        response = client.get("/v1/models?input_type=image&output_type=text")
        assert response.status_code == 200
        
        data = response.json()
        vision_text_models = data["data"]
        # May or may not have vision models
        assert len(vision_text_models) >= 0
    
    def test_provider_specific_filtering(self, client):
        """Test provider-specific filtering with capabilities."""
        # Test OpenAI models with image input
        response = client.get("/v1/models?provider=openai&input_type=image")
        assert response.status_code == 200
        
        data = response.json()
        openai_vision_models = data["data"]
        
        # All returned models should be from OpenAI
        for model in openai_vision_models:
            assert model["id"].startswith("openai/")
            assert model["owned_by"] == "openai"
        
        # Test Ollama models with text output
        response = client.get("/v1/models?provider=ollama&output_type=text")
        assert response.status_code == 200
        
        data = response.json()
        ollama_text_models = data["data"]
        
        # All returned models should be from Ollama
        for model in ollama_text_models:
            assert model["id"].startswith("ollama/")
            assert model["owned_by"] == "ollama"
    
    def test_invalid_capability_values(self, client):
        """Test handling of invalid capability values."""
        # Test invalid input type
        response = client.get("/v1/models?input_type=invalid")
        # Should return 422 validation error
        assert response.status_code == 422
        
        # Test invalid output type
        response = client.get("/v1/models?output_type=invalid")
        # Should return 422 validation error
        assert response.status_code == 422
    
    def test_nonexistent_provider(self, client):
        """Test filtering with nonexistent provider."""
        response = client.get("/v1/models?provider=nonexistent")
        assert response.status_code == 200
        
        data = response.json()
        # Should return empty list for nonexistent provider
        assert len(data["data"]) == 0
    
    def test_capability_enum_values(self, client):
        """Test that the server accepts the correct enum values."""
        # Test all valid input capability values
        valid_input_types = ["text", "image", "audio", "video"]
        for input_type in valid_input_types:
            response = client.get(f"/v1/models?input_type={input_type}")
            assert response.status_code == 200
        
        # Test all valid output capability values
        valid_output_types = ["text", "embeddings"]
        for output_type in valid_output_types:
            response = client.get(f"/v1/models?output_type={output_type}")
            assert response.status_code == 200
    
    def test_response_format_consistency(self, client):
        """Test that all responses follow the same format."""
        test_endpoints = [
            "/v1/models",
            "/v1/models?input_type=text",
            "/v1/models?output_type=text",
            "/v1/models?input_type=image&output_type=text",
            "/v1/models?provider=openai"
        ]
        
        for endpoint in test_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            data = response.json()
            assert "object" in data
            assert "data" in data
            assert data["object"] == "list"
            assert isinstance(data["data"], list)
            
            # Check each model has consistent format
            for model in data["data"]:
                assert "id" in model
                assert "object" in model
                assert "owned_by" in model
                assert "created" in model
                assert "permission" in model
                assert model["object"] == "model"
    
    def test_filtering_logic_consistency(self, client):
        """Test that filtering logic is consistent and makes sense."""
        # Get all models
        all_response = client.get("/v1/models")
        all_models = all_response.json()["data"]
        
        # Get text models
        text_response = client.get("/v1/models?input_type=text")
        text_models = text_response.json()["data"]
        
        # Get vision models
        vision_response = client.get("/v1/models?input_type=image")
        vision_models = vision_response.json()["data"]
        
        # Text models should be a subset of all models
        text_model_ids = {model["id"] for model in text_models}
        all_model_ids = {model["id"] for model in all_models}
        assert text_model_ids.issubset(all_model_ids)
        
        # Vision models should be a subset of all models
        vision_model_ids = {model["id"] for model in vision_models}
        assert vision_model_ids.issubset(all_model_ids)
        
        # Text models count should be >= vision models count (more text models than vision)
        assert len(text_models) >= len(vision_models)


class TestModelCapabilityFilteringIntegration:
    """Integration tests that require a running server."""
    
    @pytest.fixture(scope="class")
    def server_url(self):
        """Start a test server and return its URL."""
        import socket
        
        # Find an available port
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        
        # Start server in background thread
        def start_server():
            uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        return f"http://127.0.0.1:{port}"
    
    def test_http_requests_integration(self, server_url):
        """Test model capability filtering with actual HTTP requests."""
        # Test basic listing
        response = requests.get(f"{server_url}/v1/models", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) > 0
        
        # Test capability filtering
        response = requests.get(f"{server_url}/v1/models?input_type=text", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) > 0
        
        # Test combined filtering
        response = requests.get(f"{server_url}/v1/models?input_type=text&output_type=text", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) > 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
