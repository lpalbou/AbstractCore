"""
Server Media Tests - Vision Processing
Tests OpenAI-compatible endpoints with vision-capable models for image processing.
"""

import pytest
import json
import base64
import tempfile
import requests
import os
import time
from PIL import Image, ImageDraw
import io
from typing import Dict, Any, List

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
TIMEOUT = 60  # seconds

# Test models configuration
VISION_MODELS = {
    "ollama": [
        "qwen2.5vl:7b",
        "llama3.2-vision:11b",
        "gemma3:4b"
    ],
    "lmstudio": [
        "qwen/qwen2.5-vl-7b",
        "google/gemma-3n-e4b",
        "mistralai/magistral-small-2509"
    ]
}

class VisionTestHelper:
    """Helper class for vision testing utilities."""

    @staticmethod
    def create_test_image(text: str = "Test Image", size: tuple = (300, 200), color: str = "lightblue") -> bytes:
        """Create a simple test image with text."""
        img = Image.new('RGB', size, color=color)
        draw = ImageDraw.Draw(img)

        # Calculate text position (centered)
        text_bbox = draw.textbbox((0, 0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

        draw.text(position, text, fill='black')

        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def image_to_base64_url(image_data: bytes, format: str = "png") -> str:
        """Convert image bytes to base64 data URL."""
        b64_string = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/{format};base64,{b64_string}"

    @staticmethod
    def create_test_images() -> Dict[str, str]:
        """Create various test images and return as base64 data URLs."""
        images = {}

        # Simple test image
        images["simple"] = VisionTestHelper.image_to_base64_url(
            VisionTestHelper.create_test_image("Hello World")
        )

        # Chart-like image
        chart_img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(chart_img)

        # Draw simple bar chart
        bars = [80, 120, 160, 100, 140]
        bar_width = 60
        base_y = 250

        for i, height in enumerate(bars):
            x = 50 + i * 80
            draw.rectangle([x, base_y - height, x + bar_width, base_y], fill='blue', outline='black')
            draw.text((x + 15, base_y + 10), f"Q{i+1}", fill='black')

        draw.text((150, 20), "Quarterly Sales", fill='black')

        buffer = io.BytesIO()
        chart_img.save(buffer, format='PNG')
        images["chart"] = VisionTestHelper.image_to_base64_url(buffer.getvalue())

        # Geometric shapes
        shapes_img = Image.new('RGB', (300, 300), color='white')
        draw = ImageDraw.Draw(shapes_img)

        # Circle
        draw.ellipse([50, 50, 150, 150], fill='red', outline='black')
        draw.text((90, 95), "Circle", fill='white')

        # Rectangle
        draw.rectangle([200, 50, 280, 130], fill='green', outline='black')
        draw.text((210, 85), "Rect", fill='white')

        # Triangle (approximate)
        draw.polygon([(150, 200), (100, 280), (200, 280)], fill='blue', outline='black')
        draw.text((130, 240), "Triangle", fill='white')

        buffer = io.BytesIO()
        shapes_img.save(buffer, format='PNG')
        images["shapes"] = VisionTestHelper.image_to_base64_url(buffer.getvalue())

        return images

    @staticmethod
    def is_server_running() -> bool:
        """Check if the server is running."""
        try:
            response = requests.get(f"{SERVER_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """Get available models for a provider from the server."""
        try:
            response = requests.get(f"{SERVER_BASE_URL}/providers/{provider}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except:
            return []

@pytest.fixture(scope="module")
def server_check():
    """Ensure server is running before tests."""
    if not VisionTestHelper.is_server_running():
        pytest.skip("Server is not running. Start with: uvicorn abstractcore.server.app:app --port 8000")

@pytest.fixture(scope="module")
def test_images():
    """Create test images for the test session."""
    return VisionTestHelper.create_test_images()

class TestVisionOpenAIFormat:
    """Test vision processing using OpenAI Vision API format."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_single_image_analysis_openai_format(self, server_check, test_images, provider):
        """Test single image analysis using OpenAI Vision API format."""
        # Get available models
        available_models = VisionTestHelper.get_available_models(provider)
        vision_models = [m for m in VISION_MODELS.get(provider, []) if m in available_models]

        if not vision_models:
            pytest.skip(f"No vision models available for {provider}")

        model = vision_models[0]  # Use first available model

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image? Be specific about colors, text, and objects."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": test_images["simple"]
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 150,
            "temperature": 0.1
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

        content = data["choices"][0]["message"]["content"].lower()

        # Verify the model can see the image content
        assert any(word in content for word in ["hello", "world", "text", "image", "blue"]), \
            f"Model should recognize image content. Got: {content}"

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_chart_analysis_openai_format(self, server_check, test_images, provider):
        """Test chart analysis using OpenAI format."""
        available_models = VisionTestHelper.get_available_models(provider)
        vision_models = [m for m in VISION_MODELS.get(provider, []) if m in available_models]

        if not vision_models:
            pytest.skip(f"No vision models available for {provider}")

        model = vision_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this chart. What type of chart is it and what data does it show?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": test_images["chart"]
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should recognize chart elements
        assert any(word in content for word in ["chart", "bar", "quarterly", "sales", "data"]), \
            f"Model should recognize chart elements. Got: {content}"

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_multiple_images_openai_format(self, server_check, test_images, provider):
        """Test multiple image analysis in single request."""
        available_models = VisionTestHelper.get_available_models(provider)
        vision_models = [m for m in VISION_MODELS.get(provider, []) if m in available_models]

        if not vision_models:
            pytest.skip(f"No vision models available for {provider}")

        model = vision_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Compare these two images. What's different between them?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": test_images["simple"]
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": test_images["shapes"]
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 250
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should recognize differences between images
        assert any(word in content for word in ["different", "shapes", "text", "circle", "rectangle"]), \
            f"Model should recognize image differences. Got: {content}"

class TestVisionAbstractCoreFormat:
    """Test vision processing using AbstractCore @filename format."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_filename_syntax_vision(self, server_check, provider):
        """Test @filename syntax with image files."""
        available_models = VisionTestHelper.get_available_models(provider)
        vision_models = [m for m in VISION_MODELS.get(provider, []) if m in available_models]

        if not vision_models:
            pytest.skip(f"No vision models available for {provider}")

        model = vision_models[0]

        # Create a temporary image file
        image_data = VisionTestHelper.create_test_image("File Test", color="yellow")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_data)
            temp_file = f.name

        try:
            payload = {
                "model": f"{provider}/{model}",
                "messages": [
                    {
                        "role": "user",
                        "content": f"What color is the background in @{temp_file}?"
                    }
                ],
                "max_tokens": 100
            }

            response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

            assert response.status_code == 200
            data = response.json()
            content = data["choices"][0]["message"]["content"].lower()

            # Should recognize yellow background
            assert "yellow" in content, f"Model should recognize yellow background. Got: {content}"

        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)

class TestVisionStreaming:
    """Test streaming responses with vision models."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_streaming_vision_openai_format(self, server_check, test_images, provider):
        """Test streaming responses with vision in OpenAI format."""
        available_models = VisionTestHelper.get_available_models(provider)
        vision_models = [m for m in VISION_MODELS.get(provider, []) if m in available_models]

        if not vision_models:
            pytest.skip(f"No vision models available for {provider}")

        model = vision_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": test_images["shapes"]
                            }
                        }
                    ]
                }
            ],
            "stream": True,
            "max_tokens": 150
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=TIMEOUT)

        assert response.status_code == 200

        # Collect streaming response
        content_parts = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: ') and not line_str.endswith('[DONE]'):
                    try:
                        data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content_parts.append(delta['content'])
                    except json.JSONDecodeError:
                        pass

        full_content = ''.join(content_parts).lower()

        # Should have received streaming content about the image
        assert len(full_content) > 10, "Should receive substantial streaming content"
        assert any(word in full_content for word in ["shape", "circle", "rectangle", "triangle", "blue", "red", "green"]), \
            f"Streaming content should describe image shapes. Got: {full_content}"

class TestVisionErrorHandling:
    """Test error handling for vision requests."""

    def test_invalid_base64_image(self, server_check):
        """Test error handling for invalid base64 image data."""
        payload = {
            "model": "ollama/qwen2.5vl:7b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,INVALID_BASE64_DATA"
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "media_error"

    def test_missing_image_url(self, server_check):
        """Test error handling for missing image URL."""
        payload = {
            "model": "ollama/qwen2.5vl:7b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": ""  # Empty URL
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        # Should either handle gracefully or return appropriate error
        assert response.status_code in [200, 400]

if __name__ == "__main__":
    # Quick validation run
    helper = VisionTestHelper()

    if not helper.is_server_running():
        print("❌ Server not running. Start with: uvicorn abstractcore.server.app:app --port 8000")
        exit(1)

    print("✅ Server is running")

    # Check available models
    for provider in ["ollama", "lmstudio"]:
        models = helper.get_available_models(provider)
        vision_models = [m for m in VISION_MODELS.get(provider, []) if m in models]
        print(f"📝 {provider.title()} vision models available: {vision_models}")

    print("\n🧪 Run tests with: pytest tests/server/media-vision.py -v")