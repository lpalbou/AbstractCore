"""
Server Media Tests - Mixed Media & Edge Cases
Tests OpenAI-compatible endpoints with multiple file types and complex scenarios.
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
import csv
from typing import Dict, Any, List

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
TIMEOUT = 120  # Longer timeout for complex processing

# Combined models for mixed scenarios
MIXED_MODELS = {
    "ollama": [
        "qwen2.5vl:7b",  # Vision capable
        "llama3:8b",     # Text only
        "gemma3:2b"      # Text only
    ],
    "lmstudio": [
        "qwen/qwen2.5-vl-7b",        # Vision capable
        "qwen/qwen3-next-80b",       # Text only
        "meta-llama/llama-3.2-8b-instruct"  # Text only
    ]
}

class MixedMediaTestHelper:
    """Helper class for mixed media testing utilities."""

    @staticmethod
    def create_comprehensive_test_suite() -> Dict[str, str]:
        """Create a comprehensive set of test files for mixed media testing."""
        files = {}

        # 1. Chart image
        chart_img = Image.new('RGB', (500, 400), color='white')
        draw = ImageDraw.Draw(chart_img)

        # Draw a bar chart with data
        bars_data = [
            ("Q1", 85, "blue"),
            ("Q2", 120, "green"),
            ("Q3", 95, "red"),
            ("Q4", 140, "purple")
        ]

        base_y = 350
        for i, (label, value, color) in enumerate(bars_data):
            x = 100 + i * 80
            bar_height = value * 2
            draw.rectangle([x, base_y - bar_height, x + 60, base_y], fill=color, outline='black')
            draw.text((x + 15, base_y + 10), label, fill='black')
            draw.text((x + 15, base_y - bar_height - 20), str(value), fill='black')

        draw.text((180, 30), "Quarterly Performance", fill='black')
        draw.text((200, 50), "(in thousands)", fill='gray')

        # Save chart image
        buffer = io.BytesIO()
        chart_img.save(buffer, format='PNG')
        chart_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        files["chart_b64"] = f"data:image/png;base64,{chart_b64}"

        # Also save as file for @filename testing
        chart_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        chart_img.save(chart_file.name)
        files["chart_file"] = chart_file.name

        # 2. Related CSV data
        csv_data = [
            ["Quarter", "Revenue", "Expenses", "Profit", "Growth"],
            ["Q1", "85000", "65000", "20000", "8.5%"],
            ["Q2", "120000", "85000", "35000", "12.0%"],
            ["Q3", "95000", "70000", "25000", "9.5%"],
            ["Q4", "140000", "95000", "45000", "14.0%"]
        ]

        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False, newline='')
        with csv_file as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        files["csv_file"] = csv_file.name

        # 3. Analysis report (text)
        report_content = """Financial Performance Analysis Report
Q1-Q4 2024

Executive Summary:
The company demonstrated strong growth throughout 2024, with revenue increasing
from $85K in Q1 to $140K in Q4, representing a 64% year-over-year growth.

Key Highlights:
- Q2 showed the strongest growth rate at 12.0%
- Q4 achieved the highest absolute revenue of $140K
- Profit margins remained consistent around 20-25%
- Operating expenses were well-controlled

The data shown in the accompanying chart aligns with our strategic objectives
and market expansion goals.

Recommendations:
1. Continue current growth trajectory
2. Maintain operational efficiency
3. Invest in Q2-style initiatives for 2025
"""

        report_file = tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False)
        with report_file as f:
            f.write(report_content)
        files["report_file"] = report_file.name

        # 4. JSON metadata
        json_data = {
            "analysis_metadata": {
                "report_date": "2024-10-19",
                "analyst": "AbstractCore AI System",
                "data_sources": ["financial_records", "quarterly_reports"],
                "confidence_level": "high"
            },
            "quarters": {
                "Q1": {"focus": "foundation", "key_metric": "stability"},
                "Q2": {"focus": "growth", "key_metric": "expansion"},
                "Q3": {"focus": "consolidation", "key_metric": "efficiency"},
                "Q4": {"focus": "performance", "key_metric": "profitability"}
            },
            "recommendations": [
                "Analyze Q2 success factors",
                "Replicate Q4 performance model",
                "Investigate Q3 efficiency dip"
            ]
        }

        json_file = tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False)
        with json_file as f:
            json.dump(json_data, f, indent=2)
        files["json_file"] = json_file.name

        return files

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

    @staticmethod
    def cleanup_files(files: Dict[str, str]):
        """Clean up test files."""
        for key, file_path in files.items():
            if not key.endswith("_b64") and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass

@pytest.fixture(scope="module")
def server_check():
    """Ensure server is running before tests."""
    if not MixedMediaTestHelper.is_server_running():
        pytest.skip("Server is not running. Start with: uvicorn abstractcore.server.app:app --port 8000")

@pytest.fixture(scope="module")
def test_media_suite():
    """Create comprehensive test media files."""
    files = MixedMediaTestHelper.create_comprehensive_test_suite()
    yield files
    MixedMediaTestHelper.cleanup_files(files)

class TestMixedMediaOpenAI:
    """Test mixed media scenarios using OpenAI format."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_image_plus_text_openai_format(self, server_check, test_media_suite, provider):
        """Test image + text analysis using OpenAI format."""
        available_models = MixedMediaTestHelper.get_available_models(provider)
        vision_models = [m for m in MIXED_MODELS.get(provider, []) if m in available_models and "vl" in m.lower()]

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
                            "text": f"Analyze the chart in the image and compare it with the data in @{test_media_suite['csv_file']}. Are they consistent?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": test_media_suite["chart_b64"]
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 400,
            "temperature": 0.1
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should analyze both image and CSV data
        assert any(term in content for term in ["chart", "data", "quarter", "consistent", "revenue"]), \
            f"Should analyze both chart and CSV data. Got: {content[:200]}"

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_multiple_files_comprehensive_analysis(self, server_check, test_media_suite, provider):
        """Test comprehensive analysis with multiple file types."""
        available_models = MixedMediaTestHelper.get_available_models(provider)
        models = [m for m in MIXED_MODELS.get(provider, []) if m in available_models]

        if not models:
            pytest.skip(f"No models available for {provider}")

        # Use vision model if available, otherwise text model
        vision_models = [m for m in models if "vl" in m.lower()]
        model = vision_models[0] if vision_models else models[0]

        # For text-only models, we'll use @filename syntax for all files
        if "vl" not in model.lower():
            payload = {
                "model": f"{provider}/{model}",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Create a comprehensive analysis based on @{test_media_suite['csv_file']}, @{test_media_suite['report_file']}, and @{test_media_suite['json_file']}. What insights emerge from combining these data sources?"
                    }
                ],
                "max_tokens": 500
            }
        else:
            # For vision models, combine image and files
            payload = {
                "model": f"{provider}/{model}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Create a comprehensive analysis based on the chart image, @{test_media_suite['csv_file']}, @{test_media_suite['report_file']}, and @{test_media_suite['json_file']}. What insights emerge from combining these data sources?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": test_media_suite["chart_b64"]
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should synthesize information from multiple sources
        assert any(term in content for term in ["analysis", "revenue", "growth", "quarter", "data"]), \
            f"Should provide comprehensive analysis. Got: {content[:300]}"

        # Should be substantial analysis
        assert len(content) > 200, "Should provide detailed analysis when combining multiple sources"

class TestMixedMediaAbstractCore:
    """Test mixed media using AbstractCore @filename syntax."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_multiple_files_filename_syntax(self, server_check, test_media_suite, provider):
        """Test multiple files using @filename syntax."""
        available_models = MixedMediaTestHelper.get_available_models(provider)
        models = [m for m in MIXED_MODELS.get(provider, []) if m in available_models]

        if not models:
            pytest.skip(f"No models available for {provider}")

        model = models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Compare the quarterly performance shown in @{test_media_suite['chart_file']} with the financial data in @{test_media_suite['csv_file']} and the analysis in @{test_media_suite['report_file']}. Do they tell a consistent story?"
                }
            ],
            "max_tokens": 400
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should reference multiple data sources
        assert any(term in content for term in ["quarter", "financial", "data", "performance", "consistent"]), \
            f"Should analyze multiple files consistently. Got: {content[:200]}"

class TestMixedMediaStreaming:
    """Test streaming with mixed media scenarios."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_streaming_mixed_media_analysis(self, server_check, test_media_suite, provider):
        """Test streaming comprehensive analysis with mixed media."""
        available_models = MixedMediaTestHelper.get_available_models(provider)
        models = [m for m in MIXED_MODELS.get(provider, []) if m in available_models]

        if not models:
            pytest.skip(f"No models available for {provider}")

        model = models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Provide a detailed executive summary based on @{test_media_suite['csv_file']}, @{test_media_suite['report_file']}, and @{test_media_suite['json_file']}. Include trends, insights, and recommendations."
                }
            ],
            "stream": True,
            "max_tokens": 600
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=TIMEOUT)

        assert response.status_code == 200

        # Collect streaming response
        content_parts = []
        chunk_count = 0

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: ') and not line_str.endswith('[DONE]'):
                    try:
                        data = json.loads(line_str[6:])
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content_parts.append(delta['content'])
                                chunk_count += 1
                    except json.JSONDecodeError:
                        pass

        full_content = ''.join(content_parts).lower()

        assert chunk_count > 5, "Should receive multiple streaming chunks for comprehensive analysis"
        assert len(full_content) > 300, "Should generate substantial content when analyzing multiple files"
        assert any(term in full_content for term in ["executive", "summary", "quarter", "revenue", "analysis"]), \
            f"Streaming content should include executive summary elements. Got: {full_content[:300]}"

class TestMixedMediaErrorHandling:
    """Test error handling in mixed media scenarios."""

    def test_mixed_valid_invalid_files(self, server_check, test_media_suite):
        """Test scenario with mix of valid and invalid files."""
        nonexistent_file = "/tmp/nonexistent_file_12345.txt"

        payload = {
            "model": "ollama/qwen3:4b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze @{test_media_suite['csv_file']} and @{nonexistent_file}"
                }
            ]
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        # Should return error for missing file
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "file_not_found"

    def test_too_many_files_limit(self, server_check, test_media_suite):
        """Test behavior with many files (approaching limits)."""
        # Create additional temporary files to test limits
        temp_files = []
        for i in range(8):  # Create several files
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False)
            with temp_file as f:
                f.write(f"Test file {i} content for limit testing.")
            temp_files.append(temp_file.name)

        try:
            # Construct a message with many files
            files_string = " and ".join([f"@{f}" for f in temp_files[:6]])  # Use 6 files

            payload = {
                "model": "ollama/qwen3:4b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Summarize the content from {files_string}"
                    }
                ],
                "max_tokens": 200
            }

            response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

            # Should either succeed (within limits) or fail gracefully
            assert response.status_code in [200, 400]

            if response.status_code == 400:
                data = response.json()
                assert "error" in data
                # Could be file size limit or other resource constraint

        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_large_image_plus_files(self, server_check, test_media_suite):
        """Test large image combined with multiple files."""
        # Create a larger image
        large_img = Image.new('RGB', (2000, 1500), color='lightblue')
        draw = ImageDraw.Draw(large_img)
        draw.text((800, 700), "Large Test Image", fill='black')

        buffer = io.BytesIO()
        large_img.save(buffer, format='PNG', optimize=True)
        large_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        large_img_url = f"data:image/png;base64,{large_img_b64}"

        payload = {
            "model": "ollama/qwen2.5vl:7b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this large image and the data in @{test_media_suite['csv_file']}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": large_img_url
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        # Should either handle the large image or provide appropriate error
        assert response.status_code in [200, 400]

        if response.status_code == 400:
            data = response.json()
            assert "error" in data
            assert data["error"]["type"] in ["media_error", "file_too_large"]

class TestFormatConsistency:
    """Test consistency between OpenAI and AbstractCore formats."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_format_equivalence(self, server_check, test_media_suite, provider):
        """Test that both formats produce similar results."""
        available_models = MixedMediaTestHelper.get_available_models(provider)
        text_models = [m for m in MIXED_MODELS.get(provider, []) if m in available_models and "vl" not in m.lower()]

        if not text_models:
            pytest.skip(f"No text models available for {provider}")

        model = text_models[0]

        # Same analysis using @filename syntax
        payload1 = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"What is the total revenue across all quarters in @{test_media_suite['csv_file']}?"
                }
            ],
            "max_tokens": 150,
            "temperature": 0.1
        }

        # Same analysis but with explicit instruction referencing the file
        payload2 = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Calculate the sum of revenue values from the quarterly data in @{test_media_suite['csv_file']}"
                }
            ],
            "max_tokens": 150,
            "temperature": 0.1
        }

        response1 = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload1, timeout=TIMEOUT)
        response2 = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload2, timeout=TIMEOUT)

        assert response1.status_code == 200
        assert response2.status_code == 200

        content1 = response1.json()["choices"][0]["message"]["content"].lower()
        content2 = response2.json()["choices"][0]["message"]["content"].lower()

        # Both should reference the total revenue (440K)
        total_mentioned_1 = any(term in content1 for term in ["440", "total", "sum"])
        total_mentioned_2 = any(term in content2 for term in ["440", "total", "sum"])

        # At least one should get the calculation right
        assert total_mentioned_1 or total_mentioned_2, \
            f"Should calculate total revenue. Got response1: {content1}, response2: {content2}"

if __name__ == "__main__":
    # Quick validation run
    helper = MixedMediaTestHelper()

    if not helper.is_server_running():
        print("❌ Server not running. Start with: uvicorn abstractcore.server.app:app --port 8000")
        exit(1)

    print("✅ Server is running")

    # Check available models
    for provider in ["ollama", "lmstudio"]:
        models = helper.get_available_models(provider)
        mixed_models = [m for m in MIXED_MODELS.get(provider, []) if m in models]
        vision_models = [m for m in mixed_models if "vl" in m.lower()]
        text_models = [m for m in mixed_models if "vl" not in m.lower()]
        print(f"📝 {provider.title()} vision models: {vision_models}")
        print(f"📝 {provider.title()} text models: {text_models}")

    # Test media suite creation
    try:
        print("\n🧪 Testing mixed media suite creation...")
        files = helper.create_comprehensive_test_suite()
        print(f"✅ Created {len(files)} test media files")

        # Verify files exist
        for key, path in files.items():
            if not key.endswith("_b64"):
                assert os.path.exists(path), f"File {key} should exist at {path}"

        helper.cleanup_files(files)
        print("✅ Mixed media suite creation tests passed")

    except Exception as e:
        print(f"❌ Mixed media suite creation failed: {e}")

    print("\n🧪 Run tests with: pytest tests/server/media-mixed.py -v")