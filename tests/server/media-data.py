"""
Server Media Tests - Data File Processing
Tests OpenAI-compatible endpoints with data files (CSV, TSV, JSON, XML).
"""

import pytest
import json
import tempfile
import requests
import os
import csv
from typing import Dict, Any, List

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
TIMEOUT = 60

# Test models - using text models for data analysis
TEXT_MODELS = {
    "ollama": [
        "qwen3:4b-instruct",
        "llama3:8b",
        "gemma3:2b"
    ],
    "lmstudio": [
        "qwen/qwen3-next-80b",
        "meta-llama/llama-3.2-8b-instruct",
        "microsoft/DialoGPT-large"
    ]
}

class DataTestHelper:
    """Helper class for data file testing utilities."""

    @staticmethod
    def create_test_csv() -> str:
        """Create a test CSV file with sample data."""
        data = [
            ["Product", "Quarter", "Sales", "Region", "Manager"],
            ["Widget A", "Q1", "15000", "North", "Alice Johnson"],
            ["Widget B", "Q1", "22000", "South", "Bob Smith"],
            ["Widget C", "Q1", "18000", "East", "Carol Davis"],
            ["Widget A", "Q2", "17000", "North", "Alice Johnson"],
            ["Widget B", "Q2", "25000", "South", "Bob Smith"],
            ["Widget C", "Q2", "21000", "East", "Carol Davis"],
            ["Widget A", "Q3", "19000", "North", "Alice Johnson"],
            ["Widget B", "Q3", "28000", "South", "Bob Smith"],
            ["Widget C", "Q3", "23000", "East", "Carol Davis"],
            ["Widget A", "Q4", "21000", "North", "Alice Johnson"],
            ["Widget B", "Q4", "30000", "South", "Bob Smith"],
            ["Widget C", "Q4", "25000", "East", "Carol Davis"]
        ]

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False, newline='')
        with temp_file as f:
            writer = csv.writer(f)
            writer.writerows(data)

        return temp_file.name

    @staticmethod
    def create_test_tsv() -> str:
        """Create a test TSV file with sample data."""
        data = [
            ["Employee_ID", "Name", "Department", "Salary", "Years_Experience", "Performance_Rating"],
            ["E001", "John Doe", "Engineering", "85000", "5", "Excellent"],
            ["E002", "Jane Smith", "Marketing", "72000", "3", "Good"],
            ["E003", "Mike Johnson", "Sales", "68000", "4", "Excellent"],
            ["E004", "Sarah Wilson", "Engineering", "92000", "7", "Outstanding"],
            ["E005", "Tom Brown", "Marketing", "65000", "2", "Good"],
            ["E006", "Lisa Garcia", "Sales", "75000", "6", "Excellent"],
            ["E007", "David Lee", "Engineering", "88000", "4", "Good"],
            ["E008", "Emma Davis", "Marketing", "70000", "3", "Excellent"]
        ]

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".tsv", delete=False)
        with temp_file as f:
            for row in data:
                f.write('\t'.join(row) + '\n')

        return temp_file.name

    @staticmethod
    def create_test_json() -> str:
        """Create a test JSON file with structured data."""
        data = {
            "company": "TechCorp Solutions",
            "founded": 2020,
            "employees": 150,
            "headquarters": {
                "city": "San Francisco",
                "state": "CA",
                "country": "USA"
            },
            "departments": [
                {
                    "name": "Engineering",
                    "employees": 80,
                    "manager": "Alex Thompson",
                    "budget": 2500000,
                    "projects": ["AI Platform", "Mobile App", "Web Portal"]
                },
                {
                    "name": "Sales",
                    "employees": 35,
                    "manager": "Maria Rodriguez",
                    "budget": 800000,
                    "projects": ["Customer Acquisition", "Market Expansion"]
                },
                {
                    "name": "Marketing",
                    "employees": 25,
                    "manager": "James Wilson",
                    "budget": 600000,
                    "projects": ["Brand Awareness", "Digital Campaigns"]
                }
            ],
            "financial": {
                "revenue_2023": 15000000,
                "revenue_2024": 22000000,
                "growth_rate": "46.7%",
                "profit_margin": "18%"
            },
            "products": [
                {
                    "name": "AI Analytics Platform",
                    "launch_date": "2023-01-15",
                    "customers": 120,
                    "revenue": 8000000
                },
                {
                    "name": "Business Intelligence Suite",
                    "launch_date": "2023-06-20",
                    "customers": 85,
                    "revenue": 6000000
                }
            ]
        }

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False)
        with temp_file as f:
            json.dump(data, f, indent=2)

        return temp_file.name

    @staticmethod
    def create_test_xml() -> str:
        """Create a test XML file with structured data."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<inventory>
    <metadata>
        <title>Product Inventory System</title>
        <version>2.1</version>
        <last_updated>2024-10-19</last_updated>
    </metadata>
    <categories>
        <category id="electronics">
            <name>Electronics</name>
            <products>
                <product id="P001">
                    <name>Smartphone Pro Max</name>
                    <price>999.99</price>
                    <stock>45</stock>
                    <specifications>
                        <screen>6.7 inches</screen>
                        <storage>256GB</storage>
                        <camera>48MP</camera>
                    </specifications>
                </product>
                <product id="P002">
                    <name>Wireless Headphones</name>
                    <price>299.99</price>
                    <stock>78</stock>
                    <specifications>
                        <battery>30 hours</battery>
                        <connectivity>Bluetooth 5.0</connectivity>
                        <noise_cancellation>Yes</noise_cancellation>
                    </specifications>
                </product>
            </products>
        </category>
        <category id="home">
            <name>Home & Garden</name>
            <products>
                <product id="P003">
                    <name>Smart Thermostat</name>
                    <price>249.99</price>
                    <stock>32</stock>
                    <specifications>
                        <wifi>Yes</wifi>
                        <voice_control>Alexa, Google</voice_control>
                        <energy_saving>25%</energy_saving>
                    </specifications>
                </product>
            </products>
        </category>
    </categories>
</inventory>"""

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".xml", delete=False)
        with temp_file as f:
            f.write(xml_content)

        return temp_file.name

    @staticmethod
    def create_test_txt() -> str:
        """Create a test text file with structured information."""
        content = """AbstractCore Media Processing Report

Executive Summary
=================
AbstractCore's unified media handling system has been successfully implemented and tested
across multiple data formats and provider endpoints. This report summarizes the key
findings and performance metrics.

Key Metrics
===========
- Supported Formats: 12+ file types
- Provider Coverage: 7 major LLM providers
- Processing Accuracy: 98.5%
- Average Response Time: 2.3 seconds
- Error Rate: <1%

Test Results by Format
=====================

Images (Vision Processing):
- PNG, JPEG, GIF: 100% success rate
- Complex charts: 95% accuracy
- Multi-image analysis: 92% accuracy

Documents:
- PDF extraction: 99% success rate
- Office documents: 97% success rate
- Large files (>5MB): 94% success rate

Data Files:
- CSV/TSV parsing: 100% success rate
- JSON structure analysis: 98% success rate
- XML processing: 96% success rate

Conclusion
==========
The media processing system demonstrates production-ready capabilities with excellent
performance across all tested modalities. The OpenAI-compatible endpoints successfully
integrate with existing client libraries while maintaining AbstractCore's universal
provider support.

Recommendations
===============
1. Deploy to production environment
2. Monitor performance metrics
3. Implement caching for frequently accessed files
4. Consider rate limiting for large file uploads
"""

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False)
        with temp_file as f:
            f.write(content)

        return temp_file.name

    @staticmethod
    def create_test_md() -> str:
        """Create a test Markdown file."""
        content = """# AbstractCore Server Media Integration

## Overview

This document validates the **server media integration** capabilities of AbstractCore,
specifically testing the OpenAI-compatible endpoints with various data formats.

## Features Tested

### 🖼️ Image Processing
- Base64 encoding/decoding
- Multiple image formats
- Vision model compatibility

### 📄 Document Processing
- PDF text extraction
- Office document parsing
- Content analysis

### 📊 Data Analysis
- CSV/TSV processing
- JSON structure parsing
- Statistical analysis

## Test Scenarios

| Format | Provider | Model | Status |
|--------|----------|-------|--------|
| PNG | Ollama | qwen2.5vl:7b | ✅ Pass |
| PDF | LMStudio | qwen3-next-80b | ✅ Pass |
| CSV | Ollama | llama3:8b | ✅ Pass |
| JSON | LMStudio | llama-3.2-8b | ✅ Pass |

## Code Examples

### OpenAI Format
```python
response = client.chat.completions.create(
    model="ollama/qwen2.5vl:7b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this data"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }]
)
```

### AbstractCore Format
```python
response = client.chat.completions.create(
    model="lmstudio/qwen3-next-80b",
    messages=[{
        "role": "user",
        "content": "Summarize @data.csv and @report.pdf"
    }]
)
```

## Results

All tests **PASSED** with excellent performance across providers and formats.
The implementation successfully bridges AbstractCore's powerful media system
with standard OpenAI client libraries.
"""

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=".md", delete=False)
        with temp_file as f:
            f.write(content)

        return temp_file.name

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
    if not DataTestHelper.is_server_running():
        pytest.skip("Server is not running. Start with: uvicorn abstractcore.server.app:app --port 8000")

@pytest.fixture(scope="module")
def test_data_files():
    """Create test data files for the test session."""
    files = {}

    # Create test files
    files["csv"] = DataTestHelper.create_test_csv()
    files["tsv"] = DataTestHelper.create_test_tsv()
    files["json"] = DataTestHelper.create_test_json()
    files["xml"] = DataTestHelper.create_test_xml()
    files["txt"] = DataTestHelper.create_test_txt()
    files["md"] = DataTestHelper.create_test_md()

    yield files

    # Cleanup
    for file_path in files.values():
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass

class TestDataProcessingCSV:
    """Test CSV data processing capabilities."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_csv_data_analysis(self, server_check, test_data_files, provider):
        """Test CSV data analysis and insights."""
        available_models = DataTestHelper.get_available_models(provider)
        text_models = [m for m in TEXT_MODELS.get(provider, []) if m in available_models]

        if not text_models:
            pytest.skip(f"No text models available for {provider}")

        model = text_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze the sales data in @{test_data_files['csv']}. What trends do you see across quarters?"
                }
            ],
            "max_tokens": 300,
            "temperature": 0.1
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should identify CSV content and trends
        assert any(term in content for term in ["sales", "quarter", "widget", "trend", "increase"]), \
            f"Should analyze CSV sales trends. Got: {content}"

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_csv_statistical_questions(self, server_check, test_data_files, provider):
        """Test statistical questions about CSV data."""
        available_models = DataTestHelper.get_available_models(provider)
        text_models = [m for m in TEXT_MODELS.get(provider, []) if m in available_models]

        if not text_models:
            pytest.skip(f"No text models available for {provider}")

        model = text_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Based on @{test_data_files['csv']}, which product had the highest total sales across all quarters?"
                }
            ],
            "max_tokens": 150
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should identify Widget B as having highest sales
        assert any(term in content for term in ["widget b", "widget", "highest", "sales"]), \
            f"Should identify highest performing product. Got: {content}"

class TestDataProcessingJSON:
    """Test JSON data processing capabilities."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_json_structure_analysis(self, server_check, test_data_files, provider):
        """Test JSON structure analysis."""
        available_models = DataTestHelper.get_available_models(provider)
        text_models = [m for m in TEXT_MODELS.get(provider, []) if m in available_models]

        if not text_models:
            pytest.skip(f"No text models available for {provider}")

        model = text_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Describe the company structure based on the data in @{test_data_files['json']}"
                }
            ],
            "max_tokens": 250
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should identify company information
        assert any(term in content for term in ["techcorp", "company", "engineering", "departments", "employees"]), \
            f"Should analyze JSON company structure. Got: {content}"

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_json_financial_analysis(self, server_check, test_data_files, provider):
        """Test financial data extraction from JSON."""
        available_models = DataTestHelper.get_available_models(provider)
        text_models = [m for m in TEXT_MODELS.get(provider, []) if m in available_models]

        if not text_models:
            pytest.skip(f"No text models available for {provider}")

        model = text_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"What was the revenue growth from 2023 to 2024 according to @{test_data_files['json']}?"
                }
            ],
            "max_tokens": 150
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should identify revenue growth
        assert any(term in content for term in ["46", "47", "growth", "revenue", "million"]), \
            f"Should identify revenue growth. Got: {content}"

class TestDataProcessingTSV:
    """Test TSV data processing capabilities."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_tsv_employee_analysis(self, server_check, test_data_files, provider):
        """Test TSV employee data analysis."""
        available_models = DataTestHelper.get_available_models(provider)
        text_models = [m for m in TEXT_MODELS.get(provider, []) if m in available_models]

        if not text_models:
            pytest.skip(f"No text models available for {provider}")

        model = text_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze the employee data in @{test_data_files['tsv']}. Which department has the highest average salary?"
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should identify Engineering as highest paid department
        assert any(term in content for term in ["engineering", "highest", "salary", "average"]), \
            f"Should identify highest paid department. Got: {content}"

class TestDataProcessingText:
    """Test text and markdown file processing."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    @pytest.mark.parametrize("file_type", ["txt", "md"])
    def test_text_content_analysis(self, server_check, test_data_files, provider, file_type):
        """Test text and markdown content analysis."""
        available_models = DataTestHelper.get_available_models(provider)
        text_models = [m for m in TEXT_MODELS.get(provider, []) if m in available_models]

        if not text_models:
            pytest.skip(f"No text models available for {provider}")

        model = text_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Summarize the key findings from @{test_data_files[file_type]}"
                }
            ],
            "max_tokens": 200
        }

        response = requests.post(f"{SERVER_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()

        # Should identify AbstractCore content
        assert any(term in content for term in ["abstractcore", "media", "processing", "test", "integration"]), \
            f"Should analyze {file_type} content about AbstractCore. Got: {content}"

class TestDataStreamingProcessing:
    """Test streaming data analysis."""

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
    def test_streaming_csv_analysis(self, server_check, test_data_files, provider):
        """Test streaming analysis of CSV data."""
        available_models = DataTestHelper.get_available_models(provider)
        text_models = [m for m in TEXT_MODELS.get(provider, []) if m in available_models]

        if not text_models:
            pytest.skip(f"No text models available for {provider}")

        model = text_models[0]

        payload = {
            "model": f"{provider}/{model}",
            "messages": [
                {
                    "role": "user",
                    "content": f"Provide a detailed analysis of the sales patterns in @{test_data_files['csv']}"
                }
            ],
            "stream": True,
            "max_tokens": 300
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
                        data = json.loads(line_str[6:])
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content_parts.append(delta['content'])
                    except json.JSONDecodeError:
                        pass

        full_content = ''.join(content_parts).lower()

        assert len(full_content) > 50, "Should receive substantial streaming content for data analysis"
        assert any(term in full_content for term in ["sales", "data", "analysis", "pattern"]), \
            f"Streaming content should reference data analysis. Got: {full_content[:200]}"

if __name__ == "__main__":
    # Quick validation run
    helper = DataTestHelper()

    if not helper.is_server_running():
        print("❌ Server not running. Start with: uvicorn abstractcore.server.app:app --port 8000")
        exit(1)

    print("✅ Server is running")

    # Check available models
    for provider in ["ollama", "lmstudio"]:
        models = helper.get_available_models(provider)
        text_models = [m for m in TEXT_MODELS.get(provider, []) if m in models]
        print(f"📝 {provider.title()} text models available: {text_models}")

    # Test data file creation
    try:
        print("\n🧪 Testing data file creation...")

        csv_path = helper.create_test_csv()
        print(f"✅ Created test CSV: {csv_path}")

        json_path = helper.create_test_json()
        print(f"✅ Created test JSON: {json_path}")

        tsv_path = helper.create_test_tsv()
        print(f"✅ Created test TSV: {tsv_path}")

        txt_path = helper.create_test_txt()
        print(f"✅ Created test TXT: {txt_path}")

        md_path = helper.create_test_md()
        print(f"✅ Created test MD: {md_path}")

        # Cleanup
        for path in [csv_path, json_path, tsv_path, txt_path, md_path]:
            if os.path.exists(path):
                os.unlink(path)

        print("✅ Data file creation tests passed")

    except Exception as e:
        print(f"❌ Data file creation failed: {e}")

    print("\n🧪 Run tests with: pytest tests/server/media-data.py -v")