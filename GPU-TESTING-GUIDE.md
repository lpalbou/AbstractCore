# GPU Testing Guide - vLLM Provider

Complete guide for testing the vLLM provider on your GPU instance.

## 📋 Prerequisites

- NVIDIA GPU with CUDA support
- vLLM installed: `pip install vllm`
- Python 3.9+

---

## 🚀 Step-by-Step Setup

### 1. Clone the Branch

```bash
# Clone the vllm-provider branch directly
git clone -b vllm-provider https://github.com/lpalbou/AbstractCore.git
cd AbstractCore

# Or if you already have the repo:
cd AbstractCore
git fetch origin
git checkout vllm-provider
git pull origin vllm-provider
```

### 2. Install AbstractCore

```bash
# Install in development mode
pip install -e .

# Verify installation
python -c "from abstractcore.providers import VLLMProvider; print('✅ VLLMProvider installed')"
```

### 3. Start vLLM Server

Choose the configuration that matches your GPU setup:

#### Single GPU (Basic)
```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

#### Single GPU (With LoRA Support)
```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-lora \
    --max-loras 4
```

#### 4 GPUs (Production Setup) - **RECOMMENDED**
```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-lora \
    --max-loras 4 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192
```

#### With API Key (Secure)
```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key your-secret-key \
    --tensor-parallel-size 4
```

**Wait for server to fully load** (you'll see "Connected to the server")

---

## 🧪 Testing Methods

### Method 1: Automated Test Script (RECOMMENDED)

```bash
# Run the comprehensive test script
python test-gpu.py
```

This script will:
1. ✅ Test vLLM provider connectivity
2. ✅ Test basic generation, streaming, guided JSON
3. ✅ Start AbstractCore server on port 8080
4. ✅ Display curl examples for manual testing
5. ✅ Keep server running for testing

**Expected Output:**
```
================================================================================
  STEP 1: Testing vLLM Provider Connectivity
================================================================================

Creating vLLM provider instance...
✅ Provider created successfully
   Provider: vllm
   Model: Qwen/Qwen3-Coder-30B-A3B-Instruct
   Base URL: http://localhost:8000/v1

Listing available models from vLLM server...
✅ Found 1 model(s):
   - Qwen/Qwen3-Coder-30B-A3B-Instruct

Testing basic generation...
✅ Response: Hello from vLLM!

Testing streaming generation...
Response: 1 2 3 4 5
✅ Streaming works

Testing guided JSON (vLLM-specific)...
✅ Guided JSON response: {"colors": ["red", "blue", "green"]}

🎉 All vLLM provider tests passed!

================================================================================
  STEP 2: Starting AbstractCore Server
================================================================================

Starting server on port 8080...
✅ Server starting on http://0.0.0.0:8080
```

---

### Method 2: Run Official Test Suite

```bash
# Run all vLLM provider tests
pytest tests/providers/test_vllm_provider.py -v

# Run specific test class
pytest tests/providers/test_vllm_provider.py::TestVLLMProviderBasics -v

# Run with detailed output
pytest tests/providers/test_vllm_provider.py -vv -s
```

**Expected Output:**
```
tests/providers/test_vllm_provider.py::TestVLLMProviderBasics::test_provider_initialization PASSED
tests/providers/test_vllm_provider.py::TestVLLMProviderBasics::test_list_available_models PASSED
tests/providers/test_vllm_provider.py::TestVLLMGeneration::test_basic_generation PASSED
tests/providers/test_vllm_provider.py::TestVLLMGeneration::test_streaming_generation PASSED
tests/providers/test_vllm_provider.py::TestVLLMGuidedDecoding::test_guided_json PASSED
...
```

---

### Method 3: Manual Python Testing

Create a file `test_vllm_manual.py`:

```python
from abstractcore import create_llm

# Create vLLM provider
llm = create_llm('vllm', model='Qwen/Qwen3-Coder-30B-A3B-Instruct')

# 1. Basic generation
print("1. Basic generation:")
response = llm.generate("What is vLLM?", temperature=0.7, max_tokens=100)
print(f"   {response.content}\n")

# 2. Streaming
print("2. Streaming:")
print("   ", end="")
for chunk in llm.generate("Count from 1 to 5", stream=True, temperature=0):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print("\n")

# 3. Guided JSON (vLLM-specific)
print("3. Guided JSON:")
response = llm.generate(
    "List 3 programming languages",
    guided_json={
        "type": "object",
        "properties": {
            "languages": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["languages"]
    },
    temperature=0
)
print(f"   {response.content}\n")

# 4. Guided regex (vLLM-specific)
print("4. Guided regex:")
response = llm.generate(
    "Is Python a programming language? Answer yes or no.",
    guided_regex="(yes|no)",
    temperature=0
)
print(f"   {response.content}\n")

# 5. Beam search (vLLM-specific)
print("5. Beam search:")
response = llm.generate(
    "Write a creative one-line tagline for a coding bootcamp",
    use_beam_search=True,
    best_of=5,
    temperature=0.8
)
print(f"   {response.content}\n")

# 6. Async generation
import asyncio

async def test_async():
    print("6. Async generation:")
    response = await llm.agenerate("Hello async world!", temperature=0)
    print(f"   {response.content}\n")

asyncio.run(test_async())

print("✅ All manual tests completed!")
```

Run it:
```bash
python test_vllm_manual.py
```

---

## 🌐 Test OpenAI-Compatible Endpoint

Once `test-gpu.py` is running (or you start the server separately), test with curl:

### Basic Request
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {"role": "user", "content": "What is vLLM?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

### Streaming
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {"role": "user", "content": "Count from 1 to 10"}
    ],
    "stream": true,
    "temperature": 0
  }'
```

### List Models
```bash
curl http://localhost:8080/v1/models
```

### Using OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## 🎯 What to Test (Priority Order)

### ✅ Must Test (Critical)
1. **vLLM server connectivity** - `llm.list_available_models()`
2. **Basic generation** - `llm.generate("Hello")`
3. **Streaming** - `llm.generate("Test", stream=True)`
4. **Server endpoint** - curl to `/v1/chat/completions`

### ⭐ Should Test (Important)
5. **Guided JSON** - `llm.generate(..., guided_json={...})`
6. **Guided regex** - `llm.generate(..., guided_regex="(yes|no)")`
7. **Async generation** - `await llm.agenerate("Test")`
8. **Batch concurrent** - Multiple async requests

### 💡 Nice to Test (Advanced)
9. **Beam search** - `llm.generate(..., use_beam_search=True, best_of=5)`
10. **LoRA adapters** - `llm.load_adapter("name", "/path")` (if enabled)
11. **Structured output** - With Pydantic models
12. **Multi-GPU performance** - Compare throughput

---

## 📊 Success Criteria

You should see:

- ✅ vLLM provider creates without errors
- ✅ Models list returns at least 1 model
- ✅ Basic generation returns coherent text
- ✅ Streaming generates text incrementally
- ✅ Guided JSON returns valid JSON
- ✅ Guided regex respects pattern constraints
- ✅ Server starts on port 8080
- ✅ curl requests return proper responses
- ✅ No Python errors or exceptions

**Performance Expectations** (4 GPUs with tensor parallelism):
- First token latency: ~100-500ms
- Throughput: ~100-200 tokens/sec
- Concurrent requests: 10-50 simultaneous users

---

## 🐛 Troubleshooting

### vLLM Server Not Starting
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Try with smaller model first
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

### Connection Refused
```bash
# Check vLLM server is running
curl http://localhost:8000/v1/models

# Check if port is in use
lsof -i :8000
```

### Model Not Found
```bash
# List available models
curl http://localhost:8000/v1/models

# Use exact model name from list
python -c "
from abstractcore import create_llm
llm = create_llm('vllm')
print(llm.list_available_models())
"
```

### Out of Memory
```bash
# Reduce GPU memory utilization
vllm serve model --gpu-memory-utilization 0.7

# Reduce max model length
vllm serve model --max-model-len 4096

# Use fewer GPUs
vllm serve model --tensor-parallel-size 2
```

### Guided Decoding Not Working
```bash
# Check vLLM version (needs 0.2.0+)
vllm --version

# Update vLLM if needed
pip install --upgrade vllm
```

---

## 📝 Report Back

After testing, please report:

### ✅ Working
- Which tests passed
- Response times (approximate)
- Any special configuration used

### ❌ Issues
- Error messages (full traceback)
- vLLM version: `vllm --version`
- CUDA version: `nvcc --version`
- GPU model: `nvidia-smi`
- Which test failed

### 📊 Performance
- Tokens per second (if measured)
- First token latency
- Number of concurrent requests handled
- GPU utilization during inference

---

## 🎉 Expected Results

If everything works correctly, you should see:

```bash
$ python test-gpu.py

================================================================================
  vLLM Provider GPU Test Script
  AbstractCore + vLLM OpenAI-Compatible Server
================================================================================

vLLM Server URL: http://localhost:8000/v1

================================================================================
  STEP 1: Testing vLLM Provider Connectivity
================================================================================

Creating vLLM provider instance...
✅ Provider created successfully
✅ Found 1 model(s): Qwen/Qwen3-Coder-30B-A3B-Instruct
✅ Response: Hello from vLLM!
✅ Streaming works
✅ Guided JSON response: {"colors": ["red", "blue", "green"]}
✅ Capabilities: streaming, chat, tools, structured_output, guided_decoding, multi_lora, beam_search

🎉 All vLLM provider tests passed!

================================================================================
  STEP 2: Starting AbstractCore Server
================================================================================

✅ Server starting on http://0.0.0.0:8080

[Server logs will appear here]
```

---

## 🚀 Next Steps After Testing

1. **If tests pass**: Merge the `vllm-provider` branch to main
2. **Add to documentation**: Update README with vLLM provider info
3. **Performance tuning**: Optimize vLLM settings for your GPU setup
4. **Deploy to production**: Use in your GPU cluster

Good luck with testing! 🎯
