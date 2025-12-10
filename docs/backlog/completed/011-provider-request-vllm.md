This proposal outlines the architectural changes required to add a dedicated `VLLMProvider` to **AbstractCore**.

While vLLM offers an OpenAI-compatible endpoint, treating it as a generic OpenAI target masks its most powerful features (Dynamic LoRA, Guided Decoding, and Beam Search). This proposal leverages the existing `AbstractCore` structure found in `llms-full.txt` to integrate vLLM as a first-class citizen.

-----

# RFC: Dedicated vLLM Provider Integration for AbstractCore

## 1\. Executive Summary

**Objective:** Implement a dedicated `VLLMProvider` in AbstractCore to unlock high-performance inference capabilities specifically for self-hosted GPU clusters (e.g., AWS EC2, Kubernetes).

**Current State:** Users must use `OpenAIProvider` with a modified `base_url`.
**Problem:** This approach "sanitizes" requests, stripping away vLLM-exclusive parameters like `beam_search`, `guided_regex`, and prevents access to management endpoints (e.g., loading Adapters/LoRA at runtime).

**Proposed Solution:** Create `VLLMProvider` inheriting from `OpenAIProvider` that exposes these raw capabilities while maintaining the unified AbstractCore interface.

-----

## 2\. Gap Analysis: Why `OpenAIProvider` is Not Enough

| Feature | Generic OpenAI Provider | Proposed vLLM Provider | Impact |
| :--- | :--- | :--- | :--- |
| **Multi-LoRA** | ❌ Impossible | ✅ `load_adapter("sql-expert")` | Allows 1 model to act as 50 specialized agents without reloading. |
| **Guided Decoding** | ❌ Limited to `json_object` | ✅ Regex, Grammars, JSON Schemas | 100% syntax-safe code generation (vital for `Qwen-Coder`). |
| **Beam Search** | ❌ Ignored | ✅ `best_of=n`, `use_beam_search` | Higher accuracy for complex coding tasks. |
| **Parsing** | ❌ Text blob | ✅ `<think>` tag separation | Clean UI separation of "Reasoning" vs "Answer". |
| **Latency** | ⚠️ HTTP Overhead | ✅ Raw Client Optimization | Potential for lower-latency implementations if skipping standard SDK. |

-----

## 3\. Implementation Specification

### A. Core Class Structure

We will introduce `vllm_provider.py`. Instead of rewriting the connection logic, we inherit from `OpenAIProvider` to reuse the robust HTTP/Client handling but override the payload construction.

**Location:** `abstractcore/providers/vllm_provider.py`

```python
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.models.response import LLMResponse

class VLLMProvider(OpenAIProvider):
    def __init__(self, model: str, base_url: str = None, **kwargs):
        # Default to localhost if no URL provided, supporting the Docker setup
        base_url = base_url or "http://localhost:8000/v1"
        super().__init__(model=model, base_url=base_url, api_key="EMPTY", **kwargs)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # 1. Intercept AbstractCore generic params
        # 2. Map 'guided_regex' and 'guided_json' to vLLM's 'extra_body'
        if "guided_regex" in kwargs:
            kwargs["extra_body"] = kwargs.get("extra_body", {})
            kwargs["extra_body"]["guided_regex"] = kwargs.pop("guided_regex")
        
        # 3. Handle Beam Search params (not supported by OpenAI client validation usually)
        if "beam_width" in kwargs:
             kwargs["extra_body"] = kwargs.get("extra_body", {})
             kwargs["extra_body"]["use_beam_search"] = True
             kwargs["extra_body"]["best_of"] = kwargs.pop("beam_width")

        return super().generate(prompt, **kwargs)

    # --- NEW CAPABILITIES ---
    
    def load_adapter(self, name: str, path: str):
        """
        Dynamic LoRA loading via vLLM management API.
        """
        import requests
        endpoint = f"{self.client.base_url.replace('/v1', '')}/v1/load_lora_adapter"
        payload = {"lora_name": name, "lora_path": path}
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return f"Adapter {name} loaded."

    def unload_adapter(self, name: str):
        import requests
        endpoint = f"{self.client.base_url.replace('/v1', '')}/v1/unload_lora_adapter"
        requests.post(endpoint, json={"lora_name": name})
```

### B. Factory Update

Update `abstractcore/factories.py` (or `create_llm` implementation) to recognize the `vllm` string.

```python
def create_llm(provider: str, **kwargs):
    if provider == "vllm":
        from abstractcore.providers.vllm_provider import VLLMProvider
        return VLLMProvider(**kwargs)
    # ... existing logic ...
```

-----

## 4\. Configuration Schema Updates

We need to update `~/.abstractcore/config/abstractcore.json` to support vLLM specific defaults, particularly useful for your AWS 4xGPU setup.

**New Config Section:**

```json
{
  "provider_settings": {
    "vllm": {
      "base_url": "http://localhost:8000/v1",
      "default_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
      "timeout": 600,
      "parameters": {
        "max_tokens": 8192,
        "temperature": 0.1,
        "extra_body": {
           "guided_decoding_backend": "outlines" 
        }
      }
    }
  }
}
```

-----

## 5\. Integration with AbstractCore Features

### A. Media Handling (Vision)

vLLM supports OpenAI-compatible vision. The existing `AbstractCore` media handling (parsing images to base64) will work out-of-the-box with `VLLMProvider` inheritance.

  * **Usage:** `llm.generate("Describe this", media=["image.jpg"])` requires **zero changes**.

### B. Tool Calling

vLLM recently added support for tool calling, but it can be brittle depending on the model.

  * **Proposal:** The `VLLMProvider` should override `_get_tool_mode`.
      * If the model is `Qwen2.5` or `Mistral`, use native tool calling.
      * If the model is an older open-source model, fallback to AbstractCore's **"Prompted Tool Calling"** (which injects tools into the system prompt), as this is often more reliable than vLLM's native parser for non-GPT-4 models.

### C. The "Agentic" Workflow (Multi-LoRA)

This is where the new provider shines. You can build a routing agent in AbstractCore:

```python
# abstractcore/agents/dynamic_coder.py

llm = create_llm("vllm", model="Qwen/Qwen2.5-Coder-32B-Instruct")

def solve_ticket(ticket_content):
    # 1. Classify the problem
    category = llm.generate(f"Classify: {ticket_content}", guided_regex="(SQL|Python|React)")
    
    # 2. Switch context dynamically
    if category.content == "SQL":
        # Uses the SQL LoRA adapter without unloading the 32B base model
        return llm.generate(ticket_content, model="sql-adapter") 
    elif category.content == "React":
        return llm.generate(ticket_content, model="frontend-adapter")
```

-----

## 6\. Migration Plan for Your AWS Setup

### Step 1: Update Docker

Switch your `docker-compose.yml` to vLLM (as discussed previously).

### Step 2: Install AbstractCore (Dev Branch)

(Assuming you fork and implement the changes above)

```bash
pip install -e .
```

### Step 3: Run the AbstractCore CLI

You can now use the CLI with the full power of the 4 GPUs:

```bash
# Uses the vLLM provider, defaulting to localhost:8000
abstractcore-chat --provider vllm --model Qwen/Qwen2.5-Coder-32B-Instruct
```

-----

## 7\. Summary of Justification

Creating `VLLMProvider` is **highly justified** because:

1.  **Hardware ROI:** It is the only way to expose features (like tensor-parallelism-aware config and beam search) that justify paying for 4x GPUs.
2.  **Agentic Future:** vLLM's Multi-LoRA capability is the future of efficient local agents. AbstractCore currently lacks a mechanism to use this.
3.  **Code Safety:** Regex-guided generation is critical for coding assistants to prevent syntax errors. Only a dedicated provider can expose this cleanly.