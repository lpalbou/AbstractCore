# Async/Await Guide

Complete guide to using async/await with AbstractCore for concurrent LLM operations.

## Overview

AbstractCore exposes `agenerate()` for async generation across providers.

- **HTTP-based providers** (OpenAI-compatible endpoints, OpenRouter, Ollama, LMStudio, vLLM, etc.) implement native async I/O.
- **In-process local inference** providers (MLX, HuggingFace) use an `asyncio.to_thread()` fallback to avoid blocking the event loop.

Concurrency can improve throughput when requests are **I/O-bound** (network calls). For local inference, throughput is limited by your hardware and the model runtime.

## Provider support

| Provider | Async implementation |
|----------|----------------------|
| `openai`, `anthropic` | Native async SDK clients (when installed) |
| HTTP-based providers (`ollama`, `lmstudio`, `openrouter`, `vllm`, `openai-compatible`, …) | `httpx.AsyncClient` (native async HTTP) |
| `mlx`, `huggingface` | `asyncio.to_thread()` fallback (keeps the event loop responsive) |

## Basic Usage

### Single Async Request

```python
import asyncio
from abstractcore import create_llm

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")

    # Single async request
    response = await llm.agenerate("What is Python?")
    print(response.content)

asyncio.run(main())
```

### Concurrent Requests

```python
import asyncio
from abstractcore import create_llm

async def main():
    llm = create_llm("ollama", model="qwen3:4b")

    # Execute 3 requests concurrently
    tasks = [
        llm.agenerate(f"Summarize {topic}")
        for topic in ["Python", "JavaScript", "Rust"]
    ]

    # Gather runs all tasks concurrently
    responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses):
        print(f"\n{['Python', 'JavaScript', 'Rust'][i]}:")
        print(response.content)

asyncio.run(main())
```

## Async Streaming

### Basic Streaming

```python
import asyncio
from abstractcore import create_llm

async def main():
    llm = create_llm("anthropic", model="claude-haiku-4-5")

    # Step 1: await the generator
    stream_gen = await llm.agenerate(
        "Write a haiku about coding",
        stream=True
    )

    # Step 2: async for over the chunks
    async for chunk in stream_gen:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

asyncio.run(main())
```

### Concurrent Streaming

```python
import asyncio
from abstractcore import create_llm

async def stream_response(llm, topic, label):
    """Stream a single response with label."""
    print(f"\n{label}:")

    stream_gen = await llm.agenerate(f"Explain {topic} in one sentence", stream=True)

    async for chunk in stream_gen:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")

    # Stream 3 responses concurrently
    await asyncio.gather(
        stream_response(llm, "Python", "Python"),
        stream_response(llm, "JavaScript", "JavaScript"),
        stream_response(llm, "Rust", "Rust")
    )

asyncio.run(main())
```

## Session Async

### Async Conversation Management

```python
import asyncio
from abstractcore import create_llm
from abstractcore.core.session import BasicSession

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")
    session = BasicSession(provider=llm)

    # Maintain conversation history with async
    response1 = await session.agenerate("What is Python?")
    print(response1.content)

    response2 = await session.agenerate("What are its main use cases?")
    print(response2.content)

    # Session tracks full conversation history
    print(f"\nConversation length: {len(session.conversation_history)} messages")

asyncio.run(main())
```

### Concurrent Sessions

```python
import asyncio
from abstractcore import create_llm
from abstractcore.core.session import BasicSession

async def chat_session(llm, topic, name):
    """Run independent chat session."""
    session = BasicSession(provider=llm)

    response1 = await session.agenerate(f"What is {topic}?")
    response2 = await session.agenerate("Give me a simple example")

    print(f"\n{name}:")
    print(f"  Question 1: {response1.content[:50]}...")
    print(f"  Question 2: {response2.content[:50]}...")

async def main():
    llm = create_llm("anthropic", model="claude-haiku-4-5")

    # Run 3 independent conversations concurrently
    await asyncio.gather(
        chat_session(llm, "Python", "Session 1"),
        chat_session(llm, "JavaScript", "Session 2"),
        chat_session(llm, "Rust", "Session 3")
    )

asyncio.run(main())
```

## Multi-Provider Comparisons

### Concurrent Provider Queries

```python
import asyncio
from abstractcore import create_llm

async def query_provider(provider_name, model, prompt):
    """Query a single provider."""
    llm = create_llm(provider_name, model=model)
    response = await llm.agenerate(prompt)
    return {
        "provider": provider_name,
        "model": model,
        "response": response.content
    }

async def main():
    prompt = "What is the capital of France?"

    # Query multiple providers simultaneously
    results = await asyncio.gather(
        query_provider("openai", "gpt-4o-mini", prompt),
        query_provider("anthropic", "claude-haiku-4-5", prompt),
        query_provider("ollama", "qwen3:4b", prompt)
    )

    for result in results:
        print(f"\n{result['provider']} ({result['model']}):")
        print(result['response'])

asyncio.run(main())
```

### Provider Consensus

```python
import asyncio
from abstractcore import create_llm

async def main():
    prompt = "Is the Earth flat? Answer yes or no."

    # Get consensus from 3 providers
    llm_openai = create_llm("openai", model="gpt-4o-mini")
    llm_anthropic = create_llm("anthropic", model="claude-haiku-4-5")
    llm_ollama = create_llm("ollama", model="qwen3:4b")

    responses = await asyncio.gather(
        llm_openai.agenerate(prompt),
        llm_anthropic.agenerate(prompt),
        llm_ollama.agenerate(prompt)
    )

    answers = [r.content.strip().lower() for r in responses]
    print(f"Answers: {answers}")
    print(f"Consensus: {'Yes' if answers.count('no') >= 2 else 'No'}")

asyncio.run(main())
```

## FastAPI Integration

### Async HTTP Endpoints

```python
from fastapi import FastAPI
from abstractcore import create_llm

app = FastAPI()
llm = create_llm("openai", model="gpt-4o-mini")

@app.post("/generate")
async def generate(prompt: str):
    """Non-blocking LLM generation endpoint."""
    response = await llm.agenerate(prompt)
    return {"response": response.content}

@app.post("/batch")
async def batch_generate(prompts: list[str]):
    """Process multiple prompts concurrently."""
    tasks = [llm.agenerate(p) for p in prompts]
    responses = await asyncio.gather(*tasks)

    return {
        "responses": [r.content for r in responses]
    }

# Run with: uvicorn your_app:app --reload
```

### Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from abstractcore import create_llm
import asyncio

app = FastAPI()
llm = create_llm("anthropic", model="claude-haiku-4-5")

async def stream_response(prompt: str):
    """Generate streaming response."""
    stream_gen = await llm.agenerate(prompt, stream=True)

    async for chunk in stream_gen:
        if chunk.content:
            yield f"data: {chunk.content}\n\n"

@app.post("/stream")
async def stream_generate(prompt: str):
    """Streaming LLM generation endpoint."""
    return StreamingResponse(
        stream_response(prompt),
        media_type="text/event-stream"
    )
```

## Batch Document Processing

### Concurrent Document Summaries

```python
import asyncio
from abstractcore import create_llm
from abstractcore.processing import Summarizer

async def summarize_document(summarizer, doc_path):
    """Summarize single document."""
    result = summarizer.summarize(
        input_source=doc_path,
        style="executive",
        length="brief"
    )
    return {
        "path": doc_path,
        "summary": result.summary
    }

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")
    summarizer = Summarizer(llm)

    documents = [
        "report1.pdf",
        "report2.pdf",
        "report3.pdf"
    ]

    # Summarize all documents concurrently
    tasks = [summarize_document(summarizer, doc) for doc in documents]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"\n{result['path']}:")
        print(result['summary'])

asyncio.run(main())
```

## Error Handling

### Graceful Error Recovery

```python
import asyncio
from abstractcore import create_llm
from abstractcore.exceptions import ProviderAPIError

async def safe_generate(llm, prompt, label):
    """Generate with error handling."""
    try:
        response = await llm.agenerate(prompt)
        return {"label": label, "content": response.content, "error": None}
    except ProviderAPIError as e:
        return {"label": label, "content": None, "error": str(e)}

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")

    # Some requests may fail - continue processing others
    results = await asyncio.gather(
        safe_generate(llm, "Valid prompt 1", "Task 1"),
        safe_generate(llm, "Valid prompt 2", "Task 2"),
        safe_generate(llm, "Valid prompt 3", "Task 3")
    )

    for result in results:
        if result["error"]:
            print(f"{result['label']}: ERROR - {result['error']}")
        else:
            print(f"{result['label']}: {result['content']}")

asyncio.run(main())
```

## Practical tips

### 1. Prefer native-async providers when possible

```python
# ✅ Native async HTTP (I/O-bound)
llm = create_llm("ollama", model="qwen3:4b")

# ✅ Native async SDK (cloud APIs)
llm = create_llm("openai", model="gpt-4o-mini")

# ⚠️ Fallback: runs sync generation in a thread (keeps the event loop responsive)
llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")
```

### 2. Batch Similar Operations

```python
# ✅ GOOD: Single gather for all tasks
tasks = [llm.agenerate(f"Task {i}") for i in range(10)]
results = await asyncio.gather(*tasks)

# ❌ BAD: Sequential awaits lose concurrency benefit
results = []
for i in range(10):
    result = await llm.agenerate(f"Task {i}")
    results.append(result)
```

### 3. Mix Async with Sync I/O

```python
import asyncio
from abstractcore import create_llm

async def main():
    llm = create_llm("anthropic", model="claude-haiku-4-5")

    # Concurrent: LLM generation + file I/O
    llm_task = llm.agenerate("Explain async")
    file_task = asyncio.to_thread(read_large_file, "data.txt")

    response, data = await asyncio.gather(llm_task, file_task)
    # Both completed concurrently!
```

## Common Patterns

### Retry with Exponential Backoff

```python
import asyncio
from abstractcore import create_llm

async def generate_with_retry(llm, prompt, max_retries=3):
    """Generate with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await llm.agenerate(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(f"Retry {attempt + 1} after {wait_time}s...")
            await asyncio.sleep(wait_time)

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")
    response = await generate_with_retry(llm, "What is Python?")
    print(response.content)

asyncio.run(main())
```

### Rate Limiting

```python
import asyncio
from abstractcore import create_llm

class RateLimiter:
    def __init__(self, max_per_second):
        self.max_per_second = max_per_second
        self.semaphore = asyncio.Semaphore(max_per_second)
        self.reset_task = None

    async def acquire(self):
        await self.semaphore.acquire()

        # Release after 1 second
        if not self.reset_task or self.reset_task.done():
            self.reset_task = asyncio.create_task(self._release_after_delay())

    async def _release_after_delay(self):
        await asyncio.sleep(1.0)
        self.semaphore.release()

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")
    limiter = RateLimiter(max_per_second=5)

    # Process 20 requests with 5 requests/second limit
    async def limited_generate(prompt):
        await limiter.acquire()
        return await llm.agenerate(prompt)

    tasks = [limited_generate(f"Task {i}") for i in range(20)]
    results = await asyncio.gather(*tasks)

asyncio.run(main())
```

### Progress Tracking

```python
import asyncio
from abstractcore import create_llm

async def generate_with_progress(llm, prompts):
    """Generate with real-time progress tracking."""
    completed = 0
    total = len(prompts)

    async def track_task(prompt):
        nonlocal completed
        response = await llm.agenerate(prompt)
        completed += 1
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        return response

    tasks = [track_task(p) for p in prompts]
    return await asyncio.gather(*tasks)

async def main():
    llm = create_llm("ollama", model="qwen3:4b")
    prompts = [f"Task {i}" for i in range(10)]

    results = await generate_with_progress(llm, prompts)
    print(f"\nCompleted {len(results)} tasks!")

asyncio.run(main())
```

## Why MLX/HuggingFace Use Fallback

MLX and HuggingFace providers use `asyncio.to_thread()` fallback because:

1. **No Async Library APIs**: Neither `mlx_lm` nor `transformers` expose async Python APIs
2. **Direct Function Calls**: No HTTP layer to enable concurrent I/O
3. **Industry Standard**: Same pattern used by LangChain, Pydantic-AI for CPU-bound operations
4. **Event Loop Responsive**: Fallback keeps event loop responsive for mixing with I/O

```python
# MLX/HF async example (fallback keeps event loop responsive)
import asyncio
from abstractcore import create_llm

async def main():
    llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")

    # Can mix MLX inference with async I/O
    inference_task = llm.agenerate("What is Python?")
    io_task = fetch_data_from_api()  # Async I/O

    # Both run concurrently - event loop not blocked!
    response, data = await asyncio.gather(inference_task, io_task)

asyncio.run(main())
```

If you run local inference behind an OpenAI-compatible HTTP server (for example, via LM Studio), you can use the `lmstudio` (or `openai-compatible`) provider for native async I/O to the server:

```python
llm = create_llm("lmstudio", model="local-model", base_url="http://localhost:1234/v1")
```

## Best Practices

### 1. Always Use asyncio.gather() for Concurrent Tasks

```python
# ✅ CORRECT: All tasks run concurrently
results = await asyncio.gather(*[llm.agenerate(p) for p in prompts])

# ❌ WRONG: Sequential execution (no concurrency)
results = [await llm.agenerate(p) for p in prompts]
```

### 2. Await Stream Generator First

```python
# ✅ CORRECT: Two-step pattern
stream_gen = await llm.agenerate(prompt, stream=True)
async for chunk in stream_gen:
    print(chunk.content, end="")

# ❌ WRONG: Missing await before async for
async for chunk in llm.agenerate(prompt, stream=True):  # Error!
    print(chunk.content, end="")
```

### 3. Close Resources Properly

```python
# ✅ GOOD: Clean shutdown
llm = create_llm("openai", model="gpt-4o-mini")
try:
    response = await llm.agenerate("Test")
finally:
    llm.unload_model(llm.model)  # Closes async client
```

### 4. Handle Errors in Concurrent Operations

```python
# ✅ GOOD: Catch errors per-task
async def safe_task(prompt):
    try:
        return await llm.agenerate(prompt)
    except Exception as e:
        return f"Error: {e}"

results = await asyncio.gather(*[safe_task(p) for p in prompts])
```

## Learning Resources

- **Educational Demo**: [examples/async_cli_demo.py](../examples/async_cli_demo.py) - 8 core async/await patterns
- **Test Suite**: `tests/async/test_async_providers.py` - real implementation examples
- **Concurrency & Throughput**: [concurrency.md](concurrency.md) - practical guidance for local inference

## Summary

- ✅ `agenerate()` works across providers
- ✅ Use `asyncio.gather()` for concurrent (I/O-bound) requests
- ✅ HTTP-based providers use native async; MLX/HuggingFace use a thread fallback to keep the event loop responsive
- ✅ Async streaming uses a 2-step pattern: `stream_gen = await llm.agenerate(..., stream=True)` then `async for ...`
- ✅ Works well in FastAPI and other async frameworks

**Get Started**:
```bash
pip install abstractcore

# Try the educational async demo
python examples/async_cli_demo.py --provider ollama --model qwen3:4b
```
