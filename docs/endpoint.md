# Endpoint (single-model `/v1` server)

`abstractcore-endpoint` runs a **single-model** OpenAI-compatible server.

Unlike the multi-provider gateway ([Server](server.md)), this endpoint loads **one** `provider+model` once per worker process and reuses it across requests. It’s useful when you want to host a local backend (for example HF GGUF or MLX) as a stable `/v1` endpoint.

Source: `abstractcore/endpoint/app.py` (entrypoint: `abstractcore-endpoint`).

## When to use this vs the gateway

- Use **[Server](server.md)** when you want `model="provider/model"` routing across many providers/models from one gateway process.
- Use **Endpoint** when you want a dedicated “one worker = one model” process (simpler performance characteristics; fewer per-request initialization costs).

## Install

```bash
pip install "abstractcore[server]"
```

Then install the provider extra you need:

```bash
pip install "abstractcore[mlx]"         # Apple Silicon local inference
pip install "abstractcore[huggingface]" # Transformers / torch / llama-cpp-python (heavy)
```

## Run

```bash
# CLI flags
abstractcore-endpoint --provider mlx --model mlx-community/Qwen3-4B --host 0.0.0.0 --port 8001

# Or via env vars
export ABSTRACTENDPOINT_PROVIDER=mlx
export ABSTRACTENDPOINT_MODEL=mlx-community/Qwen3-4B
export ABSTRACTENDPOINT_HOST=0.0.0.0
export ABSTRACTENDPOINT_PORT=8001
abstractcore-endpoint
```

Health check:

```bash
curl http://localhost:8001/health
```

## Use with an OpenAI-compatible client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="unused")
resp = client.chat.completions.create(
    model="anything",  # ignored/validated in single-model mode
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

## Prompt cache control plane (optional)

If the underlying provider exposes prompt-cache controls, the endpoint also exposes a small control plane under `/acore/prompt_cache/*` (see `abstractcore/endpoint/app.py`):

- `GET /acore/prompt_cache/stats`
- `GET /acore/prompt_cache/capabilities`
- `POST /acore/prompt_cache/set`
- `POST /acore/prompt_cache/update`
- `POST /acore/prompt_cache/fork`
- `POST /acore/prompt_cache/clear`
- `POST /acore/prompt_cache/prepare_modules`

Response contract:

- `GET /acore/prompt_cache/capabilities` always returns the provider capability profile (`supported`, `operation="capabilities"`, `capabilities`).
- Other prompt-cache routes return structured payloads instead of ambiguous booleans:
  - success: `supported=true`
  - unsupported operation: `supported=false`, `code="prompt_cache_unsupported"`
  - runtime/provider failure: `supported=false`, `code="prompt_cache_error"`

The `capabilities` object is always included on prompt-cache control-plane responses so callers can branch on `mode` / `supports_*` flags without re-probing the provider.

For caching concepts, see [Session Management](session.md) and [Architecture](architecture.md).
For a dedicated overview, see [Prompt Caching](prompt-caching.md).
