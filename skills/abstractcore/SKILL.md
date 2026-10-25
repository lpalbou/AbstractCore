---
name: abstractcore
description: Write Python integrations and operate AbstractCore through create_llm, generate/agenerate, sessions, host-controlled tool loops, structured output, media, embeddings, the abstractcore CLI, and OpenAI-compatible HTTP serving. Use when choosing providers and exact model artifacts, configuring endpoints, downloading local models, diagnosing generation failures, or piloting an AbstractCore installation.
license: MIT
compatibility: Python 3.9+. Remote inference needs a reachable endpoint and its credentials. MLX needs Apple silicon; local engines require the apple or gpu install setting. Models must be downloaded explicitly.
metadata:
  package-version: "2.29.0"
---

# AbstractCore

## Purpose and limits

Use AbstractCore as a provider layer for Python applications and OpenAI-compatible clients.
Write integrations against `create_llm`, then choose an explicit provider, model and endpoint.
It supplies generation, tools, typed output, sessions, media and model management; your host
owns the execution loop, permissions and durable state. Use AbstractRuntime for durable agent
orchestration; assemble retrieval yourself rather than expecting a complete RAG pipeline.

## Install and verify

Choose exactly one setting: `python -m pip install abstractcore` for remote inference, tools,
media inputs, HTTP serving and remote capability plugins; `"abstractcore[apple]"` adds Apple
silicon local engines; `"abstractcore[gpu]"` adds NVIDIA/AMD local engines. These commands install
the published version. For a checkout containing this skill's version, install its source:

```bash
python -m pip install -e .                 # run at the AbstractCore repository root
# For local engines instead: python -m pip install -e ".[apple]" or -e ".[gpu]"
```

Check imports and print both code and distribution versions without contacting a provider:

```python
from importlib.metadata import PackageNotFoundError, version

try:
    import abstractcore
    installed = version("abstractcore")
    assert callable(abstractcore.create_llm), "create_llm is missing"
except (ImportError, PackageNotFoundError, AssertionError) as exc:
    raise SystemExit(f"Install verification failed: {exc}") from None
print(f"AbstractCore code={abstractcore.__version__}; distribution={installed}")
if abstractcore.__version__ != installed:
    raise SystemExit("Code/metadata mismatch: reinstall in this Python environment.")
if installed != "2.29.0":
    raise SystemExit("Version differs from this skill; use the installed version's docs/skill.")
```

For cloud access, ask the user to set the provider's environment variable locally, never paste
a key into chat. For a custom endpoint, use its own key explicitly; `api_key=""` on
`openai-compatible` prevents ambient key fallback for an unauthenticated local server.
Stop at the first authentication failure and fix configuration before another request.

## Mental model

- **Provider + model + endpoint:** provider IDs include `openai`, `anthropic`, `openrouter`,
  `portkey`, `ollama`, `lmstudio`, `vllm`, `openai-compatible`, `huggingface`, `mlx`.
  Model IDs belong to that backend. Compatible base URLs normally end in `/v1`; Ollama's
  native provider normally uses `http://127.0.0.1:11434` without `/v1`.
- **Budgets:** `max_tokens` is the total input plus output budget; `max_output_tokens` caps
  generation. Set both explicitly. The public output attribute defaults to 2048 but providers
  that allow uncapped output can omit the wire limit when you did not set it. Model metadata
  can supply sampling defaults; explicit supported call parameters win. `thinking=False`
  requests no thinking; effort strings are best-effort and can consume extra output tokens.
  Ollama raises small caps while thinking: minimum output budgets are 1024 (`minimal`),
  2048 (`low` or boolean on), 4096 (`medium`), 8192 (`high`). Disable thinking for a small cap.
- **Time and retries:** config defaults for LLM and tool timeouts are 7200 seconds;
  `timeout=None` is unlimited. Provider retry defaults are 3 attempts including the first,
  1-second initial backoff, 60-second maximum delay, jitter, and no total retry budget.
  `retry_wall_clock_budget_s` stops starting retries after that budget, not an in-flight call.
  Use `RetryConfig.single_attempt()` when the host already retries. Structured validation has
  a separate `FeedbackRetry` default of 3 attempts; tools plus `response_model` use two passes
  and do not support streaming. OpenAI/Anthropic SDK clients additionally default to 2 retries;
  `RetryConfig` controls the AbstractCore layer, not the SDK's own retry policy.
- **Results:** ordinary generation returns `GenerateResponse`; `content`, `usage`, `metadata`
  and `tool_calls` can be absent. Tool-only and reasoning-only responses can be legitimate.
  A `response_model` call returns the Pydantic instance instead. Stream chunks are deltas;
  the final chunk carries `finish_reason` and provider-reported usage. `length` means incomplete.
- **State:** `BasicSession` owns conversation history; `CachedSession` also coordinates stable
  prefixes. Cache capabilities vary by backend; inspect capabilities and hit metadata.
  A cache hit is not a guarantee of byte-identical output. Blocs store text and native cache
  artifacts, with live-binding checks on deletion.
- **Errors:** handle `AbstractCoreError` subclasses, especially `AuthenticationError`,
  `ModelNotFoundError`, `ModelArtifactMismatchError`, `ProviderAPIError`, `RateLimitError`,
  `UnsupportedFeatureError`, and `GenerationCancelledError`. Cancellation is not retried.
  Invalid options can raise `ValueError`, missing engines `ImportError`, exhausted structured
  validation `pydantic.ValidationError`; backend/transport failures can also surface as
  `RuntimeError` or `httpx.HTTPError`. Catch those at the application boundary, report failure,
  and do not treat partial or empty output as success.

## Capability map

| Job | Entry point | Guidance |
| --- | --- | --- |
| Text, async, streaming | `create_llm`, `generate`, `agenerate` | Recipes below; [integration recipes](references/integrations.md) |
| Conversations, prefix reuse | `BasicSession`, `CachedSession`, `prompt_cache_key` | Integration recipes; `get_prompt_cache_capabilities()` before cache operations |
| Tools and coding helpers | `@tool`, `tools=`, `response.tool_calls`; `abstractcore.tools.common_tools` | Integration recipes; host validates and executes |
| Typed output | `response_model=`, `FeedbackRetry` | Integration recipes |
| Media inputs and outputs | `media=`, `request=`, `output=`; `llm.voice/audio/vision/music/scene3d/camera` | Integration recipes |
| Embeddings | `abstractcore.embeddings.EmbeddingManager` | Integration recipes; explicitly select a route/model |
| Summarize, extract, classify | `BasicSummarizer`, `BasicExtractor`; `abstractcore-summarizer/extractor/judge/intent/deepsearch` | [Operations](references/operations.md) |
| Local models, engines, configuration | `abstractcore models`, `engines`, `config`, `--status` | Operations |
| HTTP and chat | `abstractcore serve`, `abstractcore-endpoint`, `abstractcore-chat` | Operations |
| Durable bloc caches and memory | `ensure_bloc_kv_artifact`, `load_bloc_kv_artifact`, `llm.unload_model(llm.model)` | Operations; inspect binding/ownership before deletion or unload |
| Observability, cancellation | `enable_tracing=True`, events, `cancel_event`, `progress_callback` | Operations |
| MCP tools | `abstractcore.mcp` | Integration recipes; tool source, not a built-in MCP server |

There is no automatic switch to a different model artifact and no complete autonomous agent
runner. Local loads are offline/cache-first; downloading is an explicit operation. Remote
providers, tools, and plugins can still use the network. Configuration normally lives under
`~/.abstractcore/config/abstractcore.json`; caches, sessions, downloads and opt-in traces write
local data. Logs/traces and prompt-cache artifacts can contain user content. Model management,
file/command/email tools and capability outputs can change local or remote state.

## Recipes

### Connect and generate

End-to-end: choose an endpoint you control, list its models (operations reference), select an
exact ID, set budgets, make one call, inspect the result, then add tools or state only as needed.
Set `ACORE_BASE_URL` and `ACORE_MODEL` locally; `ACORE_API_KEY` is the endpoint's optional key.
These three variables are recipe inputs, not AbstractCore configuration variables.

```python
import os
import httpx
from abstractcore import create_llm
from abstractcore.core.retry import RetryConfig
from abstractcore.exceptions import AbstractCoreError

try:
    llm = create_llm(
        "openai-compatible", model=os.environ["ACORE_MODEL"],
        base_url=os.environ["ACORE_BASE_URL"], api_key=os.environ.get("ACORE_API_KEY", ""),
        timeout=30, max_tokens=4096, max_output_tokens=256,
        retry_config=RetryConfig.single_attempt(), execute_tools=False, validate_model=False,
    )
    response = llm.generate("Reply with one sentence about HTTP caching.")
except KeyError as exc:
    raise SystemExit(f"Set required recipe input: {exc}") from None
except (AbstractCoreError, httpx.HTTPError, ValueError, ImportError, RuntimeError) as exc:
    raise SystemExit(f"Generation failed ({type(exc).__name__}); check endpoint/model/config.") from None
if response.finish_reason == "length":
    raise SystemExit("Incomplete answer: increase the output budget or shorten the request.")
if not (response.content or "").strip():
    raise SystemExit("No text answer: inspect tool_calls and reasoning before continuing.")
print(response.content)
```

Success is nonempty text with no length cutoff. Use a longer explicit timeout for large local
models. After explicit model discovery, `validate_model=False` skips the constructor's
best-effort chat-catalog probe; generation still reports request/model errors.
For `openai`/`anthropic`, use the provider's key variable and provider-specific model ID;
the generic endpoint recipe does not validate those cloud dialects.

### Inspect a host before making changes

```bash
python -c "from abstractcore import __version__; print(__version__)"
abstractcore host profile --json
abstractcore models list --json
abstractcore models catalog --fits --json
abstractcore engines status --json
```

These report the machine, installed weights, catalog fit and engines. Check exit status and
payload state: missing/unknown measurements do not mean a model fits. Discovery can contact
local model servers. Operations covers exact artifacts, dry-run plans and serving.

### Add state, typed answers, streaming and tools

Open [integration recipes](references/integrations.md) for complete continuations of the client
recipe: session history, typed extraction, streaming/async, an allowlisted tool round trip,
embeddings and media routing. Keep tool execution in the host; never execute model-provided
Python, shell text or an arbitrary function name.

## Rules and sharp edges

- Inference on hosted providers can incur charges; model downloads consume disk/network;
  local inference and generated media can take minutes and hold substantial memory. Inspect
  configuration, model IDs, catalog fit and dry-run plans first. Obtain user authorization for
  paid calls, downloads, installs, deletions or external effects when the task has not already
  authorized them. An already-authorized operation needs no repeated confirmation.
- Pin provider/model/base URL. Do not send a cloud credential to an arbitrary compatible host.
  Portkey additionally needs `PORTKEY_CONFIG`; optional gateway sampler fields are forwarded
  only when explicitly chosen.
- Execute only complete parsed tool calls; check `(response.metadata or {}).get("unparsed_tool_call")`
  and reject a length-cutoff turn. Preserve call IDs in assistant/tool history. Bound loop turns.
- Audio defaults to `audio_policy="native_only"`; video defaults to `video_policy="auto"`.
  Video frame sampling needs `ffmpeg`/`ffprobe`. An image needs native vision or a configured
  captioning fallback. Fallbacks can make another model call; choose them deliberately.
- Native MLX batching/MTP is opt-in and checkpoint-specific (`mlx_batching=True`, `speculation`).
  Inspect `metadata["speculation"]["used"]` and execution metadata; accepting an option is not
  evidence it ran. The shared contract does not enable HuggingFace/GGUF MTP.
- When editing AbstractCore itself, read repository instructions and `CONTRIBUTING.md`, then
  the relevant public docs and tests. Keep local-engine imports lazy and use the existing
  three install settings. Route contributor work to those docs; do not invent private APIs.

## When things fail

| Symptom | Fix |
| --- | --- |
| 401/403 or `AuthenticationError` | Stop; check the endpoint's own key and server-vs-upstream token distinction. Never retry login blindly. |
| Connection refused/timeout | Start the intended server; check host, port, `/v1` and explicit timeout. Do not switch providers silently. |
| Model missing/artifact mismatch | List installed/backend models; download the exact authorized artifact. Specify quantization rather than substituting another build. |
| Empty text, `length`, unparsed call | Inspect reasoning/tools and finish metadata. Treat cutoff as incomplete; adjust budgets and retry only within the task's policy. |
| Import error, validation or media failure | Install the matching engine setting, simplify the schema, or select a supported media policy/model; see the recipes and troubleshooting. |

## Go deeper

- [Integration guide](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/agent-usage.md): canonical workflow, recipe inputs and operating limits.
- [Getting started](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/getting-started.md): setup and first calls.
- [API](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/api.md): public entry points and response/error contracts.
- [Troubleshooting](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/troubleshooting.md): symptom-driven fixes.
- [Full documentation](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/llms-full.txt): expanded context.
- [Contributing](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/CONTRIBUTING.md): source changes and development checks.
- [Integration recipes](references/integrations.md): load for tools, structured output, async, sessions, media or embeddings.
- [Operations](references/operations.md): load for CLI, serving, model/cache lifecycle and diagnostics.
