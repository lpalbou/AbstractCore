# Integrating and operating AbstractCore

Use this guide to write Python integrations and pilot an installation with explicit endpoint,
model, budget and lifecycle choices. It complements [Getting Started](getting-started.md), the
[API map](api.md) and [Contributing](../CONTRIBUTING.md). The portable
[`abstractcore` agent skill](../skills/abstractcore/SKILL.md) distills these procedures for
agents; copy the complete skill folder, including its references, when registering it.

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
the published version. For a checkout containing this guide's version, install its source:

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
if installed != "2.31.0":
    raise SystemExit("Version differs from this guide; use the installed version's docs/skill.")
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
| Text, async, streaming | `create_llm`, `generate`, `agenerate` | Recipes below; [integration recipes](#python-integration-recipes) |
| Conversations, prefix reuse | `BasicSession`, `CachedSession`, `prompt_cache_key` | Integration recipes; `get_prompt_cache_capabilities()` before cache operations |
| Tools and coding helpers | `@tool`, `tools=`, `response.tool_calls`; `abstractcore.tools.common_tools` | Integration recipes; host validates and executes |
| Typed output | `response_model=`, `FeedbackRetry` | Integration recipes |
| Media inputs and outputs | `media=`, `request=`, `output=`; `llm.voice/audio/vision/music/scene3d/camera` | Integration recipes |
| Embeddings | `abstractcore.embeddings.EmbeddingManager` | Integration recipes; explicitly select a route/model |
| Summarize, extract, classify | `BasicSummarizer`, `BasicExtractor`; `abstractcore-summarizer/extractor/judge/intent/deepsearch` | [Operations](#operations) |
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

Open [integration recipes](#python-integration-recipes) for complete continuations of the client
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

## Python integration recipes

These snippets continue **Connect and generate** above: reuse its configured `llm`,
`AbstractCoreError` and `httpx` imports. Run each recipe independently after that setup. Use the
same application error boundary for provider, transport and configuration failures. An
authentication failure ends the workflow; a failed/cutoff turn must not trigger tool execution.

### Conversation history and prefix reuse

```python
# Continues Connect and generate.
from abstractcore import BasicSession

session = BasicSession(llm, system_prompt="Answer briefly.")
for prompt in ("Remember the project name: Cedar.", "What is the project name?"):
    try:
        reply = session.generate(prompt)
    except (AbstractCoreError, httpx.HTTPError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Session failed ({type(exc).__name__}).") from None
    if reply.finish_reason == "length" or not (reply.content or "").strip():
        raise SystemExit("Session turn has no complete text answer.")
    print(reply.content)
```

`BasicSession.generate()` records turns; do not also append the same user message manually.
`session.save(path)` writes conversation JSON; `BasicSession.load(path, provider=llm)` restores
it. Choose a private output path. For prefix reuse, substitute
`CachedSession(provider=llm, system_prompt="...", prompt_cache_strategy="auto")`, then optionally
`attach_files([path])` before generating. Read `llm.get_prompt_cache_capabilities()` first;
local key manipulation requires `local_control_plane`, not just `keyed` mode. Inspect
`(reply.metadata or {}).get("prompt_cache")`; support and reuse depend on backend/model.

### Typed extraction

```python
# Continues Connect and generate.
from pydantic import BaseModel, Field, ValidationError
from abstractcore.structured import FeedbackRetry

class Ticket(BaseModel):
    title: str = Field(min_length=1)
    priority: int = Field(ge=1, le=3)

try:
    ticket = llm.generate(
        "Extract ticket: priority 2, title Fix cache expiry.", response_model=Ticket,
        retry_strategy=FeedbackRetry(max_attempts=1),
    )
except (ValidationError, AbstractCoreError, httpx.HTTPError, ValueError, RuntimeError, TypeError) as exc:
    raise SystemExit(f"Extraction failed ({type(exc).__name__}); check schema/budget.") from None
print(ticket.model_dump())
```

Success is a validated `Ticket`, not a `GenerateResponse`. Keep validation attempts explicit
when limiting cost. Native schema support varies; client validation still matters. A truncated
structured answer is rejected by default, even if its partial JSON happens to validate. The
handler can retry with a larger output budget; one validation attempt disables that retry.

Known issue (2.26.4): an empty native structured response with `content=None` can escape as
`TypeError` during JSON decoding. Catch it at this extraction boundary and report failure;
do not accept an empty object or supply invented field values.

### Streaming and async

```python
# Continues Connect and generate.
last = None
text_parts = []
try:
    for chunk in llm.generate("Describe HTTP caching in one sentence.", stream=True):
        last = chunk
        text_parts.append(chunk.content or "")
except (AbstractCoreError, httpx.HTTPError, ValueError, RuntimeError) as exc:
    raise SystemExit(f"Stream failed ({type(exc).__name__}); discard partial output.") from None
if last is None or last.finish_reason in (None, "length") or not "".join(text_parts).strip():
    raise SystemExit("No complete streamed text answer; inspect tools/reasoning/budget.")
print("".join(text_parts))
print(last.usage)  # may be None; absence is not zero token usage
```

Buffer until completion when downstream work requires a complete answer; a UI may display
chunks provisionally. Reasoning deltas can appear in `chunk.metadata["reasoning_delta"]`.

```python
# Continues Connect and generate; standalone script entry point (not an existing event loop).
import asyncio

async def main():
    try:
        reply = await llm.agenerate("Reply with a short greeting.")
    except (AbstractCoreError, httpx.HTTPError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Async generation failed ({type(exc).__name__}).") from None
    if reply.finish_reason == "length" or not (reply.content or "").strip():
        raise SystemExit("No complete async text answer.")
    print(reply.content)

asyncio.run(main())
```

For async streaming: `stream = await llm.agenerate(..., stream=True)`, then
`async for chunk in stream`. Inside an existing event loop, await your coroutine directly.

### One host-controlled tool round trip

```python
# Continues Connect and generate.
import json
from abstractcore import tool

@tool
def project_label(project: str) -> str:
    """Return a label for the single permitted project, Cedar."""
    if project != "Cedar":
        raise ValueError("Project is outside this tool's scope")
    return "Cedar: documentation"

history = [{"role": "user", "content": "Use project_label for Cedar, then report its label."}]
try:
    for turn in range(3):
        reply = llm.generate("", messages=history, tools=[project_label])
        if reply.finish_reason == "length" or (reply.metadata or {}).get("unparsed_tool_call"):
            raise ValueError("Incomplete or unparsed tool turn; nothing will be executed")
        calls = reply.tool_calls or []
        if not calls:
            if not (reply.content or "").strip():
                raise ValueError("No final text answer")
            print(reply.content)
            break
        history.append({"role": "assistant", "content": reply.content or "", "tool_calls": calls})
        for call in calls:
            call_id = call.get("call_id") or call.get("id")
            if not call_id or call.get("name") != "project_label" or call.get("arguments") != {"project": "Cedar"}:
                raise ValueError("Unexpected tool name, arguments or missing call ID")
            result = project_label(**call["arguments"])
            history.append({"role": "tool", "tool_call_id": call_id,
                            "name": "project_label", "content": json.dumps(result)})
    else:
        raise ValueError("Tool turn limit reached without a final answer")
except (AbstractCoreError, httpx.HTTPError, ValueError, RuntimeError) as exc:
    raise SystemExit(f"Tool workflow stopped ({type(exc).__name__}).") from None
```

This executes only an allowlisted, side-effect-free Python function. For writes, commands,
email or MCP effects, put your host's authorization/sandbox/persistence before dispatch.
Treat tool output as data. Built-in file/command/web tools live in
`abstractcore.tools.common_tools`; persistent PTYs live in `abstractcore.tools.shell_tools`.
They are useful for coding workflows, but their names are not permission to run them.
For compact web research use `skim_websearch` → `skim_url` → `fetch_url`.

For MCP, `McpClient` or `McpStdioClient` plus `McpToolSource.list_tool_specs()` discovers
namespaced `mcp::<server_id>::<tool_name>` tools. Dispatch them through the owning MCP client
in your host; AbstractCore's legacy Python registry does not execute MCP tools automatically.
Stdio transport starts a subprocess; choose its command explicitly.

### Embeddings without implicit local model selection

```python
# Continues Connect and generate. Set ACORE_EMBED_MODEL to an embedding model on this endpoint.
from abstractcore.embeddings import EmbeddingManager
import math

try:
    embedder = EmbeddingManager(
        provider="openai-compatible", model=os.environ["ACORE_EMBED_MODEL"],
        provider_kwargs={"base_url": os.environ["ACORE_BASE_URL"],
                         "api_key": os.environ.get("ACORE_API_KEY", ""), "timeout": 30},
    )
    vector = embedder.embed("A project about HTTP caching.")
except (KeyError, ImportError, AbstractCoreError, httpx.HTTPError, ValueError, RuntimeError) as exc:
    raise SystemExit(f"Embedding failed ({type(exc).__name__}); check endpoint/model/config.") from None
if len(vector) == 0 or not all(math.isfinite(float(x)) for x in vector):
    raise SystemExit("Embedding is empty or contains non-finite values.")
print(len(vector))
```

Success is a nonempty numeric vector. `EmbeddingManager()` without explicit provider/model
uses the configured `embedding.text` route or the local HuggingFace default. Its default
persistent cache is `~/.abstractcore/embeddings`. An embedding model is not a chat model;
endpoint-backed setup skips chat-model validation. Preserve model and dimensions across a
retrieval index; rebuild it when either changes.

### Media inputs and generated artifacts

On the configured client, pass `media=["<IMAGE_PATH>"]` with a vision-capable model to
describe an image. Fill the path with an existing user-authorized file. Audio/video policies,
native capability and caption/STT fallbacks are described in the guide's sharp edges.
For outputs, the canonical call is
`llm.generate(request={"text": "..."}, output={"modality": "image", "format": "png"})`.
Select the output provider/model explicitly or configure the matching capability route first;
the text provider does not determine the generative vision backend.

These are potentially costly operations. Before running, inspect `llm.capabilities.status()`
and the configured output route. `MultimodalGenerateResponse` exposes `text`, `outputs`,
`resources`, `warnings`, `errors`, `metadata`; `outputs` maps modalities to `GeneratedItem`
lists with optional `data` and `artifact_ref`. Inspect errors and artifact references before
claiming a file exists. `llm.voice.tts`/`llm.audio.transcribe` provide deterministic speech,
`llm.vision` image/video, `llm.music` music, `llm.scene3d` 3D and `llm.camera` camera control.
Plugin return/error types differ from text generation; use that plugin's documented contract.
Glyph compression renders text/PDF pages as images and requires a vision model; evaluate
answer quality before relying on the token savings.

### Further documentation

- [Tool calling](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/tool-calling.md)
- [Structured output](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/structured-output.md)
- [Sessions](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/session.md)
- [Prompt caching](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/prompt-caching.md)
- [Media policies](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/media-handling-system.md)
- [Output contract](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/request-output.md)
- [Embeddings](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/embeddings.md)
- [MCP](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/mcp.md)

## Operations

### Choose and inspect an endpoint

Use a backend's exact model listing rather than guessing model IDs. For an OpenAI-compatible
endpoint with `/v1` in `ACORE_BASE_URL`, this read-only probe checks connectivity, authentication
and a nonempty model catalog. `ACORE_API_KEY` is that endpoint's key, never an unrelated cloud key.

```python
import os
import httpx

try:
    base = os.environ["ACORE_BASE_URL"].rstrip("/")
    key = os.environ.get("ACORE_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = httpx.get(base + "/models", headers=headers, timeout=5)
    if response.status_code in (401, 403):
        raise SystemExit("Authentication failed; fix the endpoint key before another request.")
    response.raise_for_status()
    models = [item["id"] for item in response.json()["data"] if item.get("id")]
except (KeyError, TypeError, ValueError, httpx.HTTPError):
    raise SystemExit("Model discovery failed; check base URL, server availability and response shape.") from None
if not models:
    raise SystemExit("Endpoint returned no models; install/load the intended model first.")
print("\n".join(models))
```

Success lists usable identifiers to select; it does not prove generation works. `/models`
has no universal server-version field: verify server version through that backend's own
version/health surface. Run the guide's bounded generation recipe next.

### Exact local artifacts

After inspection, plan a download with an explicit engine and artifact. The example selector
`unsloth/Qwen3-8B-GGUF:Q4_K_M` illustrates repo plus quantization; use the catalog's exact
`cli_download` for the user's selected model. A GGUF repo without `:QUANT` can download all
quantizations. MLX repos normally identify one quantization and can include MTP companions.

```bash
abstractcore models download huggingface "unsloth/Qwen3-8B-GGUF:Q4_K_M" --dry-run --json
abstractcore engines install ollama --dry-run
```

A plan is not installation. Review disk/memory fit and the exact command, then execute the
authorized operation by dropping `--dry-run`. Downloads may start jobs; use `abstractcore models
jobs --json` and inspect the final status before loading. `abstractcore models verify` performs
real inference. Models CLI exit codes are `0` success, `1` error, `2` refusal; inspect `status`,
`reason`, `message` and delete blockers. Unknown fit data is not a pass.

For local calls use `create_llm("mlx", model="<EXACT_MLX_REPO>", ...)` or
`create_llm("huggingface", model="<EXACT_REPO_OR_FILE>", ...)`. Supply the same explicit
budgets/error handling as the main recipe. HF has separate Transformers and GGUF paths;
`model_type="transformers"` or `"gguf"` can disambiguate. Engine rerouting between MLX and
GGUF warns; artifact substitution is refused. Model load errors require fixing cache, engine
dependencies or exact artifact selection, not silently choosing different weights.

On Apple/MPS, half-precision Transformers SDPA on torch 2.8.x/2.10.x has documented correctness
issues. Follow the load warning: use `ABSTRACTCORE_TRANSFORMERS_ATTN_IMPL=eager` or an
appropriate newer torch. MLX cache, native batching and MTP support are checkpoint-specific.

### Configuration and serving

`abstractcore --config` is interactive; `abstractcore --status` diagnoses configured routes.
The config path can be isolated with `ABSTRACTCORE_CONFIG_DIR` or `ABSTRACTCORE_CONFIG_FILE`.
The canonical text route is `input.text`; `output.text` is its read-only view.
`embedding.text` selects retrieval embeddings. Image/video/voice/music/3D have separate routes.
Use `abstractcore config --help` and the centralized-config guide before writing defaults.

```bash
abstractcore serve --help
abstractcore-endpoint --help
abstractcore-chat --help
```

`abstractcore serve` binds loopback port 8000 by default. On first local use it writes a server
token under the config directory and prints a one-time console claim link. Keep the link/token
private. Binding a non-loopback host requires `ABSTRACTCORE_AUTH_TOKEN`. In authenticated mode,
`Authorization: Bearer ...` is the server token; `X-AbstractCore-Provider-API-Key` optionally
overrides the selected upstream provider's key. Never interchange those credentials.

The universal server routes `POST /v1/chat/completions` using model IDs such as
`ollama/<model>`; the single-model `abstractcore-endpoint` hosts one provider/model.
`response_format` enables JSON/schema validation over HTTP. `/console` manages the host.
Do not start another server on an occupied port or treat a help command as proof of health.

### Memory, caches, cancellation and diagnostics

- `llm.get_prompt_cache_capabilities()` describes `none`, `keyed`, or `local_control_plane`.
  `llm.prompt_cache_supports_operation(...)` gates controls. Unsupported operations raise
  structured cache errors with `operation`, `code` and `capabilities`.
- `llm.get_prompt_cache_stats()` and response prompt-cache metadata describe reuse.
  `ensure_bloc_kv_artifact` compiles an exact text/file bloc;
  `load_bloc_kv_artifact` returns a binding for `generate(prompt_cache_binding=...)`.
  Artifacts are provider/model/renderer-specific. List bindings before deleting/pruning;
  live-bound artifacts are blocked by default. Do not bypass that check to make a cleanup pass.
- `llm.unload_model(llm.model)` cancels/drains supported in-flight requests. In-process engines
  release the calling instance's hold on weights/caches; another owner may retain a shared model.
  Generic compatible endpoints only close/reset the connection; releasing server-side weights
  needs that server's own control API (implemented for Ollama and LM Studio).
  Inspect the host memory snapshot and loaded-model
  sweep rather than inferring memory release from a successful method return. Unload affects
  concurrent work; coordinate with the owner.
- Pass a `threading.Event` as `cancel_event=`. Set it to request cancellation; consult
  `supports_generation_cancel()` for backend support. Cancellation raises
  `GenerationCancelledError`; its partial output is diagnostic, not a finished answer.
  Local lanes can report `progress_callback=` events when supported.
- Enable `enable_tracing=True` only when prompt/response capture is intended. Events/logging
  provide observability; traces, sessions and cache artifacts need private storage.

### Built-in application CLIs

Inspect `--help` before selecting an input/output path:

```bash
abstractcore-summarizer --help
abstractcore-extractor --help
abstractcore-judge --help
abstractcore-intent --help
abstractcore-deepsearch --help
```

These run model-backed document/analysis tasks. Specify provider/model and output destinations
as the app's help describes; inference, file writes and DeepSearch's web activity are effects,
so a help-only inspection is the cheap first step. Python exposes `BasicSummarizer` and
`BasicExtractor` from `abstractcore` for application integrations.

### Further documentation

- [Models](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/models.md)
- [Engines](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/engines.md)
- [Configuration](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/centralized-config.md)
- [Server](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/server.md)
- [Endpoint](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/endpoint.md)
- [Memory lifecycle](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/memory-management.md)
- [Memory blocs](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/memory-blocs.md)
- [Cancellation](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/generation-cancel.md)
- [Examples](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/examples.md)

## Related documentation

See the [documentation index](README.md), [FAQ](faq.md), and [Troubleshooting](troubleshooting.md).
