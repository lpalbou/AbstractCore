# Python integration recipes

These snippets continue **Connect and generate** in `SKILL.md`: reuse its configured `llm`,
`AbstractCoreError` and `httpx` imports. Run each recipe independently after that setup. Use the
same application error boundary for provider, transport and configuration failures. An
authentication failure ends the workflow; a failed/cutoff turn must not trigger tool execution.

## Conversation history and prefix reuse

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

## Typed extraction

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

## Streaming and async

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

## One host-controlled tool round trip

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

## Embeddings without implicit local model selection

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

## Media inputs and generated artifacts

On the configured client, pass `media=["<IMAGE_PATH>"]` with a vision-capable model to
describe an image. Fill the path with an existing user-authorized file. Audio/video policies,
native capability and caption/STT fallbacks are described in the main skill's sharp edges.
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

## Further documentation

- [Tool calling](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/tool-calling.md)
- [Structured output](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/structured-output.md)
- [Sessions](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/session.md)
- [Prompt caching](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/prompt-caching.md)
- [Media policies](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/media-handling-system.md)
- [Output contract](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/request-output.md)
- [Embeddings](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/embeddings.md)
- [MCP](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/mcp.md)
