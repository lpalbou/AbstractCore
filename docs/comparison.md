# Framework Comparison

Snapshot: May 2026.

This guide compares **AbstractCore** and the wider **AbstractFramework** ecosystem with nearby LLM libraries and application frameworks. The most important distinction is layer:

- **AbstractCore** is an LLM I/O and reliability layer.
- **AbstractFramework** is a larger agent/runtime/workflow ecosystem built on AbstractCore.

That difference matters. AbstractCore should be compared mostly with provider clients, LiteLLM, and the model I/O layer inside larger frameworks. AbstractFramework should be compared with LangChain/LangGraph, AutoGen, CrewAI, workflow runtimes, gateway systems, and local AI workbenches.

The second important distinction is deployment posture. AbstractCore is not
"local only" and not "cloud only": it is designed so the same Python application
can run fully offline after local model assets are installed, stay private on
self-hosted inference, or route to hosted APIs and gateways when managed
capacity is the better trade-off.

## Short Verdict

Yes, AbstractCore is interesting in May 2026, but not because it is the biggest or most popular framework. It is interesting because it is a focused provider layer with unusually broad local-model support and call-level infrastructure: offline-capable provider setup, tools, structured output, streaming, media policies, embeddings, MCP, prompt caching, events, tracing, and optional OpenAI-compatible server mode.

Its sharpest niche is portability across **offline, self-hosted, and remote**
LLM deployments. That makes it useful for privacy-sensitive assistants, local
developer tools, document workflows, edge deployments, research machines, and
regular cloud-backed products that may need to move between model families over
time.

Yes, AbstractFramework is also interesting, but it should be judged as an emerging integrated stack. Its strongest idea is not a single API call; it is the combination of:

- AbstractCore for provider I/O
- AbstractRuntime for durable effects, waits, ledgers, snapshots, and replay
- AbstractAgent for ReAct/CodeAct/MemAct loops
- AbstractFlow for portable visual workflow bundles
- AbstractGateway for run control, streaming ledgers, scheduling, and remote clients
- AbstractMemory and AbstractSemantics for persistent knowledge and structured assertions
- AbstractCode, AbstractAssistant, Observer, and UI packages as concrete clients

The trade-off is maturity. LangChain, LangGraph, LlamaIndex, LiteLLM, AutoGen, and CrewAI have much larger communities and more battle-tested ecosystems. AbstractFramework is more opinionated, more local-first, and more vertically integrated, but still pre-1.0 in several packages.

## Objective Ranking By Use Case

These rankings are deliberately scenario-specific. There is no single winner across all layers.

| Use case | Best default in May 2026 | Where AbstractCore / AbstractFramework ranks |
| --- | --- | --- |
| Maximum OpenAI-compatible model/API gateway coverage | LiteLLM | AbstractCore is behind LiteLLM for sheer routed provider coverage, but stronger as an embedded Python provider layer with local-model behavior, media policies, structured output, and sessions. |
| Direct Python LLM I/O across cloud, local, and self-hosted backends | AbstractCore, LiteLLM, provider SDKs | AbstractCore is near the top if you value local inference, provider capability handling, structured output, tools, media, and offline-capable behavior in one package. |
| Offline-capable Python applications | AbstractCore / AbstractFramework, Ollama or LM Studio plus custom code | AbstractCore ranks very high because local HTTP servers, MLX, HuggingFace/GGUF, vLLM, explicit model setup, and cache/local-files-only paths are core design concerns rather than afterthoughts. |
| Hybrid local/remote products | AbstractCore, LiteLLM, custom provider layers | AbstractCore is especially attractive when the same app should run on a laptop, private server, edge box, or hosted API without rewriting product code. |
| Full RAG/document applications | LlamaIndex, LangChain | AbstractCore is not a full RAG framework. It provides embeddings and generation primitives; AbstractFramework's memory layer is more KG/temporal-memory oriented than document-RAG oriented. |
| Stateful agent graphs | LangGraph | AbstractFramework is interesting here because AbstractRuntime gives durable waits, ledgers, replay, snapshots, and gateway execution, but LangGraph is more established and has a larger ecosystem. |
| Multi-agent experimentation | AutoGen, CrewAI, LangGraph | AbstractAgent covers core ReAct/CodeAct/MemAct patterns, but the broader multi-agent ecosystem is less mature than AutoGen/CrewAI/LangGraph. |
| Local-first/offline AI workbench or assistant | AbstractFramework, Ollama/LM Studio plus custom tooling | AbstractFramework ranks unusually high because local providers, voice/vision plugins, durable runtime, gateway, memory, and concrete clients such as AbstractAssistant are designed to compose offline. |
| Durable workflow orchestration for AI runs | AbstractRuntime/AbstractGateway, Temporal/custom systems, LangGraph with persistence | AbstractFramework is compelling if you want AI-specific run ledgers and replay. Temporal is more mature for general distributed workflows. |
| Coding assistant product | Cursor/Claude Code/Codex-class tools for polish; AbstractCode for open local durable stack | AbstractCode is early, but strategically interesting because it sits on the same durable runtime/gateway/core foundation. |

## Quick Comparison

| Dimension | AbstractCore | AbstractFramework | LiteLLM | LangChain | LangGraph | LlamaIndex |
| --- | --- | --- | --- | --- | --- | --- |
| Primary layer | LLM provider I/O | Full AI system stack | API compatibility/proxy | Broad app framework | Stateful graph runtime | RAG/document framework |
| Best fit | App/runtime needs stable LLM calls across offline, self-hosted, and remote models | Durable local-first agents, assistants, and workflows | One OpenAI-style API across many providers | Many integrations and common app patterns | Explicit stateful agent workflows | Retrieval-heavy document apps |
| Provider support | 10 direct adapters plus gateways | Through AbstractCore | Very broad routed coverage | Many integrations | Via model layer | Via integrations |
| Local inference | First-class: Ollama, LM Studio, MLX, HF/GGUF, vLLM | First-class across stack | Supported, often via adapters/proxy | Supported via integrations | Via model layer | Supported in RAG pipelines |
| Offline operation | First-class when models/tools are local and preinstalled | A central design goal for local assistants/workbenches | Possible through local endpoints, but not the main product thesis | Possible, depends heavily on integrations | Possible through selected model/runtime choices | Possible for local RAG, depends on stores/models |
| Remote operation | First-class: OpenAI, Anthropic, OpenRouter, Portkey, OpenAI-compatible | Through Core and Gateway clients | Core strength | Core strength through integrations | Through model layer | Through integrations |
| Tool handling | Native/prompted normalization; pass-through by default | Runtime/gateway owns durable execution policy | Mostly forwards/normalizes provider calls | Mature abstractions | Strong graph tool loops | Available, less central |
| Structured output | Pydantic-first, provider-aware strategies and retry | Used as contract layer for agents/workflows | Provider dependent | Mature APIs/parsers | Via model layer | Strong extraction/query patterns |
| Media/capability gaps | Policy-driven media handling, fallbacks, capability metadata | Used across assistant/workflow surfaces | Mostly provider/API dependent | Depends on integrations | Depends on model layer | Strong for document workflows |
| Durable execution | Sessions and prompt caches only | Core feature via AbstractRuntime | Not the main role | Via external components | Core strength | Not the main role |
| Memory | Basic sessions, embeddings | AbstractMemory/semantics plus runtime state | Not the main role | Memory abstractions | Checkpoint/state persistence | Index/storage abstractions |
| Gateway/server | Optional OpenAI-compatible server | AbstractGateway run gateway plus Core server | Core strength | Not the main role | Not the main role | Not the main role |
| Ecosystem maturity | Beta but substantial | Emerging/pre-1.0 stack | Mature and widely used | Mature and widely used | Mature and widely used | Mature and widely used |

## What AbstractCore Is

AbstractCore is best understood as an **LLM I/O layer**:

- one `create_llm(...)` interface for local, self-hosted, cloud, gateway, and OpenAI-compatible backends
- 10 registered provider/backend adapters: OpenAI, Anthropic, Ollama, LM Studio, MLX, HuggingFace, vLLM, OpenAI-compatible, OpenRouter, Portkey
- about 79k physical Python lines in the package source
- 213 model capability entries and 48 architecture format entries
- a default install with only `pydantic` and `httpx`; heavy features are optional extras
- offline-capable local operation through explicit model setup, local server backends, and cache/local-files-only behavior for in-process local runtimes
- remote operation through hosted APIs, hosted gateways, and arbitrary OpenAI-compatible endpoints
- tool-call normalization across native and prompted formats
- Pydantic structured output with provider-aware strategies and retry fallback
- sync, async, streaming, media handling, embeddings, MCP tool sources, tracing, events, and optional server mode
- prompt-cache-aware sessions, including stronger local KV/prefix-cache paths where backends expose the required control

It is not trying to be the full agent runtime, RAG framework, or graph engine. In the AbstractFramework ecosystem, those concerns move upward into AbstractRuntime, AbstractAgent, AbstractFlow, AbstractGateway, AbstractMemory, and AbstractSemantics.

## What AbstractFramework Is

AbstractFramework is the larger system built around AbstractCore. Its architectural thesis is:

```text
clients / UIs / apps
        |
AbstractGateway and concrete hosts
        |
AbstractAgent and AbstractFlow
        |
AbstractRuntime durable execution
        |
AbstractCore provider I/O
        |
LLM providers, local models, tools, media, MCP
```

The interesting part is the integration:

- **Durability**: runs can pause, resume, replay, and survive process restarts.
- **Observability**: append-only ledgers make LLM calls, tool calls, waits, and decisions inspectable.
- **Local-first design**: Ollama, LM Studio, MLX, HuggingFace/GGUF, local voice, local vision, and local workflow execution are first-class paths.
- **Offline application surface**: concrete clients such as [AbstractAssistant](https://github.com/lpalbou/abstractassistant) can be built as complete local assistants while still keeping the option to use remote providers.
- **Composable clients**: terminal, browser, tray apps, gateway clients, and visual flow bundles share the same underlying runtime contracts.
- **Separation of concerns**: AbstractCore returns tool calls; the runtime/gateway decides whether and how to execute them.

This is more ambitious than AbstractCore alone, but also less mature as a public ecosystem.

## Offline, Remote, Or Hybrid

This is where AbstractCore's positioning is most different from a plain provider
SDK or gateway proxy. The same application code can be written against
`create_llm(...)`, then deployed in several modes:

- **Fully offline**: Ollama, LM Studio, MLX, HuggingFace/GGUF, vLLM, local media handling, local embeddings, and local voice/vision plugins where installed.
- **Private/self-hosted**: local network inference servers, private OpenAI-compatible endpoints, vLLM clusters, or internal gateways.
- **Hosted/remote**: OpenAI, Anthropic, OpenRouter, Portkey, or any managed OpenAI-compatible provider.
- **Hybrid**: local models for privacy/cost/latency, remote models for peak quality, special modalities, or burst capacity.

That deployment choice is a product advantage. It means an app can offer "works
without internet" as a real mode, not just as a demo, while still remaining able
to call frontier hosted models when the user or deployment permits it.

## AbstractCore vs LiteLLM

LiteLLM is strongest when you want **OpenAI-compatible access to many model APIs**, often through a proxy/gateway. It is a good fit for routing, cost controls, standardized request/response shapes, virtual keys, spend tracking, and using existing OpenAI-style client code.

AbstractCore is strongest when you want **a Python infrastructure layer inside your application**, especially when local/offline execution and remote execution should share the same product code:

```python
from pydantic import BaseModel
from abstractcore import create_llm, tool

class UserInfo(BaseModel):
    name: str
    email: str

@tool
def get_device_status() -> str:
    return "offline mode: enabled"

# Local/offline when the model has already been pulled.
llm = create_llm("ollama", model="qwen3:4b")

resp = llm.generate("Check device status.", tools=[get_device_status])
print(resp.tool_calls)  # Host/runtime executes tool calls by default.

user = llm.generate("Extract user info: Jane <jane@example.com>", response_model=UserInfo)
print(user.email)

# The same application can later route to a hosted API.
llm = create_llm("openai", model="gpt-4o-mini")
```

Choose LiteLLM when gateway compatibility, enterprise proxy features, and maximum routed provider coverage are the product center. Choose AbstractCore when local inference, offline-capable deployments, media, tool-call normalization, structured output, sessions, tracing, and provider-aware behavior matter inside application code.

## AbstractCore vs LangChain

LangChain is a broad application framework and integration ecosystem. It is appropriate when you want ready-made components for prompts, retrievers, tools, agents, callbacks, hosted tracing, and third-party integrations.

AbstractCore is narrower. It gives you direct LLM calls and call-level reliability primitives, then leaves orchestration and product architecture to your application or to AbstractFramework. That narrower layer is useful when you want a stable provider boundary without adopting a large chain/agent abstraction throughout the product.

Choose LangChain when you want the ecosystem and are comfortable adopting its abstractions. Choose AbstractCore when you want a focused provider layer, especially for products that must keep offline/local and remote provider options open.

## AbstractFramework vs LangGraph

LangGraph is the more established answer for stateful agent graphs, especially if you already live in the LangChain ecosystem.

AbstractFramework's answer is different:

- AbstractRuntime models durable effects, waits, resumes, ledgers, snapshots, artifacts, and replay.
- AbstractAgent provides agent loops on top.
- AbstractFlow creates portable visual workflow bundles.
- AbstractGateway exposes durable run control and ledger streaming to clients.

Choose LangGraph when explicit graph modeling and ecosystem maturity are the priority. Choose AbstractFramework when the key requirements are local-first operation, durable replayable runs, gateway-controlled clients, and a single vertically integrated stack.

## AbstractCore / AbstractFramework vs LlamaIndex

LlamaIndex specializes in document ingestion, indexing, retrieval, query engines, and RAG application structure.

AbstractCore does not replace that. It provides embeddings, generation, media/document extraction primitives, and provider normalization. AbstractFramework adds temporal memory and semantics, but that is closer to persistent knowledge and provenance than to turnkey document RAG.

Choose LlamaIndex when retrieval over documents is the center of the product. Choose AbstractCore or AbstractFramework when RAG is one component inside a broader agent/workflow system.

## When To Choose What

Choose **AbstractCore alone** if you want:

- a focused LLM provider layer for Python application code
- the same application to run offline, self-hosted, remote, or hybrid
- first-class local/open-source model support with explicit model setup
- direct cloud, local server, in-process local, and gateway providers in one API
- tool-call normalization with pass-through execution by default
- Pydantic structured output with retry and provider-aware strategies
- policy-driven media handling and fallbacks
- prompt-cache-aware sessions, events, tracing, MCP, embeddings, and optional server mode

Choose **AbstractFramework** if you want:

- durable agent/workflow execution
- pause/resume/replay across restarts
- observable ledgers for LLM calls, tool calls, waits, and decisions
- visual workflows that can be bundled and deployed across clients
- a gateway that can start, schedule, control, and stream runs
- memory and semantics components for persistent knowledge
- local-first voice, vision, coding, assistant, and workflow applications
- a foundation for complete offline assistants such as [AbstractAssistant](https://github.com/lpalbou/abstractassistant)

Choose **LiteLLM** if you want:

- an OpenAI-compatible SDK/proxy across many model APIs
- routing, gateway deployment, virtual-key/cost controls, and broad API reach
- a drop-in path for existing OpenAI-style clients

Choose **LangChain** if you want:

- a broad framework with many prebuilt components
- prompt, retriever, agent, callback, and integration abstractions
- a large community and many examples

Choose **LangGraph** if you want:

- stateful graph workflows
- checkpointed agent execution
- human-in-the-loop flows
- strong compatibility with LangChain model/tool components

Choose **LlamaIndex** if you want:

- RAG pipelines
- document ingestion and indexing
- vector-store and data-source integrations
- query engines and retrieval-heavy application structure

## Maturity Notes

AbstractCore is a beta package with substantial implementation and tests. The current repository has about 79k physical Python source lines, 266 Python test files, 213 model capability entries, and 48 architecture format entries. That is enough substance to evaluate seriously, but it is still not as battle-tested or widely adopted as LiteLLM, LangChain, LangGraph, or LlamaIndex.

AbstractFramework is more uneven because it is a multi-package ecosystem. Some parts, especially AbstractRuntime's durable model and AbstractCore's provider layer, look architecturally serious. Other parts are early, pre-1.0, or still building public adoption. For production use today, pin versions, test the exact providers and workflows you plan to run, and treat the stack as promising but not yet a de facto standard.

## Summary

AbstractCore's strongest niche in May 2026 is not "smallest wrapper" or "largest ecosystem." It is a focused provider I/O layer with strong local/open-source support, offline-capable deployment paths, remote provider reach, and broad call-level infrastructure.

AbstractFramework's strongest niche is broader: a local-first, durable, observable AI systems stack where agents, workflows, memory, gateway control, and concrete clients share the same foundation.

The objective answer is:

- **Interesting?** Yes.
- **Category leader by adoption?** No.
- **Technically differentiated?** Yes, especially for offline-capable, local-first, durable AI systems.
- **Best used blindly instead of LangChain/LangGraph/LlamaIndex/LiteLLM?** No.
- **Worth evaluating seriously if you want an open, composable foundation that can run local, remote, or fully offline?** Yes.
