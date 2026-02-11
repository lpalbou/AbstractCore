# AbstractFramework: build durable, observable AI systems (fully open source)

If you’ve ever shipped an “LLM app” and then immediately had to add **retries**, **tool safety**, **resume after restart**, **run history**, and a way to **see what happened**, you’ve already felt the gap between “a prompt” and “a system”.

**AbstractFramework** is an ecosystem of open-source packages that closes that gap by giving you:
- a unified LLM + tools interface (**AbstractCore**),
- a durable workflow runtime with an append-only ledger (**AbstractRuntime**),
- ready-made durable agent patterns (**AbstractAgent**) and visual workflows (**AbstractFlow**),
- production-friendly hosting (**AbstractGateway**) and UIs (**AbstractObserver**, Flow Editor, Code Web),
- optional capability plugins (**AbstractVoice**, **AbstractVision**),
- a knowledge layer (**AbstractMemory**, **AbstractSemantics**),
- and convenient local hosts (**AbstractCode**, **AbstractAssistant**).

![AbstractFramework stack overview](assets/abstractframework-stack.svg)

## What you gain (practically)

- **Durability**: workflows and agents survive restarts (no “lost state” after a crash).
- **Observability**: every run produces a replayable, append-only ledger (debuggable by default).
- **Tool safety**: tools are explicit, allowlisted, and typically approval-gated by the host.
- **Local-first**: run fully offline with Ollama / LM Studio / any OpenAI-compatible `/v1` server.
- **Modularity**: use one package or the full stack—compose as needed.

## Two ways to run the stack

### 1) Gateway-first (recommended)

Use an **AbstractGateway** as the “run host” for durable runs, and connect from multiple clients:
- **AbstractObserver** (run observability UI)
- Flow Editor (visual authoring)
- Code Web (browser coding assistant)
- your own app/service

This is the best path when you want:
- shared history across devices,
- central storage + streaming,
- a simple deployment boundary for security.

### 2) Local in-process (alternative)

Run everything inside one process (or on one machine) using:
- **AbstractCode** (terminal TUI)
- **AbstractAssistant** (macOS tray)
- your own Python process

This is the fastest way to start if you don’t want to run a gateway service yet.

## Quick start (5 minutes)

### Step 1 — Install the pinned release

AbstractFramework is a meta-package that pins compatible versions of the ecosystem:

```bash
pip install "abstractframework==0.1.1"
```

That pulls in (at the time of writing) `abstractcore==2.11.8`, `abstractruntime==0.4.2`, `abstractagent==0.3.1`, `abstractflow==0.3.7`, `abstractcode==0.3.6`, `abstractgateway==0.2.1`, and others (see AbstractFramework’s pinned profile in `pyproject.toml` for the exact pins).

### Step 2 — Pick a provider/model

Local (recommended for local-first development):

```bash
ollama serve
ollama pull qwen3:4b
```

Or use an OpenAI-compatible local server like **LM Studio** and point clients at `--base-url http://127.0.0.1:1234/v1`.

Cloud providers work too (set keys like `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`).

### Step 3a — Start a durable terminal agent (local path)

```bash
abstractcode --provider ollama --model qwen3:4b
```

Inside the app:
- `/help` to see commands
- type a task
- tools are approval-gated by default (safer)

### Step 3b — Run a gateway + observability UI (gateway-first path)

Terminal 1 (gateway):

```bash
export ABSTRACTGATEWAY_DATA_DIR="$PWD/runtime/gateway"
export ABSTRACTGATEWAY_AUTH_TOKEN="for-my-security-my-token-must-be-at-least-15-chars"
export ABSTRACTGATEWAY_ALLOWED_ORIGINS="http://localhost:*,http://127.0.0.1:*"

abstractgateway serve --host 127.0.0.1 --port 8080
```

Terminal 2 (observer UI):

```bash
npx --yes --package @abstractframework/observer -- abstractobserver
```

Open `http://localhost:3001`, set your Gateway URL to `http://127.0.0.1:8080`, paste the auth token, then connect.

![AbstractObserver UI (illustrative)](assets/abstractobserver-ui.svg)

## The core idea: “effects” + durable waits

AbstractFramework intentionally separates:
- **planning/decision logic** (agent/workflow nodes),
- from **execution** (tools, LLM calls, IO),
- via explicit **effects**.

When a workflow wants to do something “external” (ask a human, call a tool, call an LLM), it emits an effect. The host can:
- execute it immediately,
- require approval,
- or persist a “wait” and resume later.

This is the backbone of:
- safe tool boundaries,
- crash recovery,
- and replayable run history.

## Where to go next

If you want the “minimal mental model” for the whole ecosystem:
- Start with **AbstractCore** (LLM interface): [next article](02-abstractcore-unified-llm-layer.md)
- Then learn **AbstractRuntime** (durability): [03-abstractruntime-durable-workflows.md](03-abstractruntime-durable-workflows.md)
- Then pick your UX: terminal (**AbstractCode**) or gateway + UI (**AbstractGateway** + **AbstractObserver**)

If you’re using an AI agent to navigate the docs, use the agent-friendly entrypoints:
- `llms.txt` (curated index)
- `llms-full.txt` (full concatenated context)

---

### Evidence pointers (source of truth)

- Pinned release profile: https://github.com/lpalbou/AbstractFramework/blob/main/pyproject.toml
- Ecosystem docs index: https://github.com/lpalbou/AbstractFramework/blob/main/docs/README.md
- Architecture overview: https://github.com/lpalbou/AbstractFramework/blob/main/docs/architecture.md
- Getting started: https://github.com/lpalbou/AbstractFramework/blob/main/docs/getting-started.md
