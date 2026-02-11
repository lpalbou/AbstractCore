# AbstractCode: a durable terminal UI for agentic coding (with tool approvals)

If you want an “agent in your terminal” but you *also* want:
- durability (resume after restart),
- an explicit tool boundary (approve/deny),
- a log of what happened,
- and the ability to run flows/agents with consistent semantics…

**AbstractCode** is the local host for that job.

It’s a terminal TUI built on the AbstractFramework foundation:
**AbstractAgent + AbstractRuntime + AbstractCore**.

![AbstractCode TUI (illustrative)](assets/abstractcode-tui.svg)

## What you gain

- **Durable runs**: pause/resume/cancel; state persists under `~/.abstractcode/`
- **Approval-gated tools** by default (safer)
- Built-in agents: `react`, `memact`, `codeact`
- Plan/Review modes (`--plan`, `--review`; `/plan`, `/review`)
- Optional: run **VisualFlow** workflows (`abstractcode[flow]`)
- Optional: remote tool execution via **MCP** (`/mcp`, `/executor`)

## Install

```bash
pip install abstractcode
```

Optional (run VisualFlow locally):

```bash
pip install "abstractcode[flow]"
```

## Quick start

### Local model (Ollama)

```bash
abstractcode --provider ollama --model qwen3:4b-instruct
```

### OpenAI-compatible local server (LM Studio, vLLM, proxies)

```bash
abstractcode --provider openai --base-url http://127.0.0.1:1234/v1 --model qwen/qwen3-next-80b
```

Inside the app:
- `/help` shows the authoritative command list
- type a task (or use `/task ...`)
- approve tools when prompted, or use `/auto-accept` (or `--auto-approve`)
- attach files with `@path/to/file` in your prompt

## Persistence (how it stays durable)

Default paths:
- state file: `~/.abstractcode/state.json`
- durable stores: `~/.abstractcode/state.d/`
- saved settings: `~/.abstractcode/state.config.json`

Disable persistence (ephemeral mode):

```bash
abstractcode --no-state
```

## When to use AbstractCode

Use it when you want:
- a polished local UX for durable agents,
- built-in tool approvals,
- fast iteration with local models,
- and a CLI that aligns with the gateway-first world (same runtime semantics).

If you want a deployable run host + browser observability, pair it with **AbstractGateway** + **AbstractObserver**:
[07-abstractgateway-and-observer.md](07-abstractgateway-and-observer.md).

---

### Evidence pointers (source of truth)

- AbstractCode repo: https://github.com/lpalbou/abstractcode
- App host + wiring: https://github.com/lpalbou/abstractcode/blob/main/src/abstractcode/
- Workflows support: https://github.com/lpalbou/abstractcode/blob/main/docs/workflows.md
- MCP integration: https://github.com/lpalbou/abstractcode/blob/main/docs/mcp.md
- Web host: https://github.com/lpalbou/abstractcode/blob/main/web/
