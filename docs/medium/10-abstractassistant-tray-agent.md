# AbstractAssistant: a durable tray assistant (macOS-first) with a safe tool boundary

Not every “agent” should live in a web app.

Sometimes you want:
- a local assistant that’s always available,
- durable sessions that survive restarts,
- attachments + tool approvals,
- and a clear separation between model decisions and host execution.

**AbstractAssistant** is a macOS-first tray app (plus a CLI) that hosts a **local, durable agent** on the AbstractFramework stack.

![AbstractAssistant tray (illustrative)](assets/abstractassistant-tray.svg)

## What you gain

- **Tray UI**: menu bar bubble with sessions, attachments, approvals, and optional voice
- **CLI**: run a single agentic turn from the terminal
- **Durable tool boundary**: tool calls become resumable waits; host executes only after approval
- **Local-first**: works with Ollama / LM Studio / OpenAI-compatible endpoints (or cloud keys)

## Install

```bash
pip install abstractassistant
```

Requirements summary:
- Python 3.10+
- Tray UI is macOS-first (CLI/backend may work elsewhere, but macOS is the primary target)

## Quick start

Tray UI:

```bash
assistant tray
```

CLI (one turn):

```bash
assistant run --prompt "What is in this repo and where do I start?"
```

Provider/model override:

```bash
assistant --provider ollama --model qwen3:4b-instruct run --prompt "Summarize my changes"
```

## Data & durability

Default data directory: `~/.abstractassistant/` (override with `--data-dir`).

Typical contents:
- `session.json`: transcript snapshot + last run id (fast UI state)
- `sessions.json`: session registry + active session id
- `runtime/`: AbstractRuntime stores (run state, ledger, artifacts)

This structure is what makes “close app → reopen → continue” reliable.

## When to use AbstractAssistant

Use it when you want:
- a persistent personal assistant on your machine,
- a UI that exposes approvals and attachments,
- durability without running a gateway service.

If you also want multi-client observability and deployment, the gateway-first path is the natural next step.

Next: a complete end-to-end walkthrough tying the pieces together:
[11-end-to-end-durable-agent.md](11-end-to-end-durable-agent.md).

---

### Evidence pointers (source of truth)

- AbstractAssistant repo: https://github.com/lpalbou/abstractassistant
- Host wiring: https://github.com/lpalbou/abstractassistant/blob/main/abstractassistant/core/agent_host.py
- Session/index persistence: https://github.com/lpalbou/abstractassistant/blob/main/abstractassistant/core/session_index.py
- CLI entry point: https://github.com/lpalbou/abstractassistant/blob/main/pyproject.toml
