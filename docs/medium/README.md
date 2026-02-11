# AbstractFramework — Medium Article Series (Drafts)

This folder contains a publish-ready series of **Medium-style articles** about the AbstractFramework ecosystem.

The goal is to help:
- **New users** understand the stack and pick a starting path quickly.
- **Evaluators** sanity-check what each package does (and what it deliberately does *not* do).
- **Builders** copy/paste minimal examples and get to a working system fast.

If you want the canonical ecosystem docs instead, start with:
- AbstractFramework docs index: https://github.com/lpalbou/AbstractFramework/blob/main/docs/README.md
- AbstractFramework getting started: https://github.com/lpalbou/AbstractFramework/blob/main/docs/getting-started.md

## Publishing notes (Medium)

- Each article is standalone Markdown, optimized for copy/paste into Medium.
- Figures are provided as **SVG** under `assets/`. Medium may rasterize SVG; if needed, convert to PNG before uploading.
- Code blocks are minimal and “first-run friendly”; deeper details are linked to each package’s docs.

## Recommended reading order

1. [AbstractFramework: the stack in one picture](01-abstractframework-overview.md)
2. [AbstractCore: unified LLM API + servers](02-abstractcore-unified-llm-layer.md)
3. [AbstractRuntime: durable workflows + ledger](03-abstractruntime-durable-workflows.md)
4. [AbstractAgent: durable agents (ReAct/CodeAct/MemAct)](04-abstractagent-durable-agents.md)
5. [AbstractFlow: visual workflows + bundling](05-abstractflow-visual-workflows.md)
6. [AbstractCode: terminal agentic coding](06-abstractcode-terminal-coding.md)
7. [AbstractGateway + AbstractObserver: deploy + observe](07-abstractgateway-and-observer.md)
8. [AbstractVoice + AbstractVision: capability plugins](08-voice-and-vision-capabilities.md)
9. [AbstractMemory + AbstractSemantics: knowledge layer](09-memory-and-semantics.md)
10. [AbstractAssistant: tray assistant host](10-abstractassistant-tray-agent.md)
11. [End-to-end: a durable agent you can deploy](11-end-to-end-durable-agent.md)

## Figures

- `assets/abstractframework-stack.svg` — ecosystem overview (recommended vs local path)
- `assets/abstractcore-flow.svg` — AbstractCore request flow (library + servers)
- `assets/abstractruntime-ledger.svg` — runtime tick/wait/resume + append-only ledger
- `assets/abstractagent-loop.svg` — agent loop on top of runtime + core
- `assets/abstractflow-editor.svg` — VisualFlow editor (illustrative screenshot)
- `assets/abstractcode-tui.svg` — terminal UI (illustrative screenshot)
- `assets/abstractgateway-replay-first.svg` — replay-first gateway contract (HTTP + SSE)
- `assets/abstractobserver-ui.svg` — gateway observability UI (illustrative screenshot)
- `assets/capabilities-voice-vision.svg` — voice/vision capability plugins
- `assets/memory-and-semantics.svg` — triple memory + semantics schema
- `assets/abstractassistant-tray.svg` — tray assistant (illustrative screenshot)
- `assets/end-to-end.svg` — end-to-end deployment sketch
