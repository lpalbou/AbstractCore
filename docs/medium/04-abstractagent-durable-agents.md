# AbstractAgent: durable ReAct / CodeAct / MemAct agents you can actually operate

Agent demos usually work until the moment you need to:
- pause for approval,
- survive a restart,
- replay what happened,
- or run the same “agent bundle” from different clients.

**AbstractAgent** is a set of agent patterns implemented as **durable workflows** on top of:
- **AbstractRuntime** (pause/resume + ledger),
- and **AbstractCore** (LLM calls + tool normalization).

![AbstractAgent loop](assets/abstractagent-loop.svg)

## What you gain

- **ReAct**: a tool-first Reason → Act → Observe loop
- **CodeAct**: safely execute Python blocks/tools (host-controlled)
- **MemAct**: memory-aware agent actions (runtime-owned memory)
- **Durable runs**: resume via `run_id` + stores
- **Observability**: ledger records for LLM/tool/wait steps

## Quick start (ReAct)

```python
from abstractagent import create_react_agent

agent = create_react_agent(provider="ollama", model="qwen3:4b-instruct")
agent.start("List the files in the current directory")
state = agent.run_to_completion()

print(state.output["answer"])
```

By default, factory helpers use in-memory stores (fast for prototyping).

## Persistence (resume after restart)

To make runs durable across process restarts, pass persistent stores:

```python
from abstractagent import create_react_agent
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

run_store = JsonFileRunStore(".runs")
ledger_store = JsonlLedgerStore(".runs")

agent = create_react_agent(run_store=run_store, ledger_store=ledger_store)
agent.start("Long running task")
agent.save_state("agent_state.json")

# ... later ...

agent2 = create_react_agent(run_store=run_store, ledger_store=ledger_store)
agent2.load_state("agent_state.json")
state = agent2.run_to_completion()
print(state.output["answer"])
```

This is the “durable agent” baseline: you can stop the process, restart, and continue.

## Tool control (safety boundary)

AbstractAgent intentionally does **not** “auto-execute everything”.
Instead, it produces structured tool calls via AbstractCore and relies on the host to:
- allowlist tools,
- require approval,
- enforce policies (timeouts, limits, audit logging),
- and provide the tool execution environment.

If you want a polished interactive experience with approvals and persistence out of the box,
use **AbstractCode** (terminal host) or deploy via **AbstractGateway** + **AbstractObserver**.

## When to use AbstractAgent

Use it when you want:
- a **ready-made agent loop** that composes with your tools,
- durability + audit trail by default,
- predictable integration points (effects, tool bundles, stores).

If you want node-based workflows (and packaging `.flow` bundles), add **AbstractFlow** next:
[05-abstractflow-visual-workflows.md](05-abstractflow-visual-workflows.md).

---

### Evidence pointers (source of truth)

- AbstractAgent repo: https://github.com/lpalbou/abstractagent
- Agents: https://github.com/lpalbou/abstractagent/blob/main/src/abstractagent/agents/
- Runtime adapters: https://github.com/lpalbou/abstractagent/blob/main/src/abstractagent/adapters/
- Prompting/parsing logic: https://github.com/lpalbou/abstractagent/blob/main/src/abstractagent/logic/
- Default tool bundle: https://github.com/lpalbou/abstractagent/blob/main/src/abstractagent/tools/__init__.py
