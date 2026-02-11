# AbstractRuntime: durable workflows that can pause, checkpoint, and resume

If your AI workflow needs to:
- wait for a human,
- wait for a timer or an external event,
- run for minutes/hours,
- survive restarts,
- and produce a replayable audit trail…

you need more than a Python call stack.

**AbstractRuntime** is a durable workflow runtime built around a simple idea:
every step is recorded in an **append-only execution ledger**, and blocking is modeled explicitly as **waits**.

![AbstractRuntime: tick/wait/resume + ledger](assets/abstractruntime-ledger.svg)

## What you gain

- **Durable execution**: pause/resume without keeping stacks alive.
- **Replayable history**: every node/effect/write is recorded (great for UIs + debugging).
- **Deterministic control points**: tool approvals, human input, timeouts, and retries become first-class.
- **Storage flexibility**: in-memory for prototypes, JSON/JSONL/SQLite for real workloads.

## The mental model (small, on purpose)

- A workflow is a graph: `WorkflowSpec(entry_node=..., nodes={...})`.
- Each node returns a `StepPlan` describing:
  - what effect (if any) should happen,
  - where to go next.
- The runtime drives the workflow with:
  - `start()` → run id
  - `tick()` → advance until waiting/completed
  - `resume()` → provide payload for a wait and continue

## Quick start: pause + resume

This is the smallest end-to-end example (human-in-the-loop):

```python
from abstractruntime import Effect, EffectType, Runtime, StepPlan, WorkflowSpec
from abstractruntime.storage import InMemoryLedgerStore, InMemoryRunStore

def ask(run, ctx):
    return StepPlan(
        node_id="ask",
        effect=Effect(
            type=EffectType.ASK_USER,
            payload={"prompt": "Continue?"},
            result_key="user_answer",
        ),
        next_node="done",
    )

def done(run, ctx):
    answer = run.vars.get("user_answer") or {}
    return StepPlan(node_id="done", complete_output={"answer": answer.get("text")})

wf = WorkflowSpec(workflow_id="demo", entry_node="ask", nodes={"ask": ask, "done": done})
rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

run_id = rt.start(workflow=wf)
state = rt.tick(workflow=wf, run_id=run_id)
print(state.status.value)  # "waiting"

state = rt.resume(workflow=wf, run_id=run_id, wait_key=state.waiting.wait_key, payload={"text": "yes"})
print(state.status.value)  # "completed"
```

That *waiting* state is the key: it can be stored, displayed in a UI, and resumed later.

## Where LLMs and tools fit (and why it matters)

AbstractRuntime stays dependency-light and deterministic. When a node wants an LLM call or tool call,
it emits an effect like:
- `EffectType.LLM_CALL`
- `EffectType.TOOL_CALLS`

Most hosts handle those effects using **AbstractCore**, which provides provider/tool normalization.

This separation is what makes “durable tool boundaries” possible:
the runtime can wait for approval, then resume with the approved payload.

## When to use AbstractRuntime (vs just functions)

Use AbstractRuntime when you want any of the following:
- **pause/resume** (human approval, timers, background jobs)
- an **audit log** (ledger) you can replay and render
- **idempotent** command inbox patterns
- portable workflows that can be hosted locally or via a gateway

If you just need a unified LLM client, start with **AbstractCore** instead.

## Next up

Once you have durability, agent patterns become much safer and easier to operate:
[AbstractAgent](04-abstractagent-durable-agents.md).

---

### Evidence pointers (source of truth)

- AbstractRuntime repo: https://github.com/lpalbou/abstractruntime
- Runtime core: https://github.com/lpalbou/abstractruntime/blob/main/src/abstractruntime/core/runtime.py
- WorkflowSpec / StepPlan: https://github.com/lpalbou/abstractruntime/blob/main/src/abstractruntime/core/spec.py
- Effects and waits: https://github.com/lpalbou/abstractruntime/blob/main/src/abstractruntime/core/effects.py
- Storage backends: https://github.com/lpalbou/abstractruntime/blob/main/src/abstractruntime/storage/
- AbstractCore integration: https://github.com/lpalbou/abstractruntime/blob/main/src/abstractruntime/integrations/abstractcore/
