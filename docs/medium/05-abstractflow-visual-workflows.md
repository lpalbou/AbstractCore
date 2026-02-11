# AbstractFlow: visual, durable workflows you can bundle and run anywhere

If your team needs workflows that are:
- durable (pause/resume),
- shareable (portable format),
- inspectable (diagram-first),
- and deployable (bundles),

**AbstractFlow** gives you both:
1) a small Python API (`Flow`, `FlowRunner`) and  
2) a portable **VisualFlow** JSON format + a visual editor.

![AbstractFlow visual editor (illustrative)](assets/abstractflow-editor.svg)

## What you gain

- **Diagram-first authoring** (VisualFlow JSON + editor)
- **Durable execution** via AbstractRuntime (wait nodes, subflows, retries)
- **Portable bundles** (`.flow`) for distribution via AbstractGateway
- **Two runtimes**: execute flows locally (Python) or host them behind a gateway

> Status note: AbstractFlow is pre-alpha. The core concepts are stable, but expect breaking changes.

## Quick start (programmatic Flow)

```python
from abstractflow import Flow, FlowRunner

flow = Flow("linear")
flow.add_node("double", lambda x: x * 2, input_key="value", output_key="doubled")
flow.add_node("add_ten", lambda x: x + 10, input_key="doubled", output_key="final")
flow.add_edge("double", "add_ten")
flow.set_entry("double")

print(FlowRunner(flow).run({"value": 5}))
# {"success": True, "result": 20}
```

This path is great for embedding flow concepts into your codebase quickly.

## Quick start (execute VisualFlow JSON)

```python
import json
from abstractflow.visual import VisualFlow, execute_visual_flow

with open("my-flow.json", "r", encoding="utf-8") as f:
    vf = VisualFlow.model_validate(json.load(f))

print(execute_visual_flow(vf, {"prompt": "Hello"}, flows={vf.id: vf}))
```

If your flow uses subflows, load all referenced `*.json` into the `flows={...}` mapping.

## Visual editor (local)

The editor is split into:
- a Python **FastAPI backend** (optional; shipped with `abstractflow[editor]`)
- a JS **frontend** published as `@abstractframework/flow` (run via `npx`)

Terminal 1 (backend):

```bash
pip install "abstractflow[editor]"
abstractflow serve --reload --port 8080
```

Terminal 2 (frontend):

```bash
npx @abstractframework/flow
```

Open:
- UI: `http://localhost:3003`
- Backend health: `http://localhost:8080/api/health`

## Bundling: ship workflows as a `.flow`

AbstractFlow can pack a VisualFlow JSON (and its dependencies) into a portable bundle:

```bash
abstractflow bundle pack web/flows/ac-echo.json --out /tmp/ac-echo.flow
abstractflow bundle inspect /tmp/ac-echo.flow
abstractflow bundle unpack /tmp/ac-echo.flow --dir /tmp/ac-echo
```

Those bundles are what AbstractGateway discovers and runs in “bundle mode”.

## How it fits in AbstractFramework

- AbstractFlow compiles VisualFlow → AbstractRuntime workflows/effects.
- AbstractRuntime provides durability + ledger.
- AbstractCore handles LLM/tool calls when your nodes need them.
- AbstractGateway hosts bundles and exposes them to UIs (Observer, Flow Editor, Code Web).

If you want a polished local UX for running flows and agents, the next stop is **AbstractCode**:
[06-abstractcode-terminal-coding.md](06-abstractcode-terminal-coding.md).

---

### Evidence pointers (source of truth)

- AbstractFlow repo: https://github.com/lpalbou/abstractflow
- Flow runner: https://github.com/lpalbou/abstractflow/blob/main/abstractflow/runner.py
- VisualFlow execution: https://github.com/lpalbou/abstractflow/blob/main/abstractflow/visual/executor.py
- CLI: https://github.com/lpalbou/abstractflow/blob/main/abstractflow/cli.py
- Editor backend: https://github.com/lpalbou/abstractflow/blob/main/web/backend/
