# End-to-end: prototype locally, then deploy a durable agent via AbstractGateway

This walkthrough shows a realistic “promotion path”:

1) Start **local-first** and iterate quickly.  
2) Then deploy the same durable semantics behind a gateway so you can observe and control runs from the browser.

![End-to-end sketch](assets/end-to-end.svg)

## Prereqs

- Python 3.10+
- Node.js 18+ (for browser UIs)
- A provider/model available locally (recommended) or via cloud keys

Local provider examples:
- **Ollama** (recommended for local-first): `ollama serve`
- **LM Studio**: OpenAI-compatible server at `http://127.0.0.1:1234/v1`

## Step 1 — Install the pinned ecosystem

```bash
pip install "abstractframework==0.1.1"
```

## Step 2 — Prototype locally (fastest feedback loop)

Start a durable terminal agent:

```bash
abstractcode --provider ollama --model qwen3:4b-instruct
```

Why start here?
- you get tool approvals + persistence immediately,
- you can validate provider/model + tool behavior,
- and you can iterate on prompts and tool boundaries safely.

## Step 3 — Start a gateway (deployment boundary)

```bash
export ABSTRACTGATEWAY_DATA_DIR="$PWD/runtime/gateway"
export ABSTRACTGATEWAY_AUTH_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export ABSTRACTGATEWAY_ALLOWED_ORIGINS="http://localhost:*,http://127.0.0.1:*"

export ABSTRACTGATEWAY_FLOWS_DIR="$PWD/workflows"   # directory containing *.flow bundles
mkdir -p "$ABSTRACTGATEWAY_FLOWS_DIR"

abstractgateway serve --host 127.0.0.1 --port 8080
```

Smoke checks:

```bash
export BASE_URL="http://127.0.0.1:8080"
export AUTH="Authorization: Bearer $ABSTRACTGATEWAY_AUTH_TOKEN"

curl -sS "$BASE_URL/api/health"
curl -sS -H "$AUTH" "$BASE_URL/api/gateway/bundles"
```

## Step 4 — Launch the observability UI

```bash
npx --yes --package @abstractframework/observer -- abstractobserver
```

Open `http://localhost:3001` and connect to:
- Gateway URL: `http://127.0.0.1:8080`
- Auth token: your `ABSTRACTGATEWAY_AUTH_TOKEN`

At this point you have:
- a deployable host for durable runs,
- and a replay-first UI for debugging and control.

## Step 5 — Create a workflow bundle (`.flow`)

Workflows are distributed as **WorkflowBundles** (`.flow` files), typically authored in the Flow Editor.

Run the editor:

```bash
npx @abstractframework/flow
```

Open `http://localhost:3003`, build your workflow, then **export a `.flow`** from the UI.

Copy the resulting bundle into your gateway’s flows directory, for example:

```bash
cp my-bundle@0.1.0.flow "$ABSTRACTGATEWAY_FLOWS_DIR/"
```

Restart the gateway (or use the UI’s reload controls if available), then confirm it’s visible:

```bash
curl -sS -H "$AUTH" "$BASE_URL/api/gateway/bundles"
```

## Step 6 — Start a run (UI or curl)

From curl (bundle mode):

```bash
curl -sS -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"bundle_id":"my-bundle","input_data":{"prompt":"Hello"}}' \
  "$BASE_URL/api/gateway/runs/start"
```

Then replay the durable ledger:

```bash
curl -sS -H "$AUTH" "$BASE_URL/api/gateway/runs/<run_id>/ledger?after=0&limit=200"
```

And stream live updates (SSE):

```bash
curl -N -H "$AUTH" "$BASE_URL/api/gateway/runs/<run_id>/ledger/stream?after=0"
```

In the browser (AbstractObserver), you’ll see the run timeline as the ledger grows.

## Step 7 — Control runs via durable commands

Pause:

```bash
curl -sS -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"command_id":"'"$(python -c 'import uuid; print(uuid.uuid4())')"'", "run_id":"<run_id>", "type":"pause", "payload":{"reason":"operator_pause"}}' \
  "$BASE_URL/api/gateway/commands"
```

Resume a waiting run with payload (for approvals or human input):

```bash
curl -sS -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"command_id":"'"$(python -c 'import uuid; print(uuid.uuid4())')"'", "run_id":"<run_id>", "type":"resume", "payload":{"payload":{"approved":true}}}' \
  "$BASE_URL/api/gateway/commands"
```

Cancel:

```bash
curl -sS -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"command_id":"'"$(python -c 'import uuid; print(uuid.uuid4())')"'", "run_id":"<run_id>", "type":"cancel", "payload":{"reason":"operator_cancel"}}' \
  "$BASE_URL/api/gateway/commands"
```

## Optional upgrades

- Add **voice** (AbstractVoice) if you want push-to-talk + TTS.
- Add **vision** (AbstractVision) if you want image generation or vision tooling.
- Add **memory** (AbstractMemory + AbstractSemantics) if you want a durable KG-backed “agent memory”.

## Where to read deeper

- AbstractGateway API contract: https://github.com/lpalbou/abstractgateway/blob/main/docs/api.md (and `/openapi.json` at runtime)
- Flow bundle lifecycle: https://github.com/lpalbou/AbstractFramework/blob/main/docs/scenarios/workflow-bundle-lifecycle.md
- Gateway-first local dev scenario: https://github.com/lpalbou/AbstractFramework/blob/main/docs/scenarios/gateway-first-local-dev.md
