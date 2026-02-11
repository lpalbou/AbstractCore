# AbstractGateway + AbstractObserver: deploy durable runs, then replay and control them from the browser

Once you have durability (AbstractRuntime), the next question is operational:

> “Where do runs live, how do I start them remotely, and how do I see what happened?”

**AbstractGateway** is the deployable “run host” for AbstractRuntime runs.  
**AbstractObserver** is a gateway-only UI (Web/PWA) that renders runs by replaying and streaming the durable ledger.

![Replay-first contract](assets/abstractgateway-replay-first.svg)

## What you gain

- A clean deployment boundary for:
  - starting and scheduling runs,
  - submitting durable commands (pause/resume/cancel),
  - replaying run history,
  - streaming live updates via SSE.
- A security baseline (token + origin allowlist + limits).
- A browser UI that can reconnect safely because it’s **replay-first**.

## Quick start (gateway)

```bash
pip install "abstractgateway[http]"

export ABSTRACTGATEWAY_FLOWS_DIR="/path/to/bundles"   # *.flow dir (or a single .flow file)
export ABSTRACTGATEWAY_DATA_DIR="$PWD/runtime/gateway"

# Required by default: the server refuses to start without a token.
export ABSTRACTGATEWAY_AUTH_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export ABSTRACTGATEWAY_ALLOWED_ORIGINS="http://localhost:*,http://127.0.0.1:*"

abstractgateway serve --host 127.0.0.1 --port 8080
```

OpenAPI docs: `http://127.0.0.1:8080/docs`

Smoke checks:

```bash
curl -sS "http://127.0.0.1:8080/api/health"

curl -sS -H "Authorization: Bearer $ABSTRACTGATEWAY_AUTH_TOKEN" \
  "http://127.0.0.1:8080/api/gateway/bundles"
```

## Quick start (observer UI)

```bash
npx --yes --package @abstractframework/observer -- abstractobserver
```

Open `http://localhost:3001`, then configure:
- Gateway URL (example: `http://127.0.0.1:8080`)
- Auth token (the value of `ABSTRACTGATEWAY_AUTH_TOKEN`)

![AbstractObserver UI (illustrative)](assets/abstractobserver-ui.svg)

## Why “replay-first” matters

In many systems, UIs depend on continuous websocket state. If the UI refreshes, you lose context.

With AbstractGateway + AbstractObserver:
- the UI can **replay** the durable ledger from storage,
- then **stream** new ledger records as they happen,
- and submit control commands as durable inbox events.

This makes run observability resilient to refreshes, reconnects, and client restarts.

## How it fits in AbstractFramework

- AbstractFlow packages workflows into `.flow` bundles.
- AbstractGateway discovers those bundles and runs them durably.
- AbstractAgent patterns can also be embedded into bundles (Visual Agent nodes).
- AbstractObserver (and other clients) connect to the gateway for run control + rendering.

Next: optional capability plugins (voice and vision) that integrate cleanly with the stack:
[08-voice-and-vision-capabilities.md](08-voice-and-vision-capabilities.md).

---

### Evidence pointers (source of truth)

- AbstractGateway repo: https://github.com/lpalbou/abstractgateway
- Gateway runner/hosts: https://github.com/lpalbou/abstractgateway/blob/main/src/abstractgateway/runner.py and https://github.com/lpalbou/abstractgateway/blob/main/src/abstractgateway/hosts/
- API contract: https://github.com/lpalbou/abstractgateway/blob/main/docs/api.md (plus live `/openapi.json`)
- AbstractObserver repo: https://github.com/lpalbou/abstractobserver
- Observer client: https://github.com/lpalbou/abstractobserver/blob/main/src/lib/gateway_client.ts
