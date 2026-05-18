# AbstractCore Server Image

This image packages the OpenAI-compatible AbstractCore HTTP server from the
published PyPI package. It is intended for release images such as:

```bash
ghcr.io/lpalbou/abstractcore-server:2.13.8
```

Release images are published for `linux/amd64` and `linux/arm64`.

The image installs:

```bash
abstractcore[server,remote,media,tokens,compression]==<version>
```

It includes remote chat/responses, remote embeddings, remote STT/TTS routing,
remote OpenAI-compatible image proxying, server dependencies, media parsing,
token counting, and compression helpers. It intentionally does not include local
model runtimes, local embedding dependencies, or the AbstractVoice/AbstractVision
local plugin runtimes because those pull large native inference stacks. Install
`abstractcore[voice]` or `abstractcore[vision]` in a custom image when local
voice/vision plugin execution is required.

## Run

For local development, keep secrets in an uncommitted `.env` file:

```bash
ABSTRACTCORE_AUTH_TOKEN=replace-with-a-server-token
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
PORTKEY_API_KEY=pk_...
PORTKEY_CONFIG=pcfg_...
OPENAI_BASE_URL=http://host.docker.internal:1234/v1
```

Then run the image with that environment file:

```bash
docker run --rm -p 8000:8000 --env-file .env \
  ghcr.io/lpalbou/abstractcore-server:2.13.8
```

`ABSTRACTCORE_AUTH_TOKEN` is the server auth token. Clients send it as
`Authorization: Bearer <token>`. At `/docs`, use Swagger UI's normal
`Authorize` button when server auth is enabled; AbstractCore validates that
token before Swagger treats it as authorized. Provider keys are read by the
server and are not sent to clients.
For production, inject the same environment variables from your orchestrator's
secret store.

```bash
docker run --rm -p 8000:8000 \
  -e ABSTRACTCORE_AUTH_TOKEN="$ABSTRACTCORE_AUTH_TOKEN" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  ghcr.io/lpalbou/abstractcore-server:2.13.8
```

For local OpenAI-compatible endpoints such as LM Studio or Ollama's `/v1`
server, point the container at a reachable base URL:

```bash
docker run --rm -p 8000:8000 \
  -e ABSTRACTCORE_AUTH_TOKEN="$ABSTRACTCORE_AUTH_TOKEN" \
  -e OPENAI_BASE_URL="http://host.docker.internal:1234/v1" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/lpalbou/abstractcore-server:2.13.8
```

Set `ABSTRACTCORE_SERVER_PROTECT_DOCS=1` if `/docs`, `/docs-lite`, `/redoc`,
and `/openapi.json` should require the same server token.

For image generation through `/v1/images/*`, set `OPENAI_BASE_URL` for a
remote OpenAI-compatible image endpoint. Use `OPENAI_API_KEY` when that
endpoint requires a bearer token.

## Smoke Checks

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer $ABSTRACTCORE_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/text-embedding-3-small","input":"hello"}'
```
