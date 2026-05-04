# AbstractCore Server Image

This image packages the OpenAI-compatible AbstractCore HTTP server from the
published PyPI package. It is intended for release images such as:

```bash
ghcr.io/lpalbou/abstractcore-server:2.13.4
```

Release images are published for `linux/amd64` and `linux/arm64`.

The image installs:

```bash
abstractcore[server,remote,media,tokens,compression]==<version>
```

It includes the dependency-light OpenAI-compatible image proxy routes, but
intentionally does not include local model runtimes (`vllm`, `mlx`,
`huggingface`), local Diffusers/sdcpp vision backends, or local embedding
dependencies (`sentence-transformers`).

## Run

For local development, keep secrets in an uncommitted `.env` file:

```bash
ABSTRACTCORE_SERVER_API_KEY=replace-with-a-server-token
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
PORTKEY_API_KEY=pk_...
PORTKEY_CONFIG=pcfg_...
OPENAI_COMPATIBLE_BASE_URL=http://host.docker.internal:1234/v1
OPENAI_COMPATIBLE_API_KEY=optional
```

Then run the image with that environment file:

```bash
docker run --rm -p 8000:8000 --env-file .env \
  ghcr.io/lpalbou/abstractcore-server:2.13.4
```

`ABSTRACTCORE_SERVER_API_KEY` is the server auth token. Clients send it as
`Authorization: Bearer <token>`. Provider keys are read by the server and are
not sent to clients. For production, inject the same environment variables from
your orchestrator's secret store.

```bash
docker run --rm -p 8000:8000 \
  -e ABSTRACTCORE_SERVER_API_KEY="$ABSTRACTCORE_SERVER_API_KEY" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  ghcr.io/lpalbou/abstractcore-server:2.13.4
```

For local OpenAI-compatible endpoints such as LM Studio or Ollama's `/v1`
server, point the container at a reachable base URL:

```bash
docker run --rm -p 8000:8000 \
  -e ABSTRACTCORE_SERVER_API_KEY="$ABSTRACTCORE_SERVER_API_KEY" \
  -e OPENAI_COMPATIBLE_BASE_URL="http://host.docker.internal:1234/v1" \
  -e OPENAI_COMPATIBLE_API_KEY="$OPENAI_COMPATIBLE_API_KEY" \
  ghcr.io/lpalbou/abstractcore-server:2.13.4
```

Set `ABSTRACTCORE_SERVER_PROTECT_DOCS=1` if `/docs`, `/redoc`, and
`/openapi.json` should require the same server token.

## Smoke Checks

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/text-embedding-3-small","input":"hello"}'
```
