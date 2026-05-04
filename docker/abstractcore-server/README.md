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
  ghcr.io/lpalbou/abstractcore-server:2.13.4
```

## Smoke Checks

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/text-embedding-3-small","input":"hello"}'
```
