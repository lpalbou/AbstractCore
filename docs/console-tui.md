# Terminal console (`abstractcore-console`)

`abstractcore-console` is AbstractCore's keyboard-first terminal console.
It configures `abstractcore.json` (default models, providers and API
keys, capability routes, media, embeddings, server) and manages local
models and engines on this machine.

```bash
pip install abstractcore               # the CLI the console drives
cargo install abstractcore-console     # the console (Rust 1.87+)
abstractcore-console                   # --wizard for the guided setup
```

The console reads the config file directly and runs the `abstractcore`
CLI for everything derived or written. It finds the CLI through
`$ABSTRACTCORE_CLI`, then `abstractcore` on `PATH`, then
`~/.local/bin/abstractcore` (the `uv tool` shim), then
`./.venv/bin/abstractcore` in the current directory.

## Screens

| Key | Screen | What it is for |
|---|---|---|
| 1 | Overview | every config section's state; Enter jumps to its owner |
| 2 | Model | default models and app defaults |
| 3 | Providers | providers, endpoint profiles, API keys, model discovery |
| 4 | Routes | capability routes (vision, audio, image, video, embeddings); `w` downloads a route's weights |
| 5 | Media | vision / audio / video settings |
| 6 | Embeddings | the embeddings route |
| 7 | Server | server, logging, timeouts, cache |
| 8 | Review | the file identity and test evidence |
| 9 | **Models** | the model catalog fitted to this host: `w` download, `d` delete, `/` filter, `f` fits only, `e` engine, `v` installed ⇄ catalog, `c` cancel |
| 0 | **Engines** | Ollama, LM Studio, MLX, llama.cpp, vLLM, Hugging Face: `i` install (confirmed; shows the command and the host), `o` open the download page, `r` probe |

Models and Engines speak the same vocabulary as the web console and the
gateway console: weights `installed / not downloaded / unknown /
remote`, fit `fits / tight / too large / partial offload / unknown`.
Every action there is a CLI command you can run yourself:

| Console action | CLI equivalent |
|---|---|
| Models list | `abstractcore models catalog --json` / `models search <q> [--engine X] [--fits] --json` |
| Installed view | `abstractcore models list --json` |
| `w` download | `abstractcore models download <provider> <artifact> --json` |
| `d` delete | `abstractcore models delete <provider> <artifact> --yes [--force] --json` |
| Engines list | `abstractcore engines status [--probe] --json` |
| `i` install | `abstractcore engines install <id> --yes [--dry-run] --json` |

Downloads and installs run on the machine that runs the console. One job
runs at a time; `c` cancels it, and `q` will not quit while it runs.

## Same screens in the gateway console

The Models and Engines screens are a library inside the crate
(`abstractcore_console::screens`). `abstractgateway-console` mounts them
over an HTTP transport to the gateway's `/api/gateway/models/*` and
`/api/gateway/engines/*` routes, so a gateway operator sees the same
screens, acting on the gateway host. See the crate README
(`console-tui/README.md`) for the library API.
