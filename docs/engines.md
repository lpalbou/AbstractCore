# Local engines: detect, install, open

An **engine** is the software that runs local model weights: Ollama, LM Studio, MLX (`mlx-lm`),
llama.cpp, vLLM, and the Hugging Face transformers stack. AbstractCore detects which engines are
on this machine, whether their servers answer, and can install a missing engine with the
vendor's documented command after showing you that command.

Related pages: [Local models](models.md) (browse, download, delete weights),
[Prerequisites](prerequisites.md) (provider setup and base URLs), [Server](server.md).

## Status

```bash
abstractcore engines status            # installed / version / supported on this host
abstractcore engines status --probe    # also GET each local server once (running, models_count)
abstractcore engines status --json     # engines_status_v1
```

HTTP: `GET /acore/engines?probe=true`, `GET /acore/engines/{id}`. Python:
`abstractcore.config.engines.engine_inventory(probe=True)`.

Each row carries `id`, `name`, `kind` (`local_server`, `local_engine`, `remote_only`),
`supported_on_host` and `unsupported_reason`, `installed`, `version`, `install_location`,
`running`, `base_url`, `reachable`, `models_count`, `install` (the plan below) and `docs_url`.
`running`/`reachable` are `null` unless you probe.

| Engine | Detected by | Version from |
|---|---|---|
| `ollama` | `ollama` on PATH, `/Applications/Ollama.app`, `%LOCALAPPDATA%\Programs\Ollama` | `ollama --version` (app bundle version as fallback) |
| `lmstudio` | the `lms` CLI (PATH or `~/.lmstudio/bin/lms`), `/Applications/LM Studio.app`, `%LOCALAPPDATA%\Programs\LM Studio` | app bundle version, else `lms version` (`cli-<commit>`) |
| `mlx` | `import mlx` | package metadata (`mlx`, `mlx-lm`) |
| `llamacpp` | `import llama_cpp`, or `llama-server` on PATH | `llama-cpp-python` metadata |
| `vllm` | `import vllm` or `vllm` on PATH | package metadata |
| `huggingface` | `import transformers` | package metadata |

Servers are read at `OLLAMA_BASE_URL` (default `http://localhost:11434`),
`LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`) and `VLLM_BASE_URL`.

## Install plans

`abstractcore engines install <id> --dry-run` shows the exact command for this host without
running anything. Commands come from a fixed table; nothing in them comes from a request.

| Engine | macOS | Linux | Windows |
|---|---|---|---|
| Ollama | `brew install ollama` when Homebrew is present, else the official script `curl -fsSL https://ollama.com/install.sh \| sh` (may ask for your password to link `/usr/local/bin/ollama`) | official script; **needs sudo** (installs to `/usr/local`, creates a systemd `ollama.service`) | `winget install --id Ollama.Ollama` (per-user), else `irm https://ollama.com/install.ps1 \| iex` |
| LM Studio | desktop app from https://lmstudio.ai/download (or `brew install --cask lm-studio`); the command installs the headless daemon and `lms` CLI: `curl -fsSL https://lmstudio.ai/install.sh \| bash`. Apple silicon only | same headless install (may need `libatomic1`) | desktop app (or `winget install --id ElementLabs.LMStudio`); headless: `irm https://lmstudio.ai/install.ps1 \| iex` |
| MLX | `python -m pip install mlx-lm` into this Python environment | not supported | not supported |
| llama.cpp | `python -m pip install llama-cpp-python` (Metal build); alternative `brew install llama.cpp` | same pip install (CPU unless `CMAKE_ARGS=-DGGML_CUDA=on`) | same pip install (needs a C/C++ compiler without a matching wheel); alternative `winget install --id ggml.llamacpp` |
| vLLM | not supported (use a remote vLLM through `VLLM_BASE_URL`) | NVIDIA GPU only: `python -m pip install vllm` | not supported |
| Hugging Face | `python -m pip install "abstractcore[huggingface]"` | same | same |

Python-package installs target the interpreter running AbstractCore. When that environment has
no `pip` (a `uv` virtual environment), the plan uses
`uv pip install --python <this interpreter> ...` instead. The plan block also lists
`alternatives`, `requires_admin`, `url` and human `notes`.

## Installing

```bash
abstractcore engines install ollama --dry-run     # show the command
abstractcore engines install ollama               # asks for confirmation, then streams the output
abstractcore engines install ollama --yes --json  # NDJSON host_job_v1 lines, final job last
abstractcore engines open lmstudio                # print and open the download page
```

HTTP: `POST /acore/engines/{id}/install` with `{"dry_run": false, "force": false}` returns a
job; follow it with `GET /acore/jobs/{id}`.

- An engine that is already installed finishes immediately as `already_installed`; pass
  `--force` to run the installer anyway.
- Only one engine install runs at a time; a second request is refused (exit `2`, HTTP `409`).
- The installer runs with no terminal input: a step that needs a password (for example `sudo`
  in the Linux Ollama script) fails with the tool's own message. Run the shown command in a
  terminal in that case.
- Cancel (`abstractcore models cancel <job_id>`, `POST /acore/jobs/{id}/cancel`, or SIGTERM /
  Ctrl-C on the CLI) stops the installer and every process it started.
- After a successful install the result reports `installed_after` and `version_after`. A new
  shell may be needed before a freshly installed command is on your PATH.

## Safety and policy

- Installs run on the machine that runs AbstractCore. For `abstractcore serve`, that is the
  server host, not the browser's machine.
- The CLI allows installs by default. The server allows them by default only when bound to a
  loopback address (`abstractcore serve --host 127.0.0.1`); set
  `ABSTRACTCORE_ALLOW_ENGINE_INSTALL=1` or `=0` to override either default. `GET /acore/engines`
  reports the effective `install_allowed`. A dry run is always allowed.
- Every `POST` under `/acore/models`, `/acore/engines` and `/acore/jobs` needs a server
  principal: the `ABSTRACTCORE_AUTH_TOKEN` bearer token, or
  `ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED=1` for local development. An upstream provider key
  alone does not authorize host actions.
- Refusals are structured: `{"ok": false, "status": "refused", "reason": ..., "message": ...}`
  with HTTP `403` (not allowed), `404` (unknown engine) or `409` (unsupported, busy).
