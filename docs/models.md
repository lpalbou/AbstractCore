# Local models: browse, check fit, download, delete

AbstractCore can show which local models suit your machine, where they are installed, and
fetch or remove them with each engine's own tool. The same data is available from the CLI
(`abstractcore models ...`, with `--json`), from the HTTP server (`/acore/models/*`), and from
Python. AbstractGateway re-exposes the same payloads, so every console shows the same answers.

Related pages: [Engines](engines.md) (installing Ollama, LM Studio, MLX, llama.cpp),
[Centralized Config](centralized-config.md) (which model each capability uses and
`abstractcore models status`), [Memory and model residency](memory-management.md)
(what is loaded right now), [Server](server.md).

## What this covers

| Verb | CLI | HTTP | Python |
|---|---|---|---|
| Describe this machine | `abstractcore host profile --json` | `GET /acore/host/profile` | `abstractcore.utils.host_profile.host_profile()` |
| Browse downloadable models | `abstractcore models catalog [--engine X] [--fits] [--tag T] [--hub] --json` | `GET /acore/models/catalog?q=&engine=&fits=&hub=&tag=` | `abstractcore.config.model_catalog.catalog(...)` |
| Search | `abstractcore models search "<query>" [--engine X] [--fits] [--hub] --json` | same route with `q=` | `model_catalog.search(q, ...)` |
| List installed models | `abstractcore models list [--provider X] --json` | `GET /acore/models/installed?provider=` | `model_materializer.list_installed(provider)` |
| Download | `abstractcore models download <provider> <artifact> [--dry-run] [--detach] [--json]` | `POST /acore/models/download` | `host_jobs.start_download_job(...)` / `model_materializer.download(...)` |
| Delete | `abstractcore models delete <provider> <artifact> [--yes] [--dry-run] [--force] [--json]` | `POST /acore/models/delete` | `model_materializer.delete_artifact(...)` |
| Follow background work | `abstractcore models jobs [<job_id>] [--kind K] [--status S] --json`, `abstractcore models cancel <job_id>` | `GET /acore/jobs`, `GET /acore/jobs/{id}`, `POST /acore/jobs/{id}/cancel` | `host_jobs.default_registry()` |

Exit codes for every verb: `0` success, `1` error, `2` refused (a policy, a delete blocker, or
a destructive action without `--yes`). A refusal still prints its JSON (`status`, `reason`,
`message`, and `delete_blockers` for deletes).

## Artifact references

An **artifact** names the exact weights to fetch, quantization included:

| Provider | Example artifact | Notes |
|---|---|---|
| `ollama` | `qwen3:8b` | An Ollama tag. |
| `lmstudio` | `qwen/qwen3.5-9b@4bit` | LM Studio model id with `@quant`. A bare id lets LM Studio pick its default build. |
| `mlx` | `mlx-community/Qwen3-8B-4bit` | A Hugging Face repo; MLX repos hold one quantization. |
| `huggingface` | `unsloth/Qwen3-8B-GGUF:Q4_K_M` | A GGUF repo plus `:QUANT`. Only the matching `*Q4_K_M*.gguf` files are fetched. Without `:QUANT` the whole repo is downloaded. |
| `mlx-gen`, `supertonic` | `AbstractFramework/flux.2-klein-4b-8bit`, `supertonic-3` | Image and voice starters. |

## The host profile

`abstractcore host profile --json` (`host_profile_v1`) reports:

- `os`, `arch`, `accelerator` (`metal`, `cuda`, `rocm`, `none`), `gpu_name`, `unified_memory`;
- `ram_bytes`, `vram_bytes` (CUDA, from `nvidia-smi`);
- `ceiling_bytes` and `ceiling_source`: the most memory a model can use. On Apple silicon this is
  the `iogpu.wired_limit_mb` sysctl when set (`metal_wired_limit`), else Metal's recommended
  working set (`metal_recommended`, needs `mlx`), else 75% of RAM (`ram_75pct`). On NVIDIA it is
  total VRAM (`cuda_total`). This is the same ceiling the context estimator uses;
- `free_now_bytes`: memory available right now (available RAM capped at the ceiling; free VRAM on CUDA);
- `disk`: free space for the Hugging Face cache, the LM Studio models folder and the Ollama store.

Values that cannot be measured are `null`.

## The catalog

`abstractcore models catalog --json` returns `model_catalog_v1`: the host profile plus one row
per model. The curated seed ships with AbstractCore
(`abstractcore/assets/model_downloads_catalog.json`, schema alongside) and covers chat, coding,
vision and embedding models across Qwen, Llama, Gemma, Mistral, DeepSeek, GPT-OSS, Phi,
SmolLM, Granite, GLM and common embedding families, plus the recommended fresh-install set
(`qwen/qwen3.5-9b@4bit`, FLUX.2 klein 4B, Supertonic 3).

Each row carries `capabilities` (text, vision, audio, tools, thinking, max_tokens, embedding)
joined from AbstractCore's model capability registry, and one entry per artifact with:

- `presence.status`: `installed`, `absent`, `unknown` (the engine could not be asked), or
  `not_applicable` (remote providers);
- `download_bytes` and `size_source`: `catalog` (observed on a real engine store), `engine`
  (reported by the installed engine), `hf_api` (Hugging Face file listing), `estimate`
  (parameters x bits), or `unknown`;
- `fit` (see below), `downloadable`, `supported_on_host`, and the `cli_download` command;
- `recommended`: exactly one artifact per row is pre-selected for this machine. The curated
  starter wins when the host can run it; otherwise the order is LM Studio, MLX, Ollama,
  Hugging Face on Apple silicon and Ollama, LM Studio, Hugging Face elsewhere, preferring
  artifacts that fit and engines that are installed.

Filters: `q` (every word must start a word of the id, name, vendor, tags or artifacts),
`--engine` (`ollama`, `lmstudio`, `mlx`, `huggingface`, or `llamacpp` for GGUF artifacts),
`--fits` (keep `fits` and `tight` only), `--tag` (`chat`, `coding`, `vision`, `embedding`, ...).

### Hugging Face enrichment (`--hub`)

The catalog works offline. With `--hub` (`hub=true`), AbstractCore asks the Hugging Face API
for exact file sizes and parameter counts of the Hugging Face-hosted artifacts and, when a
query is given, appends up to 20 `hf_search` rows per lane (MLX repos on Apple silicon, GGUF
repos everywhere; a GGUF search result gets the best available quant, Q4_K_M first).
Answers are cached for 24 hours in `~/.abstractcore/config/cache/hf_hub_catalog.json`. A rate
limit (HTTP 429) or an offline machine is reported in `hub.errors` and the offline answer is
returned. Set `HF_TOKEN` for gated repos and higher rate limits.

## Fit verdicts

`fit` answers "will this run here" before any byte is downloaded:

```
W    weights       exact artifact size, else parameters x bits / 8 x 1.03
KV   KV cache      n x 2 x layers x kv_heads x head_dim x 2 bytes (f16), when the geometry is known;
                   otherwise n x 0.5 MiB x (parameters / 8e9)
O    overhead      max(0.5 GiB, 5% of W)
Ceff usable       ceiling - max(2 GiB, 5% of ceiling)
need = W + KV + O          (n = min(8192, context window))
```

| Verdict | Meaning |
|---|---|
| `fits` | `need <= 0.8 x Ceff` |
| `tight` | `need <= Ceff`: runs, with little headroom |
| `partial_offload` | CUDA only: too big for VRAM, but fits VRAM + 75% of RAM (slower) |
| `too_large` | does not fit |
| `unknown` | no size and no parameter count, or no measurable ceiling |

The block also reports `need_bytes`, `ceiling_bytes`, `free_now_bytes`, `fits_now` (whether it
fits in memory that is free right now), `disk_ok` (download size plus 5 GiB headroom against
free disk), `max_context` (largest context beside the weights, capped at the model window),
`confidence` (`exact`, `estimated`, `rough`, `unknown`) and human `notes`. Mixture-of-experts
models are sized by total parameters: every expert is resident. Verdicts are advisory; nothing
blocks a download or a load on them.

Python:

```python
from abstractcore.utils.host_profile import host_profile
from abstractcore.utils.model_fit import estimate_fit

fit = estimate_fit(host=host_profile(), params_total=8_200_000_000, quant="q4_k_m", max_tokens=131072)
print(fit["verdict"], fit["need_bytes"], fit["max_context"])
```

## Installed models

`abstractcore models list --json` returns `models_installed_v1`: one row per installed
artifact with `provider`, `artifact`, `quant`, `size_bytes`, `params_total`, `location`,
`loaded`, `catalog_id`, `deletable` and `delete_blockers`, plus `engines_probed`, per-engine
`errors` and `totals`.

| Engine | Source | Size fields |
|---|---|---|
| Ollama | `GET /api/tags` (`/api/ps` for `loaded`) | `size`, `details.parameter_size`, `details.quantization_level` |
| LM Studio | `lms ls --json` (`lms ps --json` for `loaded`) | `sizeBytes`, `paramsString`, `quantization.name`, `maxContextLength` |
| MLX / Hugging Face | the Hugging Face cache (`scan_cache_dir()`) | repo size on disk; rows split into `mlx` and `huggingface` by AbstractCore's MLX detection |

When the local Ollama server is not running, its tags are listed from the on-disk manifests
(marked `engine_not_running`). An engine that cannot be read appears in `errors`, never as an
empty list.

## Downloading

```bash
abstractcore models download ollama qwen3:8b
abstractcore models download lmstudio qwen/qwen3.5-9b@4bit
abstractcore models download huggingface unsloth/Qwen3-8B-GGUF:Q4_K_M
abstractcore models download lmstudio qwen/qwen3.5-9b@4bit --dry-run     # show the command only
```

- Ollama downloads use `POST /api/pull` with real byte progress; LM Studio uses
  `lms get <artifact> --yes` and verifies afterwards that the requested model landed;
  Hugging Face uses `snapshot_download` (narrowed by `:QUANT`), with a byte total read from the
  hub first.
- A download that cannot fit on disk with 5 GiB to spare is refused before it starts (Hugging
  Face computes the size itself; `expected_bytes` arms the check for other engines).
- `--json` streams one `host_job_v1` object per line while the download runs and ends with the
  final job. SIGTERM or Ctrl-C cancels the download, stops the engine tool, and the last line
  reports `cancelled`. `--detach` starts the download in the background and prints the job;
  follow it with `abstractcore models jobs <job_id> --json`.

## Deleting

```bash
abstractcore models delete ollama qwen3:8b --dry-run
abstractcore models delete ollama qwen3:8b --yes
abstractcore models delete lmstudio qwen/qwen3.8-27b@q4_k_m --yes --force
```

| Engine | Mechanism |
|---|---|
| Ollama | `DELETE /api/delete` (with `--force`, a loaded model is unloaded first) |
| LM Studio | `lms unload` when loaded (`--force`), then removal of the model's files under the LM Studio models folder and its hub entry |
| MLX / Hugging Face | `scan_cache_dir().delete_revisions(...).execute()` for the whole repo |

A delete is refused (exit `2`, HTTP `409`) when a blocker applies, unless `--force`:

- `loaded`: the model is in memory;
- `remote_engine`: the Ollama server is on another machine;
- `shared_cache:mlx,huggingface`: the Hugging Face cache entry is classified for the other
  engine (the files are shared).

`unknown_location` (a model bundled inside LM Studio) and `engine_not_running` cannot be
forced. Paths outside the known model stores are never removed.

## Jobs

Downloads, deletes and engine installs are jobs (`host_job_v1`): `job_id`, `kind`
(`download`, `delete`, `engine_install`), `status` (`queued`, `running`, `completed`, `failed`,
`cancelled`), `percent`, `downloaded_bytes`, `total_bytes`, `message`, `log_tail` (last 60
lines), `command`, `dry_run`, `started_at`, `finished_at`, `error`, `joined`,
`cli_equivalent`, and `result` (the verb's own outcome).

- A second request for the same download joins the running job (`joined` counts them).
- At most 40 finished jobs are kept per process.
- Job snapshots are written to `~/.abstractcore/config/jobs/` (override with
  `ABSTRACTCORE_JOBS_DIR`; disable with `ABSTRACTCORE_JOBS_PERSIST=0`), so
  `abstractcore models jobs` lists work started by `abstractcore serve` or by
  `--detach`, and `abstractcore models cancel <job_id>` can stop it.
