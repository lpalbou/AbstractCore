# Data-Home Registry

One machine-level registry of every data directory the framework writes — so an operator can always answer "what is on my disk, where, and what is safe to delete" from one view instead of digging through app folders.

## Why it exists

AbstractFramework packages co-locate their own data: model caches, prompt-cache artifacts, run/session stores, logs, entity homes. After months of experiments this data spreads across the machine with no map. The ruled design (cross-package vote + operator sign-off, 2026-07-13):

- **Data stays co-located per app** — each package keeps owning its files.
- **Every package registers its data homes** in one machine-level registry at first write.
- **One management surface** (the AbstractGateway console) enumerates all registered homes with live sizes and offers purge actions; observability apps get read-only telemetry.
- **AbstractCore owns the primitive** (this module); everything else reads it, nothing re-derives it.

## The registry file

`~/.abstractframework/data_registry.json` — override with the `ABSTRACTFRAMEWORK_DATA_REGISTRY` environment variable. Writes are atomic (temp file + `os.replace`) under a cross-process lock (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows), so many processes can register at first write concurrently.

A corrupt registry file is refused loudly and never silently regenerated — repair or remove it explicitly.

## Row shape

Each registered home is one row:

| Field | Meaning |
|---|---|
| `name` | Unique row id (e.g. `huggingface-hub-cache`) |
| `path` | Absolute, resolved directory path |
| `kind` | One of the ruled set below |
| `owner` | Package/app id that owns the data (e.g. `abstractgateway`) |
| `safe_to_purge` | **Owner-declared.** `False` rows are structurally refused by the purge verb |
| `description` | Human sentence for the management view (include re-download/restore cost) |
| `meta` | Optional owner dict |

`kind` is a closed set, widened only by owning-seat proposal + a vocabulary ruling:

- `model-cache` — downloadable model weights (e.g. the Hugging Face hub cache)
- `prompt-cache` — KV/prefix cache artifacts
- `runs` — durable run/ledger stores
- `sessions` — session/conversation state
- `logs` — log directories
- `entity-home` — a summoned entity's home (a LIFE: memory + book + spark + artifacts). Registers `safe_to_purge=False` by construction; an entity home's `artifacts/` stays inside its `entity-home` row, never a separate registration
- `artifacts` — content-addressed artifact stores with rebuildable catalogs; purge semantics are per-store (prunable run-media vs load-bearing stores are separate rows)

## API

```python
from abstractcore.utils import (
    register_data_home, list_data_homes, get_data_home,
    data_home_size, purge_data_home, unregister_data_home,
    register_core_data_homes, DataHome, DataRegistryError, DATA_HOME_KINDS,
)

# Register-at-first-write (idempotent upsert; call when your app first writes the dir)
register_data_home(
    "myapp-logs",
    path="~/.myapp/logs",
    kind="logs",
    owner="myapp",
    safe_to_purge=True,
    description="MyApp rotating logs; safe to purge anytime.",
)

# Enumerate (the management view's read)
rows = list_data_homes(include_sizes=True)   # + size_bytes (recursive), + exists

# Purge safely (dry_run first for a confirm dialog)
plan = purge_data_home("myapp-logs", dry_run=True)   # would-purge accounting
result = purge_data_home("myapp-logs")               # deletes CONTENTS only
```

### Best-effort lane for hot paths

`register_data_home` refuses loudly — right for a management surface, wrong
inside a data-writing code path. Hot paths call the best-effort twin:

```python
from abstractcore.utils import ensure_data_home_registered, ensure_core_data_homes

# Never raises: dedupes per process, degrades to ONE #FALLBACK warning per
# name when the registry refuses or the disk write fails, retries silently
# on the next call. A broken registry must never break embeddings/logging.
ensure_data_home_registered("myapp-logs", path="~/.myapp/logs", kind="logs",
                            owner="myapp", safe_to_purge=True)

# Once-per-process register_core_data_homes() (providers call this).
ensure_core_data_homes()
```

Kill switch: set `ABSTRACTFRAMEWORK_DATA_REGISTRY_DISABLE=1` to turn all
best-effort registration off (the strict verbs are unaffected).

## Safety model (what purge will and will not do)

- **Owner-declared safety is enforced**: purging a `safe_to_purge=False` row raises, naming the owner and the rule. Entity homes are the canonical case — deleting one is the owner's designed lifecycle act, never a cache cleanup.
- **Contents only**: the home directory itself and its registration survive a purge.
- **No nesting**: registering a home that is an ancestor or descendant of an existing home is refused — an ancestor purged as "safe" must never eat a nested home that declared itself unsafe. The directory containing the registry file itself is also refused.
- **No symlink escapes**: symlinks inside a home are unlinked, never followed. If the registered path itself no longer resolves to its registration-time location (a component was swapped for a symlink), purge refuses instead of deleting through the swap.
- **Hand-edited registries still can't cross-purge**: at purge time, other registered homes and the registry file are skipped with `skipped_protected` accounting even if a nesting guard was bypassed by editing the JSON directly.
- **Unregistered paths cannot be purged** through this API at all; `unregister_data_home` removes the row only and never touches disk.

## Core's own registrations

Core registers its data homes at first write (no setup call needed):

- **Hugging Face hub cache** (`HF_HUB_CACHE` > `HUGGINGFACE_HUB_CACHE` > `$HF_HOME/hub` > `~/.cache/huggingface/hub`) — `model-cache`, safe to purge (models re-download on demand; large models cost bandwidth/time to restore). Registered when a HuggingFace/MLX/LMStudio provider or a local embedding model is constructed (`register_core_data_homes()` via the once-per-process ensure lane).
- **LM Studio model directory** (when present) — `model-cache`, **report-only** (`safe_to_purge=False`, owner `lmstudio`): that directory belongs to LM Studio and is never touched by the framework.
- **`abstractcore-blocs`** — `~/.abstractcore/blocs`, the file-bloc store (extracted file-text snapshots + per-(provider, model) KV prompt-cache artifacts; the KV artifacts are the bulk — hundreds of GB on long-lived machines). `prompt-cache`, safe to purge: content re-extracts and KV recompiles on demand, but strict durable-bloc bindings go cold until recompiled. Registered by `FileBlocStore.upsert` at first write and by the core probe when the directory exists.
- **`abstractcore-prompt-cache-repl-sessions`** — `~/.abstractcore/prompt_cache_repl_sessions`, saved sessions from the RETIRED save feature of the prompt-cache REPL demo (no current writer). `sessions`, **report-only**: the `*.artifacts` KV files are re-derivable bulk but the JSON transcripts are user-elected saves — review and delete manually.
- **`abstractcore-logs`** — the file-logging directory, registered when file logging first creates it (`logs`, safe to purge).
- **`abstractcore-embeddings-cache`** — the default `~/.abstractcore/embeddings` vector cache (`artifacts`, safe to purge: vectors recompute). Only the machine-level default dir registers; a constructor-custom cache dir lives inside its caller's data home and rides that container's row (self-registering ephemeral dirs spammed 372 pytest-tmp rows in the live incident that set this rule).
- **`abstractcore-glyph-cache`** — glyph-compression PNG renders from PDFs (`artifacts`, safe to purge: re-rendered from source documents), registered at first render.
- **`abstractcore-local-models`** — `~/.abstractcore/models` vision/caption model downloads (`model-cache`, safe to purge), registered by `abstractcore --download-vision-model`.

## Notes

- `size_bytes` is an apparent-size estimate (per-name `lstat`, symlinks not followed, hardlinks counted once per name) — a management view number, not `du` disk accounting.
- Consumers rendering rows should display unknown `kind` values labeled-unknown rather than refusing — the closed set is enforced at registration, not at read.
