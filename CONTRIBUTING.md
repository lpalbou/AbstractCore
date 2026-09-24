# Contributing to AbstractCore

Thanks for contributing to AbstractCore. This guide is written for external contributors and focuses on a fast setup, practical repo conventions, and a smooth PR process.

## Quick start

```bash
git clone https://github.com/lpalbou/AbstractCore.git
cd AbstractCore

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -U pip

# Tooling + tests (recommended baseline for contributors)
pip install -e ".[dev,test]"

pytest -q
```

### Optional extras (install only what you need)

AbstractCore’s default install is intentionally lightweight. Most features and provider SDKs are behind extras:

```bash
pip install -e ".[remote]"       # OpenAI + Anthropic SDKs (OpenRouter/Portkey use core httpx)
pip install -e ".[openai]"       # OpenAI SDK
pip install -e ".[anthropic]"    # Anthropic SDK
pip install -e ".[tools]"        # requests/bs4/lxml/ddgs for built-in tools
pip install -e ".[media]"        # Pillow + PDF/Office extraction
pip install -e ".[embeddings]"   # sentence-transformers + numpy
pip install -e ".[server]"       # FastAPI gateway
```

Extras compose, so a realistic app setup might be `pip install -e ".[remote,tools,media,server]"`.

If you want a “kitchen sink” contributor environment, `full-dev` is a convenient superset, but it may not install everywhere (for example MLX vs CUDA-only stacks):

```bash
pip install -e ".[full-dev]"
```

## Repository conventions

### Dependency and import-safety policy (important)

AbstractCore is designed so:
- `pip install abstractcore` stays small.
- `import abstractcore` stays import-safe.

When contributing:
- Don’t add heavy libraries to core `dependencies` in `pyproject.toml`.
- Keep optional subsystems behind explicit extras (`[tools]`, `[media]`, `[embeddings]`, `[server]`, provider SDKs).
- Avoid importing optional dependencies on default import paths (for example `abstractcore/__init__.py`). Prefer lazy imports and clear install hints like `pip install "abstractcore[media]"`.

### Formatting, linting, and typing

These tools are useful, but the full-repo baselines are not currently clean.
Treat them as diagnostics unless a maintainer explicitly asks for a full-repo
cleanup.

- `black` is the code formatter. It rewrites layout/spacing; most failures are
  style drift, not runtime bugs.
- `ruff` is the fast linter. Some findings are cosmetic, but `F821` undefined
  names, broad `except`, unused imports, and similar findings can point to real
  bugs.
- `mypy` is the static type checker. The repo has a strict target config, but
  dynamic provider code and optional dependencies still produce known legacy
  errors.

For normal PRs, format and lint the files you touched when they already have a
clean local baseline:

```bash
black path/to/changed_file.py
ruff check path/to/changed_file.py
```

If a touched file has legacy style/lint debt, avoid unrelated churn and keep the
high-signal package check clean:

```bash
ruff check --select F821 abstractcore
```

Full-repo checks are still useful for maintainers tracking cleanup progress:

```bash
black --check abstractcore tests
ruff check abstractcore
mypy abstractcore
```

### Pre-commit (recommended)

This repo has `pre-commit` hooks for formatting/lint checks. The expensive hooks
are configured for manual use, so run them explicitly when you want them.

One-time setup:

```bash
pip install -e ".[dev,test]"
pre-commit install
```

Run on all files:

```bash
pre-commit run --all-files
```

### Tests

```bash
pytest -q
```

Some provider-/network-/hardware-dependent tests are intentionally opt-in and may
skip locally. When local LLM services or heavyweight inference tests are enabled,
the suite can take a long time; during development, run the focused test file or
marker first, then a broader pass before release. See
`tests/README_VISION_TESTING.md` and `tests/README_SEED_TESTING.md`.

#### Tests never touch your home or the network

`tests/conftest.py` makes every run hermetic, whatever your shell exports:

- **Home and caches.** `HOME` (and `USERPROFILE` on Windows) points at a
  temporary directory for the whole session and a fresh one for each test, so
  everything the code derives from the home directory lands there: the
  AbstractCore config, models, embeddings and blocs under `~/.abstractcore`,
  the Hugging Face cache (`HF_HOME`, `HF_HUB_CACHE`), the data registry and the context calibration store.
  Path settings exported in your shell (the `ABSTRACT*`/`HF_*` directory, file
  and cache variables, `XDG_*`) are cleared for the run. This happens when the
  conftest is imported, before any package or `huggingface_hub` loads; a test
  fails loudly if `huggingface_hub` froze its cache path on your real home.
- **Network guard.** Sockets refuse any non-loopback destination and name
  lookup, and also the live local services on loopback: the gateway (8080), LM
  Studio (1234), Ollama (11434) and 18850. Any other loopback port stays open,
  so `TestClient`, fake servers and scratch-port fixtures work. A refused
  attempt fails the test and is listed under "network guard" at the end of the
  run with the host and port it tried to reach. Point such a test at a fake or
  a scratch port; the `fake_public_dns` fixture answers name lookups for code
  that resolves a host before a faked fetch.
- **Opting out, with a reason.** `@pytest.mark.network("reason")` marks a test
  that genuinely needs the network (a Hub lookup, a real download, a live
  provider). Such tests are skipped unless you run `pytest --allow-network`.
  `@pytest.mark.real_home("reason")` marks a test that READS your real home
  (for example installed tokenizers); `HOME` still stays temporary, and the
  test gets the real path as `ABSTRACT_TEST_REAL_HOME`. A marker without its
  reason is a collection error, as is a test module that reads the real-home
  path without the marker. `pytest --markers` lists both.

## Documentation

If a change affects user-facing behavior, update the docs entry points:
- `README.md`
- `docs/README.md`
- `docs/getting-started.md`
- `docs/architecture.md`
- `docs/api.md`
- `docs/faq.md`
- `docs/server.md` (if the HTTP gateway is affected)

Keep language clear, user-oriented, and accurate to the code (the code is the source of truth).

## Pull request checklist

- Add or update tests where appropriate.
- Run relevant tests; run `pytest -q` when feasible.
- Run `black` and `ruff check` on changed files.
- Keep `ruff check --select F821 abstractcore` passing.
- Update relevant documentation.
- Add a changelog entry when the change is user-visible.

## Versioning (maintainers)

The package version is sourced from `abstractcore/utils/version.py`.

For a release:
1. Bump `abstractcore/utils/version.py`.
2. Add a new section to `CHANGELOG.md`.
3. Verify: `python -c "import abstractcore; print(abstractcore.__version__)"`

## Security

If you believe you found a security vulnerability, please follow `SECURITY.md` for responsible disclosure.
