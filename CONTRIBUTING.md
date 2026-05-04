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
