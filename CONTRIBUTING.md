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
pip install -e ".[openai]"       # OpenAI SDK
pip install -e ".[anthropic]"    # Anthropic SDK
pip install -e ".[tools]"        # requests/bs4/lxml/ddgs for built-in tools
pip install -e ".[media]"        # Pillow + PDF/Office extraction
pip install -e ".[embeddings]"   # sentence-transformers + numpy
pip install -e ".[server]"       # FastAPI gateway
```

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

### Style

```bash
black .
ruff check .
```

### Tests

```bash
pytest -q
```

Some provider-/network-/hardware-dependent tests are intentionally opt-in and may skip locally. See `tests/README_VISION_TESTING.md` and `tests/README_SEED_TESTING.md`.

## Documentation

If a change affects user-facing behavior, update the docs entry points:
- `README.md`
- `docs/getting-started.md`
- `docs/faq.md`

Keep language clear, user-oriented, and accurate to the code (the code is the source of truth).

## Pull request checklist

- Add or update tests where appropriate.
- Run `pytest -q`.
- Run `black .` and `ruff check .`.
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

