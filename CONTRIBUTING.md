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

### Local engines (optional)

The editable install above is the light setting: every remote provider, tools, media, the server
and the capability plugins. To work on the local engines, add your platform's setting:

```bash
pip install -e ".[apple,dev,test]"   # Apple silicon: MLX, HuggingFace/GGUF, local media engines
pip install -e ".[gpu,dev,test]"     # NVIDIA / AMD: vLLM, HuggingFace/GGUF, local media engines
```

## Repository conventions

### Dependency and import-safety policy (important)

AbstractCore has exactly three install settings (see [Installation](docs/installation.md)):
- `pip install abstractcore` (light) runs every remote provider and carries tools, media inputs,
  the server and the capability plugins, with no local engine.
- `abstractcore[apple]` and `abstractcore[gpu]` add every local engine for their platform.
- `import abstractcore` stays import-safe.

When contributing:
- Never add a fourth setting or a feature extra. A dependency the light install needs goes in
  core `dependencies`; a local engine (heavy or platform-specific) goes in `apple` and/or `gpu`.
  `tests/test_install_settings.py` enforces this.
- Keep local engines out of default import paths (for example `abstractcore/__init__.py`). Prefer
  lazy imports and install hints that name one of the three settings (`abstractcore.utils.install_settings`).
- Contributor tooling lives in the `dev`, `test` and `docs` extras (`pip install -e ".[dev,test]"`);
  they are not install settings.

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

#### Live email tests (opt-in)

`tests/email/live/` runs the email library against a **real test mailbox**
(framework backlog 0992): verified TLS with the default trust store (plus a
negative control that must fail), connect and test through `EmailAccountStore`,
capabilities and folders, a uniquely tagged message sent to the account's own
address and delivered through `fetch_new`, read, search, a threaded reply, the
mailbox staying read-only (`\Seen` unchanged by our reads), and the recipient
policy and send limits refusing before any SMTP call. They carry the
`live_email` marker and `@pytest.mark.network`, so they are skipped by default
and in CI.

The test harness reads the mailbox from these environment variables (the tests'
input only; product code takes accounts through AbstractCore settings, never
environment variables): `AF_TEST_EMAIL_IMAP_HOST`, `AF_TEST_EMAIL_IMAP_PORT`,
`AF_TEST_EMAIL_IMAP_SECURITY`, `AF_TEST_EMAIL_SMTP_HOST`,
`AF_TEST_EMAIL_SMTP_PORT`, `AF_TEST_EMAIL_SMTP_SECURITY`,
`AF_TEST_EMAIL_USERNAME`, `AF_TEST_EMAIL_ADDRESS`, `AF_TEST_EMAIL_PASSWORD`.
Any missing variable skips the tests with the list of missing names. Keep them in
a private file (mode 0600) and load it into the test process only:

```bash
set -a; . ~/.config/abstractframework-test/email.env; set +a; \
  python -m pytest tests/email/live -m live_email --allow-network -s --durations=0
```

`-s` shows one `LIVE-FACT` line per observation (TLS version and issuer,
capabilities such as IDLE, UIDVALIDITY, delivery time); no credential value
is ever printed: the harness keeps the values in redacting objects and scrubs
them from failure reports and captured output. Mail only goes to the test
account's own address (a guard refuses any other recipient before `MAIL FROM`),
and every message is small and tagged `[af-live-test]`: the mailbox is read-only,
so nothing can be cleaned up afterwards. Use a dedicated test mailbox.

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
