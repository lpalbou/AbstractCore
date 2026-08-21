# Planned: clarify and break the `architectures ↔ utils ↔ media` import cycle

## Metadata
- Created: 2026-08-21
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: None known covering import-graph policy
- ADR impact: Needs new ADR — the project's stated intent is "no circular imports"; if that is a
  durable rule it belongs in an ADR with an enforcement mechanism, not only in prose and habit.

## Context

Found incidentally on 2026-08-21 while scripting `analyze_media` verification from the framework
root. Importing a core submodule without the package root having been initialized first raises:

```
ImportError: cannot import name 'get_model_capabilities' from partially initialized module
'abstractcore.architectures' (most likely due to a circular import)
```

The cycle is real and it is in the module graph today. Traced by execution, not by reading:

```
tools/__init__.py:77        from .handler import (...)
tools/handler.py:12         from ..architectures import detect_architecture, get_model_capabilities, ...
architectures/__init__.py:8 from .detection import (...)
architectures/detection.py:13 from ..utils.structured_logging import get_logger
utils/__init__.py:5         from .file_filters import (...)
utils/file_filters.py:12    from abstractcore.media.types import MediaType, detect_media_type, ...
media/__init__.py:27        from .capabilities import (...)
media/capabilities.py:14    from ..architectures import get_model_capabilities   ← closes the cycle
```

Four packages participate: `tools → architectures → utils → media → architectures`.

## Current code reality

- The cycle exists but is normally **latent**, because `abstractcore/__init__.py:30` begins with
  `from .utils.version import __version__`, so the package root establishes an import order in
  which the participants are already initialized by the time anything closes the loop.
- Reproduces reliably when `abstractcore/__init__.py` is **skipped**. The trigger observed is
  namespace-package shadowing: run Python from `/Users/albou/tmp/abstractframework`, where the
  sibling directory named `abstractcore` (the repo root, which has no `__init__.py`) makes
  `abstractcore` resolve as a *namespace* package. Verified:
  - `cd /tmp && python3 -c "import abstractcore.tools.common_tools"` → OK
  - `cd .../abstractframework/abstractcore && python3 -c "import abstractcore.tools.common_tools"` → OK
  - `cd .../abstractframework && python3 -c "import abstractcore.tools.common_tools"` → **ImportError**
  - in that failing case, `abstractcore.__file__ is None` and
    `abstractcore.__path__ == ['<repo>/abstractcore', '<repo>/abstractcore']` (namespace package,
    contributed twice — cwd plus the editable finder `__editable__.abstractcore-2.13.39.pth`).
- Invisible to CI: `tests/conftest.py` imports `from abstractcore import create_llm` at collection
  time, so every test process has the package root initialized before any submodule import. No
  existing test asserts anything about the import graph.

## Problem

Two distinct concerns, and the item should separate them:

1. **The cycle itself.** A dependency loop across four subpackages is present in the graph and is
   masked only by one import statement's position in `__init__.py`. Any reorder of that file, a
   lazy import added inside a function, a plugin/entry-point loader, `python -m` against a
   submodule, or a consumer whose cwd shadows the package name will surface it. The project set
   out to avoid circular imports; this one is load-bearing on an accident.
2. **The shadowing.** A repo checkout whose top directory shares the distribution name silently
   turns the package into a namespace package and skips `__init__.py`. That is a foot-gun for
   anyone (human or agent) scripting from the framework root, and it changes import semantics
   without any diagnostic.

## What we want to do

Establish whether the cycle is intended-and-managed or unnoticed drift; then either break it or
document and enforce the invariant that keeps it safe.

## Why

`get_model_capabilities` is consumed by `media/capabilities.py` to answer vision questions, and
`utils/file_filters.py` reaches into `media.types` for extension tables — so the loop sits on the
capability path that delegated sight (`analyze_media`), media routing, and provider selection all
depend on. A latent cycle on that path fails at import time, in a consumer's process, with a
message that names neither the real culprit nor the fix.

## Requirements

- The cycle is either removed, or documented as accepted with a test that fails if the masking
  import order changes.
- `import abstractcore.<any submodule>` succeeds without the package root having been imported
  first, from any working directory.
- The failure mode, if any remains, is a named error explaining the constraint — never a bare
  partially-initialized `ImportError`.

## Suggested implementation

Options, cheapest first — the investigation should pick one, not assume:

1. **Break the `media → architectures` edge.** `media/capabilities.py:14` imports
   `get_model_capabilities` at module scope; a function-local import, or moving the capability
   lookup behind a small accessor module that neither side owns, closes the loop at its narrowest
   point.
2. **Break the `utils → media` edge.** `utils/file_filters.py:12` imports `media.types` only for
   `MediaType`/`detect_media_type`/`get_supported_extensions_by_type`. Extension tables are data;
   a leaf module owned by neither `utils` nor `media` would remove the dependency entirely.
3. **Break the `architectures → utils` edge.** `architectures/detection.py:13` needs only
   `get_logger`; logging is the most-imported leaf in the codebase and arguably should not pull
   `utils/__init__` (which transitively drags in `file_filters` and media).

Option 3 plus option 2 likely removes the loop without touching the capability API at all.

## Scope

- The four edges listed above, and whichever one is chosen to cut.
- A regression test for the import graph.

## Non-goals

- Reorganizing the package layout.
- Changing the capability API or vision detection behavior.
- Fixing the namespace-shadowing ergonomics of the monorepo checkout (worth its own item if the
  investigation finds it affects more than this cycle).

## Dependencies and related tasks

- `0837_delegated_sight_session_route_and_first_run_seeding.md` — same capability path.
- `0834_analyze_media_enforce_abstractignore.md` — same tool.
- Found during the 2026-08-21 cross-package tool-failure investigation
  (`abstractcode-tui/docs/reports/2026-08-21-tool-failure-taxonomy-and-fixes.md`), which fixed
  `analyze_media` attachment addressing, its workspace walling, and an undelivered-image guard.
  That work did not touch the import graph.

## Expected outcomes

- A clear answer to "is this cycle intended?", recorded in the ADR.
- Either no cycle, or a cycle that cannot silently start failing.

## Validation

- A test that imports each top-level subpackage **in a subprocess, in isolation**, with no prior
  `import abstractcore`: `python -c "import abstractcore.tools.common_tools"` and the same for
  `architectures`, `media`, `utils`, `providers`, `processing`. Must pass from a neutral cwd.
- A test that reproduces the namespace-shadow case (cwd containing a directory named
  `abstractcore`) and asserts the import either succeeds or fails with a named, explanatory error.
- Existing suites unchanged: `tests/tools`, `tests/media`-adjacent capability tests.
- Deliberately NOT satisfied by running pytest alone — `conftest.py` masks the defect.

## Progress checklist

- [ ] Reproduce from a clean checkout and confirm the trigger list above
- [ ] Decide: break the cycle, or accept + enforce
- [ ] Land the chosen edge cut
- [ ] Add the isolated-import regression test
- [ ] Write the ADR recording the import-graph rule

## Guidance for the implementing agent

Do not "fix" this by adding another import to `abstractcore/__init__.py` to force the order. That
deepens the dependence on a masking side effect and makes the next reorder a mystery. The cycle
either goes away or becomes an explicit, tested invariant.

Reproduction is order-sensitive and cwd-sensitive; always verify in a **fresh subprocess** and
state the cwd in any result you report.
