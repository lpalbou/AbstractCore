# Completed: Permissive PDF document media dependencies

## Metadata
- Created: 2026-06-05
- Status: Completed
- Completed: 2026-06-05

## ADR status
- Governing ADRs: root `docs/adr/0029-permissive-dependency-and-licensing-policy.md`, root `docs/adr/0033-install-profiles-config-entrypoints-and-server-boundaries.md`
- ADR impact: Known drift tracked by this item. No new ADR is required unless maintainers intentionally approve PyMuPDF-family packages as default install dependencies.

## Context
AbstractFlow and AbstractRuntime now have first-class `read_pdf` and `write_pdf` VisualFlow nodes backed by permissive dependencies (`pypdf` and `reportlab`). That fixes the workflow-authoring path, but AbstractCore still exposes PyMuPDF-family packages through its media and aggregate extras.

PyPI metadata currently lists:
- `PyMuPDF` and `pymupdf4llm`: dual AGPL-3.0/commercial.
- `pymupdf-layout`: Polyform Noncommercial/commercial.
- `pypdf`: BSD-3-Clause.
- `reportlab`: BSD.

The remaining Core exposure is real because Core owns the older PDF media-processing stack used by general media attachments and server/media install profiles.

## Current code reality
- `abstractcore/pyproject.toml` includes `pymupdf4llm` and `pymupdf-layout` in `media`, `all-apple`, `all-gpu`, legacy `all-non-mlx`, and broad aggregate profiles.
- `abstractcore/abstractcore/media/processors/pdf_processor.py` imports PyMuPDF-family modules for PDF text/Markdown extraction.
- Core docs and changelog historically describe PyMuPDF4LLM as the PDF extraction backend.
- Runtime's new VisualFlow PDF nodes are separate and permissive, but Runtime hardware extras still depend on Core aggregate extras that currently include the PyMuPDF-family packages.

## Problem
The default Core media/document path is not aligned with the framework's permissive dependency preference. Users installing broad or hardware-oriented profiles may receive AGPL/commercial or noncommercial/commercial PDF dependencies without making an explicit licensing decision.

## What we want to do
Move Core's default PDF/document handling to permissive dependencies and make high-fidelity PyMuPDF-family processing an explicit, license-reviewed opt-in.

## Why
This keeps AbstractFramework easier to adopt, redistribute, and deploy in commercial or mixed environments while preserving a path for users who knowingly choose commercial/high-fidelity PDF processing.

## Requirements
- Remove `pymupdf4llm` and `pymupdf-layout` from Core's default, `media`, server, `all-apple`, `all-gpu`, legacy aggregate, and broad install profiles unless a profile name explicitly communicates the licensing choice.
- Provide a permissive PDF reader path for Core media attachments, likely using `pypdf` for text extraction and metadata.
- Preserve clear limitations for scanned PDFs, complex layout, tables, OCR, and high-fidelity Markdown extraction.
- Add an explicit optional extra for PyMuPDF-family support only if maintainers want to keep that backend, for example `pdf-pymupdf-commercial` or `documents-pymupdf-commercial`.
- Fail closed when the optional PyMuPDF backend is requested but unavailable or unapproved; do not silently fall back across licensing boundaries.
- Add packaging tests that assert PyMuPDF-family packages are absent from permissive/default profiles and present only in the explicit opt-in profile.
- Update Core docs, media docs, install docs, changelog, and AI-readable docs.

## Suggested implementation
1. Introduce a small backend abstraction for PDF extraction in Core media processing.
2. Implement a permissive default backend with `pypdf`.
3. Keep the existing PyMuPDF-backed behavior behind an explicit optional backend and extra, or remove it entirely if maintainers do not want to carry the license boundary.
4. Update `pyproject.toml` extras and profile tests.
5. Update server/media docs and examples to describe the default quality envelope and opt-in backend.

## Scope
- AbstractCore dependency extras.
- AbstractCore media PDF processor.
- Core packaging and import-safety tests.
- Core media/server documentation and generated AI docs.

## Non-goals
- Do not change AbstractFlow's Runtime-owned `read_pdf` / `write_pdf` nodes; those are already permissive.
- Do not add OCR or pixel-perfect layout extraction as a default dependency.
- Do not silently route to PyMuPDF when the permissive backend is insufficient.
- Do not remove user ability to opt into a commercial/high-fidelity backend when explicitly configured.

## Dependencies and related tasks
- Runtime follow-up: `abstractruntime/docs/backlog/completed/0041_runtime_hardware_extras_avoid_nonpermissive_document_stacks.md`
- AbstractFlow completed PDF node work: `abstractflow/docs/backlog/completed/0101_permissive_pdf_document_nodes.md`
- Root ADR-0029 permissive dependency policy.
- Root ADR-0033 install-profile boundaries.

## Expected outcomes
- Core default/media/server install paths no longer install PyMuPDF-family dependencies.
- PyMuPDF-family dependencies, if retained, are isolated behind an explicit opt-in extra and documented license warning.
- Core media attachment PDF reading remains available with a permissive baseline.
- Packaging tests prevent accidental reintroduction into default or aggregate profiles.

## Validation
- `python -m pytest` for Core packaging/import-safety tests.
- Focused PDF media processor tests using generated fixture PDFs.
- Metadata inspection proving default/media/server/all hardware profiles do not include `PyMuPDF`, `pymupdf4llm`, or `pymupdf-layout`.
- Docs check confirming install examples no longer promote nonpermissive PDF dependencies by default.

## Progress checklist
- [x] Audit every Core extra/profile containing PyMuPDF-family packages.
- [x] Implement permissive Core PDF extraction backend.
- [x] Move or remove PyMuPDF-family dependencies.
- [x] Add packaging and media processor regression tests.
- [x] Update docs, changelog, and AI-readable context.

## Guidance for the implementing agent
Re-check current dependency metadata before editing. Treat this as a licensing boundary, not just a dependency cleanup. Any opt-in PyMuPDF path must be explicit in package extras, docs, and tests.

## Completion report

Completed on 2026-06-05.

### What changed
- `PDFProcessor` now defaults to a `pypdf` backend for text and metadata extraction.
- PyMuPDF-family imports are lazy and only used when `pdf_backend="pymupdf4llm"` is requested.
- `AutoMediaHandler` checks `pypdf` for default PDF availability and no longer pretends missing PDF dependencies can be handled as plain text.
- `abstractcore[media]`, `all`, `all-apple`, `all-gpu`, `all-non-mlx`, `full-dev`, and `test` now use `pypdf` instead of PyMuPDF-family packages.
- `pymupdf4llm` and `pymupdf-layout` remain only under the explicit `pdf-pymupdf-commercial` extra.
- Docs, changelog, acknowledgements, examples, and `llms-full.txt` now describe the permissive default and the explicit commercial-license opt-in.

### Validation
- `python -m py_compile abstractcore/media/processors/pdf_processor.py abstractcore/media/auto_handler.py`
- `python -m pytest tests/test_packaging_extras.py tests/test_import_safety.py tests/media_handling/test_pdf_processor_pdf_version_compat.py tests/media_handling/test_media_processors.py -q`
  - Result: 25 passed, 4 skipped (`TEST_WITH_OFFICE_DOCS` not enabled).
- `rg -n "pymupdf4llm|pymupdf-layout" abstractcore/pyproject.toml abstractcore/abstractcore/media/processors/pdf_processor.py abstractcore/abstractcore/media/auto_handler.py`
  - Result: package dependencies appear only in `pdf-pymupdf-commercial`; remaining code references are lazy optional backend paths.

### Residual risks
- `glyph_pdf_processor.py` remains a PyMuPDF-dependent optional/legacy path and is documented as requiring the explicit opt-in extra.
- Historical changelog/archive/research reports still mention the old PyMuPDF implementation because they describe past releases and investigations.
- Runtime follow-up is complete in `abstractruntime/docs/backlog/completed/0041_runtime_hardware_extras_avoid_nonpermissive_document_stacks.md`.
