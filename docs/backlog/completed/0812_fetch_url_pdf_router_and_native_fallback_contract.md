# Completed: fetch_url PDF router and native fallback contract

## Metadata
- Created: 2026-06-21
- Status: Completed
- Completed: 2026-06-21

## ADR status
- Governing ADRs: root `docs/adr/0029-permissive-dependency-and-licensing-policy.md`, root `docs/adr/0033-install-profiles-config-entrypoints-and-server-boundaries.md`
- ADR impact: No new ADR required. This item applies the existing dependency/profile boundary to web-tool PDF handling and adds explicit provenance for optional remote/native behavior.

## Context
`fetch_url` and `skim_url` had their own inline PDF parsing path, separate from the shared media `PDFProcessor`. That left Core with two unsynchronized PDF contracts:

- web tools used a local `pypdf` helper only;
- media/document attachments used `PDFProcessor` with a richer optional backend surface.

At the same time, users wanted better PDF handling for fetched remote documents, especially scanned or image-only PDFs, without making PyMuPDF-family packages a default dependency and without silently hiding remote/provider work behind a deterministic fetch tool.

## Problem
The previous design was brittle in three ways:

1. tool-owned PDF extraction drifted from the shared media stack;
2. scanned/image-only PDFs degraded into empty text or low-signal image placeholders;
3. there was no truthful provenance showing whether a PDF result came from `pypdf`, PyMuPDF-family extraction, or an optional native model route.

## What we wanted to do
Create one media-owned PDF routing seam for byte-oriented callers, keep permissive/local behavior as the baseline, and allow explicitly authorized native PDF model augmentation with clear fallback/provenance.

## Requirements
- Move `fetch_url`/`skim_url` PDF handling onto a shared media-owned path.
- Keep the permissive/local baseline intact.
- Support explicit backend ordering `native_llm > pymupdf > pypdf` when the corresponding route is both available and authorized.
- Do not silently imply native PDF capability from generic `document_support`.
- Expose backend provenance and attempt history in tool outputs.
- Improve scanned/image-only PDF behavior so low-signal image placeholder output does not masquerade as usable text.

## Scope
- Shared media PDF routing helper for byte-oriented callers.
- `fetch_url` and `skim_url` PDF delegation.
- `PDFProcessor` backend surface for explicit `pymupdf`.
- Focused docs and regression tests.

## Non-goals
- Do not make native PDF model routing the global media default.
- Do not make PyMuPDF-family packages default dependencies.
- Do not solve full OCR/table/layout strategy for all PDF workflows; broader follow-up remains in `planned/0806_pdf_images_tables_and_extraction_strategy.md`.

## Dependencies and related tasks
- `completed/0805_permissive_pdf_document_media_dependencies.md`
- `planned/0806_pdf_images_tables_and_extraction_strategy.md`

## Expected outcomes
- Web tools and media processing now share one PDF routing authority for fetched bytes.
- Tool outputs explicitly reveal `pdf_text_backend`, `pdf_summary_backend`, `pdf_backend_attempts`, and native transport provenance.
- Explicit native PDF routing can improve scanned/image-only fetches while staying opt-in.

## Validation
- Focused pytest for router behavior, PDF processor backends, fetch/skim PDF handling, and packaging boundary checks.
- Real end-to-end `fetch_url` runs against local PDF URLs using:
  - local-only extraction;
  - a non-native OpenAI-compatible endpoint (`localhost:8090/v1`);
  - official OpenAI native PDF file input.

## Progress checklist
- [x] Add a shared media-owned PDF routing helper for bytes.
- [x] Make `fetch_url` and `skim_url` delegate to that helper.
- [x] Expose explicit PDF backend provenance in `fetch_url`.
- [x] Make explicit `pymupdf` a first-class `PDFProcessor` backend.
- [x] Prove native success, native-unavailable fallback, and scanned-PDF improvement with live tests.

## Completion report

Completed on 2026-06-21.

### What changed
- Added `abstractcore/media/pdf_routing.py` as the shared PDF routing seam for byte-oriented callers.
- `fetch_url` now delegates PDF handling to the router and returns:
  - `pdf_text_backend`
  - `pdf_summary_backend`
  - `pdf_backend_attempts`
  - `pdf_native_available`
  - `pdf_native_used`
  - `pdf_native_model`
  - `pdf_native_transport`
  - `pdf_degraded`
- `skim_url` now uses the same router for PDF preview/refetch behavior.
- `PDFProcessor` now supports explicit `pymupdf` in addition to `pypdf` and `pymupdf4llm`.
- The router treats image-only/placeholder PyMuPDF4LLM markdown as low-signal and falls back to native preview text when available.
- Native PDF routing is automatic whenever a normal OpenAI-compatible client is configured (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`), with optional model override through `ABSTRACTCORE_FETCH_URL_PDF_NATIVE_MODEL`. There is no separate enable flag. Native calls try `file_id` first, then fall back to `data_url` for endpoints that reject uploaded PDF files but still support inline PDF inputs.

### Validation
- `python -m py_compile abstractcore/media/pdf_routing.py abstractcore/media/processors/pdf_processor.py abstractcore/tools/common_tools.py`
- `python -m pytest -q tests/media_handling/test_pdf_routing.py tests/media_handling/test_media_processors.py tests/tools/test_common_tools_fetch_url_asset_parsing.py tests/tools/test_common_tools_skim_url_and_websearch.py tests/test_packaging_extras.py -k 'pdf'`
  - Result: `13 passed, 27 deselected`
- `python -m pytest -q tests/tools/test_common_tools_fetch_url_integration_html_normalization.py tests/tools/test_common_tools_fetch_url_e2e_real_html_text.py`
  - Result: `1 passed, 1 skipped` (`ABSTRACT_E2E_FETCH_URL=1` not enabled)
- Live proof artifacts:
  - `untracked/tests-pdf-fetch/current/manifest.json`
  - `untracked/tests-pdf-fetch/current/local_only_text.json`
  - `untracked/tests-pdf-fetch/current/local_only_scan.json`
  - `untracked/tests-pdf-fetch/current/native_8090_text.json`
  - `untracked/tests-pdf-fetch/current/native_openai_text.json`
  - `untracked/tests-pdf-fetch/current/native_openai_scan.json`

### Residual risks
- Native PDF routing is still tool-level opt-in, not a general provider/media capability contract for all `media=[...]` document flows.
- Official native models can still make OCR-style mistakes on poor scans; the router improves evidence quality, but it does not make scanned-PDF extraction deterministic.
- Broader OCR/table/layout strategy remains open in `planned/0806_pdf_images_tables_and_extraction_strategy.md`.
