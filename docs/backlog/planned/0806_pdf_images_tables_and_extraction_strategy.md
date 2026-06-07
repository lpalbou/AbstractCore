# Planned: PDF images, tables, and extraction strategy

## Metadata
- Created: 2026-06-06
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: root `docs/adr/0029-permissive-dependency-and-licensing-policy.md`, root `docs/adr/0033-install-profiles-config-entrypoints-and-server-boundaries.md`
- ADR impact: None yet. This item must not reintroduce PyMuPDF-family packages into default install profiles unless the ADR policy is explicitly revised.

## Context
Core PDF media handling now defaults to the permissive `pypdf` route. Runtime PDF writing uses ReportLab. That satisfies the default read/write requirement for text-first PDFs, but it does not solve richer document understanding:

- embedded images inside PDFs;
- scanned PDFs that need page rendering and OCR;
- table structure reconstruction;
- high-fidelity Markdown layout.

The current code must stay honest about those limits. The default pypdf processor extracts text and metadata, reports table/image limitations as warnings, and does not claim vision/image extraction support.

## Problem
Users will ask for PDF reading that includes images and tables. The old PyMuPDF-family route could cover some of that, but it carries licensing concerns and must remain explicit opt-in. We need a permissive default strategy for richer extraction, or a clear documented reason that the default route is intentionally text-only.

## What we want to do
Design and implement richer PDF extraction without compromising the permissive default install policy.

## Requirements
- Keep `pypdf` as the default text/metadata route.
- Keep PyMuPDF-family packages only in the explicit `pdf-pymupdf-commercial` extra.
- Evaluate permissive alternatives for page rasterization and image extraction, such as `pypdfium2`, before adding a dependency.
- Evaluate table extraction options with MIT/Apache/BSD-compatible licensing; do not add GPL/AGPL/noncommercial dependencies to default profiles.
- Surface capability truth separately for:
  - text extraction;
  - embedded image extraction;
  - page rendering for OCR/VLM use;
  - table structure extraction.
- Add tests that prove capability flags match the installed backend.
- Update media docs and AI-readable docs with the default quality envelope and optional richer routes.

## Suggested implementation
1. Add a small `PDFExtractionCapabilities` helper or equivalent metadata model so route consumers can distinguish text, embedded images, page images, tables, and OCR.
2. Implement pypdf embedded-image extraction only if it is reliable enough and can be tested without heavyweight dependencies.
3. If page rendering is needed, prefer a permissive optional dependency such as `pypdfium2` after license and platform checks.
4. Treat table extraction as separate from text extraction. Basic Markdown tables from pypdf text are not a real structure guarantee.
5. Keep scanned-PDF OCR as a separate optional capability, because it depends on OCR engines and image preprocessing.

## Scope
- Core PDF processor capability reporting.
- Optional PDF image/table extraction backend(s).
- Packaging tests for dependency policy.
- Media docs and AI-readable docs.

## Non-goals
- Do not make PyMuPDF-family packages default dependencies.
- Do not claim scanned PDF support without OCR/page-rendering tests.
- Do not claim table preservation from plain text extraction.
- Do not change Runtime's existing `read_pdf` / `write_pdf` nodes unless they need to consume a new Core capability surface later.

## Dependencies and related tasks
- Completed Core item: `completed/0805_permissive_pdf_document_media_dependencies.md`
- Completed Runtime item: `../abstractruntime/docs/backlog/completed/0041_runtime_hardware_extras_avoid_nonpermissive_document_stacks.md`
- Completed Gateway item: `../abstractgateway/docs/backlog/completed/0056_gateway_pdf_runtime_floor_and_e2e_contract.md`

## Expected outcomes
- Default PDF handling remains permissive and text-first.
- Richer PDF extraction is either permissive and tested, or clearly documented as requiring explicit optional dependencies.
- Thin clients can stop guessing whether "PDF support" means text, images, tables, OCR, or layout.

## Validation
- Core packaging tests proving default and aggregate profiles remain free of PyMuPDF-family dependencies.
- Core media processor tests for text, metadata, and any added image/table route.
- Capability-contract tests showing unsupported image/table/OCR features are not advertised.

## Progress checklist
- [ ] Evaluate permissive image/page-rendering libraries.
- [ ] Decide embedded-image vs page-rendering API shape.
- [ ] Evaluate table extraction libraries and license compatibility.
- [ ] Implement selected route(s) with truthful capability metadata.
- [ ] Update docs and AI-readable context.
