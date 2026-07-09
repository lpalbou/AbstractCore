# Proposed: Optional Trafilatura HTML extractor for web tools

## Metadata
- Created: 2026-06-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
`skim_url(...)` and `fetch_url(...)` now handle mislabelled feeds, RSS/Atom summaries, PDF text extraction through `pypdf`, and downstream `BasicDeepSearch` consumption of structured `web_search(...)` / `fetch_url(...)` outputs. That closes the immediate correctness gap for web-search agents without adding a new dependency.

Official Trafilatura documentation still makes it a credible next-step candidate for harder HTML/article extraction work: it is purpose-built for main-text extraction, metadata, comments, and feed/sitemap-aware crawling, and documents benchmark-oriented extraction tradeoffs (`fast`, `favor_precision`, `favor_recall`).

## Current code reality
- Inspected `abstractcore/tools/common_tools.py`.
  Current HTML extraction is custom `BeautifulSoup` + heuristic container scoring + Markdown rendering.
- Inspected `abstractcore/processing/basic_deepsearch.py`.
  Current search agent flow now consumes structured `web_search(...)` results and `fetch_url(...)` dict outputs.
- Inspected `tests/tools/test_common_tools_*` and `tests/tools/test_basic_deepsearch_tool_contracts.py`.
  Coverage now exists for feed/PDF sniffing and deep-search consumer compatibility, but not for article-extraction quality on a broad live corpus.
- `trafilatura` is not installed in the current environment.

## Problem or opportunity
The current in-house HTML extractor is good enough for many pages, but it still:
- reparses content in multiple places;
- depends on custom heuristics for article-body selection;
- has no reproducible benchmark that compares extraction quality or token efficiency against a stronger article extractor.

That leaves a credible optimization opportunity, but not yet a promotion-ready implementation mandate.

## Proposed direction
Evaluate an optional `trafilatura` integration for the HTML path used by `skim_url(...)` and `fetch_url(...)`, keeping the current extractor as the default fallback.

The likely shape is:
- new optional extra rather than a hard default dependency;
- one extractor boundary that can return shared structured fields (`title`, `description`, `text`, `links`, diagnostics);
- benchmark harness comparing current extraction against `trafilatura` on representative article, docs, blog, and noisy marketing pages;
- no change to XML/feed or PDF parsing ownership in this item.

## Why it might matter
- Better main-text extraction can reduce boilerplate and lower token waste for agent prompts.
- Higher-fidelity article extraction can improve answer quality for web-search agents without forcing every caller to maintain its own parser stack.
- Making it optional preserves the lightweight default install and avoids dependency/maintenance cost until evidence justifies it.

## Promotion criteria
- A reproducible benchmark on a representative URL set shows a clear win over the current extractor in at least one of:
  - lower prompt/output size for equivalent information;
  - less boilerplate/noise in extracted text;
  - better downstream answer quality in a fixed QA harness.
- No regression on the current `skim_url(...)` / `fetch_url(...)` HTML tests.
- Optional-dependency packaging, import cost, and maintenance burden are acceptable.

## Validation ideas
- Build a small live URL corpus covering documentation, blog posts, news/articles, and noisy marketing pages.
- Compare current extractor vs `trafilatura.extract(...)` using:
  - extracted-character count;
  - boilerplate ratio heuristics;
  - fixed-answer evaluation with a stable hosted model profile.
- Verify fallback behavior when `trafilatura` is not installed.

## Non-goals
- Replacing the current XML/RSS/Atom parser.
- Replacing the current PDF extraction path.
- Making `trafilatura` a mandatory dependency without benchmark evidence.

## Guidance for future agents
Re-check the current `common_tools.py` implementation and tests first. If the current extractor has already been refactored into a parse-once structured content model, evaluate `trafilatura` against that new boundary instead of bolting it onto multiple call sites.
