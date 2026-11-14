# 1213: Bound Hugging Face discovery requests and rate-limit retries

> Package: AbstractCore, AbstractRuntime, AbstractGateway

## Metadata
- Created: 2026-11-14
- Status: Planned — in progress
- Completed: N/A
- Source: operator release recovery, root backlog 1209

## ADR status
- Governing ADRs: framework ADR 0050/0051 release gates; existing package contracts.
- ADR impact: implementation correction within existing contracts.

## Current code reality

Core catalog currently hydrates every matching curated repo and every search hit with model_info under one lock. Offline counts: 42 calls per Apple page, 87 for qwen with no remote hits; repeated 429s have no shared cooldown.

## Scope and expected outcome

Separate bounded summary listings from explicit single-repo detail; retain summary metadata and unknowns; cursor pages never refetch prefixes. Discovery-owned finite transport, no SDK pagination retry or user-agent registry fetch, scoped caching/cooldown respecting Retry-After and RateLimit. Runtime/Gateway forward detail_repo; update floors and console.

Deferred governance, spoken-language, Core reasoning/usage, composer redesign and breaking TUI migration remain outside this item.

## Validation

42 offline HTTP-count/catalog tests; asymmetric three-page lane fixture; privacy/token policy; concurrency; cooldown; facade and UI race regressions. Full gates and scratch HTTP proof pending.

## Release acceptance

Code, tests and coredoc reach remote main; all exact-commit gates green; changed artifacts published in dependency order. Root/site publication remains held for operator GO.
