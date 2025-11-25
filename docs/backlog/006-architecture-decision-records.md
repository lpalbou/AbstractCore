# ENHANCEMENT-001: Architecture Decision Records (ADRs)

**Status**: Proposed
**Priority**: P3 - Low (Documentation)
**Effort**: Small (6-12 hours initial, ongoing)
**Type**: Documentation / Knowledge Management
**Target Version**: 2.6.0 (Minor Release)

## Executive Summary

Implement Architecture Decision Records (ADRs) to document key architectural decisions, their context, and rationale. This industry best practice captures the "why" behind design choices, helping current and future contributors understand the codebase evolution.

**Expected Benefits**:
- Preserve architectural knowledge
- Prevent re-litigating past decisions
- Faster onboarding for new contributors
- Better understanding of system evolution
- Reference for similar future decisions

---

## Problem Statement

### Missing "Why" Documentation

**Current State**: Code and documentation explain "what" and "how", but not "why"

**Example Questions Without Answers**:
- Why offline-first design?
- Why no mocking in tests?
- Why unified token terminology?
- Why provider abstraction over direct SDK usage?
- Why interaction tracing uses ring buffer?
- Why BasicDeepSearch instead of BasicDeepResearcher A/B?

**Impact**:
- Contributors may propose already-rejected approaches
- Knowledge loss when team members leave
- Repeated debates on settled topics
- Difficult to understand design trade-offs

---

## Proposed Solution

### Architecture Decision Records (ADRs)

**Format**: Markdown documents in `docs/adr/` directory

**Naming Convention**: `NNN-title-of-decision.md` where NNN is sequential number

**Template** (following Michael Nygard's ADR format):

```markdown
# ADR-NNN: Title of Decision

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Context
What is the issue we're facing that motivates this decision?

## Decision
What is the change that we're proposing and/or have agreed to make?

## Consequences
### Positive
- Benefits of this decision

### Negative
- Trade-offs and downsides

### Neutral
- Other impacts

## Alternatives Considered
What other options were evaluated?

## References
- Links to relevant discussions, PRs, issues
```

---

## Initial ADRs to Create

### ADR-001: Provider Abstraction Strategy

```markdown
# ADR-001: Provider Abstraction Strategy

## Status
Accepted (2024-Q3)

## Context
AbstractCore needs to support multiple LLM providers (OpenAI, Anthropic, Ollama, etc.) with vastly different APIs and capabilities. Users want:
- Single API that works everywhere
- Provider-specific features when needed
- Easy switching between providers
- Consistent error handling

## Decision
Implement a provider abstraction layer using:
1. `AbstractCoreInterface` - Common interface all providers implement
2. `BaseProvider` - Shared functionality (events, telemetry, retry)
3. Provider-specific implementations extending BaseProvider
4. Factory pattern (`create_llm()`) for instantiation

## Consequences
### Positive
- Unified API across all providers
- Easy to add new providers (6 providers in 18 months)
- Consistent error handling and retry logic
- Provider-agnostic application code

### Negative
- Abstraction layer adds slight complexity
- Provider-specific features require careful API design
- Lowest common denominator for some features
- Must maintain multiple provider implementations

### Neutral
- Requires documentation of provider-specific behaviors
- Trade-off between consistency and provider uniqueness

## Alternatives Considered
1. **Direct SDK Usage**: Let users use native SDKs
   - Rejected: No consistency, user must learn multiple APIs
2. **Adapter Pattern per Provider**: Separate adapters without base class
   - Rejected: Code duplication, inconsistent features
3. **Plugin Architecture**: Load providers dynamically
   - Deferred: Current approach sufficient, may revisit for v3.0

## References
- Initial discussion: Issue #1
- Base implementation: PR #45
- Provider registry: PR #234
```

### ADR-002: Offline-First Design Philosophy

```markdown
# ADR-002: Offline-First Design Philosophy

## Status
Accepted (2024-Q2)

## Context
LLM applications face two deployment scenarios:
1. Cloud: API-based models (OpenAI, Anthropic)
2. Offline: Local models (Ollama, MLX, HuggingFace)

Many use cases (research, healthcare, finance) require offline capability for:
- Data privacy
- Network restrictions
- Cost control
- Reliability

## Decision
Design AbstractCore with offline-first philosophy:
1. Local providers as primary use case
2. API providers as optional extras
3. All features work offline when possible
4. Caching and persistence by default
5. Network operations explicit and optional

Implementation:
- HuggingFace provider loads from local cache
- Models downloaded once, used forever
- `TRANSFORMERS_OFFLINE=1` by default
- No required network calls after setup

## Consequences
### Positive
- Works in air-gapped environments
- No ongoing costs for local models
- Better privacy and security
- Predictable behavior (no API changes)
- Faster inference (no network latency)

### Negative
- Larger initial downloads (model weights)
- More complex local setup
- Hardware requirements (RAM, disk)
- Slower cold start (model loading)

### Neutral
- Clear documentation needed for online vs offline

## Alternatives Considered
1. **API-First**: Assume cloud APIs, local optional
   - Rejected: Excludes major use cases
2. **Hybrid**: Equal weight to both
   - Rejected: Confusing value proposition
3. **Separate Packages**: abstractcore-local, abstractcore-cloud
   - Rejected: Splits community, more maintenance

## References
- Design document: docs/architecture.md
- HuggingFace offline: PR #123
- User feedback: Issue #67
```

### ADR-003: No Mocking Testing Policy

```markdown
# ADR-003: No Mocking Testing Policy

## Status
Accepted (2024-Q3)

## Context
Testing LLM applications presents unique challenges:
- Non-deterministic outputs
- API costs
- Network dependencies
- Complex integration flows

Traditional testing often uses mocks to:
- Avoid API costs
- Speed up tests
- Ensure determinism

However, mocks have limitations:
- Don't catch integration bugs
- Can drift from real API behavior
- Give false confidence
- Maintenance overhead

## Decision
Adopt "No Mocking" policy for AbstractCore tests:
1. Use real LLM providers (local models for cost)
2. Use real implementations for all unit tests
3. Accept non-determinism with appropriate assertions
4. Use `pytest.mark.slow` for expensive tests
5. Only mock when fundamentally impossible (e.g., testing error handling)

Implementation:
- Default test provider: Ollama (free, local)
- Use `temperature=0` for determinism where needed
- Focus on functional behavior, not exact outputs
- Mock only infrastructure (filesystem, network) when necessary

## Consequences
### Positive
- Catches real integration bugs
- High confidence in test results
- Tests validate actual behavior
- No mock-reality drift
- Better understanding of actual LLM behavior

### Negative
- Tests slower (~30s vs instant)
- Requires local Ollama installation
- Some tests non-deterministic
- Higher resource usage

### Neutral
- Clear guidelines needed
- Selective slow test execution in CI

## Alternatives Considered
1. **Mock Everything**: Standard TDD approach
   - Rejected: Misses integration issues
2. **Hybrid**: Mock API providers, real local
   - Considered: May adopt for API-only tests
3. **Fixtures**: Recorded responses
   - Rejected: Still doesn't test integration

## References
- Testing philosophy: tests/README.md
- Discussion: Issue #89
- Implementation: PR #145
```

### ADR-004: Unified Token Terminology

```markdown
# ADR-004: Unified Token Terminology

## Status
Accepted (2024-Q4)

## Context
LLM providers use inconsistent token parameter names:
- OpenAI: `max_tokens` (used to mean completion tokens)
- Anthropic: `max_tokens` (means output tokens)
- Google: `max_output_tokens`
- Ollama: `num_predict`
- HuggingFace: `max_length`, `max_new_tokens`

This creates confusion:
- Users must learn provider-specific names
- Code not portable across providers
- Documentation unclear
- Errors in token budgeting

## Decision
Implement unified token terminology:
- `max_tokens`: Total context window budget
- `max_input_tokens`: Maximum input size
- `max_output_tokens`: Maximum generation length

Rules:
1. AbstractCore uses unified names in API
2. Providers translate to native names internally
3. Response uses unified names (`input_tokens`, `output_tokens`)
4. Documentation explains mapping

## Consequences
### Positive
- Consistent API across providers
- Portable code between providers
- Clear token budgeting
- Less confusion for users
- Better error messages

### Negative
- Abstraction layer adds slight complexity
- Must document provider mappings
- Some providers don't support all parameters
- Translation overhead (minimal)

### Neutral
- Requires comprehensive documentation
- May evolve as providers change

## Alternatives Considered
1. **Provider-Specific Names**: Use each provider's terms
   - Rejected: No consistency
2. **Union of All Names**: Support all variants
   - Rejected: Too complex, confusing
3. **Most Common Name**: Use OpenAI's convention
   - Rejected: Still inconsistent with other providers

## References
- Token management: docs/generation-parameters.md
- Implementation: abstractcore/core/interface.py
- Discussion: Issue #234
```

### ADR-005: Interaction Tracing with Ring Buffer

```markdown
# ADR-005: Interaction Tracing Storage Design

## Status
Accepted (2025-Q1)

## Context
Interaction tracing needs to capture complete LLM interactions for debugging and observability. Storage options:
1. In-memory: Fast, ephemeral
2. Database: Persistent, queryable
3. File: Persistent, simple
4. Queue: Streaming, scalable

Requirements:
- Minimal performance impact
- Memory efficient
- Simple API
- No external dependencies
- Opt-in (disabled by default)

## Decision
Use ring buffer (collections.deque) for in-memory storage:
- Fixed size buffer (default 100 traces)
- Automatic eviction (oldest first)
- Thread-safe
- No external dependencies
- Optional export to file

Implementation:
```python
self._traces = deque(maxlen=100)  # Ring buffer
```

## Consequences
### Positive
- Zero performance impact when disabled
- Minimal memory footprint (controlled size)
- No external dependencies
- Simple implementation
- Thread-safe by default

### Negative
- Limited to recent traces only
- Lost on process restart
- No queryability (use export for analysis)
- Manual export required for persistence

### Neutral
- Users must export for long-term storage
- Good for debugging, not audit trails

## Alternatives Considered
1. **SQLite Database**: Persistent, queryable
   - Rejected: Adds complexity, dependencies
2. **File per Trace**: Simple persistence
   - Rejected: Filesystem overhead, management
3. **External Service**: Elasticsearch, DataDog
   - Deferred: Optional integration later
4. **Unlimited Memory**: No size limit
   - Rejected: Memory leak risk

## References
- Implementation: abstractcore/providers/base.py:76-78
- Documentation: docs/interaction-tracing.md
- Export utilities: abstractcore/utils/trace_export.py
```

---

## Implementation Plan

### Phase 1: Setup ADR Infrastructure (2 hours)

1. Create `docs/adr/` directory
2. Create ADR template (`docs/adr/000-template.md`)
3. Create ADR index (`docs/adr/README.md`)
4. Add to documentation navigation

### Phase 2: Write Initial ADRs (6-8 hours)

Priority order:
1. ADR-001: Provider Abstraction Strategy
2. ADR-002: Offline-First Design
3. ADR-003: No Mocking Testing Policy
4. ADR-004: Unified Token Terminology
5. ADR-005: Interaction Tracing Storage

### Phase 3: Integration (1-2 hours)

1. Link ADRs from relevant documentation
2. Add ADRs to CONTRIBUTING.md
3. Create ADR process documentation

### Phase 4: Future ADRs (Ongoing)

Create ADRs for future decisions:
- When to create: Significant architectural decisions
- Who creates: Decision maker or tech lead
- Review: Same as code review process
- Updates: Deprecation, superseding

**Total Estimated Time**: 9-12 hours initial, ongoing

---

## Success Criteria

1. **Coverage**: 5+ initial ADRs documenting key decisions
2. **Findability**: ADRs linked from relevant docs
3. **Process**: ADR creation process documented
4. **Adoption**: Team uses ADRs for new decisions

---

## ADR Writing Guidelines

### When to Write an ADR

Write an ADR for decisions that:
- Are architecturally significant
- Affect multiple components
- Have long-term implications
- Involve trade-offs
- May be questioned later

### When NOT to Write an ADR

Skip ADRs for:
- Implementation details
- Temporary solutions
- Obvious choices
- Local/scoped decisions

### Best Practices

1. **Be Concise**: 1-2 pages maximum
2. **Date Context**: When was this decision made?
3. **Preserve History**: Don't delete old ADRs
4. **Link Generously**: Reference issues, PRs, docs
5. **Update Status**: Mark superseded/deprecated

---

## References

- ADR concept: https://adr.github.io/
- Michael Nygard's format: https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- ADR tools: https://github.com/npryce/adr-tools
- Examples: https://github.com/joelparkerhenderson/architecture-decision-record

---

**Document Version**: 1.0
**Created**: 2025-11-25
**Author**: Expert Code Review
**Status**: Ready for Implementation
