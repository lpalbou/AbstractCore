# ADR 0004: Operator control and server trust boundary

Status: Accepted.

## Context

AbstractCore can run as a local library, an OpenAI-compatible HTTP gateway, and a single-model
endpoint. In server mode it can hold provider credentials, accept request-level upstream routing
overrides, fetch remote URLs, process local files when enabled, and keep local runtimes warm.

Those features are useful only if the operator remains in control. The current implementation
already encodes important protections: server auth is separate from upstream provider keys,
provider keys are not accepted in request bodies, non-loopback `base_url` overrides require an
allowlist, and unauthenticated clients cannot implicitly use server-held provider credentials.

## Decision

AbstractCore will keep server-mode behavior operator-controlled and deny-by-default where a route
could expose credentials, private network access, local files, or expensive runtime state.

The accepted rules are:

- Environment variables override centralized config, and centralized config overrides defaults.
- Server-held provider credentials require inbound AbstractCore server auth before clients can use
  them implicitly.
- Per-request upstream provider keys use explicit headers, primarily
  `X-AbstractCore-Provider-API-Key`, not request bodies.
- Request-level `base_url` overrides allow loopback by default and require allowlists for
  non-loopback targets.
- Remote URL fetch and local file access stay constrained by explicit operator settings.
- New server control-plane routes must reuse the same trust model unless a later ADR changes it.

## Consequences

### Positive

- Operators can reason about credential use and upstream routing.
- Accidental credential forwarding and unsafe request routing are harder.
- Local runtime management remains a trusted control-plane concern.

### Negative

- Advanced routing setups require explicit configuration.
- Some discovery or media routes need extra code to preserve the same checks.

### Neutral

- This ADR does not require every deployment to expose the HTTP server publicly.
- This ADR does not ban custom upstream routing; it requires operator opt-in.

## Enforcement

- Reviews should reject provider secrets in HTTP request bodies.
- New routes that accept `base_url`, file paths, remote URLs, or provider keys must use the shared
  server hardening rules.
- Docs for new server behavior must describe auth, provider-key, and allowlist requirements.
- Server-held credentials must not become available to unauthenticated clients by convenience
  fallback.

## Validation

- Server hardening tests should cover provider key handling, allowlist behavior, and unauthenticated
  server-held key refusal.
- Config tests should cover environment/config/default precedence for new settings.
- Route tests should include at least one forbidden path when adding new control-plane access.

## Related

- `abstractcore/server/app.py`
- `abstractcore/server/audio_endpoints.py`
- `abstractcore/server/vision_endpoints.py`
- `abstractcore/config/manager.py`
- `docs/server.md`
- `docs/centralized-config.md`
- `docs/backlog/planned/789_server-auth-rate-limits.md`
