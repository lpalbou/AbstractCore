# 789 — Server access control + tenant rate limits

## Summary
Add first-class inbound authentication for the AbstractCore server with per-tenant API keys, quotas, and rate limiting.

## Why
The server currently accepts unauthenticated requests and relies on external proxies for protection. That is risky for production: it enables abuse, uncontrolled spend, and unpredictable load. The server docs already list rate limiting and multi-tenancy as future enhancements, and these are foundational for any public or shared deployment.

## Scope
### In scope
- Auth middleware for inbound requests (static API keys in config/env).
- Tenant registry with per-tenant limits (requests/minute, tokens/day, optional cost budget).
- Provider/model allowlists per tenant.
- Usage accounting per tenant (requests, tokens, cost estimate).
- Clear warnings when auth/limits are disabled (#FALLBACK).
- Admin endpoints for managing keys/limits (guarded by admin key).
- Docs updates (`docs/server.md`, `docs/api.md`, config schema).

### Out of scope
- OAuth/OIDC and external IdP integration.
- Billing, invoicing, or payment processing.
- Full RBAC or user management UI.

## Dependencies
- Optional Redis for shared rate limits across workers.
- Optional rate-limit helper (custom token-bucket or `fastapi-limiter`).

## Design details (draft)
### Config shape (centralized config)
Add `server` to the centralized config (persisted via `abstractcore --config`):
- `server.auth.enabled` (bool)
- `server.auth.admin_key_hash` + `admin_key_salt` (hash only, no raw keys stored)
- `server.auth.allowed_headers` (default: `Authorization`, `X-API-Key`)
- `server.auth.exempt_paths` (default: `/healthz`, `/docs`, `/openapi.json`, `/redoc`)
- `server.tenants[]`:
  - `tenant_id`, `label`, `enabled`
  - `key_hashes[]` (per-tenant keys, hashed + salt + key_prefix)
  - `allow_providers[]` / `allow_models[]` (optional)
  - `limits` (see below)
- `server.limits.defaults`:
  - `requests_per_minute`
  - `tokens_per_minute`
  - `tokens_per_day`
  - `burst_multiplier`
  - `estimate_tokens` (bool; default true)

### Auth flow
1. Middleware extracts API key from `Authorization: Bearer <key>` or `X-API-Key`.
2. Validate against hashed keys (constant-time compare).
3. Attach `tenant_id` to `request.state`.
4. If auth disabled, log `#FALLBACK : server auth disabled` once at startup.

### Rate limiting flow
Two-phase enforcement:
1. **Preflight (middleware)**: request-per-minute (RPM) bucket per tenant.
2. **Post-parse (endpoint dependency)**:
   - Validate model/provider allowlists.
   - Estimate tokens: `TokenUtils.estimate_tokens(messages) + max_output_tokens`.
   - Enforce tokens-per-minute (TPM) and tokens-per-day (TPD).
3. **Post-response**:
   - Reconcile actual usage with estimate.
   - If usage missing, log `#FALLBACK : token usage unavailable`.

### Storage & scale
- Default: in-memory token buckets (single-process).
- Optional: Redis backend for multi-worker deployments.
- If multiple workers and Redis not configured, log `#FALLBACK : per-process rate limits`.

### Headers and responses
- 401 if missing/invalid key.
- 403 if key disabled or tenant disabled.
- 429 for limits; include standard rate-limit headers:
  - `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

### CLI + Wizard
- `abstractcore --config` adds a **Server Security** step:
  - enable auth, generate admin key, create first tenant + key
  - set defaults for RPM/TPM/TPD
- Additional CLI helpers:
  - `--server-add-tenant`, `--server-add-key`, `--server-remove-key`, `--server-list-keys`

### ADR
Add ADR defining server auth boundary and fallback rules.

## Expected outcomes
- Safe public exposure of the server with tenant API keys.
- Predictable spend and abuse prevention.
- Clear multi-tenant boundary for future product work.

## Testing
- Unit tests for auth header parsing, key validation, and quota checks.
- Integration tests with FastAPI TestClient for 401/429 paths.
- Load test for rate limiter correctness under concurrency.

## ADR note
If approved, add an ADR clarifying the server security boundary (first-party auth vs reverse-proxy-only).
