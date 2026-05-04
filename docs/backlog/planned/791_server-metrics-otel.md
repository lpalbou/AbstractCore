# 791 — Metrics endpoint + OpenTelemetry tracing

## Summary
Add first-class operational telemetry for the server: Prometheus metrics and optional OpenTelemetry tracing.

## Why
The server emits events and interaction traces, but production systems need standardized metrics and tracing for alerting and debugging. The server docs list a Prometheus metrics endpoint and OpenTelemetry tracing as future enhancements.

## Scope
### In scope
- Prometheus `/metrics` endpoint (requests, latency, tokens, errors, cache hits).
- OpenTelemetry spans for request lifecycle and provider calls.
- Trace correlation IDs in logs and responses.
- Config toggles for metrics/tracing (disabled by default).
- Docs updates with deployment examples.

### Out of scope
- Full APM dashboards or managed observability hosting.
- Vendor-specific integrations beyond standard OTel exporters.

## Dependencies
- Optional: `prometheus_client`.
- Optional: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`.

## Expected outcomes
- Standard metrics for capacity planning and alerting.
- End-to-end traces for debugging latency and failures.
- Consistent observability surface across deployments.

## Testing
- Unit tests for metric counters and labels.
- Integration tests for `/metrics` endpoint output.
- Smoke test for OTel exporter wiring (no external collector required).

## ADR note
If approved, add an ADR for observability standards (metric names, trace semantics).
