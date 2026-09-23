# Native inference follow-ups

Planned, audited 2026-09-20. These items extend the existing providers; they do
not introduce a second MLX provider or require oMLX. The tested two-model native
MLX implementation is described in [native runtime](../../../native-mlx-runtime.md).

Implementation order: shared negotiation (0849), then independently the Flash
sidecar (0848), Transformers (0850), GGUF binding (0851), and discovery/UI (0853).
Pressure eviction (0852) is independent and disabled by default. Runtime's
immediate transport/scheduling repairs are tracked by abstractruntime-0846.

- [0848: Flash separate head](0848_flash_external_mtp_head.md)
- [0849: Shared speculation negotiation](0849_shared_speculation_backend_contract.md)
- [0850: Transformers MTP adapter](0850_transformers_native_mtp_adapter.md)
- [0851: In-process GGUF MTP adapter](0851_gguf_native_mtp_adapter.md)
- [0852: Optional pressure eviction](0852_opt_in_model_pressure_eviction.md)
- [0853: Capability discovery and application controls](0853_execution_capabilities_and_controls.md)

Read existing model/cache ownership work 0847 and ADR 0008 before implementation.
No automatic downloads, external inference services, TTL enforcement, or broad
UI redesign are authorized by these records alone. Backend completion requires
real execution evidence, not model-name or metadata inference.
