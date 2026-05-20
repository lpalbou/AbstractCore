# Technical Reports (Non-Authoritative)

Files in `docs/reports/` are point-in-time engineering notes, investigations, and release write-ups.

They are **not guaranteed to match the current codebase**. Use the main docs index (`docs/README.md`) for
the canonical, up-to-date documentation.

## How to read these

- Treat each report as a snapshot taken on the date in its filename.
- Performance numbers (latency, compression ratios, etc.) are highly environment-dependent and should be
  considered **illustrative**, not guarantees.
- When a report references code paths or APIs, verify against the current implementation before relying on it.

## Recent reports

- `2026-05-20-durable-memory-bloc-cache-validation.md` — MLX, HuggingFace transformers, and
  HuggingFace GGUF durable memory-bloc cache proof with processing-phase timing split.
