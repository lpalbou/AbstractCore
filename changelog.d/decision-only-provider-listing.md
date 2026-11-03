### Fixed

- **Hosted decision models are listed as decision models, and only there** (`providers/typesafe_provider.py`,
  `server/app.py`). `GET /v1/models` showed `typesafe/jev-latest` (and any `systemone` model) as a text model
  (`capabilities.output: ["text"]`, no `decision` block), and listed it under every capability filter, so text,
  image and embedding model pickers offered it and a chat on it then failed with `decision_model`. The `typesafe` and
  `systemone` providers now list their models only for a filter a decision model satisfies (`input.text`,
  `output.decision`), and `/v1/models` gives their entries `capabilities: {input: [text], output: [decision]}` and
  `decision: {endpoint: /v1/decisions, kind: systemone, ...}`, like the local Kev, Lev and Laya models.
- **An `output.decision` default on the hosted Jev runs the hosted Jev** (`providers/base.py`, `decision/pool.py`).
  With the default set to provider `typesafe` and model `jev-latest`, a decision from a workflow (or any
  `generate(..., output="decision")` on a text provider) read `jev-latest` as the alias of the most recently used
  local decision model: it ran that local model instead, or failed with "names no resident decision model yet" when
  none was loaded. `kev-latest` / `jev-latest` are aliases only on a local route; on `typesafe` and `systemone` they
  are the hosted model's own name.
