### Fixed

- **Hosted decision models are listed as decision models, and only there** (`providers/typesafe_provider.py`,
  `server/app.py`). `GET /v1/models` showed `typesafe/jev-latest` (and any `systemone` model) as a text model
  (`capabilities.output: ["text"]`, no `decision` block), and listed it under every capability filter, so text,
  image and embedding model pickers offered it and a chat on it then failed with `decision_model`. The `typesafe` and
  `systemone` providers now list their models only for a filter a decision model satisfies (`input.text`,
  `output.decision`), and `/v1/models` gives their entries `capabilities: {input: [text], output: [decision]}` and
  `decision: {endpoint: /v1/decisions, kind: systemone, ...}`, like the local Kev, Lev and Laya models.
