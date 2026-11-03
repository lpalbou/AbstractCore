### Fixed

- **A decision model answers its own decisions; the `output.decision` default is for models that cannot decide**
  (`providers/base.py` `_decisions_stay_on_this_model`). Before, with a default of `typesafe` + `jev-latest`, a local
  Kev instance (`create_llm("mlx", model="…kev…").generate(doc, output="decision", questions=…)`, no route) sent the
  document to the hosted Jev because of a global setting. Now an instance whose own model is a decision model (a
  local Kev / Lev / Laya checkpoint on mlx or huggingface, or a `typesafe` / `systemone` instance) answers itself when
  the request names no decision provider or model. A route the request names still wins, and a text model still
  uses the default. Review of #41, Q1.
