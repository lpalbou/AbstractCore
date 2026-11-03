# abstractcore.decision

Decision models: a document (the *state*) in, calibrated typed decisions out. This package runs the
Kev family (Jev-like decision models: a LoRA adapter plus a pointer head on a Qwen3.5 base) without
any text generation, and is what `generate(..., output="decision")` and the server's `POST /v1/decisions` call.

```
decision/
├── types.py        Choice / Noul / Score questions, DecisionRequest, typed answers, DecisionResponse
├── render.py       request -> state text + option strings; probabilities -> answers (confidence formulas)
├── encoding.py     token packing: <state> ... then per question <q> instr <opt> o </opt>... <decide>
├── head.py         the pointer head (two projections, scale, temperature) in numpy; head.safetensors + decision.json
├── tokenizers.py   transformers and llama.cpp tokenizer adapters (same ids on the Qwen vocabulary)
├── runtime.py      DecisionModel: encode -> backend -> answers, state-prefix LRU cache
├── loader.py       detect a decision checkpoint (adapter / merged / GGUF) and load it for a provider
├── convert.py      adapter -> merged transformers or mlx-lm checkpoint (bf16, 4/8-bit, mixed)
├── kinds.py        the prompt/readout facts per decision kind (kev, lev, openjev, laya, systemone)
├── engines/        base.py (the DecisionEngine protocol), remote.py (hosted System One: Jev and
│                   compatible hosts), labels.py (label-logit kinds), laya.py (ModernBERT + scorer)
└── backends/       hf.py / mlx.py / gguf.py (pointer-head rows -> logits), logits.py (last-token
                    logits on transformers / mlx-lm / llama.cpp for the label-logit kinds)
```

The pointer-head runtime (`runtime.DecisionModel`) is one engine among several; every engine
answers the same `DecisionRequest` with the same `DecisionResponse`, so providers, the server and
callers never depend on the model behind it.

How a request runs:

1. `render.to_record` flattens the state and each question into text (dict keys become labels).
2. `encoding.encode` packs the state once and every question as its own branch; user text can
   never forge a delimiter (`<|name|>` is escaped before tokenisation).
3. The backend prefills the state into a cache and runs every question as a causal row continuing
   from a copy of that cache, so questions never see each other and the state is computed once.
4. The pointer head scores each option's `</opt>` hidden state against the question's `<decide>`
   hidden state; a softmax per question gives the probabilities, divided by the checkpoint's
   calibration temperature.
5. `render.to_answers` maps probabilities to the typed answer (argmax + confidence, P(true),
   expected level).

Public docs: `docs/decision-models.md`. Tests: `tests/decision/`.
