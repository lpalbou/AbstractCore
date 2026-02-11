# AbstractMemory + AbstractSemantics: a knowledge layer that stays explainable

RAG isn’t the only kind of “memory”.

Sometimes you want:
- a compact, deterministic representation of what an agent believes,
- a provenance trail of when/how it was asserted,
- the ability to query it predictably,
- and (optionally) a semantic retrieval signal.

In AbstractFramework, that’s the role of:
- **AbstractSemantics**: a shared registry of allowed predicate + entity-type ids (and schema helpers)
- **AbstractMemory**: an append-only temporal triple store with provenance-aware assertions

![Memory + semantics overview](assets/memory-and-semantics.svg)

## What you gain

- **Deterministic structure**: triples + explicit ids, not “whatever JSON the model felt like”.
- **Append-only updates**: changes are new assertions, not mutations (great for auditing and replay).
- **Provenance hooks**: record where an assertion came from (span ids, tool evidence, timestamps).
- **Optional semantic retrieval**: plug in embeddings without coupling memory storage to an LLM client.

## AbstractSemantics: shared ids + schema builders

AbstractSemantics ships:
- a default YAML registry (predicates + entity types),
- a loader returning an immutable `SemanticsRegistry`,
- helpers to build a compact JSON Schema contract for structured outputs.

Minimal use:

```python
from abstractsemantics import load_semantics_registry, build_kg_assertion_schema_v0

reg = load_semantics_registry()
schema = build_kg_assertion_schema_v0(registry=reg, include_predicate_aliases=True)
```

That schema can be used by:
- ingestion boundaries (validate model output),
- UIs (dropdowns for allowed predicates/types),
- and LLM structured output (contract-first extraction).

## AbstractMemory: append-only triple assertions

Minimal use:

```python
from abstractmemory import InMemoryTripleStore, TripleAssertion, TripleQuery

store = InMemoryTripleStore()
store.add(
    [
        TripleAssertion(
            subject="Scrooge",
            predicate="related_to",
            object="Christmas",
            scope="session",
            owner_id="sess-1",
            provenance={"span_id": "span_123"},
        )
    ]
)

hits = store.query(TripleQuery(subject="scrooge", scope="session", owner_id="sess-1"))
print(hits[0].object)  # "christmas" (terms are canonicalized)
```

For persistence + vector search, AbstractMemory can use a LanceDB-backed store (`AbstractMemory[lancedb]`).

## A practical pattern: “extract → validate → append”

One common production pattern looks like this:

1) Ask an LLM to produce KG assertions *against a schema* (from AbstractSemantics).  
2) Validate the assertions (ids, required fields, provenance).  
3) Append them to AbstractMemory’s store (never mutate).  
4) Query deterministically for control logic, and optionally use semantic similarity for ranking.

The important part is the separation:
- semantics and contracts are centralized (AbstractSemantics),
- storage/query is explicit (AbstractMemory),
- embeddings can be injected (for example via an AbstractGateway embeddings endpoint) without coupling the store to an LLM client.

## When to use this layer

Use AbstractMemory/Semantics when you want:
- explicit, explainable memory updates,
- durable knowledge you can replay and audit,
- a shared predicate/type vocabulary across agents, flows, and UIs.

Next: a local host with a UI — **AbstractAssistant** (tray agent):
[10-abstractassistant-tray-agent.md](10-abstractassistant-tray-agent.md).

---

### Evidence pointers (source of truth)

- AbstractMemory repo: https://github.com/lpalbou/abstractmemory
- AbstractMemory exports: https://github.com/lpalbou/abstractmemory/blob/main/src/abstractmemory/__init__.py
- Triple models + stores: https://github.com/lpalbou/abstractmemory/blob/main/src/abstractmemory/models.py and https://github.com/lpalbou/abstractmemory/blob/main/src/abstractmemory/
- Gateway embedder adapter: https://github.com/lpalbou/abstractmemory/blob/main/src/abstractmemory/embeddings.py
- AbstractSemantics repo: https://github.com/lpalbou/abstractsemantics
- AbstractSemantics registry + schema: https://github.com/lpalbou/abstractsemantics/blob/main/src/abstractsemantics/registry.py and https://github.com/lpalbou/abstractsemantics/blob/main/src/abstractsemantics/schema.py
