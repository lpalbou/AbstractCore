# RAG Examples

## What This Folder Teaches

How to build a small, end-to-end Retrieval-Augmented Generation (RAG) pipeline using AbstractCore:

1) embed documents  
2) retrieve relevant context  
3) build a prompt with that context  
4) ask an LLM to answer grounded in retrieved snippets

## Prereqs

- Local embeddings: `pip install "abstractcore[embeddings]"`
- An LLM backend (local or hosted) depending on how you configure the script.

## Key AbstractCore Concepts

- Retrieval is a separate step from generation (easy to swap embedding model without touching your LLM provider).
- “RAG quality” depends more on retrieval + chunking + prompt formatting than on any single model knob.

## Scripts

- `complete_rag_example.py`
  - Demonstrates: a compact, end-to-end RAG flow you can copy into an application.
  - How it works: builds an embedding index in-memory, retrieves top-k, then generates an answer with context.

## Key Takeaways

- Start simple: in-memory top-k similarity search is enough to learn the pattern.
- Only add a vector DB once you understand your retrieval/chunking requirements.
