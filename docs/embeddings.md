# Vector Embeddings Guide

AbstractCore includes built-in support for vector embeddings using state-of-the-art open-source models. This guide shows you how to use embeddings for semantic search, RAG applications, and similarity analysis.

## Quick Start

### Installation

```bash
# Install with embeddings support
pip install abstractcore[embeddings]
```

### First Embeddings

```python
from abstractllm.embeddings import EmbeddingManager

# Create embedder with default model (Google's EmbeddingGemma)
embedder = EmbeddingManager()

# Generate embedding for a single text
embedding = embedder.embed("Machine learning transforms how we process information")
print(f"Embedding dimension: {len(embedding)}")  # 768

# Compute similarity between texts
similarity = embedder.compute_similarity(
    "artificial intelligence",
    "machine learning"
)
print(f"Similarity: {similarity:.3f}")  # 0.847
```

## Available Models

AbstractCore includes several SOTA open-source embedding models:

| Model | Size | Dimensions | Languages | Best For |
|-------|------|------------|-----------|----------|
| **embeddinggemma** (default) | 300M | 768 | 100+ | General purpose, multilingual |
| **granite** | 278M | 768 | 100+ | Enterprise applications |

### Model Selection

```python
# Default: Google's EmbeddingGemma (recommended)
embedder = EmbeddingManager()

# IBM Granite for enterprise use
embedder = EmbeddingManager(model="granite")

# Direct HuggingFace model ID
embedder = EmbeddingManager(model="google/embeddinggemma-300m")
```

## Core Features

### Single Text Embeddings

```python
embedder = EmbeddingManager()

text = "Python is a versatile programming language"
embedding = embedder.embed(text)

print(f"Text: {text}")
print(f"Embedding: {len(embedding)} dimensions")
print(f"First 5 values: {embedding[:5]}")
```

### Batch Processing (More Efficient)

```python
texts = [
    "Python programming language",
    "JavaScript for web development",
    "Machine learning with Python",
    "Data science and analytics"
]

# Process multiple texts at once (much faster)
embeddings = embedder.embed_batch(texts)

print(f"Generated {len(embeddings)} embeddings")
for i, embedding in enumerate(embeddings):
    print(f"Text {i+1}: {len(embedding)} dimensions")
```

### Similarity Analysis

```python
# Compare different concepts
pairs = [
    ("cat", "kitten"),
    ("car", "automobile"),
    ("happy", "joyful"),
    ("python", "snake"),
    ("python", "programming")
]

for text1, text2 in pairs:
    similarity = embedder.compute_similarity(text1, text2)
    print(f"{text1} ↔ {text2}: {similarity:.3f}")

# Output:
# cat ↔ kitten: 0.789
# car ↔ automobile: 0.845
# happy ↔ joyful: 0.712
# python ↔ snake: 0.423
# python ↔ programming: 0.687
```

## Practical Applications

### Semantic Search

```python
from abstractllm.embeddings import EmbeddingManager

embedder = EmbeddingManager()

# Document collection
documents = [
    "Python is excellent for data science and machine learning applications",
    "JavaScript enables interactive web pages and modern frontend development",
    "React is a popular library for building user interfaces with JavaScript",
    "SQL databases store and query structured data efficiently",
    "Machine learning algorithms can predict patterns from historical data"
]

def semantic_search(query, documents, top_k=3):
    """Find most relevant documents for a query."""
    similarities = []

    for i, doc in enumerate(documents):
        similarity = embedder.compute_similarity(query, doc)
        similarities.append((i, similarity, doc))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

# Search for relevant documents
query = "web development frameworks"
results = semantic_search(query, documents)

print(f"Query: {query}\n")
for rank, (idx, similarity, doc) in enumerate(results, 1):
    print(f"{rank}. Score: {similarity:.3f}")
    print(f"   {doc}\n")
```

### Simple RAG Pipeline

```python
from abstractllm import create_llm
from abstractllm.embeddings import EmbeddingManager

# Setup
embedder = EmbeddingManager()
llm = create_llm("openai", model="gpt-4o-mini")

# Knowledge base
knowledge_base = [
    "The Eiffel Tower is 330 meters tall and was completed in 1889.",
    "Paris is the capital city of France with over 2 million inhabitants.",
    "The Louvre Museum in Paris houses the famous Mona Lisa painting.",
    "French cuisine is known for its wine, cheese, and pastries.",
    "The Seine River flows through central Paris."
]

def rag_query(question, knowledge_base, llm, embedder):
    """Answer question using relevant context from knowledge base."""

    # Step 1: Find most relevant context
    similarities = []
    for doc in knowledge_base:
        similarity = embedder.compute_similarity(question, doc)
        similarities.append((similarity, doc))

    # Get top 2 most relevant documents
    similarities.sort(reverse=True)
    top_contexts = [doc for _, doc in similarities[:2]]
    context = "\n".join(top_contexts)

    # Step 2: Generate answer using context
    prompt = f"""Context:
{context}

Question: {question}

Based on the context above, please answer the question:"""

    response = llm.generate(prompt)
    return response.content, top_contexts

# Usage
question = "How tall is the Eiffel Tower?"
answer, contexts = rag_query(question, knowledge_base, llm, embedder)

print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"\nUsed context:")
for ctx in contexts:
    print(f"- {ctx}")
```

### Document Clustering

```python
from abstractllm.embeddings import EmbeddingManager
import numpy as np
from sklearn.cluster import KMeans

embedder = EmbeddingManager()

# Documents to cluster
documents = [
    "Python programming tutorial for beginners",
    "Introduction to machine learning concepts",
    "JavaScript web development guide",
    "Advanced Python data structures",
    "Machine learning with neural networks",
    "Building web apps with JavaScript",
    "Python for data analysis",
    "Deep learning fundamentals",
    "React.js frontend development",
    "Statistical analysis with Python"
]

# Generate embeddings
embeddings = embedder.embed_batch(documents)

# Cluster documents
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Group documents by cluster
clustered_docs = {i: [] for i in range(n_clusters)}
for doc, cluster in zip(documents, clusters):
    clustered_docs[cluster].append(doc)

# Display results
for cluster_id, docs in clustered_docs.items():
    print(f"\nCluster {cluster_id + 1}:")
    for doc in docs:
        print(f"  - {doc}")
```

## Performance Optimization

### ONNX Backend (2-3x Faster)

```python
# Enable ONNX for faster inference
embedder = EmbeddingManager(
    model="embeddinggemma",
    backend="onnx"  # 2-3x speedup
)

# Performance comparison
import time

texts = ["Sample text for performance testing"] * 100

# Time the embedding generation
start_time = time.time()
embeddings = embedder.embed_batch(texts)
duration = time.time() - start_time

print(f"Generated {len(embeddings)} embeddings in {duration:.2f} seconds")
print(f"Speed: {len(embeddings)/duration:.1f} embeddings/second")
```

### Dimension Truncation (Memory/Speed Trade-off)

```python
# Truncate embeddings for faster processing
embedder = EmbeddingManager(
    model="embeddinggemma",
    output_dims=256  # Reduce from 768 to 256 dimensions
)

embedding = embedder.embed("Test text")
print(f"Truncated embedding dimension: {len(embedding)}")  # 256
```

### Caching

```python
# Configure caching for better performance
embedder = EmbeddingManager(
    cache_size=5000,  # Larger memory cache
    cache_dir="./embeddings_cache"  # Persistent disk cache
)

# First call: computes embedding
embedding1 = embedder.embed("This text will be cached")

# Second call: returns cached result (much faster)
embedding2 = embedder.embed("This text will be cached")

# Verify they're identical
print(f"Cached result identical: {embedding1 == embedding2}")  # True
```

## Integration with LLM Providers

### Enhanced Context Selection

```python
from abstractllm import create_llm
from abstractllm.embeddings import EmbeddingManager

def smart_context_selection(query, documents, max_context_length=2000):
    """Select most relevant context that fits within token limits."""
    embedder = EmbeddingManager()

    # Score all documents
    scored_docs = []
    for doc in documents:
        similarity = embedder.compute_similarity(query, doc)
        scored_docs.append((similarity, doc))

    # Sort by relevance
    scored_docs.sort(reverse=True)

    # Select documents that fit within context limit
    selected_context = ""
    for similarity, doc in scored_docs:
        test_context = selected_context + "\n" + doc
        if len(test_context) <= max_context_length:
            selected_context = test_context
        else:
            break

    return selected_context.strip()

# Usage with LLM
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")

documents = [
    "Long document about machine learning...",
    "Another document about data science...",
    # ... many more documents
]

query = "What is supervised learning?"
context = smart_context_selection(query, documents)

response = llm.generate(f"Context: {context}\n\nQuestion: {query}")
print(response.content)
```

### Multi-language Support

```python
# EmbeddingGemma supports 100+ languages
embedder = EmbeddingManager(model="embeddinggemma")

# Cross-language similarity
similarity = embedder.compute_similarity(
    "Hello world",      # English
    "Bonjour le monde"  # French
)
print(f"Cross-language similarity: {similarity:.3f}")

# Multilingual semantic search
documents_multilingual = [
    "Machine learning is transforming technology",  # English
    "L'intelligence artificielle change le monde",  # French
    "人工智能正在改变世界",                        # Chinese
    "Künstliche Intelligenz verändert die Welt"    # German
]

query = "artificial intelligence"
for doc in documents_multilingual:
    similarity = embedder.compute_similarity(query, doc)
    print(f"{similarity:.3f}: {doc}")
```

## Production Considerations

### Error Handling

```python
from abstractllm.embeddings import EmbeddingManager

def safe_embedding(text, embedder, fallback_value=None):
    """Generate embedding with error handling."""
    try:
        return embedder.embed(text)
    except Exception as e:
        print(f"Embedding failed for text: {text[:50]}...")
        print(f"Error: {e}")
        return fallback_value or [0.0] * 768  # Return zero vector as fallback

embedder = EmbeddingManager()

# Safe embedding generation
text = "Some text that might cause issues"
embedding = safe_embedding(text, embedder)

if embedding:
    print(f"Successfully generated embedding: {len(embedding)} dimensions")
else:
    print("Using fallback embedding")
```

### Monitoring and Metrics

```python
import time
from abstractllm.embeddings import EmbeddingManager

class MonitoredEmbeddingManager:
    def __init__(self, *args, **kwargs):
        self.embedder = EmbeddingManager(*args, **kwargs)
        self.stats = {
            'total_calls': 0,
            'total_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def embed(self, text):
        start_time = time.time()
        result = self.embedder.embed(text)
        duration = time.time() - start_time

        self.stats['total_calls'] += 1
        self.stats['total_time'] += duration

        return result

    def get_stats(self):
        avg_time = self.stats['total_time'] / max(self.stats['total_calls'], 1)
        return {
            **self.stats,
            'average_time': avg_time,
            'calls_per_second': 1 / avg_time if avg_time > 0 else 0
        }

# Usage
monitored_embedder = MonitoredEmbeddingManager()

# Generate some embeddings
for i in range(10):
    monitored_embedder.embed(f"Test text number {i}")

# Check performance
stats = monitored_embedder.get_stats()
print(f"Total calls: {stats['total_calls']}")
print(f"Average time per call: {stats['average_time']:.3f}s")
print(f"Calls per second: {stats['calls_per_second']:.1f}")
```

## When to Use Embeddings

### ✅ Good Use Cases

- **Semantic Search**: Find relevant documents based on meaning, not keywords
- **RAG Applications**: Select relevant context for language model queries
- **Content Recommendation**: Find similar articles, products, or content
- **Clustering**: Group similar documents or texts together
- **Duplicate Detection**: Find near-duplicate content
- **Multi-language Search**: Search across different languages

### ❌ Not Ideal For

- **Exact Matching**: Use traditional text search for exact matches
- **Structured Data**: Use SQL databases for structured queries
- **Real-time Critical Applications**: Embedding computation has latency
- **Very Short Texts**: Embeddings work better with meaningful content
- **High-frequency Operations**: Consider caching for repeated queries

## Next Steps

- **Start Simple**: Try the semantic search example with your own data
- **Experiment with Models**: Compare different embedding models for your use case
- **Optimize Performance**: Use batch processing and caching for production
- **Build RAG**: Combine embeddings with AbstractCore LLMs for RAG applications

For more information:
- [Examples](examples.md) - More practical examples
- [API Reference](api_reference.md) - Complete EmbeddingManager API
- [Getting Started](getting-started.md) - Basic AbstractCore setup

---

**Remember**: Embeddings are the foundation for semantic understanding. Combined with AbstractCore's LLM capabilities, you can build sophisticated AI applications that understand meaning, not just keywords.