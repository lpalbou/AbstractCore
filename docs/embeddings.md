# Vector Embeddings Guide

AbstractCore includes built-in support for vector embeddings with **multiple providers** (HuggingFace, Ollama, LMStudio). This guide shows you how to use embeddings for semantic search, RAG applications, and similarity analysis.

**Two ways to use embeddings:**
1. **Python Library** (this guide) - Direct programmatic usage via `EmbeddingManager`
2. **REST API** - HTTP endpoints via AbstractCore server (see [Server API Reference](server.md#embeddings-endpoint))

## Quick Start

### Installation

```bash
# Install with embeddings support
pip install abstractcore[embeddings]
```

### First Embeddings

```python
from abstractcore.embeddings import EmbeddingManager

# Option 1: HuggingFace (default) - Local models with ONNX acceleration
embedder = EmbeddingManager()  # Uses all-MiniLM-L6-v2 by default

# Option 2: Ollama - Local models via Ollama API
embedder = EmbeddingManager(
    provider="ollama",
    model="granite-embedding:278m"
)

# Option 3: LMStudio - Local models via LMStudio API
embedder = EmbeddingManager(
    provider="lmstudio",
    model="text-embedding-all-minilm-l6-v2"
)

# Generate embedding for a single text (works with all providers)
embedding = embedder.embed("Machine learning transforms how we process information")
print(f"Embedding dimension: {len(embedding)}")  # 384 for MiniLM

# Compute similarity between texts (works with all providers)
similarity = embedder.compute_similarity(
    "artificial intelligence",
    "machine learning"
)
print(f"Similarity: {similarity:.3f}")  # 0.847
```

## Available Providers & Models

AbstractCore supports multiple embedding providers:

### HuggingFace Provider (Default)

Local sentence-transformers models with ONNX acceleration for 2-3x speedup.

| Model | Size | Dimensions | Languages | Primary Use Cases |
|-------|------|------------|-----------|----------|
| **all-minilm** (default) | 90M | 384 | English | Fast local development, testing |
| **qwen3-embedding** | 1.5B | 1536 | 100+ | Qwen-based multilingual, instruction-tuned |
| **embeddinggemma** | 300M | 768 | 100+ | General purpose, multilingual |
| **granite** | 278M | 768 | 100+ | Enterprise applications |

```python
# Default: all-MiniLM-L6-v2 (fast and lightweight)
embedder = EmbeddingManager()

# Qwen-based embedding model for multilingual support
embedder = EmbeddingManager(model="qwen3-embedding")

# Google's EmbeddingGemma for multilingual support
embedder = EmbeddingManager(model="embeddinggemma")

# Direct HuggingFace model ID
embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")
```

### Ollama Provider

Local embedding models via Ollama API. Requires Ollama running locally.

```python
# Setup: Install Ollama and pull an embedding model
# ollama pull granite-embedding:278m

# Use Ollama embeddings
embedder = EmbeddingManager(
    provider="ollama",
    model="granite-embedding:278m"
)

# Other popular Ollama embedding models:
# - nomic-embed-text (274MB)
# - granite-embedding:107m (smaller, faster)
```

### LMStudio Provider

Local embedding models via LMStudio API. Requires LMStudio running with a loaded model.

```python
# Setup: Start LMStudio and load an embedding model

# Use LMStudio embeddings
embedder = EmbeddingManager(
    provider="lmstudio",
    model="text-embedding-all-minilm-l6-v2"
)
```

### Provider Comparison

| Provider | Speed | Setup | Privacy | Cost | Primary Use Cases |
|----------|-------|-------|---------|------|----------|
| **HuggingFace** | Fast | Easy | Full | Free | Development, production |
| **Ollama** | Medium | Medium | Full | Free | Privacy, custom models |
| **LMStudio** | Medium | Easy (GUI) | Full | Free | GUI management, testing |

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
# Basic similarity between two texts
similarity = embedder.compute_similarity("cat", "kitten")
print(f"Similarity: {similarity:.3f}")  # 0.804

# NEW: Batch similarity - compare one text against many
query = "Python programming"
docs = ["Learn Python basics", "JavaScript guide", "Cooking recipes", "Data science with Python"]
similarities = embedder.compute_similarities(query, docs)
print(f"Batch similarities: {[f'{s:.3f}' for s in similarities]}")
# Output: ['0.785', '0.155', '0.145', '0.580']

# NEW: Similarity matrix - compare all texts against all texts
texts = ["Python programming", "JavaScript development", "Python data science", "Web frameworks"]
matrix = embedder.compute_similarities_matrix(texts)
print(f"Matrix shape: {matrix.shape}")  # (4, 4) symmetric matrix

# NEW: Asymmetric matrix for query-document matching
queries = ["Learn Python", "Web development guide"]
knowledge_base = ["Python tutorial", "JavaScript guide", "React framework", "Python for beginners"]
search_matrix = embedder.compute_similarities_matrix(queries, knowledge_base)
print(f"Search matrix: {search_matrix.shape}")  # (2, 4) - 2 queries × 4 documents
```

## Practical Applications

### Semantic Search

```python
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager()

# Document collection
documents = [
    "Python is strong for data science and machine learning applications",
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
from abstractcore import create_llm
from abstractcore.embeddings import EmbeddingManager

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

### Document Clustering (NEW)

```python
from abstractcore.embeddings import EmbeddingManager

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

# NEW: Automatic semantic clustering
clusters = embedder.find_similar_clusters(
    documents,
    threshold=0.6,      # 60% similarity required
    min_cluster_size=2  # At least 2 documents per cluster
)

print(f"Found {len(clusters)} clusters:")
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i+1} ({len(cluster)} documents):")
    for idx in cluster:
        print(f"  - {documents[idx]}")

# Example output:
# Cluster 1 (4 documents): Python-related content
# Cluster 2 (2 documents): JavaScript-related content
# Cluster 3 (2 documents): Machine learning content
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

### Advanced Caching (NEW)

```python
# Configure dual-layer caching system
embedder = EmbeddingManager(
    cache_size=5000,  # Larger memory cache
    cache_dir="./embeddings_cache"  # Persistent disk cache
)

# Regular embedding with standard caching
embedding1 = embedder.embed("Machine learning text")

# NEW: Normalized embedding with dedicated cache (2x faster for similarity)
normalized = embedder.embed_normalized("Machine learning text")
print(f"Normalized embedding length: {sum(x*x for x in normalized)**0.5:.3f}")  # 1.0 (unit length)

# Check comprehensive cache stats
stats = embedder.get_cache_stats()
print(f"Regular cache: {stats['persistent_cache_size']} embeddings")
print(f"Normalized cache: {stats['normalized_cache_size']} embeddings")
print(f"Memory cache hits: {stats['memory_cache_info']['hits']}")
```

## Integration with LLM Providers

### Enhanced Context Selection

```python
from abstractcore import create_llm
from abstractcore.embeddings import EmbeddingManager

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
llm = create_llm("anthropic", model="claude-haiku-4-5")

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
from abstractcore.embeddings import EmbeddingManager

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
from abstractcore.embeddings import EmbeddingManager

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

### Good Use Cases

- **Semantic Search**: Find relevant documents based on meaning, not keywords
- **RAG Applications**: Select relevant context for language model queries
- **Content Recommendation**: Find similar articles, products, or content
- **Clustering**: Group similar documents or texts together
- **Duplicate Detection**: Find near-duplicate content
- **Multi-language Search**: Search across different languages

### Not Ideal For

- **Exact Matching**: Use traditional text search for exact matches
- **Structured Data**: Use SQL databases for structured queries
- **Real-time Critical Applications**: Embedding computation has latency
- **Very Short Texts**: Embeddings work better with meaningful content
- **High-frequency Operations**: Consider caching for repeated queries

## Using Embeddings via REST API

If you prefer HTTP endpoints over Python code, use the AbstractCore server:

```bash
# Start the server
pip install abstractcore[server]
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

**HTTP Request:**
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Machine learning is fascinating",
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
  }'
```

**Supported providers via REST API:**
- `huggingface/model-name` - HuggingFace models
- `ollama/model-name` - Ollama models
- `lmstudio/model-name` - LMStudio models

**Complete REST API documentation:** [Server API Reference](server.md#embeddings-endpoint)

## Provider-Specific Features

### HuggingFace Features
- **ONNX Acceleration**: 2-3x faster inference
- **Matryoshka Truncation**: Reduce dimensions for efficiency
- **Persistent Caching**: Automatic disk caching of embeddings

### Ollama Features
- **Simple Setup**: Just `ollama pull <model>`
- **Full Privacy**: No data leaves your machine
- **Custom Models**: Use any Ollama-compatible model

### LMStudio Features
- **GUI Management**: Easy model loading via GUI
- **Testing Friendly**: Suitable for experimentation
- **OpenAI Compatible**: Standard API format

## Next Steps

- **Start Simple**: Try the semantic search example with your own data
- **Experiment with Providers**: Compare HuggingFace, Ollama, and LMStudio
- **Optimize Performance**: Use batch processing and caching for production
- **Build RAG**: Combine embeddings with AbstractCore LLMs for RAG applications
- **Use REST API**: Deploy embeddings as HTTP service with the server

## Related Documentation

**Core Library:**
- **[Python API Reference](api-reference.md)** - Complete EmbeddingManager API
- **[Getting Started](getting-started.md)** - Basic AbstractCore setup
- **[Examples](examples.md)** - More practical examples

**Server (REST API):**
- **[Server Guide](server.md)** - Server setup and deployment
- **[Server API Reference](server.md)** - REST API endpoints including embeddings
- **[Troubleshooting](troubleshooting.md)** - Common embedding issues

---

**Remember**: Embeddings are the foundation for semantic understanding. Combined with AbstractCore's multi-provider LLM capabilities, you can build sophisticated AI applications that understand meaning, not just keywords.
