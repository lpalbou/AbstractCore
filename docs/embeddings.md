# Vector Embeddings Documentation

AbstractLLM Core provides a comprehensive embeddings system with state-of-the-art (SOTA) open-source models for semantic search, RAG applications, and AI-powered text understanding.

## Table of Contents

- [Quick Start](#quick-start)
- [Model Selection](#model-selection)
- [Performance Optimization](#performance-optimization)
- [Real-World Examples](#real-world-examples)
- [Production Deployment](#production-deployment)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Install with embeddings support
pip install abstractcore[embeddings]

# Or install sentence-transformers directly
pip install sentence-transformers
```

### Basic Usage

```python
from abstractllm.embeddings import EmbeddingManager

# Initialize with default model (EmbeddingGemma)
embedder = EmbeddingManager()

# Generate embedding for a single text
text = "Machine learning enables intelligent applications"
embedding = embedder.embed(text)
print(f"Embedding: {len(embedding)}D vector")

# Process multiple texts efficiently
texts = ["AI is powerful", "ML advances technology", "Data drives insights"]
embeddings = embedder.embed_batch(texts)

# Compute semantic similarity
similarity = embedder.compute_similarity("AI models", "Machine learning algorithms")
print(f"Similarity: {similarity:.3f}")
```

## Model Selection

### SOTA Models Available

#### 🔥 **EmbeddingGemma (Recommended)**
- **Model ID**: `google/embeddinggemma-300m`
- **Dimensions**: 768D with Matryoshka support (768→512→256→128)
- **Languages**: 100+ multilingual support
- **Performance**: ~67ms embedding time, SOTA quality
- **Use Cases**: General purpose, multilingual applications, production

```python
# Default - automatically uses EmbeddingGemma
embedder = EmbeddingManager()

# Explicit configuration
embedder = EmbeddingManager(model="embeddinggemma")

# With optimization
embedder = EmbeddingManager(
    model="embeddinggemma",
    output_dims=512,  # Matryoshka truncation for speed
    backend="onnx"    # 2-3x faster inference
)
```

#### 🏢 **IBM Granite**
- **Model ID**: `ibm-granite/granite-embedding-278m-multilingual`
- **Dimensions**: 768D
- **Languages**: 100+ multilingual support
- **Performance**: Enterprise-grade quality
- **Use Cases**: Business applications, enterprise deployment, multilingual

```python
embedder = EmbeddingManager(model="granite")

# Direct HuggingFace ID
embedder = EmbeddingManager(model="ibm-granite/granite-embedding-278m-multilingual")
```

#### ⚡ **all-MiniLM-L6-v2 (Baseline)**
- **Model ID**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384D (compact)
- **Languages**: English optimized
- **Performance**: ~94ms, lightweight, reliable
- **Use Cases**: Prototyping, English-only applications, resource-constrained

```python
embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")
```

### Model Comparison

| Model | Dimensions | Languages | Matryoshka | Speed | Quality | Best For |
|-------|------------|-----------|------------|-------|---------|----------|
| **EmbeddingGemma** | 768D | 100+ | ✅ | Fast | ⭐⭐⭐⭐⭐ | Production, multilingual |
| **IBM Granite** | 768D | 100+ | ❌ | Good | ⭐⭐⭐⭐ | Enterprise, business |
| **all-MiniLM-L6-v2** | 384D | English | ❌ | Very Fast | ⭐⭐⭐ | Prototyping, baseline |

## Performance Optimization

### 1. ONNX Backend (2-3x Speedup)

```python
from abstractllm.embeddings import EmbeddingManager

# Automatic ONNX optimization if available
embedder = EmbeddingManager(
    model="embeddinggemma",
    backend="onnx"  # Significant speedup
)
```

### 2. Matryoshka Dimension Truncation

```python
# Full quality (default)
embedder = EmbeddingManager(model="embeddinggemma")  # 768D

# Balanced performance/quality
embedder = EmbeddingManager(
    model="embeddinggemma",
    output_dims=512  # 768D → 512D truncation
)

# High speed
embedder = EmbeddingManager(
    model="embeddinggemma",
    output_dims=256  # 768D → 256D truncation
)
```

### 3. Smart Caching

```python
# Production caching configuration
embedder = EmbeddingManager(
    model="embeddinggemma",
    cache_size=5000,  # Large memory cache
    cache_dir="/path/to/persistent/cache"  # Custom cache location
)

# Check cache performance
stats = embedder.get_cache_stats()
print(f"Cache hits: {stats['memory_cache_info']['hits']}")
print(f"Persistent cache: {stats['persistent_cache_size']} embeddings")
```

### 4. Batch Processing

```python
# Process large datasets efficiently
documents = [f"Document {i} content..." for i in range(1000)]

# Batch processing is much faster than individual calls
embeddings = embedder.embed_batch(documents)  # Significant speedup

# For very large datasets, process in chunks
chunk_size = 100
for i in range(0, len(documents), chunk_size):
    chunk = documents[i:i + chunk_size]
    chunk_embeddings = embedder.embed_batch(chunk)
    # Process chunk_embeddings...
```

## Real-World Examples

### 1. Semantic Document Search

```python
from abstractllm.embeddings import EmbeddingManager

class DocumentSearchSystem:
    def __init__(self):
        self.embedder = EmbeddingManager(model="embeddinggemma")
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs):
        """Add documents and pre-compute embeddings."""
        self.documents.extend(docs)
        new_embeddings = self.embedder.embed_batch(docs)
        self.embeddings.extend(new_embeddings)

    def search(self, query, top_k=5):
        """Search for most relevant documents."""
        similarities = []
        for i, doc in enumerate(self.documents):
            sim = self.embedder.compute_similarity(query, doc)
            similarities.append((sim, i, doc))

        # Return top_k most similar
        return sorted(similarities, reverse=True)[:top_k]

# Usage
search_system = DocumentSearchSystem()
search_system.add_documents([
    "Python is a programming language for AI development",
    "Machine learning algorithms process data patterns",
    "React builds interactive web user interfaces"
])

results = search_system.search("artificial intelligence programming")
for score, idx, doc in results:
    print(f"Score: {score:.3f} - {doc}")
```

### 2. Complete RAG Application

```python
from abstractllm.embeddings import EmbeddingManager
from abstractllm import create_llm

class RAGSystem:
    def __init__(self, llm_provider="openai"):
        self.embedder = EmbeddingManager(model="embeddinggemma")
        self.llm = create_llm(llm_provider, model="gpt-4o-mini")
        self.knowledge_base = []
        self.embeddings_cache = []

    def add_knowledge(self, documents):
        """Add documents to knowledge base."""
        self.knowledge_base.extend(documents)
        # Pre-compute embeddings for fast retrieval
        new_embeddings = self.embedder.embed_batch(documents)
        self.embeddings_cache.extend(new_embeddings)

    def ask(self, question, top_k=3):
        """Answer question using RAG."""
        # Step 1: Retrieve relevant context
        similarities = []
        for i, doc in enumerate(self.knowledge_base):
            sim = self.embedder.compute_similarity(question, doc)
            similarities.append((sim, doc))

        # Get top contexts
        top_contexts = sorted(similarities, reverse=True)[:top_k]
        context = "\\n\\n".join([doc for _, doc in top_contexts])

        # Step 2: Generate answer with context
        prompt = f\"\"\"Context:
{context}

Question: {question}

Based on the provided context, please answer the question:\"\"\"

        response = self.llm.generate(prompt)
        return response.content

# Usage
rag = RAGSystem(llm_provider="openai")
rag.add_knowledge([
    "AbstractLLM Core provides unified access to multiple LLM providers",
    "The embeddings system uses SOTA models like EmbeddingGemma for semantic search"
])

answer = rag.ask("What embedding models does AbstractLLM use?")
print(answer)
```

### 3. Multilingual Content Classification

```python
from abstractllm.embeddings import EmbeddingManager

class MultilingualClassifier:
    def __init__(self):
        # Use multilingual model
        self.embedder = EmbeddingManager(model="embeddinggemma")
        self.categories = {}

    def add_category(self, name, examples):
        """Add a category with example texts."""
        # Create category representation from examples
        embeddings = self.embedder.embed_batch(examples)
        # Average embeddings to create category centroid
        import numpy as np
        centroid = np.mean(embeddings, axis=0).tolist()
        self.categories[name] = centroid

    def classify(self, text):
        """Classify text into predefined categories."""
        text_embedding = self.embedder.embed(text)

        best_category = None
        best_score = -1

        for category, centroid in self.categories.items():
            # Compute similarity to category centroid
            import numpy as np
            text_emb = np.array(text_embedding)
            cat_emb = np.array(centroid)
            similarity = np.dot(text_emb, cat_emb) / (
                np.linalg.norm(text_emb) * np.linalg.norm(cat_emb)
            )

            if similarity > best_score:
                best_score = similarity
                best_category = category

        return best_category, best_score

# Usage with multilingual content
classifier = MultilingualClassifier()

# Add categories with multilingual examples
classifier.add_category("technology", [
    "Artificial intelligence and machine learning",
    "Intelligence artificielle et apprentissage automatique",
    "Inteligencia artificial y aprendizaje automático"
])

classifier.add_category("sports", [
    "Football and basketball games",
    "Jeux de football et de basket-ball",
    "Juegos de fútbol y baloncesto"
])

# Classify multilingual content
result = classifier.classify("Los algoritmos de IA son muy poderosos")
print(f"Category: {result[0]}, Confidence: {result[1]:.3f}")
```

## Production Deployment

### Environment Setup

```python
import os
from abstractllm.embeddings import EmbeddingManager

# Production configuration
embedder = EmbeddingManager(
    model="embeddinggemma",
    backend="onnx",  # Optimized inference
    cache_dir=os.getenv("EMBEDDINGS_CACHE_DIR", "/var/cache/embeddings"),
    cache_size=10000,  # Large cache for production
    output_dims=512,  # Balanced performance
)
```

### Performance Monitoring

```python
from abstractllm.events import EventType, on_global
import time

# Monitor embedding performance
def monitor_embeddings(event):
    if event.type == "embedding_generated":
        duration = event.data.get('duration_ms', 0)
        dimension = event.data.get('dimension', 0)
        print(f"Embedding: {dimension}D in {duration:.1f}ms")

on_global("embedding_generated", monitor_embeddings)

# Monitor cache performance
start_time = time.time()
embeddings = embedder.embed_batch(texts)
end_time = time.time()

stats = embedder.get_cache_stats()
print(f"Batch processing: {end_time - start_time:.3f}s")
print(f"Cache hit rate: {stats['memory_cache_info']['hits']}")
```

### Scaling Considerations

```python
# For high-throughput applications
class ProductionEmbeddingService:
    def __init__(self):
        self.embedder = EmbeddingManager(
            model="embeddinggemma",
            backend="onnx",
            cache_size=50000,  # Large cache
            output_dims=512    # Balanced performance
        )

    def embed_with_fallback(self, text):
        """Embed with error handling and fallback."""
        try:
            return self.embedder.embed(text)
        except Exception as e:
            # Log error and return zero vector fallback
            print(f"Embedding error: {e}")
            return [0.0] * self.embedder.get_dimension()

    def batch_embed_chunked(self, texts, chunk_size=100):
        """Process large batches in chunks."""
        results = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_embeddings = self.embedder.embed_batch(chunk)
            results.extend(chunk_embeddings)
        return results
```

## API Reference

### EmbeddingManager

#### Constructor

```python
EmbeddingManager(
    model: str = None,              # Model name or HuggingFace ID
    backend: str = "auto",          # "auto", "pytorch", "onnx"
    cache_dir: Path = None,         # Cache directory
    cache_size: int = 1000,         # Memory cache size
    output_dims: int = None,        # Matryoshka truncation
    trust_remote_code: bool = False # Security setting
)
```

#### Methods

**`embed(text: str) -> List[float]`**
- Generate embedding for single text
- Returns list of float values
- Automatically cached for repeated calls

**`embed_batch(texts: List[str]) -> List[List[float]]`**
- Generate embeddings for multiple texts efficiently
- Significantly faster than individual calls
- Cache-aware processing

**`compute_similarity(text1: str, text2: str) -> float`**
- Compute cosine similarity between two texts
- Returns float between -1 and 1
- Higher values indicate greater similarity

**`get_dimension() -> int`**
- Get embedding dimension
- Accounts for Matryoshka truncation

**`get_cache_stats() -> Dict`**
- Get cache performance statistics
- Memory and persistent cache metrics

**`clear_cache()`**
- Clear both memory and persistent caches
- Use for testing or cache reset

### Model Configuration

```python
from abstractllm.embeddings import get_model_config, list_available_models

# List available models
models = list_available_models()
print(models)  # ['embeddinggemma', 'granite', 'stella-400m', ...]

# Get model details
config = get_model_config("embeddinggemma")
print(f"Model: {config.model_id}")
print(f"Dimension: {config.dimension}")
print(f"Multilingual: {config.multilingual}")
print(f"Matryoshka: {config.supports_matryoshka}")
```

## Troubleshooting

### Common Issues

#### 1. Model Download Errors

```python
# Issue: Model not found or download fails
# Solution: Check model ID and network connection

try:
    embedder = EmbeddingManager(model="embeddinggemma")
except Exception as e:
    print(f"Model loading failed: {e}")
    # Fallback to reliable baseline model
    embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")
```

#### 2. Performance Issues

```python
# Issue: Slow embedding generation
# Solution: Enable optimizations

embedder = EmbeddingManager(
    model="embeddinggemma",
    backend="onnx",      # Enable ONNX optimization
    output_dims=256,     # Reduce dimensions if acceptable
    cache_size=5000      # Increase cache size
)

# Use batch processing for multiple texts
embeddings = embedder.embed_batch(texts)  # Much faster than individual calls
```

#### 3. Memory Issues

```python
# Issue: High memory usage
# Solution: Optimize configuration

embedder = EmbeddingManager(
    model="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
    cache_size=1000,     # Reduce cache size
    output_dims=256      # Use smaller dimensions
)

# Process large datasets in chunks
def process_large_dataset(texts, chunk_size=100):
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        embeddings = embedder.embed_batch(chunk)
        # Process embeddings immediately, don't store all
        yield embeddings
```

#### 4. Cache Issues

```python
# Issue: Cache not working or corrupted
# Solution: Reset cache

embedder = EmbeddingManager()
embedder.clear_cache()  # Reset all caches

# Or specify different cache directory
embedder = EmbeddingManager(cache_dir="/tmp/new_cache")
```

### Performance Benchmarks

Based on testing with Apple M4 Max (128GB RAM):

| Operation | EmbeddingGemma | all-MiniLM-L6-v2 | Notes |
|-----------|----------------|------------------|-------|
| Single embed | ~67ms | ~94ms | Includes model overhead |
| Batch (10 texts) | ~76ms | ~52ms | Significant speedup |
| Cache hit | <1ms | <1ms | Near-instantaneous |
| Memory usage | ~300MB | ~150MB | Including model weights |

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create embedder
embedder = EmbeddingManager(model="embeddinggemma")

# Monitor performance
import time
start = time.time()
embedding = embedder.embed("Test text")
print(f"Embedding time: {time.time() - start:.3f}s")

# Check cache stats
stats = embedder.get_cache_stats()
print(f"Cache stats: {stats}")
```

---

## Next Steps

1. **Explore Examples**: Check `examples/` directory for complete applications
2. **Integration**: Combine with AbstractLLM providers for full RAG pipelines
3. **Production**: Deploy with monitoring and caching optimizations
4. **Contributing**: Report issues and contribute improvements

For more information, see the main [README.md](../README.md) or visit the [GitHub repository](https://github.com/lpalbou/AbstractCore).