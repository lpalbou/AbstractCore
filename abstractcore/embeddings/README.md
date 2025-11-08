# Embeddings Module

## Purpose

The embeddings module provides production-ready vector embedding generation for semantic search, similarity computation, and retrieval-augmented generation (RAG) applications. It transforms text into high-dimensional numerical vectors that capture semantic meaning, enabling AI applications to measure text similarity, cluster documents, and perform efficient semantic search.

**Core Capabilities:**
- Generate vector embeddings from text using state-of-the-art open-source models
- Multi-provider support (HuggingFace, Ollama, LMStudio)
- ONNX-accelerated inference for 2-3x performance improvement
- Two-layer caching system (memory + persistent disk)
- Batch processing optimization for large-scale operations
- Similarity computation and clustering algorithms
- Matryoshka dimension truncation for flexible embedding sizes

## Architecture Position

**Layer:** Core Infrastructure (alongside providers, structured output, compression)

**Dependencies:**
- `sentence-transformers` (optional, for HuggingFace provider)
- `numpy` (for numerical operations and similarity computation)
- `onnxruntime` (optional, for ONNX acceleration)
- `abstractcore.providers` (for Ollama and LMStudio embedding support)
- `abstractcore.events` (optional, for event emission)
- `abstractcore.config` (for default provider/model configuration)
- `abstractcore.utils.token_utils` (for token estimation)

**Used By:**
- RAG applications requiring semantic search
- Document clustering and similarity analysis
- Content recommendation systems
- Duplicate content detection
- Semantic caching implementations
- Question-answering systems

## Component Structure

```
abstractcore/embeddings/
├── __init__.py          # Public API exports
├── models.py            # Embedding model configurations
└── manager.py           # EmbeddingManager class (core functionality)
```

### File Descriptions

1. **models.py** - Embedding model registry and configuration
   - Defines `EmbeddingModelConfig` dataclass for model metadata
   - Registry of curated open-source embedding models
   - Model compatibility and feature tracking (Matryoshka support, dimensions, etc.)
   - Utility functions for model discovery

2. **manager.py** - Core embedding manager implementation
   - `EmbeddingManager` class for unified embedding generation
   - Multi-provider abstraction (HuggingFace, Ollama, LMStudio)
   - Two-layer caching system (memory + persistent)
   - Batch processing optimization
   - Similarity computation algorithms
   - Clustering and matrix operations

## Detailed Components

### models.py

Provides a curated registry of high-quality open-source embedding models with comprehensive metadata.

#### EmbeddingModelConfig

Dataclass defining embedding model metadata:

```python
@dataclass
class EmbeddingModelConfig:
    name: str                           # Short name (e.g., "all-minilm-l6-v2")
    model_id: str                       # HuggingFace model ID
    dimension: int                      # Embedding dimension
    max_sequence_length: int            # Maximum input token length
    supports_matryoshka: bool          # Whether model supports dimension truncation
    matryoshka_dims: Optional[List[int]] # Available truncation dimensions
    description: str                    # Human-readable description
    multilingual: bool                  # Multilingual support flag
    size_mb: Optional[float]           # Model size for download planning
```

#### Curated Model Registry

The module provides carefully selected models optimized for different use cases:

**Fast & Lightweight:**
- `all-minilm-l6-v2`: 384 dims, 90MB - Default, perfect for development (fastest)
- `granite-30m`: 384 dims, 30MB - Ultra-lightweight, English only

**Balanced Performance:**
- `granite-107m`: 768 dims, 107MB - Multilingual, balanced size
- `granite-278m`: 768 dims, 278MB - High quality, multilingual

**State-of-the-Art:**
- `embeddinggemma`: 768 dims, 300MB - Google's 2025 SOTA on-device model, multilingual, Matryoshka
- `nomic-embed-v1.5`: 768 dims, 550MB - High-quality English, Matryoshka
- `nomic-embed-v2-moe`: 768 dims, 800MB - Mixture of Experts architecture
- `qwen3-embedding`: 1024 dims, 600MB - Efficient multilingual support

#### Key Functions

```python
get_model_config(model_name: str) -> EmbeddingModelConfig
    # Get configuration for a specific model

list_available_models() -> List[str]
    # List all available embedding model names

get_default_model() -> str
    # Get default model name (all-minilm-l6-v2)
```

### manager.py

Core embedding generation and similarity computation engine.

#### EmbeddingManager Class

Production-ready embedding manager with multi-provider support.

**Initialization Parameters:**

```python
EmbeddingManager(
    model: str = None,                  # Model identifier (None uses config default)
    provider: str = None,               # Provider: "huggingface", "ollama", "lmstudio"
    backend: str = "auto",              # HF backend: "auto", "pytorch", "onnx", "openvino"
    cache_dir: Optional[Path] = None,   # Cache directory (default: ~/.abstractcore/embeddings)
    cache_size: int = 1000,             # Memory cache size (LRU)
    output_dims: Optional[int] = None,  # Matryoshka truncation dimension
    trust_remote_code: bool = False     # Trust remote code for HF models
)
```

**Provider-Specific Behavior:**

- **HuggingFace** (default): Local sentence-transformers models with optional ONNX acceleration
  - Automatic ONNX backend selection for compatible models
  - Support for Matryoshka dimension truncation
  - Model aliases from curated registry
  - Direct HuggingFace model IDs supported

- **Ollama**: Local embedding models via Ollama API
  - Requires Ollama server running
  - OpenAI-compatible API format
  - No ONNX acceleration (server-side inference)

- **LMStudio**: Local embedding models via LMStudio API
  - Requires LMStudio server running
  - OpenAI-compatible API format
  - No ONNX acceleration (server-side inference)

**Configuration Integration:**

The EmbeddingManager respects AbstractCore's configuration system:
- If `model` is None, uses `config.embeddings.model` (default: "all-minilm-l6-v2")
- If `provider` is None, uses `config.embeddings.provider` (default: "huggingface")
- Explicit parameters always override configuration defaults

#### Core Methods

**Single Text Embedding:**

```python
embed(text: str) -> List[float]
    # Generate embedding for single text with caching
    # - Memory cache (LRU, 1000 entries)
    # - Persistent disk cache
    # - Auto-save every 10 embeddings
    # - Returns zero vector for empty text
```

**Batch Embedding:**

```python
embed_batch(texts: List[str]) -> List[List[float]]
    # Efficient batch embedding generation
    # - Separates cached/uncached texts
    # - Batch processing for uncached texts
    # - Provider-optimized batching
    # - Automatic cache population
```

**Normalized Embedding:**

```python
embed_normalized(text: str) -> List[float]
    # Generate L2-normalized embedding (unit length)
    # - Separate cache for normalized embeddings
    # - Enables faster similarity with dot product
    # - 2x speedup for similarity computations
```

**Similarity Computation:**

```python
compute_similarity(text1: str, text2: str) -> float
    # Compute cosine similarity between two texts
    # Returns: -1.0 to 1.0

compute_similarity_direct(embedding1: List[float], embedding2: List[float]) -> float
    # Compute cosine similarity from embeddings directly
    # Useful when embeddings are pre-computed

compute_similarities(text: str, texts: List[str]) -> List[float]
    # Compute similarities between one text and multiple texts
    # Uses batch embedding for efficiency
```

**Matrix Operations:**

```python
compute_similarities_matrix(
    texts_left: List[str],
    texts_right: Optional[List[str]] = None,
    chunk_size: int = 500,
    normalized: bool = True,
    dtype: str = "float32",
    max_memory_gb: float = 4.0
) -> np.ndarray
    # Compute L×C similarity matrix
    # - Symmetric matrix if texts_right is None
    # - Pre-normalization for 2x speedup
    # - Chunked processing for memory efficiency
    # - Automatic memory management
    # Returns: NumPy array of shape (L, C)
```

**Clustering:**

```python
find_similar_clusters(
    texts: List[str],
    threshold: float = 0.8,
    min_cluster_size: int = 2,
    max_memory_gb: float = 4.0
) -> List[List[int]]
    # Find clusters of similar texts
    # - Uses similarity matrix + graph traversal
    # - Threshold-based clustering
    # - Returns list of text indices per cluster
    # - Useful for duplicate detection, grouping
```

**Utility Methods:**

```python
get_dimension() -> int
    # Get embedding dimension (respects output_dims)

estimate_tokens(text: str) -> int
    # Estimate token count for embedding

get_cache_stats() -> Dict[str, Any]
    # Get cache statistics and metadata

clear_cache()
    # Clear all caches (memory + disk)

save_caches()
    # Explicitly save caches to disk
```

#### Caching System

**Two-Layer Cache Architecture:**

1. **Memory Cache (L1):**
   - LRU cache via `@lru_cache` decorator
   - Default: 1000 most recent embeddings
   - Fast lookup, no I/O overhead
   - Cleared on process exit

2. **Persistent Cache (L2):**
   - Pickle-based disk storage
   - Location: `~/.abstractcore/embeddings/`
   - Automatic save every 10 embeddings
   - Separate cache per model and dimension
   - Survives process restarts

3. **Normalized Cache:**
   - Dedicated cache for normalized embeddings
   - Independent of regular cache
   - Enables performance optimization

**Cache File Naming:**
```
{provider}_{model_id}_{dim}_cache.pkl
{provider}_{model_id}_{dim}_normalized_cache.pkl
```

#### ONNX Backend Selection

The manager intelligently selects the best inference backend:

**Automatic Backend Selection:**
1. Check if `onnxruntime` is installed
2. Check if model has pre-exported ONNX files
3. Check if model type is ONNX-compatible
4. Fallback to PyTorch if ONNX unavailable or incompatible

**ONNX-Compatible Models:**
- Sentence-transformers models (all-MiniLM, all-MPNet, etc.)
- BERT-based models (bert-, distilbert-, roberta-)
- Popular embeddings (GTE, BGE)

**ONNX-Problematic Models:**
- Qwen models (export issues)
- LLaMA-based models
- Mixtral, DeepSeek, CodeLLaMA

**Performance Improvement:**
- ONNX typically provides 2-3x speedup
- No quality degradation
- Automatically tries optimized ONNX files (O3 optimization)
- Graceful fallback to PyTorch on failure

#### Event System Integration

When events are available, the manager emits:

- `embedding_generated`: New embedding created
- `embedding_cached`: Cache hit occurred
- `embedding_batch_generated`: Batch processing completed
- `similarity_matrix_computed`: Matrix computation finished
- `clustering_completed`: Clustering analysis done

Event data includes timing, model info, cache statistics.

## Supported Providers

### HuggingFace (Default)

Local sentence-transformers models with advanced features.

**Features:**
- ONNX acceleration (2-3x speedup)
- Matryoshka dimension truncation
- Curated model registry
- Pre-exported ONNX models
- Direct HuggingFace model ID support

**Installation:**
```bash
pip install sentence-transformers onnxruntime
```

**Usage:**
```python
from abstractcore.embeddings import EmbeddingManager

# Using curated model alias
embedder = EmbeddingManager(model="embeddinggemma", provider="huggingface")

# Using direct HuggingFace ID
embedder = EmbeddingManager(
    model="sentence-transformers/all-MiniLM-L6-v2",
    provider="huggingface"
)

# With ONNX backend (automatic)
embedder = EmbeddingManager(model="all-minilm-l6-v2", backend="auto")

# With Matryoshka truncation
embedder = EmbeddingManager(
    model="embeddinggemma",
    output_dims=256  # Truncate from 768 to 256
)
```

### Ollama

Local embedding models via Ollama API.

**Features:**
- Uses Ollama's embedding endpoint
- OpenAI-compatible format
- Server-side inference
- No model download (managed by Ollama)

**Prerequisites:**
```bash
# Start Ollama server
ollama serve

# Pull embedding model
ollama pull nomic-embed-text
```

**Usage:**
```python
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager(
    model="nomic-embed-text",
    provider="ollama"
)

embedding = embedder.embed("Hello world")
```

### LMStudio

Local embedding models via LMStudio API.

**Features:**
- Uses LMStudio's embedding endpoint
- OpenAI-compatible format
- Server-side inference
- GUI-based model management

**Prerequisites:**
1. Start LMStudio
2. Load an embedding model
3. Enable API server

**Usage:**
```python
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager(
    model="nomic-embed-text-v1.5",
    provider="lmstudio"
)

embedding = embedder.embed("Hello world")
```

## Usage Patterns

### Basic Embedding Generation

```python
from abstractcore.embeddings import EmbeddingManager

# Initialize with defaults (all-minilm-l6-v2, huggingface)
embedder = EmbeddingManager()

# Generate single embedding
text = "Artificial intelligence is transforming the world"
embedding = embedder.embed(text)
print(f"Dimension: {len(embedding)}")  # 384

# Generate batch embeddings (efficient)
texts = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural language processing enables text understanding"
]
embeddings = embedder.embed_batch(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### Similarity Search

```python
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager(model="embeddinggemma")

# Compare two texts
text1 = "Python is a programming language"
text2 = "Python is great for data science"
similarity = embedder.compute_similarity(text1, text2)
print(f"Similarity: {similarity:.3f}")  # 0.825

# Find most similar document
query = "How to learn machine learning?"
documents = [
    "Machine learning tutorial for beginners",
    "Advanced deep learning techniques",
    "Python programming basics",
    "Machine learning course with Python"
]

similarities = embedder.compute_similarities(query, documents)
best_match_idx = similarities.index(max(similarities))
print(f"Best match: {documents[best_match_idx]}")
print(f"Similarity: {similarities[best_match_idx]:.3f}")
```

### Document Clustering

```python
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager(model="all-minilm-l6-v2")

documents = [
    "Python programming tutorial",
    "Machine learning with Python",
    "JavaScript web development",
    "React framework guide",
    "Data science using Python",
    "Vue.js tutorial",
    "Python for data analysis"
]

# Find clusters of similar documents
clusters = embedder.find_similar_clusters(
    documents,
    threshold=0.7,
    min_cluster_size=2
)

for i, cluster in enumerate(clusters):
    print(f"\nCluster {i + 1}:")
    for idx in cluster:
        print(f"  - {documents[idx]}")

# Output:
# Cluster 1:
#   - Python programming tutorial
#   - Machine learning with Python
#   - Data science using Python
#   - Python for data analysis
# Cluster 2:
#   - JavaScript web development
#   - React framework guide
#   - Vue.js tutorial
```

### Similarity Matrix Operations

```python
from abstractcore.embeddings import EmbeddingManager
import numpy as np

embedder = EmbeddingManager(model="embeddinggemma")

# Compute symmetric similarity matrix
texts = [
    "AI and machine learning",
    "Deep learning neural networks",
    "Natural language processing",
    "Computer vision applications"
]

# Symmetric matrix (4x4)
matrix = embedder.compute_similarities_matrix(texts)
print(f"Matrix shape: {matrix.shape}")  # (4, 4)

# Find most similar pairs
n = len(texts)
for i in range(n):
    for j in range(i + 1, n):
        similarity = matrix[i, j]
        if similarity > 0.7:
            print(f"\n{texts[i]}")
            print(f"  ↔ {texts[j]}")
            print(f"  Similarity: {similarity:.3f}")

# Asymmetric matrix (queries vs documents)
queries = [
    "What is deep learning?",
    "How does NLP work?"
]
documents = [
    "Deep learning is a subset of machine learning",
    "Natural language processing enables text understanding",
    "Computer vision detects objects in images"
]

# Asymmetric matrix (2x3)
matrix = embedder.compute_similarities_matrix(queries, documents)
print(f"Query-document matrix: {matrix.shape}")  # (2, 3)

# Find best document for each query
for i, query in enumerate(queries):
    best_doc_idx = np.argmax(matrix[i])
    best_similarity = matrix[i, best_doc_idx]
    print(f"\nQuery: {query}")
    print(f"Best match: {documents[best_doc_idx]}")
    print(f"Similarity: {best_similarity:.3f}")
```

### Different Providers

```python
from abstractcore.embeddings import EmbeddingManager

# HuggingFace with ONNX acceleration
hf_embedder = EmbeddingManager(
    model="embeddinggemma",
    provider="huggingface",
    backend="onnx"
)

# Ollama (requires ollama serve + model pulled)
ollama_embedder = EmbeddingManager(
    model="nomic-embed-text",
    provider="ollama"
)

# LMStudio (requires LMStudio running + model loaded)
lmstudio_embedder = EmbeddingManager(
    model="nomic-embed-text-v1.5",
    provider="lmstudio"
)

# All have the same interface
text = "Embedding test"
embedding_hf = hf_embedder.embed(text)
embedding_ollama = ollama_embedder.embed(text)
embedding_lmstudio = lmstudio_embedder.embed(text)

# Compare dimensions
print(f"HuggingFace dimension: {len(embedding_hf)}")
print(f"Ollama dimension: {len(embedding_ollama)}")
print(f"LMStudio dimension: {len(embedding_lmstudio)}")
```

### Matryoshka Dimension Truncation

```python
from abstractcore.embeddings import EmbeddingManager

# Models with Matryoshka support: embeddinggemma, nomic-embed-v1.5, nomic-embed-v2-moe
embedder_full = EmbeddingManager(
    model="embeddinggemma"  # Full 768 dimensions
)

embedder_truncated = EmbeddingManager(
    model="embeddinggemma",
    output_dims=256  # Truncated to 256 dimensions
)

text = "Matryoshka embedding test"

full_embedding = embedder_full.embed(text)
truncated_embedding = embedder_truncated.embed(text)

print(f"Full dimension: {len(full_embedding)}")  # 768
print(f"Truncated dimension: {len(truncated_embedding)}")  # 256

# Truncated embeddings are faster to compute and store
# Trade-off: slight quality reduction for significant performance gain
```

### Normalized Embeddings for Performance

```python
from abstractcore.embeddings import EmbeddingManager
import numpy as np

embedder = EmbeddingManager(model="all-minilm-l6-v2")

# Regular embeddings
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = [embedder.embed(text) for text in texts]

# Normalized embeddings (unit length)
normalized = [embedder.embed_normalized(text) for text in texts]

# Verify normalization
norms = [np.linalg.norm(emb) for emb in normalized]
print(f"Norms: {norms}")  # All close to 1.0

# Fast similarity with dot product (instead of cosine)
# For normalized embeddings: cosine_similarity = dot_product
def fast_similarity(norm_emb1, norm_emb2):
    return np.dot(norm_emb1, norm_emb2)

similarity = fast_similarity(normalized[0], normalized[1])
print(f"Fast similarity: {similarity:.3f}")

# 2x faster than full cosine similarity computation
```

### Cache Management

```python
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager(
    model="all-minilm-l6-v2",
    cache_size=5000  # Increase memory cache
)

# Generate some embeddings
texts = [f"Document {i}" for i in range(100)]
embeddings = embedder.embed_batch(texts)

# Check cache statistics
stats = embedder.get_cache_stats()
print(f"Provider: {stats['provider']}")
print(f"Model: {stats['model_id']}")
print(f"Dimension: {stats['embedding_dimension']}")
print(f"Persistent cache size: {stats['persistent_cache_size']}")
print(f"Memory cache: {stats['memory_cache_info']}")

# Explicitly save caches to disk
embedder.save_caches()

# Clear all caches
embedder.clear_cache()

# Cache files location
print(f"Cache file: {stats['cache_file']}")
print(f"Normalized cache: {stats['normalized_cache_file']}")
```

### Configuration Integration

```python
from abstractcore import set_embeddings_config
from abstractcore.embeddings import EmbeddingManager

# Set global defaults via configuration
set_embeddings_config(
    provider="huggingface",
    model="embeddinggemma"
)

# EmbeddingManager respects config defaults
embedder = EmbeddingManager()  # Uses embeddinggemma via config

# Explicit parameters override config
embedder_override = EmbeddingManager(
    model="all-minilm-l6-v2"  # Override config default
)

print(f"Default model: {embedder.model_id}")
print(f"Override model: {embedder_override.model_id}")
```

## Integration Points

### RAG Applications

```python
from abstractcore.embeddings import EmbeddingManager

# Initialize embedder
embedder = EmbeddingManager(model="embeddinggemma")

# Index documents
knowledge_base = [
    "Python is a high-level programming language",
    "Machine learning enables computers to learn from data",
    "RAG combines retrieval and generation for better answers"
]

# Pre-compute document embeddings
doc_embeddings = embedder.embed_batch(knowledge_base)

# Query processing
query = "What is machine learning?"
query_embedding = embedder.embed(query)

# Find most relevant documents
similarities = [
    embedder.compute_similarity_direct(query_embedding, doc_emb)
    for doc_emb in doc_embeddings
]

# Get top-k documents
top_k = 2
top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

print("Most relevant documents:")
for idx in top_indices:
    print(f"  - {knowledge_base[idx]} (similarity: {similarities[idx]:.3f})")
```

### Semantic Caching

```python
from abstractcore.embeddings import EmbeddingManager

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.embedder = EmbeddingManager(model="all-minilm-l6-v2")
        self.threshold = similarity_threshold
        self.cache = {}  # query_embedding -> response

    def get(self, query: str):
        query_emb = self.embedder.embed(query)

        # Check for semantically similar cached queries
        for cached_emb, response in self.cache.items():
            similarity = self.embedder.compute_similarity_direct(query_emb, cached_emb)
            if similarity >= self.threshold:
                return response

        return None

    def set(self, query: str, response: str):
        query_emb = self.embedder.embed(query)
        self.cache[tuple(query_emb)] = response

# Usage
cache = SemanticCache()

# Cache response
cache.set("What is AI?", "AI is artificial intelligence...")

# Similar query returns cached response
response = cache.get("What is artificial intelligence?")
print(response)  # "AI is artificial intelligence..."
```

### Duplicate Detection

```python
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager(model="all-minilm-l6-v2")

documents = [
    "Machine learning is a subset of AI",
    "AI includes machine learning as a subfield",  # Near-duplicate
    "Python is a programming language",
    "ML is part of artificial intelligence",  # Near-duplicate
    "JavaScript for web development"
]

# Find duplicate clusters
clusters = embedder.find_similar_clusters(
    documents,
    threshold=0.85,  # High threshold for duplicates
    min_cluster_size=2
)

print("Potential duplicate groups:")
for i, cluster in enumerate(clusters):
    print(f"\nGroup {i + 1}:")
    for idx in cluster:
        print(f"  - {documents[idx]}")
```

## Best Practices

### Model Selection

**For Development/Testing:**
- Use `all-minilm-l6-v2` (default) - Fast, lightweight, good quality
- Use `granite-30m` - Ultra-lightweight, fastest inference

**For Production (English):**
- Use `nomic-embed-v1.5` - High quality, Matryoshka support
- Use `embeddinggemma` - SOTA quality, multilingual capable

**For Multilingual:**
- Use `embeddinggemma` - Google's 2025 SOTA, 8K context
- Use `granite-107m` or `granite-278m` - Balanced multilingual
- Use `qwen3-embedding` - Efficient 1024-dim multilingual

**For Memory-Constrained:**
- Use `granite-30m` (30MB) or `all-minilm-l6-v2` (90MB)
- Enable Matryoshka truncation for supported models
- Use `output_dims` to reduce embedding size

**For Maximum Quality:**
- Use `embeddinggemma` (300M params, 768 dims)
- Use `nomic-embed-v2-moe` (MoE architecture)
- Use full dimensions, no truncation

### Batch Processing

Always use `embed_batch()` for multiple texts:

```python
# ❌ BAD: Inefficient loop
embeddings = [embedder.embed(text) for text in texts]

# ✅ GOOD: Efficient batch processing
embeddings = embedder.embed_batch(texts)
```

Batch processing provides:
- 5-10x faster than individual calls
- Better cache utilization
- Provider-optimized batching
- Automatic cache population

### Cache Configuration

**Increase cache size for large-scale applications:**

```python
embedder = EmbeddingManager(
    model="embeddinggemma",
    cache_size=10000  # Larger memory cache
)
```

**Custom cache directory for shared caching:**

```python
from pathlib import Path

embedder = EmbeddingManager(
    model="all-minilm-l6-v2",
    cache_dir=Path("/shared/embeddings/cache")
)
```

**Periodic cache saves for reliability:**

```python
# Caches auto-save every 10 embeddings
# For critical applications, explicitly save
embedder.save_caches()
```

### Normalization for Performance

Use normalized embeddings when computing many similarities:

```python
# For repeated similarity computations
texts = [...]  # Large corpus

# Pre-compute normalized embeddings once
normalized_embeddings = [embedder.embed_normalized(text) for text in texts]

# Fast similarity using dot product (2x faster)
import numpy as np
query_norm = np.array(embedder.embed_normalized(query))
similarities = [np.dot(query_norm, emb) for emb in normalized_embeddings]
```

### Matrix Operations for Scale

Use matrix operations for large-scale similarity computations:

```python
# ❌ BAD: Nested loops for 1000x1000 = 1M similarities
similarities = []
for text1 in texts:
    row = []
    for text2 in texts:
        sim = embedder.compute_similarity(text1, text2)
        row.append(sim)
    similarities.append(row)

# ✅ GOOD: Vectorized matrix computation (100x faster)
matrix = embedder.compute_similarities_matrix(texts)
```

Matrix operations provide:
- Vectorized NumPy operations (100x faster)
- Automatic memory management
- Chunked processing for large matrices
- Pre-normalization optimization

### Dimension Truncation Strategy

For Matryoshka-capable models, use dimension truncation strategically:

```python
# Full dimensions (768) - Maximum quality
embedder_full = EmbeddingManager(model="embeddinggemma")

# Truncated (256) - 3x faster, ~5% quality loss
embedder_fast = EmbeddingManager(model="embeddinggemma", output_dims=256)

# Use case based selection:
# - User-facing search: full dimensions
# - Background clustering: truncated dimensions
# - Large-scale indexing: truncated dimensions
# - Critical matching: full dimensions
```

### Provider Selection Strategy

**HuggingFace:**
- Best for offline/local deployments
- ONNX acceleration available
- Full control over model selection
- No API dependencies

**Ollama:**
- Best for unified LLM + embedding workflow
- Easy model management (`ollama pull`)
- Good for development/prototyping
- Requires Ollama server

**LMStudio:**
- Best for GUI-based workflows
- Easy model switching
- Good for experimentation
- Requires LMStudio running

## Common Pitfalls

### Dimension Mismatches

**Problem:** Comparing embeddings from different models or dimensions.

```python
# ❌ WRONG: Different models/dimensions
embedder1 = EmbeddingManager(model="all-minilm-l6-v2")  # 384 dims
embedder2 = EmbeddingManager(model="embeddinggemma")     # 768 dims

emb1 = embedder1.embed("Text 1")
emb2 = embedder2.embed("Text 2")

# This will fail or give meaningless results
similarity = embedder1.compute_similarity_direct(emb1, emb2)
```

**Solution:** Use consistent model and dimensions.

```python
# ✅ CORRECT: Same model
embedder = EmbeddingManager(model="embeddinggemma")
emb1 = embedder.embed("Text 1")
emb2 = embedder.embed("Text 2")
similarity = embedder.compute_similarity_direct(emb1, emb2)
```

### Missing Dependencies

**Problem:** ONNX acceleration fails silently if `onnxruntime` not installed.

```python
# Fails to use ONNX, falls back to PyTorch without clear indication
embedder = EmbeddingManager(model="all-minilm-l6-v2", backend="onnx")
```

**Solution:** Install optional dependencies and check backend.

```bash
pip install onnxruntime
```

```python
embedder = EmbeddingManager(model="all-minilm-l6-v2")
stats = embedder.get_cache_stats()
print(f"Backend: {stats['backend']}")  # Verify ONNX is used
```

### Empty Text Handling

**Problem:** Unexpected zero vectors for empty strings.

```python
# Returns zero vector, not error
embedding = embedder.embed("")  # [0.0, 0.0, ..., 0.0]
```

**Solution:** Validate input text before embedding.

```python
def safe_embed(text: str):
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    return embedder.embed(text)
```

### Cache Pollution

**Problem:** Testing with garbage data pollutes persistent cache.

```python
# This caches garbage embeddings
test_texts = ["asdfgh", "qwerty", "zxcvbn"]
embeddings = embedder.embed_batch(test_texts)
```

**Solution:** Clear cache after testing or use separate cache directory.

```python
# Test with isolated cache
test_embedder = EmbeddingManager(
    model="all-minilm-l6-v2",
    cache_dir=Path("/tmp/test_embeddings")
)

# Or clear after testing
embedder.clear_cache()
```

### Memory Issues with Large Matrices

**Problem:** Computing huge similarity matrices exhausts memory.

```python
# 10,000 x 10,000 = 100M float32 values = 400MB + overhead
huge_texts = [f"Document {i}" for i in range(10000)]
matrix = embedder.compute_similarities_matrix(huge_texts)  # May OOM
```

**Solution:** Use chunked processing or reduce chunk_size.

```python
# Chunked processing with memory limit
matrix = embedder.compute_similarities_matrix(
    huge_texts,
    chunk_size=500,      # Process 500 rows at a time
    max_memory_gb=2.0    # Limit memory usage
)
```

### Provider Availability

**Problem:** Ollama/LMStudio provider fails if server not running.

```python
# Fails if Ollama server not running
embedder = EmbeddingManager(model="nomic-embed-text", provider="ollama")
embedding = embedder.embed("Test")  # ConnectionError
```

**Solution:** Verify server is running or use try-except with fallback.

```python
try:
    embedder = EmbeddingManager(model="nomic-embed-text", provider="ollama")
except Exception:
    # Fallback to HuggingFace
    embedder = EmbeddingManager(model="all-minilm-l6-v2", provider="huggingface")
```

### Matryoshka on Incompatible Models

**Problem:** Specifying output_dims on non-Matryoshka models.

```python
# all-minilm-l6-v2 doesn't support Matryoshka
embedder = EmbeddingManager(
    model="all-minilm-l6-v2",
    output_dims=128  # Ignored with warning
)
```

**Solution:** Use Matryoshka-capable models or check support.

```python
from abstractcore.embeddings import get_model_config

config = get_model_config("embeddinggemma")
if config.supports_matryoshka:
    embedder = EmbeddingManager(
        model="embeddinggemma",
        output_dims=256  # Valid truncation
    )
```

## Testing Strategy

### Unit Testing Embeddings

```python
import pytest
from abstractcore.embeddings import EmbeddingManager

def test_basic_embedding():
    embedder = EmbeddingManager(model="all-minilm-l6-v2")

    # Test single embedding
    text = "Test embedding"
    embedding = embedder.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-minilm-l6-v2 dimension
    assert all(isinstance(x, float) for x in embedding)

def test_batch_embedding():
    embedder = EmbeddingManager(model="all-minilm-l6-v2")

    texts = ["Text 1", "Text 2", "Text 3"]
    embeddings = embedder.embed_batch(texts)

    assert len(embeddings) == len(texts)
    assert all(len(emb) == 384 for emb in embeddings)

def test_similarity_computation():
    embedder = EmbeddingManager(model="all-minilm-l6-v2")

    # Identical texts should have similarity ~1.0
    text = "Test text"
    similarity = embedder.compute_similarity(text, text)
    assert 0.99 <= similarity <= 1.01

    # Different texts should have lower similarity
    similarity = embedder.compute_similarity("AI", "cooking")
    assert similarity < 0.5

def test_empty_text_handling():
    embedder = EmbeddingManager(model="all-minilm-l6-v2")

    # Empty text returns zero vector
    embedding = embedder.embed("")
    assert embedding == [0.0] * 384

def test_cache_functionality():
    embedder = EmbeddingManager(model="all-minilm-l6-v2")

    # First call computes embedding
    text = "Cache test"
    emb1 = embedder.embed(text)

    # Second call uses cache (identical result)
    emb2 = embedder.embed(text)
    assert emb1 == emb2

    # Cache stats should show hits
    stats = embedder.get_cache_stats()
    assert stats['persistent_cache_size'] > 0
```

### Integration Testing

```python
def test_rag_workflow():
    """Test realistic RAG workflow."""
    embedder = EmbeddingManager(model="embeddinggemma")

    # Knowledge base
    docs = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "RAG combines retrieval and generation"
    ]

    # Index documents
    doc_embeddings = embedder.embed_batch(docs)

    # Query
    query = "What is machine learning?"
    query_emb = embedder.embed(query)

    # Retrieve relevant documents
    similarities = [
        embedder.compute_similarity_direct(query_emb, doc_emb)
        for doc_emb in doc_embeddings
    ]

    # Most relevant should be doc[1]
    best_idx = similarities.index(max(similarities))
    assert best_idx == 1
    assert similarities[best_idx] > 0.5

def test_clustering_workflow():
    """Test document clustering."""
    embedder = EmbeddingManager(model="all-minilm-l6-v2")

    docs = [
        "Python programming",
        "Python coding",
        "JavaScript development",
        "JavaScript programming"
    ]

    clusters = embedder.find_similar_clusters(
        docs,
        threshold=0.7,
        min_cluster_size=2
    )

    # Should find 2 clusters (Python and JavaScript)
    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert len(clusters[1]) == 2
```

### Performance Testing

```python
import time

def test_onnx_performance():
    """Verify ONNX provides speedup."""
    embedder_pytorch = EmbeddingManager(
        model="all-minilm-l6-v2",
        backend="pytorch"
    )

    embedder_onnx = EmbeddingManager(
        model="all-minilm-l6-v2",
        backend="onnx"
    )

    texts = ["Test text"] * 100

    # PyTorch timing
    start = time.time()
    embedder_pytorch.embed_batch(texts)
    pytorch_time = time.time() - start

    # ONNX timing
    start = time.time()
    embedder_onnx.embed_batch(texts)
    onnx_time = time.time() - start

    # ONNX should be faster
    speedup = pytorch_time / onnx_time
    print(f"ONNX speedup: {speedup:.2f}x")
    assert speedup > 1.5  # Expect at least 1.5x speedup

def test_batch_performance():
    """Verify batch is faster than sequential."""
    embedder = EmbeddingManager(model="all-minilm-l6-v2")

    texts = ["Test text"] * 50

    # Sequential timing
    start = time.time()
    for text in texts:
        embedder.embed(text)
    sequential_time = time.time() - start

    # Clear cache for fair comparison
    embedder.clear_cache()

    # Batch timing
    start = time.time()
    embedder.embed_batch(texts)
    batch_time = time.time() - start

    # Batch should be significantly faster
    speedup = sequential_time / batch_time
    print(f"Batch speedup: {speedup:.2f}x")
    assert speedup > 5  # Expect at least 5x speedup
```

## Public API

### Recommended Imports

```python
# Core embedding manager
from abstractcore.embeddings import EmbeddingManager

# Model configuration utilities
from abstractcore.embeddings import (
    EmbeddingModelConfig,
    get_model_config,
    list_available_models
)
```

### Complete Public API

**EmbeddingManager:**
- `__init__(model, provider, backend, cache_dir, cache_size, output_dims, trust_remote_code)`
- `embed(text) -> List[float]`
- `embed_batch(texts) -> List[List[float]]`
- `embed_normalized(text) -> List[float]`
- `compute_similarity(text1, text2) -> float`
- `compute_similarity_direct(embedding1, embedding2) -> float`
- `compute_similarities(text, texts) -> List[float]`
- `compute_similarities_matrix(texts_left, texts_right, ...) -> np.ndarray`
- `find_similar_clusters(texts, threshold, min_cluster_size, ...) -> List[List[int]]`
- `get_dimension() -> int`
- `estimate_tokens(text) -> int`
- `get_cache_stats() -> Dict[str, Any]`
- `clear_cache()`
- `save_caches()`

**Model Configuration:**
- `get_model_config(model_name) -> EmbeddingModelConfig`
- `list_available_models() -> List[str]`
- `get_default_model() -> str`

**EmbeddingModelConfig:**
- `name: str`
- `model_id: str`
- `dimension: int`
- `max_sequence_length: int`
- `supports_matryoshka: bool`
- `matryoshka_dims: Optional[List[int]]`
- `description: str`
- `multilingual: bool`
- `size_mb: Optional[float]`

---

**Version:** AbstractCore 2.x
**Last Updated:** 2025-01
**Maintainer:** AbstractCore Team

## Related Modules

**Direct dependencies**:
- [`providers/`](../providers/README.md) - Provider embedding implementations
- [`core/`](../core/README.md) - Factory pattern for embedding creation
- [`config/`](../config/README.md) - Default embedding configuration
- [`exceptions/`](../exceptions/README.md) - Embedding error handling

**Used by**:
- [`processing/`](../processing/README.md) - Document processing with embeddings
- [`apps/`](../apps/README.md) - RAG and semantic search applications
- [`server/`](../server/README.md) - Embedding API endpoints

**Related systems**:
- [`utils/`](../utils/README.md) - Vector utilities, normalization
