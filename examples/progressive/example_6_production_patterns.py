#!/usr/bin/env python3
"""
Example 6: Production Patterns - Building Robust AI Applications
=================================================================

This example demonstrates production-ready patterns for AbstractLLM:
- Session management and conversation history
- Structured output with validation
- Embeddings and RAG patterns
- Event-driven architectures
- Testing and monitoring strategies
- Cost optimization techniques

Technical Architecture Highlights:
- BasicSession for conversation management
- Structured output with Pydantic
- EmbeddingManager for vector operations
- Event system for observability
- Production testing patterns

Required: pip install abstractllm[all]
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

# Add project root to path for development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from abstractllm import create_llm, BasicSession, GenerateResponse
from abstractllm.events import EventType, subscribe, unsubscribe_all
from abstractllm.processing import BasicSummarizer, SummaryStyle, SummaryLength
from abstractllm.tools import ToolDefinition, tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def session_management_patterns():
    """
    Demonstrates session management for multi-turn conversations.

    Architecture Notes:
    - BasicSession manages conversation history
    - Automatic context window management
    - Token counting and truncation strategies
    """
    print("=" * 70)
    print("EXAMPLE 6: Session Management Patterns")
    print("=" * 70)

    # Create LLM and session
    llm = create_llm("mock", "mock-model", max_tokens=2048)
    session = BasicSession(llm)

    print("\n🗣️ Multi-Turn Conversation Management:")

    # Conversation flow
    conversation = [
        ("What is machine learning?", "system"),
        ("Machine learning is a branch of AI...", "assistant"),
        ("Can you give me an example?", "user"),
        ("Sure! Image recognition is a great example...", "assistant"),
        ("How does it work?", "user"),
    ]

    print("\n📝 Building conversation history:")
    for i, (content, role) in enumerate(conversation[:-1], 1):
        session.add_message(role, content)
        print(f"   {i}. [{role:9s}] {content[:40]}...")

    # Generate with context
    print("\n🤖 Generating with context:")
    response = session.generate(conversation[-1][0])
    print(f"   Response: {response.content[:100]}...")

    # Context window management
    print("\n📊 Context Window Management:")
    print(f"   • Total messages: {len(session.messages)}")
    print(f"   • Estimated tokens: ~{len(str(session.messages)) * 0.25:.0f}")
    print(f"   • Max context: {llm.max_tokens if hasattr(llm, 'max_tokens') else 'N/A'}")

    # Truncation strategies
    print("\n✂️ Truncation Strategies:")
    strategies = [
        ("FIFO", "Remove oldest messages first"),
        ("Summarize", "Summarize older messages"),
        ("Importance", "Keep important messages, drop filler"),
        ("Sliding Window", "Keep last N messages"),
    ]

    for strategy, description in strategies:
        print(f"   • {strategy:15s}: {description}")

    # Session persistence
    print("\n💾 Session Persistence:")
    print("   ```python")
    print("   # Save session")
    print("   session_data = session.to_dict()")
    print("   with open('session.json', 'w') as f:")
    print("       json.dump(session_data, f)")
    print("   ")
    print("   # Load session")
    print("   with open('session.json') as f:")
    print("       session_data = json.load(f)")
    print("   session = BasicSession.from_dict(session_data, llm)")
    print("   ```")


def structured_output_patterns():
    """
    Demonstrates structured output with validation.

    Architecture Notes:
    - Pydantic models for type safety
    - JSON schema generation
    - Output validation and retry
    """
    print("\n" + "=" * 70)
    print("Structured Output Patterns")
    print("=" * 70)

    from pydantic import BaseModel, Field, validator
    from typing import Literal

    # Define structured output models
    class ProductAnalysis(BaseModel):
        """Product analysis structure."""
        name: str = Field(description="Product name")
        category: Literal["electronics", "clothing", "food", "other"]
        price_range: Literal["budget", "mid-range", "premium"]
        sentiment: Literal["positive", "neutral", "negative"]
        key_features: List[str] = Field(max_items=5)
        score: float = Field(ge=0, le=10)

        @validator('key_features')
        def validate_features(cls, v):
            if len(v) < 2:
                raise ValueError("At least 2 features required")
            return v

    print("\n📋 Structured Output Schema:")
    print(json.dumps(ProductAnalysis.schema(), indent=2)[:300] + "...")

    # Generate structured output
    print("\n🎯 Generating Structured Output:")

    def generate_structured(llm, model_class: BaseModel, prompt: str):
        """Generate and validate structured output."""
        # Add JSON schema to prompt
        schema_prompt = f"""
{prompt}

Return your response as valid JSON matching this schema:
{json.dumps(model_class.schema(), indent=2)}
"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.generate(schema_prompt)

                # Parse and validate
                # In production, you'd extract JSON from response
                mock_json = {
                    "name": "iPhone 15 Pro",
                    "category": "electronics",
                    "price_range": "premium",
                    "sentiment": "positive",
                    "key_features": ["Titanium design", "A17 Pro chip", "Action button"],
                    "score": 9.2
                }

                result = model_class(**mock_json)
                print(f"   ✅ Valid output generated (attempt {attempt + 1})")
                return result

            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise

    llm = create_llm("mock", "mock-model")
    analysis = generate_structured(
        llm,
        ProductAnalysis,
        "Analyze the iPhone 15 Pro"
    )

    print(f"\n📦 Structured Result:")
    print(f"   Product: {analysis.name}")
    print(f"   Category: {analysis.category}")
    print(f"   Sentiment: {analysis.sentiment}")
    print(f"   Score: {analysis.score}/10")


def embeddings_and_rag_patterns():
    """
    Demonstrates embeddings and RAG patterns.

    Architecture Notes:
    - EmbeddingManager for vector operations
    - Similarity search strategies
    - RAG pipeline implementation
    """
    print("\n" + "=" * 70)
    print("Embeddings & RAG Patterns")
    print("=" * 70)

    try:
        from abstractllm.embeddings import EmbeddingManager
        embeddings_available = True
    except ImportError:
        print("   ⚠️ Embeddings module not available")
        embeddings_available = False

    if embeddings_available:
        print("\n🔍 RAG (Retrieval-Augmented Generation) Pipeline:")
        print("""
        Documents ──► Chunking ──► Embedding ──► Vector Store
                                                      │
        Query ──► Embedding ──► Similarity Search ◄──┘
                                      │
                                      ▼
                              Retrieved Context
                                      │
                                      ▼
                            LLM + Context ──► Answer
        """)

        # Example RAG implementation
        print("\n📚 Example RAG Implementation:")
        print("""
```python
class SimpleRAG:
    def __init__(self, llm, embedding_model="BAAI/bge-small-en-v1.5"):
        self.llm = llm
        self.embeddings = EmbeddingManager(model_name=embedding_model)
        self.documents = []
        self.vectors = []

    def add_document(self, text: str):
        # Chunk document
        chunks = self.chunk_text(text, chunk_size=500)

        # Generate embeddings
        for chunk in chunks:
            vector = self.embeddings.embed_text(chunk)
            self.documents.append(chunk)
            self.vectors.append(vector)

    def query(self, question: str, k=3):
        # Embed query
        query_vector = self.embeddings.embed_text(question)

        # Find similar documents
        similarities = self.compute_similarities(query_vector, self.vectors)
        top_k_indices = sorted(range(len(similarities)),
                              key=lambda i: similarities[i],
                              reverse=True)[:k]

        # Build context
        context = "\\n\\n".join([self.documents[i] for i in top_k_indices])

        # Generate answer with context
        prompt = f'''Context: {context}

Question: {question}

Answer based on the context:'''

        return self.llm.generate(prompt)

    def chunk_text(self, text, chunk_size=500, overlap=50):
        # Simple chunking with overlap
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def compute_similarities(self, query_vec, doc_vecs):
        # Cosine similarity
        import numpy as np
        similarities = []
        for doc_vec in doc_vecs:
            sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            similarities.append(sim)
        return similarities
```
        """)

    # Advanced embedding strategies
    print("\n🎯 Advanced Embedding Strategies:")
    strategies = [
        ("Hybrid Search", "Combine vector + keyword search"),
        ("Multi-Vector", "Multiple embeddings per document"),
        ("Cross-Encoder", "Re-rank retrieved documents"),
        ("Query Expansion", "Generate multiple query variants"),
        ("Hierarchical", "Embed at different granularities"),
    ]

    for strategy, description in strategies:
        print(f"   • {strategy:15s}: {description}")


def event_driven_architecture():
    """
    Demonstrates event-driven patterns for observability.

    Architecture Notes:
    - Event system for decoupled monitoring
    - Metrics collection and aggregation
    - Audit logging and compliance
    """
    print("\n" + "=" * 70)
    print("Event-Driven Architecture Patterns")
    print("=" * 70)

    print("\n📡 Event System Architecture:")
    print("""
    LLM Operations ──► Event Bus ──► Event Handlers
                           │
                           ├──► Metrics Collector
                           ├──► Audit Logger
                           ├──► Error Handler
                           └──► Custom Handlers
    """)

    # Create comprehensive event handlers
    @dataclass
    class MetricsCollector:
        """Collect and aggregate metrics."""
        total_requests: int = 0
        total_tokens: int = 0
        total_errors: int = 0
        latencies: List[float] = None

        def __post_init__(self):
            if self.latencies is None:
                self.latencies = []

        def handle_event(self, event_data: Dict[str, Any]):
            event_type = event_data.get("type")

            if event_type == EventType.GENERATION_STARTED.value:
                self.total_requests += 1

            elif event_type == EventType.GENERATION_COMPLETED.value:
                if "usage" in event_data:
                    self.total_tokens += event_data["usage"].get("total_tokens", 0)
                if "duration" in event_data:
                    self.latencies.append(event_data["duration"])

            elif event_type == EventType.GENERATION_ERROR.value:
                self.total_errors += 1

        def get_summary(self):
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_errors": self.total_errors,
                "avg_latency_ms": avg_latency * 1000,
                "error_rate": self.total_errors / self.total_requests if self.total_requests else 0
            }

    # Set up event handling
    metrics = MetricsCollector()

    # Subscribe to events
    subscribe(EventType.GENERATION_STARTED, metrics.handle_event)
    subscribe(EventType.GENERATION_COMPLETED, metrics.handle_event)
    subscribe(EventType.GENERATION_ERROR, metrics.handle_event)

    print("\n📊 Testing Event System:")
    llm = create_llm("mock", "mock-model")

    # Generate some requests
    test_prompts = [
        "What is AI?",
        "Explain ML",
        "Define NLP",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"   Request {i}: '{prompt}'")
        try:
            response = llm.generate(prompt)
            # Simulate token usage
            if not response.usage:
                response.usage = {"total_tokens": len(prompt.split()) * 10}
        except Exception as e:
            print(f"      Error: {e}")

    # Display metrics
    summary = metrics.get_summary()
    print("\n📈 Metrics Summary:")
    for key, value in summary.items():
        print(f"   • {key}: {value}")

    # Clean up
    unsubscribe_all()


def cost_optimization_patterns():
    """
    Demonstrates cost optimization strategies.

    Architecture Notes:
    - Model selection based on task complexity
    - Caching strategies
    - Batch processing optimization
    """
    print("\n" + "=" * 70)
    print("Cost Optimization Patterns")
    print("=" * 70)

    print("\n💰 Cost Optimization Strategies:")

    # Strategy 1: Model Routing
    print("\n1️⃣ Intelligent Model Routing:")

    class CostOptimizedRouter:
        """Route requests to appropriate models based on complexity."""

        def __init__(self):
            self.models = {
                "simple": ("mock", "mock-mini", 0.0001),     # $0.0001/1K tokens
                "medium": ("mock", "mock-standard", 0.001),  # $0.001/1K tokens
                "complex": ("mock", "mock-pro", 0.01),       # $0.01/1K tokens
            }

        def classify_complexity(self, prompt: str) -> str:
            """Classify prompt complexity."""
            # Simple heuristics (in production, use ML classifier)
            word_count = len(prompt.split())
            has_code = "```" in prompt or "function" in prompt
            has_analysis = any(word in prompt.lower() for word in
                             ["analyze", "compare", "evaluate", "explain"])

            if has_code or has_analysis or word_count > 100:
                return "complex"
            elif word_count > 20:
                return "medium"
            else:
                return "simple"

        def route(self, prompt: str):
            """Route to appropriate model."""
            complexity = self.classify_complexity(prompt)
            provider, model, cost = self.models[complexity]

            print(f"   Complexity: {complexity}")
            print(f"   Model: {model} (${cost}/1K tokens)")

            return create_llm(provider, model)

    router = CostOptimizedRouter()
    test_prompts = [
        "Hi",  # Simple
        "What is the capital of France?",  # Medium
        "Analyze this code and suggest improvements: def f(x): return x*2",  # Complex
    ]

    print("\n   Testing router:")
    for prompt in test_prompts:
        print(f"\n   Prompt: '{prompt[:50]}...'")
        llm = router.route(prompt)

    # Strategy 2: Response Caching
    print("\n\n2️⃣ Response Caching:")

    class CachedLLM:
        """LLM wrapper with caching."""

        def __init__(self, llm, cache_ttl=3600):
            self.llm = llm
            self.cache = {}
            self.cache_ttl = cache_ttl
            self.hits = 0
            self.misses = 0

        def generate(self, prompt: str) -> str:
            # Generate cache key
            cache_key = hash(prompt)

            # Check cache
            if cache_key in self.cache:
                timestamp, response = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self.hits += 1
                    print(f"   💾 Cache hit! Saved API call.")
                    return response

            # Cache miss - generate new response
            self.misses += 1
            response = self.llm.generate(prompt)
            self.cache[cache_key] = (time.time(), response)
            return response

        def get_stats(self):
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1%}",
                "cost_savings": f"${self.hits * 0.001:.4f}"  # Assuming $0.001 per call
            }

    # Test caching
    llm = create_llm("mock", "mock-model")
    cached_llm = CachedLLM(llm)

    print("\n   Testing cache:")
    test_queries = [
        "What is AI?",
        "What is AI?",  # Duplicate
        "What is ML?",
        "What is AI?",  # Duplicate again
    ]

    for query in test_queries:
        print(f"   Query: '{query}'")
        cached_llm.generate(query)

    stats = cached_llm.get_stats()
    print(f"\n   📊 Cache Stats:")
    for key, value in stats.items():
        print(f"      • {key}: {value}")

    # Strategy 3: Batch Processing
    print("\n3️⃣ Batch Processing Optimization:")
    print("   • Batch similar requests together")
    print("   • Use batch APIs when available")
    print("   • Reduces per-request overhead")
    print("   • Example savings: 10 requests → 1 batch = 30% cost reduction")


def testing_patterns():
    """
    Demonstrates testing patterns for LLM applications.

    Architecture Notes:
    - Mock providers for unit testing
    - Integration testing strategies
    - Performance benchmarking
    """
    print("\n" + "=" * 70)
    print("Testing Patterns for LLM Applications")
    print("=" * 70)

    print("\n🧪 Testing Strategy Layers:")
    print("""
    ┌─────────────────────────────────┐
    │     End-to-End Tests           │ ← Real providers
    ├─────────────────────────────────┤
    │     Integration Tests          │ ← Mock + real mix
    ├─────────────────────────────────┤
    │     Unit Tests                 │ ← Mock providers
    └─────────────────────────────────┘
    """)

    # Unit testing example
    print("\n1️⃣ Unit Testing with Mock Provider:")
    print("""
```python
import pytest
from abstractllm import create_llm

def test_generation():
    # Mock provider for deterministic testing
    llm = create_llm("mock", "mock-model")

    response = llm.generate("test prompt")

    assert response is not None
    assert response.content != ""
    assert response.model == "mock-model"

def test_tool_calling():
    llm = create_llm("mock", "mock-model")

    tools = [{"name": "calculator", "description": "Calculate math"}]
    response = llm.generate_with_tools("Calculate 2+2", tools)

    assert "calculator" in str(response)

@pytest.mark.parametrize("prompt,expected", [
    ("short", True),
    ("very" * 1000, False),  # Too long
])
def test_token_limits(prompt, expected):
    llm = create_llm("mock", "mock-model", max_tokens=100)

    try:
        response = llm.generate(prompt)
        assert expected == True
    except Exception:
        assert expected == False
```
    """)

    # Integration testing
    print("\n2️⃣ Integration Testing Patterns:")
    print("""
```python
@pytest.mark.integration
class TestProviderIntegration:
    @pytest.fixture
    def llm_factory(self):
        '''Factory for creating test LLMs.'''
        def _create(provider="mock", **kwargs):
            return create_llm(provider, **kwargs)
        return _create

    def test_streaming_integration(self, llm_factory):
        llm = llm_factory(stream=True)

        chunks = []
        for chunk in llm.stream_generate("test"):
            chunks.append(chunk)

        assert len(chunks) > 1
        assert all(c.content for c in chunks)

    def test_session_persistence(self, llm_factory):
        llm = llm_factory()
        session = BasicSession(llm)

        session.add_message("user", "Hello")
        session.generate("How are you?")

        # Save and reload
        data = session.to_dict()
        new_session = BasicSession.from_dict(data, llm)

        assert len(new_session.messages) == len(session.messages)
```
    """)

    # Performance testing
    print("\n3️⃣ Performance Benchmarking:")
    print("""
```python
import time
import statistics

def benchmark_latency(llm, prompts, iterations=10):
    '''Benchmark response latency.'''
    latencies = []

    for _ in range(iterations):
        for prompt in prompts:
            start = time.perf_counter()
            llm.generate(prompt)
            latencies.append(time.perf_counter() - start)

    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18],
        "p99": statistics.quantiles(latencies, n=100)[98],
    }

def test_performance_sla():
    llm = create_llm("mock", "mock-model")

    results = benchmark_latency(llm, ["test"] * 5)

    # Assert SLA requirements
    assert results["median"] < 0.1  # 100ms median
    assert results["p99"] < 0.5      # 500ms p99
```
    """)


def monitoring_and_alerting():
    """
    Demonstrates monitoring and alerting patterns.

    Architecture Notes:
    - Prometheus metrics export
    - Custom alerting rules
    - SLA monitoring
    """
    print("\n" + "=" * 70)
    print("Monitoring & Alerting Patterns")
    print("=" * 70)

    print("\n📊 Metrics to Monitor:")
    metrics_categories = {
        "Performance": [
            "Request latency (p50, p95, p99)",
            "Tokens per second",
            "First token latency",
            "Queue depth",
        ],
        "Reliability": [
            "Error rate",
            "Success rate",
            "Retry rate",
            "Circuit breaker trips",
        ],
        "Usage": [
            "Requests per minute",
            "Token consumption",
            "Unique users",
            "Model distribution",
        ],
        "Business": [
            "Cost per request",
            "Revenue per user",
            "Feature adoption",
            "User satisfaction",
        ],
    }

    for category, metrics in metrics_categories.items():
        print(f"\n   {category}:")
        for metric in metrics:
            print(f"      • {metric}")

    print("\n🚨 Alerting Rules:")
    print("""
```yaml
# prometheus-rules.yml
groups:
  - name: abstractllm
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(llm_errors_total[5m]) > 0.01
        for: 5m
        annotations:
          summary: "High LLM error rate"

      - alert: HighLatency
        expr: llm_latency_seconds{quantile="0.99"} > 2
        for: 10m
        annotations:
          summary: "P99 latency above 2 seconds"

      - alert: TokenBudgetExceeded
        expr: sum(rate(llm_tokens_total[1h])) > 1000000
        annotations:
          summary: "Token usage exceeding budget"
```
    """)


def main():
    """
    Main entry point - demonstrates production patterns.
    """
    print("\n" + "🏭 " * 20)
    print(" AbstractLLM Core - Example 6: Production Patterns")
    print("🏭 " * 20)

    # Run all demonstrations
    session_management_patterns()
    structured_output_patterns()
    embeddings_and_rag_patterns()
    event_driven_architecture()
    cost_optimization_patterns()
    testing_patterns()
    monitoring_and_alerting()

    print("\n" + "=" * 70)
    print("✅ Example 6 Complete!")
    print("\n🎯 Production Checklist:")
    print("✓ Session management for conversations")
    print("✓ Structured output with validation")
    print("✓ RAG patterns for knowledge augmentation")
    print("✓ Event-driven architecture for observability")
    print("✓ Cost optimization strategies")
    print("✓ Comprehensive testing patterns")
    print("✓ Monitoring and alerting setup")
    print("\n🚀 You're now ready to build production AI applications!")
    print("=" * 70)


if __name__ == "__main__":
    main()