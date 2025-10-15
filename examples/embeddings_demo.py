#!/usr/bin/env python3
"""
Embeddings Demo - Real-world usage examples
============================================

Demonstrates how to use the AbstractCore Core embeddings system
for semantic search and RAG scenarios.
"""

import sys
import os
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abstractcore.embeddings import EmbeddingManager
from abstractcore import create_llm


def demo_basic_embeddings():
    """Demo basic embedding functionality."""
    print("🔢 Basic Embeddings Demo")
    print("=" * 50)

    # Create embedding manager with lightweight model
    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=Path(temp_dir)
        )

        # Single embedding
        text = "Machine learning transforms how we process information."
        embedding = embedder.embed(text)
        print(f"Text: {text}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

        # Similarity computation
        text1 = "Artificial intelligence is the future"
        text2 = "AI will shape tomorrow's technology"
        text3 = "Cooking pasta requires boiling water"

        sim_high = embedder.compute_similarity(text1, text2)
        sim_low = embedder.compute_similarity(text1, text3)

        print(f"\nSimilarity Examples:")
        print(f"'{text1}' vs '{text2}': {sim_high:.3f}")
        print(f"'{text1}' vs '{text3}': {sim_low:.3f}")

        # Cache stats
        stats = embedder.get_cache_stats()
        print(f"\nCache Stats: {stats['memory_cache_info']['hits']} hits, {stats['persistent_cache_size']} stored")


def demo_semantic_search():
    """Demo semantic search with embeddings."""
    print("\n\n🔍 Semantic Search Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=Path(temp_dir)
        )

        # Knowledge base
        documents = [
            "Python is a versatile programming language used for web development, data science, and automation.",
            "JavaScript enables interactive web pages and is essential for frontend development.",
            "Machine learning algorithms can analyze data patterns to make intelligent predictions.",
            "React is a popular JavaScript library for building user interfaces and single-page applications.",
            "Docker containers provide lightweight, portable environments for application deployment.",
            "Kubernetes orchestrates containerized applications across distributed computing clusters.",
            "Natural language processing helps computers understand and generate human language.",
            "Database systems store and retrieve structured information efficiently."
        ]

        # Search queries
        queries = [
            "programming languages for web development",
            "container deployment tools",
            "AI and machine learning"
        ]

        for query in queries:
            print(f"\n🔍 Query: '{query}'")

            # Find most relevant documents
            similarities = []
            for doc in documents:
                sim = embedder.compute_similarity(query, doc)
                similarities.append(sim)

            # Get top 2 results
            top_indices = sorted(range(len(similarities)),
                               key=lambda i: similarities[i], reverse=True)[:2]

            print("📄 Top Results:")
            for rank, idx in enumerate(top_indices, 1):
                score = similarities[idx]
                doc = documents[idx]
                print(f"   {rank}. Score: {score:.3f} - {doc[:60]}...")


def demo_rag_pipeline():
    """Demo RAG pipeline structure."""
    print("\n\n🤖 RAG Pipeline Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=Path(temp_dir)
        )

        # Simulated knowledge base about AbstractCore
        knowledge_base = [
            "AbstractCore Core provides a unified interface to multiple LLM providers including OpenAI, Anthropic, and local models.",
            "The retry system in AbstractCore implements exponential backoff and circuit breaker patterns for production reliability.",
            "Event system in AbstractCore enables real-time monitoring and observability of LLM operations.",
            "Structured output support allows generating and validating Pydantic models from LLM responses.",
            "The embeddings module provides SOTA open-source models like EmbeddingGemma for semantic search capabilities."
        ]

        # User question
        question = "How does AbstractCore handle errors and reliability?"

        print(f"Question: {question}")

        # Step 1: Find relevant context
        similarities = []
        for doc in knowledge_base:
            sim = embedder.compute_similarity(question, doc)
            similarities.append(sim)

        best_idx = similarities.index(max(similarities))
        context = knowledge_base[best_idx]
        score = similarities[best_idx]

        print(f"\nSelected Context (score: {score:.3f}):")
        print(f"'{context}'")

        # Step 2: Create RAG prompt
        rag_prompt = f"""Context: {context}

Question: {question}

Based on the provided context, please answer the question:"""

        print(f"\nRAG Prompt prepared:")
        print(f"'{rag_prompt[:100]}...'")

        # In a real scenario, you would now call an LLM:
        # llm = create_llm("openai", model="gpt-4o-mini")
        # response = llm.generate(rag_prompt)
        print("\n(In production, this prompt would be sent to an LLM for final answer generation)")


def demo_performance():
    """Demo performance characteristics."""
    print("\n\n⚡ Performance Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=Path(temp_dir)
        )

        texts = [
            "First sample text for performance testing",
            "Second sample text for batch processing",
            "Third sample text for speed comparison",
            "Fourth sample text for throughput measurement"
        ]

        import time

        # Individual processing
        start_time = time.time()
        individual_embeddings = [embedder.embed(text) for text in texts]
        individual_time = time.time() - start_time

        # Clear cache for fair comparison
        embedder.embed.cache_clear()

        # Batch processing
        start_time = time.time()
        batch_embeddings = embedder.embed_batch(texts)
        batch_time = time.time() - start_time

        print(f"Individual processing: {individual_time:.3f}s ({individual_time/len(texts):.3f}s per text)")
        print(f"Batch processing: {batch_time:.3f}s ({batch_time/len(texts):.3f}s per text)")
        print(f"Speedup: {individual_time/batch_time:.1f}x")

        # Cache performance
        start_time = time.time()
        cached_embedding = embedder.embed(texts[0])  # Should be cached from batch
        cache_time = time.time() - start_time

        print(f"Cache retrieval: {cache_time:.4f}s (cached)")

        stats = embedder.get_cache_stats()
        print(f"Final cache stats: {stats['memory_cache_info']}")


def main():
    """Run all demos."""
    print("🚀 AbstractCore Embeddings Demo")
    print("=" * 70)

    try:
        demo_basic_embeddings()
        demo_semantic_search()
        demo_rag_pipeline()
        demo_performance()

        print("\n\n✅ All demos completed successfully!")
        print("\nNext steps:")
        print("- Install different embedding models: 'embeddinggemma', 'stella-400m'")
        print("- Try ONNX backend for 2-3x speedup: backend='onnx'")
        print("- Use Matryoshka truncation: output_dims=256")
        print("- Integrate with your LLM provider for full RAG pipeline")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install sentence-transformers")
    except Exception as e:
        print(f"❌ Error: {e}")
        if "offline" in str(e).lower():
            print("Note: This demo requires internet access to download models on first run.")


if __name__ == "__main__":
    main()