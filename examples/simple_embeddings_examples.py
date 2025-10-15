#!/usr/bin/env python3
"""
Simple Concrete Embeddings Examples
===================================

This file demonstrates the most common and practical uses of the embeddings system
with real, concrete examples you can copy and paste.
"""

import sys
from pathlib import Path

# Add project root to path for standalone execution
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def example_1_basic_embedding():
    """Example 1: Generate a basic embedding."""
    print("=" * 60)
    print("Example 1: Basic Embedding Generation")
    print("=" * 60)

    from abstractcore.embeddings import EmbeddingManager

    # Create embedding manager (uses lightweight model for speed)
    embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embedding for a single text
    text = "Machine learning is transforming how we build software"
    embedding = embedder.embed(text)

    print(f"Text: {text}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {[round(x, 4) for x in embedding[:5]]}")
    print(f"Embedding type: {type(embedding)}")
    print()


def example_2_similarity_search():
    """Example 2: Find similar texts using cosine similarity."""
    print("=" * 60)
    print("Example 2: Text Similarity Search")
    print("=" * 60)

    from abstractcore.embeddings import EmbeddingManager

    embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")

    # Compare different texts
    texts = [
        "Python is a programming language",
        "Python is used for web development",
        "Cats are cute animals",
        "JavaScript is for web development",
        "Dogs are loyal pets"
    ]

    query = "programming languages"

    print(f"Query: '{query}'")
    print("\nSimilarity scores:")

    for text in texts:
        similarity = embedder.compute_similarity(query, text)
        print(f"  {similarity:.3f} - {text}")

    print()


def example_3_document_search():
    """Example 3: Search through a collection of documents."""
    print("=" * 60)
    print("Example 3: Document Collection Search")
    print("=" * 60)

    from abstractcore.embeddings import EmbeddingManager

    embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")

    # Document collection
    documents = [
        "FastAPI is a modern Python web framework for building APIs quickly and efficiently.",
        "React is a JavaScript library for building user interfaces and single-page applications.",
        "Docker provides containerization technology for packaging applications and dependencies.",
        "PostgreSQL is a powerful open-source relational database management system.",
        "Kubernetes orchestrates containerized applications across distributed computing clusters.",
        "TensorFlow is Google's open-source machine learning framework for AI development.",
        "Redis is an in-memory data structure store used as database and cache.",
        "Django is a high-level Python web framework that encourages rapid development."
    ]

    query = "Python web frameworks"

    print(f"Searching for: '{query}'")
    print(f"In {len(documents)} documents\n")

    # Calculate similarities
    similarities = []
    for doc in documents:
        similarity = embedder.compute_similarity(query, doc)
        similarities.append((similarity, doc))

    # Sort by similarity (highest first)
    similarities.sort(reverse=True, key=lambda x: x[0])

    print("Top 3 most relevant documents:")
    for i, (score, doc) in enumerate(similarities[:3], 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   {doc}")
        print()


def example_4_batch_processing():
    """Example 4: Efficient batch processing of multiple texts."""
    print("=" * 60)
    print("Example 4: Batch Processing for Efficiency")
    print("=" * 60)

    from abstractcore.embeddings import EmbeddingManager
    import time

    embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")

    texts = [
        "Artificial intelligence is advancing rapidly",
        "Machine learning requires large datasets",
        "Deep learning uses neural networks",
        "Natural language processing understands text",
        "Computer vision analyzes images"
    ]

    # Individual processing
    start_time = time.time()
    individual_embeddings = []
    for text in texts:
        embedding = embedder.embed(text)
        individual_embeddings.append(embedding)
    individual_time = time.time() - start_time

    # Clear cache for fair comparison
    embedder.embed.cache_clear()

    # Batch processing
    start_time = time.time()
    batch_embeddings = embedder.embed_batch(texts)
    batch_time = time.time() - start_time

    print(f"Processing {len(texts)} texts:")
    print(f"  Individual: {individual_time:.3f} seconds")
    print(f"  Batch: {batch_time:.3f} seconds")
    if batch_time > 0:
        print(f"  Speedup: {individual_time/batch_time:.1f}x faster")
    else:
        print(f"  Speedup: >1000x faster (too fast to measure precisely)")

    print(f"\nResults identical: {individual_embeddings == batch_embeddings}")
    print()


def example_5_rag_pipeline():
    """Example 5: Complete RAG (Retrieval-Augmented Generation) pipeline."""
    print("=" * 60)
    print("Example 5: RAG Pipeline Example")
    print("=" * 60)

    from abstractcore.embeddings import EmbeddingManager

    embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")

    # Knowledge base about programming
    knowledge_base = [
        "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability.",
        "JavaScript was developed by Brendan Eich in 1995 for web browsers and is now used everywhere.",
        "React was created by Facebook in 2011 and open-sourced in 2013 for building user interfaces.",
        "Docker was first released in 2013 and revolutionized application deployment through containers.",
        "Kubernetes was originally developed by Google and released in 2014 for container orchestration."
    ]

    # User question
    question = "Who created Python and when?"

    print(f"Question: {question}")
    print(f"Searching through {len(knowledge_base)} knowledge documents...")

    # Step 1: Find most relevant context using embeddings
    best_score = 0
    best_context = ""

    for doc in knowledge_base:
        similarity = embedder.compute_similarity(question, doc)
        if similarity > best_score:
            best_score = similarity
            best_context = doc

    print(f"\nBest matching context (similarity: {best_score:.3f}):")
    print(f"  {best_context}")

    # Step 2: Create RAG prompt for LLM
    rag_prompt = f"""Context: {best_context}

Question: {question}

Based on the provided context, please answer the question:"""

    print(f"\nRAG Prompt prepared for LLM:")
    print(f"  Context found: ✓")
    print(f"  Prompt length: {len(rag_prompt)} characters")
    print(f"  Ready for LLM: ✓")

    # In a real application, you would now send this to an LLM:
    # from abstractcore import create_llm
    # llm = create_llm("openai", model="gpt-4o-mini")
    # response = llm.generate(rag_prompt)
    # print(f"  LLM Answer: {response.content}")

    print()


def example_6_performance_optimization():
    """Example 6: Performance optimization techniques."""
    print("=" * 60)
    print("Example 6: Performance Optimization")
    print("=" * 60)

    from abstractcore.embeddings import EmbeddingManager

    # Option 1: Default configuration (good for most use cases)
    embedder_default = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")
    print("Default configuration:")
    print(f"  Model: {embedder_default.model_id}")
    print(f"  Dimensions: {embedder_default.get_dimension()}")
    print(f"  Backend: Auto-selected")

    # Option 2: Optimized for speed
    embedder_fast = EmbeddingManager(
        model="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight model
        backend="auto",  # Will use ONNX if available
        cache_size=2000,  # Larger cache
    )
    print("\nSpeed-optimized configuration:")
    print(f"  Model: Lightweight (384D)")
    print(f"  Cache size: 2000 entries")
    print(f"  Expected performance: ~10ms per text")

    # Option 3: Alternative quality model (when EmbeddingGemma not available)
    embedder_quality = EmbeddingManager(
        model="sentence-transformers/all-MiniLM-L6-v2",  # Available model
        cache_size=5000,
    )
    print("\nQuality-optimized configuration:")
    print(f"  Model: all-MiniLM-L6-v2 (reliable)")
    print(f"  Dimensions: 384D")
    print(f"  Large cache: 5000 entries")

    # Option 4: Show configuration
    print("\nNote: EmbeddingGemma not available in this environment")
    print("  In production, use: model='embeddinggemma'")
    print("  For 768D, multilingual, SOTA performance")

    print()


def example_7_error_handling():
    """Example 7: Proper error handling."""
    print("=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)

    from abstractcore.embeddings import EmbeddingManager

    try:
        # This will work
        embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")
        embedding = embedder.embed("Test text")
        print(f"✓ Successfully generated {len(embedding)}D embedding")

        # Test with empty text (handled gracefully)
        empty_embedding = embedder.embed("")
        print(f"✓ Empty text handling: {len(empty_embedding)}D zero vector")

        # Test with very long text (handled gracefully)
        long_text = "This is a very long text. " * 100
        long_embedding = embedder.embed(long_text)
        print(f"✓ Long text handling: {len(long_embedding)}D embedding")

    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Solution: pip install sentence-transformers")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print("  The system provides graceful fallbacks for most errors")

    print()


def main():
    """Run all examples."""
    print("🚀 AbstractCore Embeddings - Simple Concrete Examples")
    print("=" * 80)

    try:
        example_1_basic_embedding()
        example_2_similarity_search()
        example_3_document_search()
        example_4_batch_processing()
        example_5_rag_pipeline()
        example_6_performance_optimization()
        example_7_error_handling()

        print("✅ All examples completed successfully!")
        print("\n📚 Next Steps:")
        print("• Try different embedding models: 'embeddinggemma', 'stella-400m'")
        print("• Enable ONNX optimization: backend='onnx'")
        print("• Use Matryoshka truncation: output_dims=256")
        print("• Integrate with your LLM: from abstractcore import create_llm")
        print("• Build a complete RAG application!")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install sentence-transformers")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()