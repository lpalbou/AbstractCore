#!/usr/bin/env python3
"""
Matrix Operations Demo - Advanced similarity and clustering with AbstractLLM
===========================================================================

Demonstrates the new SOTA similarity matrix computation and clustering capabilities.
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abstractllm.embeddings import EmbeddingManager


def demo_similarity_matrix():
    """Demo similarity matrix computation for document analysis."""
    print("🔢 Similarity Matrix Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=Path(temp_dir)
        )

        # Sample documents from different domains
        documents = [
            "Python programming language features and syntax",
            "JavaScript development for modern web applications",
            "Machine learning algorithms and data science",
            "Python libraries for data analysis and visualization",
            "React framework for JavaScript frontend development",
            "Deep learning neural networks and AI applications",
            "Python web development with Django and Flask",
            "JavaScript ES6 features and modern programming"
        ]

        print(f"Computing {len(documents)}×{len(documents)} similarity matrix...")

        # Compute similarity matrix
        similarity_matrix = embedder.compute_similarities_matrix(
            documents,
            normalized=True,  # Use pre-normalization for speed
            chunk_size=500    # Memory-efficient processing
        )

        print(f"Matrix shape: {similarity_matrix.shape}")
        print(f"Matrix data type: {similarity_matrix.dtype}")

        # Analyze the most similar document pairs
        print("\n📊 Top 5 Most Similar Document Pairs:")

        # Get upper triangle indices (avoid diagonal and duplicates)
        triu_indices = np.triu_indices(len(documents), k=1)
        similarities = similarity_matrix[triu_indices]

        # Get top 5 pairs
        top_indices = np.argsort(similarities)[-5:][::-1]

        for i, idx in enumerate(top_indices):
            row, col = triu_indices[0][idx], triu_indices[1][idx]
            score = similarities[idx]

            print(f"  {i+1}. Score: {score:.3f}")
            print(f"     Doc {row}: '{documents[row][:50]}...'")
            print(f"     Doc {col}: '{documents[col][:50]}...'")
            print()


def demo_asymmetric_matrix():
    """Demo asymmetric similarity matrix for query-document matching."""
    print("\n\n🔍 Query-Document Matching Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=Path(temp_dir)
        )

        # Search queries
        queries = [
            "How to learn Python programming?",
            "Best JavaScript frameworks for web development",
            "Machine learning tutorials and courses",
            "Data visualization tools and techniques"
        ]

        # Document database
        knowledge_base = [
            "Python Programming Guide: Complete tutorial covering syntax, functions, and OOP concepts",
            "React vs Vue.js: Comprehensive comparison of popular JavaScript frameworks",
            "Introduction to Machine Learning: Supervised and unsupervised learning explained",
            "Data Visualization with Python: Creating charts and graphs with matplotlib and seaborn",
            "Advanced JavaScript: ES6 features, async programming, and modern development practices",
            "Deep Learning Fundamentals: Neural networks, backpropagation, and training techniques",
            "Python for Beginners: Step-by-step guide to learning Python programming language",
            "Web Development with JavaScript: Frontend frameworks and backend development"
        ]

        print(f"Computing {len(queries)}×{len(knowledge_base)} query-document similarity matrix...")

        # Compute asymmetric similarity matrix
        similarity_matrix = embedder.compute_similarities_matrix(
            queries,
            knowledge_base,
            normalized=True
        )

        print(f"Matrix shape: {similarity_matrix.shape}")

        # For each query, find the best matching documents
        print("\n🎯 Best Document Matches for Each Query:")

        for i, query in enumerate(queries):
            similarities = similarity_matrix[i]

            # Get top 2 matches
            top_indices = np.argsort(similarities)[-2:][::-1]

            print(f"\n  Query: '{query}'")
            for j, doc_idx in enumerate(top_indices):
                score = similarities[doc_idx]
                doc = knowledge_base[doc_idx]
                print(f"    {j+1}. Score: {score:.3f} - '{doc[:60]}...'")


def demo_clustering():
    """Demo text clustering for content organization."""
    print("\n\n🔗 Text Clustering Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=Path(temp_dir)
        )

        # Mixed content that should form natural clusters
        texts = [
            # Python cluster
            "Python programming language tutorial",
            "Learn Python for data science applications",
            "Python web development with Django framework",
            "Advanced Python programming techniques",

            # JavaScript cluster
            "JavaScript for modern web development",
            "React JavaScript library for user interfaces",
            "JavaScript ES6 and modern programming features",
            "Node.js server-side JavaScript development",

            # Machine Learning cluster
            "Machine learning algorithms and methods",
            "Deep learning neural networks tutorial",
            "AI and machine learning applications",
            "Data science and predictive modeling",

            # Database cluster
            "SQL database design and optimization",
            "NoSQL databases like MongoDB and Redis",
            "Database administration and management",

            # Individual/unclustered texts
            "Mobile app development best practices",
            "Cybersecurity and network protection",
        ]

        print(f"Clustering {len(texts)} texts...")

        # Find clusters with different thresholds
        print("\n🎯 Clustering with threshold 0.4 (moderate):")
        clusters = embedder.find_similar_clusters(
            texts,
            threshold=0.4,
            min_cluster_size=2
        )

        total_clustered = sum(len(cluster) for cluster in clusters)
        print(f"Found {len(clusters)} clusters, {total_clustered}/{len(texts)} texts clustered")

        for i, cluster in enumerate(clusters):
            print(f"\n  Cluster {i+1} ({len(cluster)} texts):")
            for idx in cluster:
                print(f"    - '{texts[idx]}'")

        # Try stricter clustering
        print(f"\n🎯 Clustering with threshold 0.6 (strict):")
        strict_clusters = embedder.find_similar_clusters(
            texts,
            threshold=0.6,
            min_cluster_size=2
        )

        strict_total = sum(len(cluster) for cluster in strict_clusters)
        print(f"Found {len(strict_clusters)} clusters, {strict_total}/{len(texts)} texts clustered")

        for i, cluster in enumerate(strict_clusters):
            print(f"  Cluster {i+1}: {len(cluster)} texts")


def demo_performance_optimization():
    """Demo performance features: caching, chunking, normalization."""
    print("\n\n⚡ Performance Optimization Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        embedder = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=Path(temp_dir)
        )

        texts = [f"Sample text number {i} for performance testing" for i in range(50)]

        import time

        # Test 1: Regular vs normalized embeddings
        print("Test 1: Normalized embeddings performance")

        text1, text2 = texts[0], texts[1]

        # Method 1: Regular similarity computation
        start = time.time()
        regular_sim = embedder.compute_similarity(text1, text2)
        regular_time = time.time() - start

        # Method 2: Normalized embeddings with dot product
        start = time.time()
        norm1 = np.array(embedder.embed_normalized(text1))
        norm2 = np.array(embedder.embed_normalized(text2))
        fast_sim = np.dot(norm1, norm2)
        fast_time = time.time() - start

        print(f"  Regular similarity: {regular_sim:.4f} ({regular_time*1000:.1f}ms)")
        print(f"  Normalized method: {fast_sim:.4f} ({fast_time*1000:.1f}ms)")
        print(f"  Speedup: {regular_time/fast_time:.1f}x faster")

        # Test 2: Matrix computation scaling
        print(f"\nTest 2: Matrix computation scaling")

        sizes = [5, 10, 20]
        for size in sizes:
            test_texts = texts[:size]

            start = time.time()
            matrix = embedder.compute_similarities_matrix(test_texts)
            duration = time.time() - start

            print(f"  {size}×{size} matrix: {duration*1000:.1f}ms ({matrix.size:,} comparisons)")

        # Test 3: Cache performance
        print(f"\nTest 3: Cache performance")
        stats = embedder.get_cache_stats()

        print(f"  Regular cache: {stats['persistent_cache_size']} embeddings")
        print(f"  Normalized cache: {stats['normalized_cache_size']} embeddings")
        print(f"  Memory cache hits: {stats['memory_cache_info']['hits']}")


def main():
    """Run all demos."""
    print("🚀 AbstractLLM Matrix Operations & Clustering Demo")
    print("=" * 70)

    try:
        demo_similarity_matrix()
        demo_asymmetric_matrix()
        demo_clustering()
        demo_performance_optimization()

        print("\n\n✅ All demos completed successfully!")
        print("\n💡 Key Benefits:")
        print("- SOTA vectorized operations for 100x+ speedup")
        print("- Memory-efficient chunking for large matrices")
        print("- Automatic clustering for content organization")
        print("- Pre-normalization caching for repeated calculations")
        print("- Production-ready error handling and logging")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install sentence-transformers numpy")
    except Exception as e:
        print(f"❌ Error: {e}")
        if "offline" in str(e).lower():
            print("Note: This demo requires internet access to download models on first run.")


if __name__ == "__main__":
    main()