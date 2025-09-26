"""
Comprehensive tests for embedding matrix operations and clustering functionality.
Tests compute_similarities_matrix, find_similar_clusters, and optimizations.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path

from abstractllm.embeddings import EmbeddingManager


class TestEmbeddingMatrixOperations:
    """Test matrix operations and clustering with real embeddings."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_similarities_matrix_symmetric(self):
        """Test symmetric similarity matrix computation."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            texts = [
                "Python programming language",
                "JavaScript for web development",
                "Python for data science",
                "Machine learning with Python",
                "Web development with JavaScript"
            ]

            # Compute symmetric matrix
            matrix = manager.compute_similarities_matrix(texts)

            # Check matrix properties
            assert matrix.shape == (5, 5)
            assert isinstance(matrix, np.ndarray)

            # Check symmetry
            np.testing.assert_array_almost_equal(matrix, matrix.T, decimal=6)

            # Check diagonal is 1.0 (text similar to itself)
            np.testing.assert_array_almost_equal(np.diag(matrix), np.ones(5), decimal=6)

            # Check reasonable similarity values
            assert np.all(matrix >= -1.0) and np.all(matrix <= 1.0)

            # Check that Python texts are more similar to each other
            python_indices = [0, 2, 3]  # Python-related texts
            js_indices = [1, 4]  # JavaScript-related texts

            # Python texts should be more similar to each other than to JS texts
            python_similarities = []
            cross_similarities = []

            for i in python_indices:
                for j in python_indices:
                    if i != j:
                        python_similarities.append(matrix[i, j])
                for j in js_indices:
                    cross_similarities.append(matrix[i, j])

            avg_python_sim = np.mean(python_similarities)
            avg_cross_sim = np.mean(cross_similarities)

            assert avg_python_sim > avg_cross_sim, f"Python similarity ({avg_python_sim:.3f}) should be > cross similarity ({avg_cross_sim:.3f})"

            print(f"✅ Symmetric matrix computed: {matrix.shape}, avg Python sim: {avg_python_sim:.3f}")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_compute_similarities_matrix_asymmetric(self):
        """Test asymmetric similarity matrix computation."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            queries = [
                "How to learn Python programming?",
                "Best JavaScript frameworks for beginners",
                "Data science tutorials"
            ]

            docs = [
                "Python programming guide for beginners",
                "JavaScript development best practices",
                "Data science with Python and pandas",
                "Web development using React framework",
                "Machine learning fundamentals"
            ]

            # Compute asymmetric matrix
            matrix = manager.compute_similarities_matrix(queries, docs)

            # Check matrix properties
            assert matrix.shape == (3, 5)
            assert isinstance(matrix, np.ndarray)

            # Check similarity values are reasonable
            assert np.all(matrix >= -1.0) and np.all(matrix <= 1.0)

            # Check that queries match relevant docs
            # Query 0 (Python) should match doc 0 (Python guide) and doc 2 (Python data science)
            python_query_similarities = matrix[0, [0, 2]]
            other_similarities = matrix[0, [1, 3, 4]]

            assert np.mean(python_query_similarities) > np.mean(other_similarities)

            # Query 1 (JavaScript) should match doc 1 (JS practices) and doc 3 (React)
            js_query_similarities = matrix[1, [1, 3]]
            other_js_similarities = matrix[1, [0, 2, 4]]

            assert np.mean(js_query_similarities) > np.mean(other_js_similarities)

            print(f"✅ Asymmetric matrix computed: {matrix.shape}")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_compute_similarities_matrix_chunked(self):
        """Test chunked matrix computation for memory efficiency."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Create a moderate-sized test to trigger chunking
            texts = [f"Sample text number {i} about various topics" for i in range(20)]

            # Force chunked computation with small chunk size and memory limit
            matrix_chunked = manager.compute_similarities_matrix(
                texts,
                chunk_size=5,  # Small chunks
                max_memory_gb=0.001  # Force chunking
            )

            # Compare with direct computation
            matrix_direct = manager.compute_similarities_matrix(
                texts,
                chunk_size=100,  # Large chunks (direct)
                max_memory_gb=10.0  # Allow direct computation
            )

            # Results should be nearly identical
            np.testing.assert_array_almost_equal(matrix_chunked, matrix_direct, decimal=6)

            print(f"✅ Chunked computation verified: {matrix_chunked.shape}")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_find_similar_clusters(self):
        """Test text clustering functionality."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            texts = [
                "Python programming is powerful",      # Cluster 1: Python
                "Learn Python for data science",      # Cluster 1: Python
                "Python development best practices",  # Cluster 1: Python
                "JavaScript for web development",     # Cluster 2: JavaScript
                "React JavaScript framework",         # Cluster 2: JavaScript
                "Machine learning algorithms",        # Standalone (ML)
                "Database design principles",         # Standalone (Database)
                "Python machine learning libraries"   # Could join Cluster 1 or be standalone
            ]

            # Find clusters with moderate threshold
            clusters = manager.find_similar_clusters(
                texts,
                threshold=0.4,  # Moderate threshold
                min_cluster_size=2
            )

            # Should find at least 2 clusters
            assert len(clusters) >= 2, f"Expected at least 2 clusters, got {len(clusters)}"

            # Check cluster properties
            all_clustered_indices = set()
            for cluster in clusters:
                assert len(cluster) >= 2, f"Cluster {cluster} is smaller than min_cluster_size"
                assert len(cluster) == len(set(cluster)), f"Cluster {cluster} has duplicates"

                # Check no overlap between clusters
                for idx in cluster:
                    assert idx not in all_clustered_indices, f"Index {idx} appears in multiple clusters"
                    all_clustered_indices.add(idx)

            # Verify Python cluster exists (texts 0, 1, 2 should be similar)
            python_cluster_found = False
            for cluster in clusters:
                python_texts_in_cluster = sum(1 for idx in cluster if idx in [0, 1, 2, 7])
                if python_texts_in_cluster >= 2:
                    python_cluster_found = True
                    break

            assert python_cluster_found, f"No Python-related cluster found. Clusters: {clusters}"

            # Test with different thresholds
            strict_clusters = manager.find_similar_clusters(
                texts,
                threshold=0.7,  # Strict threshold
                min_cluster_size=2
            )

            loose_clusters = manager.find_similar_clusters(
                texts,
                threshold=0.3,  # Loose threshold
                min_cluster_size=2
            )

            # Stricter threshold should produce fewer/smaller clusters
            strict_clustered = sum(len(cluster) for cluster in strict_clusters)
            loose_clustered = sum(len(cluster) for cluster in loose_clusters)

            assert strict_clustered <= loose_clustered, "Strict threshold should cluster fewer texts"

            print(f"✅ Clustering completed: {len(clusters)} clusters, {len(all_clustered_indices)} texts clustered")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_normalized_embeddings_cache(self):
        """Test normalized embeddings caching functionality."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            text = "Test text for normalized embedding caching"

            # Get normalized embedding twice
            norm_emb1 = manager.embed_normalized(text)
            norm_emb2 = manager.embed_normalized(text)

            # Should be identical due to caching
            assert norm_emb1 == norm_emb2

            # Check that embedding is actually normalized (unit length)
            norm_array = np.array(norm_emb1)
            length = np.linalg.norm(norm_array)
            np.testing.assert_almost_equal(length, 1.0, decimal=6)

            # Check cache stats include normalized cache
            stats = manager.get_cache_stats()
            assert "normalized_cache_size" in stats
            assert stats["normalized_cache_size"] > 0

            # Compare speed with regular similarity computation
            import time

            texts = ["Sample text A", "Sample text B"]

            # Method 1: Using normalized embeddings (should be faster)
            start_time = time.time()
            norm_a = np.array(manager.embed_normalized(texts[0]))
            norm_b = np.array(manager.embed_normalized(texts[1]))
            sim_fast = np.dot(norm_a, norm_b)  # Simple dot product
            fast_time = time.time() - start_time

            # Method 2: Using regular compute_similarity
            start_time = time.time()
            sim_regular = manager.compute_similarity(texts[0], texts[1])
            regular_time = time.time() - start_time

            # Results should be nearly identical
            np.testing.assert_almost_equal(sim_fast, sim_regular, decimal=5)

            print(f"✅ Normalized caching works: cache size {stats['normalized_cache_size']}")
            print(f"   Fast method: {fast_time*1000:.2f}ms, Regular: {regular_time*1000:.2f}ms")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_matrix_edge_cases(self):
        """Test edge cases for matrix operations."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test empty lists
            matrix_empty = manager.compute_similarities_matrix([])
            assert matrix_empty.shape == (0, 0)

            # Test single text
            matrix_single = manager.compute_similarities_matrix(["Single text"])
            assert matrix_single.shape == (1, 1)
            np.testing.assert_almost_equal(matrix_single[0, 0], 1.0, decimal=6)

            # Test empty string
            matrix_empty_str = manager.compute_similarities_matrix(["", "test"])
            assert matrix_empty_str.shape == (2, 2)

            # Test clustering edge cases
            clusters_empty = manager.find_similar_clusters([])
            assert clusters_empty == []

            clusters_too_small = manager.find_similar_clusters(["single"], min_cluster_size=2)
            assert clusters_too_small == []

            print("✅ Edge cases handled correctly")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_performance_and_memory_estimation(self):
        """Test memory estimation and performance characteristics."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test with different matrix sizes
            small_texts = [f"Text {i}" for i in range(5)]
            medium_texts = [f"Text {i}" for i in range(20)]

            import time

            # Small matrix
            start_time = time.time()
            small_matrix = manager.compute_similarities_matrix(small_texts)
            small_time = time.time() - start_time

            # Medium matrix
            start_time = time.time()
            medium_matrix = manager.compute_similarities_matrix(medium_texts)
            medium_time = time.time() - start_time

            assert small_matrix.shape == (5, 5)
            assert medium_matrix.shape == (20, 20)

            # Medium should take more time but not exponentially more
            # (due to efficient vectorized operations)
            time_ratio = medium_time / small_time
            assert time_ratio < 20, f"Time scaling is too high: {time_ratio:.1f}x"

            print(f"✅ Performance scaling reasonable: 5x5 in {small_time*1000:.1f}ms, 20x20 in {medium_time*1000:.1f}ms")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])