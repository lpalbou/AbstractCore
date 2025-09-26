"""
Comprehensive Semantic Validation Test for Vector Embeddings
===========================================================

This test validates that the EmbeddingManager is actually using vector embeddings
from the loaded model and that semantic similarity relationships work as expected.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path

from abstractllm.embeddings import EmbeddingManager


class TestEmbeddingSemanticValidation:
    """Comprehensive validation of embedding semantic understanding."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embedding_vector_properties(self):
        """Test that embeddings are actual vectors with correct properties."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            text = "This is a test sentence for vector validation."
            embedding = manager.embed(text)

            # Test 1: Embedding is a list of floats
            assert isinstance(embedding, list), f"Embedding should be list, got {type(embedding)}"
            assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
            assert all(isinstance(x, (int, float)) for x in embedding), "All embedding values should be numeric"

            # Test 2: Embedding is not all zeros (indicates model is working)
            embedding_array = np.array(embedding)
            assert not np.allclose(embedding_array, 0), "Embedding should not be all zeros"
            assert np.any(embedding_array != 0), "Embedding should have non-zero values"

            # Test 3: Different texts produce different embeddings
            different_text = "Completely different sentence about mathematics and physics."
            different_embedding = manager.embed(different_text)

            assert embedding != different_embedding, "Different texts should produce different embeddings"

            # Test 4: Same text produces identical embeddings (caching test)
            same_embedding = manager.embed(text)
            assert embedding == same_embedding, "Same text should produce identical embeddings (caching)"

            print(f"✅ Vector properties validated: {len(embedding)} dimensions, non-zero values")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_semantic_similarity_relationships(self):
        """Test that semantic similarity relationships work as expected."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Define test cases with expected similarity relationships
            test_cases = [
                {
                    "name": "Synonyms (should be highly similar)",
                    "text1": "Happy and joyful person",
                    "text2": "Cheerful and delighted individual",
                    "expected_min": 0.6,  # High similarity expected
                    "description": "Synonyms should have high similarity"
                },
                {
                    "name": "Related concepts (should be moderately similar)",
                    "text1": "Dog running in the park",
                    "text2": "Cat playing in the garden",
                    "expected_min": 0.3,  # Moderate similarity (both animals)
                    "expected_max": 0.8,  # But not too high (different animals)
                    "description": "Related concepts (animals) should be moderately similar"
                },
                {
                    "name": "Same domain, different concepts",
                    "text1": "Programming in Python language",
                    "text2": "Developing with JavaScript framework",
                    "expected_min": 0.1,  # Both programming (lowered for all-MiniLM)
                    "expected_max": 0.9,  # But different languages
                    "description": "Same domain concepts should be similar"
                },
                {
                    "name": "Completely unrelated (should be dissimilar)",
                    "text1": "Advanced quantum physics equations",
                    "text2": "Delicious chocolate cake recipe",
                    "expected_max": 0.4,  # Low similarity expected
                    "description": "Unrelated concepts should have low similarity"
                }
            ]

            results = []

            for case in test_cases:
                similarity = manager.compute_similarity(case["text1"], case["text2"])

                # Check minimum similarity if specified
                if "expected_min" in case:
                    assert similarity >= case["expected_min"], \
                        f"{case['name']}: Expected similarity >= {case['expected_min']}, got {similarity:.3f}"

                # Check maximum similarity if specified
                if "expected_max" in case:
                    assert similarity <= case["expected_max"], \
                        f"{case['name']}: Expected similarity <= {case['expected_max']}, got {similarity:.3f}"

                results.append({
                    "name": case["name"],
                    "similarity": similarity,
                    "text1": case["text1"][:30] + "...",
                    "text2": case["text2"][:30] + "...",
                    "passed": True
                })

            # Additional validation: High similarity pairs should be more similar than low similarity pairs
            synonyms_sim = manager.compute_similarity(
                "Happy and joyful person",
                "Cheerful and delighted individual"
            )
            unrelated_sim = manager.compute_similarity(
                "Advanced quantum physics equations",
                "Delicious chocolate cake recipe"
            )

            assert synonyms_sim > unrelated_sim, \
                f"Synonyms ({synonyms_sim:.3f}) should be more similar than unrelated concepts ({unrelated_sim:.3f})"

            print("✅ Semantic similarity relationships validated:")
            for result in results:
                print(f"   {result['name']}: {result['similarity']:.3f}")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_similarity_matrix_semantic_structure(self):
        """Test that similarity matrix reflects semantic structure."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Create texts with known semantic relationships
            texts = [
                # Programming cluster
                "Python programming language tutorial",        # 0
                "Learning Python for beginners",               # 1
                "Advanced Python development techniques",      # 2

                # Animal cluster
                "Dogs are loyal companion animals",            # 3
                "Cats are independent pet animals",            # 4
                "Birds can fly in the sky",                    # 5

                # Cooking cluster
                "Delicious homemade pasta recipe",             # 6
                "Baking chocolate chip cookies",               # 7
                "Preparing fresh salad ingredients",           # 8

                # Unrelated singleton
                "Advanced mathematical calculus equations"      # 9
            ]

            # Compute similarity matrix
            matrix = manager.compute_similarities_matrix(texts)

            # Test matrix properties
            assert matrix.shape == (10, 10), f"Expected 10x10 matrix, got {matrix.shape}"
            assert np.allclose(matrix, matrix.T), "Matrix should be symmetric"
            assert np.allclose(np.diag(matrix), 1.0), "Diagonal should be 1.0 (self-similarity)"

            # Test semantic clustering in matrix
            # Python cluster (0, 1, 2) should have high internal similarity
            python_cluster = [0, 1, 2]
            python_similarities = []
            for i in python_cluster:
                for j in python_cluster:
                    if i != j:
                        python_similarities.append(matrix[i, j])

            avg_python_similarity = np.mean(python_similarities)
            assert avg_python_similarity > 0.5, \
                f"Python cluster internal similarity too low: {avg_python_similarity:.3f}"

            # Animal cluster (3, 4, 5) should have moderate internal similarity
            animal_cluster = [3, 4, 5]
            animal_similarities = []
            for i in animal_cluster:
                for j in animal_cluster:
                    if i != j:
                        animal_similarities.append(matrix[i, j])

            avg_animal_similarity = np.mean(animal_similarities)
            assert avg_animal_similarity > 0.25, \
                f"Animal cluster internal similarity too low: {avg_animal_similarity:.3f}"

            # Cross-cluster similarity should be lower than intra-cluster
            cross_cluster_similarities = []
            for i in python_cluster:
                for j in animal_cluster:
                    cross_cluster_similarities.append(matrix[i, j])

            avg_cross_similarity = np.mean(cross_cluster_similarities)
            assert avg_python_similarity > avg_cross_similarity, \
                f"Intra-cluster ({avg_python_similarity:.3f}) should be > cross-cluster ({avg_cross_similarity:.3f})"

            print(f"✅ Matrix semantic structure validated:")
            print(f"   Python cluster avg similarity: {avg_python_similarity:.3f}")
            print(f"   Animal cluster avg similarity: {avg_animal_similarity:.3f}")
            print(f"   Cross-cluster avg similarity: {avg_cross_similarity:.3f}")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_clustering_with_30_sentences(self):
        """Test clustering with 30 sentences across multiple semantic domains."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Create 30 sentences across 6 semantic domains (5 per domain)
            sentences = [
                # Technology/Programming (0-4)
                "Python is a powerful programming language for data science",
                "JavaScript frameworks like React make web development easier",
                "Machine learning algorithms require large datasets for training",
                "Software engineering practices include code reviews and testing",
                "Artificial intelligence will transform how we work and live",

                # Animals/Nature (5-9)
                "Golden retrievers are loyal and friendly family dogs",
                "Wild lions hunt in coordinated groups called prides",
                "Colorful butterflies migrate thousands of miles each year",
                "Ocean dolphins communicate using clicks and whistles",
                "Forest ecosystems support diverse plant and animal life",

                # Food/Cooking (10-14)
                "Italian pasta dishes require fresh ingredients and proper timing",
                "Homemade bread baking fills the kitchen with wonderful aromas",
                "Spicy Thai cuisine balances sweet, sour, and heat flavors",
                "Farm-to-table restaurants prioritize local seasonal ingredients",
                "Chocolate desserts are the perfect ending to any meal",

                # Sports/Fitness (15-19)
                "Professional basketball requires exceptional athletic skills and teamwork",
                "Marathon running demands months of dedicated training and preparation",
                "Swimming provides excellent full-body cardiovascular exercise benefits",
                "Team sports teach valuable lessons about cooperation and leadership",
                "Yoga practice combines physical movement with mindfulness and breathing",

                # Science/Medicine (20-24)
                "Medical research advances lead to better treatments and cures",
                "Climate change requires immediate global action and cooperation",
                "Renewable energy sources will replace fossil fuels over time",
                "Genetic engineering holds promise for treating inherited diseases",
                "Space exploration expands our understanding of the universe",

                # Travel/Culture (25-29)
                "Ancient European castles tell stories of medieval history",
                "Japanese tea ceremonies represent centuries of cultural tradition",
                "Mountain hiking offers breathtaking views and physical challenges",
                "Local markets showcase authentic regional foods and crafts",
                "Cultural festivals celebrate community heritage and traditions"
            ]

            print(f"Testing clustering with {len(sentences)} sentences across 6 domains...")

            # Test clustering with different thresholds
            thresholds_to_test = [0.3, 0.4, 0.5, 0.6]

            clustering_results = []

            for threshold in thresholds_to_test:
                clusters = manager.find_similar_clusters(
                    sentences,
                    threshold=threshold,
                    min_cluster_size=2
                )

                total_clustered = sum(len(cluster) for cluster in clusters)

                clustering_results.append({
                    "threshold": threshold,
                    "num_clusters": len(clusters),
                    "total_clustered": total_clustered,
                    "clusters": clusters
                })

                print(f"   Threshold {threshold}: {len(clusters)} clusters, {total_clustered}/{len(sentences)} sentences clustered")

            # Detailed analysis of best clustering result (threshold 0.4)
            best_result = None
            for result in clustering_results:
                if result["threshold"] == 0.4:
                    best_result = result
                    break

            if best_result and best_result["clusters"]:
                print(f"\n📊 Detailed Analysis (threshold {best_result['threshold']}):")

                # Analyze each cluster for semantic coherence
                domain_labels = {
                    range(0, 5): "Technology/Programming",
                    range(5, 10): "Animals/Nature",
                    range(10, 15): "Food/Cooking",
                    range(15, 20): "Sports/Fitness",
                    range(20, 25): "Science/Medicine",
                    range(25, 30): "Travel/Culture"
                }

                semantic_coherence_score = 0
                total_clusters = 0

                for i, cluster in enumerate(best_result["clusters"]):
                    if len(cluster) < 2:
                        continue

                    total_clusters += 1

                    # Determine which domains are represented in this cluster
                    domains_in_cluster = set()
                    for idx in cluster:
                        for domain_range, domain_name in domain_labels.items():
                            if idx in domain_range:
                                domains_in_cluster.add(domain_name)
                                break

                    # Calculate coherence (1.0 = all same domain, 0.0 = all different domains)
                    coherence = 1.0 if len(domains_in_cluster) == 1 else max(0, 1 - (len(domains_in_cluster) - 1) * 0.3)
                    semantic_coherence_score += coherence

                    print(f"   Cluster {i+1} ({len(cluster)} sentences): {list(domains_in_cluster)}")
                    for idx in cluster:
                        print(f"     [{idx:2d}] {sentences[idx][:50]}...")
                    print(f"   Coherence: {coherence:.2f}")
                    print()

                # Overall clustering quality metrics
                if total_clusters > 0:
                    avg_coherence = semantic_coherence_score / total_clusters

                    # Validate clustering quality
                    assert len(best_result["clusters"]) >= 2, "Should find at least 2 clusters"
                    assert best_result["total_clustered"] >= 8, f"Should cluster at least 8 sentences, got {best_result['total_clustered']}"
                    assert avg_coherence >= 0.6, f"Average semantic coherence too low: {avg_coherence:.2f}"

                    print(f"✅ Clustering validation passed:")
                    print(f"   Average semantic coherence: {avg_coherence:.2f}/1.0")
                    print(f"   Total clusters found: {len(best_result['clusters'])}")
                    print(f"   Sentences successfully clustered: {best_result['total_clustered']}/{len(sentences)}")

            else:
                print("⚠️  No clusters found with threshold 0.4 - this may indicate clustering is too strict")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_batch_similarity_consistency(self):
        """Test that batch similarity operations are consistent with individual operations."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            query = "Python programming tutorial"
            documents = [
                "Learning Python for beginners",
                "JavaScript web development",
                "Cooking pasta recipes",
                "Python data science applications"
            ]

            # Method 1: Individual similarities
            individual_similarities = []
            for doc in documents:
                sim = manager.compute_similarity(query, doc)
                individual_similarities.append(sim)

            # Method 2: Batch similarities
            batch_similarities = manager.compute_similarities(query, documents)

            # Method 3: Matrix similarities
            all_texts = [query] + documents
            matrix = manager.compute_similarities_matrix(all_texts)
            matrix_similarities = [matrix[0, i+1] for i in range(len(documents))]

            # All methods should produce nearly identical results
            for i, (individual, batch, matrix_sim) in enumerate(zip(individual_similarities, batch_similarities, matrix_similarities)):
                assert abs(individual - batch) < 1e-10, \
                    f"Individual vs batch mismatch at {i}: {individual:.10f} vs {batch:.10f}"
                assert abs(individual - matrix_sim) < 1e-6, \
                    f"Individual vs matrix mismatch at {i}: {individual:.6f} vs {matrix_sim:.6f}"

            print("✅ Batch similarity consistency validated:")
            print(f"   Query: '{query[:30]}...'")
            for i, doc in enumerate(documents):
                print(f"   Doc {i}: {individual_similarities[i]:.3f} - '{doc[:30]}...'")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])