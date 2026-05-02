#!/usr/bin/env python3
"""
Clean HuggingFace Embedding Demo
===============================

Simple demonstration of the clean EmbeddingManager interface using HuggingFace models only.
All models run locally with no external API dependencies.
"""

from abstractcore.embeddings import EmbeddingManager


def main():
    """Demonstrate clean HuggingFace embedding usage."""
    print("🚀 HuggingFace Embedding Demo")
    print("=" * 50)

    # Test 1: Default model (sentence-transformers/all-MiniLM-L6-v2)
    print("\n1️⃣ Default Model")
    embedder = EmbeddingManager()
    print(f"   Model: {embedder.model_id}")
    print(f"   Dimensions: {embedder.get_dimension()}")

    # Test 2: Generate single embedding
    print("\n2️⃣ Single Embedding")
    text = "Machine learning transforms how we process information"
    embedding = embedder.embed(text)
    print(f"   Text: {text}")
    print(f"   Embedding: {len(embedding)} dimensions")
    print(f"   First 5 values: {[f'{x:.3f}' for x in embedding[:5]]}")

    # Test 3: Batch embeddings
    print("\n3️⃣ Batch Embeddings")
    texts = [
        "Python programming language",
        "JavaScript web development",
        "Machine learning with Python",
        "Data science and analytics"
    ]

    batch_embeddings = embedder.embed_batch(texts)
    print(f"   Generated {len(batch_embeddings)} embeddings")
    for i, (text, emb) in enumerate(zip(texts, batch_embeddings)):
        print(f"   {i+1}. {text[:30]}... -> {len(emb)} dims")

    # Test 4: Similarity analysis
    print("\n4️⃣ Similarity Analysis")
    pairs = [
        ("cat", "kitten"),
        ("Python programming", "JavaScript development"),
        ("cooking recipes", "machine learning"),
    ]

    for text1, text2 in pairs:
        similarity = embedder.compute_similarity(text1, text2)
        print(f"   '{text1}' vs '{text2}': {similarity:.3f}")

    # Test 5: Semantic search
    print("\n5️⃣ Semantic Search")
    documents = [
        "Python is excellent for data science and machine learning applications",
        "JavaScript enables interactive web pages and modern frontend development",
        "SQL databases store and query structured data efficiently",
        "Machine learning algorithms can predict patterns from historical data"
    ]

    query = "data science programming"
    print(f"   Query: {query}")
    print(f"   Documents:")

    similarities = []
    for i, doc in enumerate(documents):
        similarity = embedder.compute_similarity(query, doc)
        similarities.append((similarity, i, doc))

    # Sort by similarity (highest first)
    similarities.sort(reverse=True)

    for rank, (sim, idx, doc) in enumerate(similarities, 1):
        print(f"   {rank}. Score: {sim:.3f} - {doc[:50]}...")

    # Test 6: Different model
    print("\n6️⃣ Different Model (EmbeddingGemma)")
    try:
        embedder_gemma = EmbeddingManager(model="google/embeddinggemma-300m")
        print(f"   Model: {embedder_gemma.model_id}")
        print(f"   Dimensions: {embedder_gemma.get_dimension()}")

        # Test with same text
        gemma_embedding = embedder_gemma.embed(text)
        print(f"   Same text embedding: {len(gemma_embedding)} dimensions")

    except Exception as e:
        print(f"   ⚠️  EmbeddingGemma unavailable: {e}")

    print(f"\n✅ Demo completed successfully!")
    print(f"   All embeddings generated locally using HuggingFace models")
    print(f"   No external API dependencies required")


if __name__ == "__main__":
    main()
