#!/usr/bin/env python3
"""
Favored HuggingFace Embedding Models Demo
=========================================

Demonstrates all the favored embedding models with proper HuggingFace caching.
All models are cached in ~/.cache/huggingface/ and reused automatically.
"""

import sys
from pathlib import Path

# Add parent directory to path to import abstractcore
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractcore.embeddings import EmbeddingManager
from abstractcore.embeddings.models import list_available_models, get_model_config


def main():
    """Demonstrate all favored HuggingFace embedding models."""
    print("🎯 Favored HuggingFace Embedding Models Demo")
    print("=" * 60)

    # Show available models
    available = list_available_models()
    print(f"\n📋 Available Model Aliases ({len(available)} total):")
    for model in available:
        config = get_model_config(model)
        multilingual = "✅" if config.multilingual else "❌"
        matryoshka = "✅" if config.supports_matryoshka else "❌"
        print(f"  {model:20} -> {config.model_id}")
        print(f"    {config.description}")
        print(f"    📏 {config.dimension:4d} dims | 💾 {config.size_mb:3.0f}MB | 🌍 {multilingual} | 🎯 {matryoshka}")
        print()

    # Test sample from each category
    test_models = {
        "Default (Lightweight)": "all-minilm-l6-v2",
        "Ultra-Lightweight (English-only)": "granite-30m",
        "Balanced Multilingual": "granite-107m",
        "High-Quality Multilingual": "granite-278m",
        "SOTA Google Model": "embeddinggemma",
        "Large Qwen Model": "qwen3-embedding",
        "Nomic v1.5": "nomic-embed-v1.5"
    }

    print("\n🚀 Testing Representative Models:")
    print("=" * 60)

    test_text = "Machine learning transforms how we process information"
    results = []

    for category, model_alias in test_models.items():
        print(f"\n{category}:")
        try:
            # Initialize model (will use HF cache)
            embedder = EmbeddingManager(model=model_alias)
            config = get_model_config(model_alias)

            print(f"  📋 Model: {embedder.model_id}")
            print(f"  📏 Dimensions: {embedder.get_dimension()}")
            print(f"  🌍 Multilingual: {'Yes' if config.multilingual else 'No (English-only)'}")
            print(f"  💾 Size: {config.size_mb}MB")

            # Generate embedding
            embedding = embedder.embed(test_text)
            print(f"  ✅ Generated: {len(embedding)} dimensions")
            print(f"  📊 Sample values: {[f'{x:.3f}' for x in embedding[:3]]}")

            # Test similarity
            similarity = embedder.compute_similarity("artificial intelligence", "machine learning")
            print(f"  🔗 AI/ML similarity: {similarity:.3f}")

            results.append({
                'category': category,
                'model': model_alias,
                'dimensions': len(embedding),
                'similarity': similarity,
                'multilingual': config.multilingual,
                'size_mb': config.size_mb
            })

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}...")

    # Summary comparison
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("=" * 60)
    print(f"{'Model':<20} {'Dims':<5} {'Size':<6} {'Multi':<6} {'AI/ML Sim':<10}")
    print("-" * 60)

    for result in results:
        multilingual = "Yes" if result['multilingual'] else "No"
        print(f"{result['model']:<20} {result['dimensions']:<5} {result['size_mb']:<4.0f}MB {multilingual:<6} {result['similarity']:<10.3f}")

    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 30)
    print("🔸 **Default choice**: all-minilm-l6-v2 (fast, lightweight, reliable)")
    print("🔸 **Resource-constrained**: granite-30m (30MB, English-only)")
    print("🔸 **Production multilingual**: granite-278m (high quality, 278MB)")
    print("🔸 **SOTA performance**: embeddinggemma (Google's latest, 300MB)")
    print("🔸 **Large capacity**: qwen3-embedding (1024 dims, multilingual)")

    print(f"\n✅ All models cached in ~/.cache/huggingface/ for reuse!")
    print("🚀 Ready for production with your favored embedding models!")


if __name__ == "__main__":
    main()