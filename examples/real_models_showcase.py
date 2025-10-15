#!/usr/bin/env python3
"""
Real SOTA Models Showcase
=========================

Demonstrates the latest SOTA embedding models available in AbstractCore Core:
- Google EmbeddingGemma (300M params, multilingual)
- IBM Granite (278M params, enterprise-grade)
- Performance comparison and real-world usage

This showcases actual model capabilities with real embeddings.
"""

import sys
from pathlib import Path
import time
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abstractcore.embeddings import EmbeddingManager, get_model_config, list_available_models


def demo_model_overview():
    """Overview of available SOTA models."""
    print("🚀 AbstractCore Core - SOTA Embedding Models")
    print("=" * 70)

    print("\n📋 Available Models:")
    models = list_available_models()
    for model_name in models:
        try:
            config = get_model_config(model_name)
            print(f"\n  🔹 {config.name}")
            print(f"     Model ID: {config.model_id}")
            print(f"     Dimensions: {config.dimension}D")
            print(f"     Multilingual: {'✅' if config.multilingual else '❌'}")
            print(f"     Matryoshka: {'✅' if config.supports_matryoshka else '❌'}")
            print(f"     Size: {config.size_mb}MB")
            print(f"     Description: {config.description}")
        except Exception as e:
            print(f"  ❌ {model_name}: Configuration error - {e}")


def demo_embeddinggemma():
    """Demonstrate Google EmbeddingGemma capabilities."""
    print("\n\n🔥 Google EmbeddingGemma Demo")
    print("=" * 50)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Initializing EmbeddingGemma (may take a moment to download)...")

            start_time = time.time()
            embedder = EmbeddingManager(
                model="embeddinggemma",
                cache_dir=temp_dir
            )
            init_time = time.time() - start_time
            print(f"✓ Model loaded in {init_time:.2f}s")

            # Test basic embedding
            print("\n1. Basic Embedding Generation:")
            text = "Artificial intelligence is revolutionizing technology and society."

            start_time = time.time()
            embedding = embedder.embed(text)
            embed_time = time.time() - start_time

            print(f"   Text: {text}")
            print(f"   Embedding: {len(embedding)}D vector")
            print(f"   Time: {embed_time*1000:.1f}ms")
            print(f"   Sample values: {[round(x, 4) for x in embedding[:5]]}")

            # Test multilingual capability
            print("\n2. Multilingual Support:")
            multilingual_texts = [
                "Hello, how are you today?",                    # English
                "Bonjour, comment allez-vous aujourd'hui?",     # French
                "Hola, ¿cómo estás hoy?",                      # Spanish
                "Guten Tag, wie geht es Ihnen heute?",         # German
                "こんにちは、今日はいかがですか？",                    # Japanese
                "Привет, как дела сегодня?"                     # Russian
            ]

            print("   Processing 6 languages...")
            start_time = time.time()
            multilingual_embeddings = embedder.embed_batch(multilingual_texts)
            batch_time = time.time() - start_time

            print(f"   ✓ Processed {len(multilingual_texts)} languages in {batch_time*1000:.1f}ms")
            print(f"   Average: {batch_time/len(multilingual_texts)*1000:.1f}ms per language")

            # Test Matryoshka truncation
            print("\n3. Matryoshka Dimension Truncation:")
            dimensions_to_test = [768, 512, 256, 128]

            for dim in dimensions_to_test:
                truncated_embedder = EmbeddingManager(
                    model="embeddinggemma",
                    cache_dir=temp_dir,
                    output_dims=dim
                )

                start_time = time.time()
                truncated_embedding = truncated_embedder.embed(text)
                truncate_time = time.time() - start_time

                print(f"   {dim}D: {truncate_time*1000:.1f}ms")
                assert len(truncated_embedding) == dim

            # Test semantic similarity
            print("\n4. Semantic Understanding:")
            test_pairs = [
                ("machine learning", "artificial intelligence"),
                ("dog", "puppy"),
                ("car", "vehicle"),
                ("happy", "joyful"),
                ("computer", "laptop")
            ]

            for text1, text2 in test_pairs:
                similarity = embedder.compute_similarity(text1, text2)
                print(f"   '{text1}' ↔ '{text2}': {similarity:.3f}")

            print("\n✅ EmbeddingGemma demo completed successfully!")

    except Exception as e:
        print(f"❌ EmbeddingGemma demo failed: {e}")
        if "offline" in str(e).lower() or "not found" in str(e).lower():
            print("   Note: Model download requires internet connection")


def demo_granite():
    """Demonstrate IBM Granite capabilities."""
    print("\n\n🏢 IBM Granite Demo")
    print("=" * 40)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Initializing IBM Granite...")

            # Try with short model name first
            try:
                embedder = EmbeddingManager(
                    model="granite",
                    cache_dir=temp_dir
                )
                print("✓ Loaded via short name: 'granite'")
            except:
                # Fallback to direct HuggingFace ID
                embedder = EmbeddingManager(
                    model="ibm-granite/granite-embedding-278m-multilingual",
                    cache_dir=temp_dir
                )
                print("✓ Loaded via HuggingFace ID")

            # Test enterprise/business content
            print("\n1. Enterprise Content Processing:")
            business_texts = [
                "Q4 financial results show 15% revenue growth with strong market positioning.",
                "Supply chain optimization initiatives reduced operational costs by 12% this quarter.",
                "Customer satisfaction scores improved to 4.2/5 with enhanced service delivery.",
                "Digital transformation roadmap includes cloud migration and AI integration.",
                "Risk management framework ensures compliance with regulatory requirements."
            ]

            start_time = time.time()
            business_embeddings = embedder.embed_batch(business_texts)
            batch_time = time.time() - start_time

            print(f"   ✓ Processed {len(business_texts)} business documents")
            print(f"   Time: {batch_time*1000:.1f}ms ({batch_time/len(business_texts)*1000:.1f}ms per doc)")

            # Test business semantic search
            print("\n2. Business Semantic Search:")
            business_queries = [
                "financial performance metrics",
                "cost reduction strategies",
                "customer experience improvements",
                "technology modernization plans"
            ]

            for query in business_queries:
                print(f"\n   Query: '{query}'")
                similarities = []
                for doc in business_texts:
                    sim = embedder.compute_similarity(query, doc)
                    similarities.append((sim, doc))

                # Find best match
                best_score, best_doc = max(similarities, key=lambda x: x[0])
                print(f"   Best match ({best_score:.3f}): {best_doc[:60]}...")

            print("\n✅ Granite demo completed successfully!")

    except Exception as e:
        print(f"❌ Granite demo failed: {e}")
        if "offline" in str(e).lower() or "not found" in str(e).lower():
            print("   Note: Model may not be available or requires internet connection")


def demo_performance_comparison():
    """Compare performance across models."""
    print("\n\n⚡ Performance Comparison")
    print("=" * 45)

    test_text = "Advanced machine learning algorithms enable intelligent applications."
    models_to_test = [
        ("sentence-transformers/all-MiniLM-L6-v2", "Baseline"),
        ("embeddinggemma", "Google SOTA"),
        ("granite", "IBM Enterprise")
    ]

    results = {}

    print(f"Test text: '{test_text}'")
    print("\nModel Performance:")

    for model_id, description in models_to_test:
        print(f"\n🔹 Testing {description} ({model_id})")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialization time
                start_time = time.time()
                embedder = EmbeddingManager(model=model_id, cache_dir=temp_dir)
                init_time = time.time() - start_time

                # First embedding time (includes any setup)
                start_time = time.time()
                embedding = embedder.embed(test_text)
                first_embed_time = time.time() - start_time

                # Cached embedding time
                start_time = time.time()
                cached_embedding = embedder.embed(test_text)
                cached_time = time.time() - start_time

                # Verify consistency
                assert embedding == cached_embedding

                results[model_id] = {
                    "description": description,
                    "dimension": len(embedding),
                    "init_time": init_time,
                    "first_embed_time": first_embed_time,
                    "cached_time": cached_time,
                    "status": "✅ Success"
                }

                print(f"   Dimension: {len(embedding)}D")
                print(f"   Init time: {init_time:.2f}s")
                print(f"   First embed: {first_embed_time*1000:.1f}ms")
                print(f"   Cached embed: {cached_time*1000:.3f}ms")

        except Exception as e:
            results[model_id] = {
                "description": description,
                "status": f"❌ Failed: {str(e)[:50]}..."
            }
            print(f"   ❌ Failed: {e}")

    # Summary table
    print("\n📊 Performance Summary:")
    print("┌─────────────────────────┬───────────┬──────────────┬─────────────┐")
    print("│ Model                   │ Dimension │ First Embed  │ Cached      │")
    print("├─────────────────────────┼───────────┼──────────────┼─────────────┤")

    for model_id, result in results.items():
        if "Success" in result.get("status", ""):
            model_name = result["description"][:23]
            dimension = f"{result['dimension']}D"
            first_time = f"{result['first_embed_time']*1000:.1f}ms"
            cached_time = f"{result['cached_time']*1000:.3f}ms"
            print(f"│ {model_name:<23} │ {dimension:<9} │ {first_time:<12} │ {cached_time:<11} │")

    print("└─────────────────────────┴───────────┴──────────────┴─────────────┘")


def demo_real_world_rag():
    """Demonstrate real-world RAG application."""
    print("\n\n🤖 Real-World RAG Application")
    print("=" * 50)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use EmbeddingGemma for best quality
            print("Setting up RAG system with EmbeddingGemma...")
            embedder = EmbeddingManager(model="embeddinggemma", cache_dir=temp_dir)

            # Knowledge base about AbstractCore
            knowledge_base = [
                "AbstractCore Core provides a unified interface to multiple LLM providers including OpenAI, Anthropic, Ollama, and local models.",
                "The embeddings system in AbstractCore uses state-of-the-art models like Google EmbeddingGemma and IBM Granite for semantic search.",
                "Vector embeddings convert text into high-dimensional numerical representations that capture semantic meaning and enable similarity search.",
                "RAG (Retrieval-Augmented Generation) combines information retrieval with language generation to provide more accurate and contextual responses.",
                "AbstractCore Core includes production features like automatic retry logic, comprehensive event systems, and structured output validation.",
                "The library supports streaming responses, tool calling, and session management for building complex AI applications.",
                "Performance optimizations include ONNX backend support, Matryoshka dimension truncation, and intelligent caching mechanisms."
            ]

            print(f"Knowledge base: {len(knowledge_base)} documents")

            # Real user questions
            questions = [
                "What LLM providers does AbstractCore support?",
                "What embedding models are available?",
                "How do vector embeddings work?",
                "What is RAG and why is it useful?",
                "What production features are included?",
                "What performance optimizations are available?"
            ]

            print("\nProcessing questions with RAG pipeline...")

            for i, question in enumerate(questions, 1):
                print(f"\n{i}. Question: {question}")

                # Find best context
                best_score = 0
                best_context = ""

                for doc in knowledge_base:
                    similarity = embedder.compute_similarity(question, doc)
                    if similarity > best_score:
                        best_score = similarity
                        best_context = doc

                print(f"   Best context (score: {best_score:.3f}):")
                print(f"   {best_context[:80]}...")

                # In production, you would send this to an LLM:
                rag_prompt = f"""Context: {best_context}

Question: {question}

Based on the provided context, please provide a helpful answer:"""

                print(f"   ✓ RAG prompt prepared ({len(rag_prompt)} chars)")

            print("\n✅ RAG demo completed - ready for LLM integration!")

    except Exception as e:
        print(f"❌ RAG demo failed: {e}")


def main():
    """Run all demonstrations."""
    try:
        demo_model_overview()
        demo_embeddinggemma()
        demo_granite()
        demo_performance_comparison()
        demo_real_world_rag()

        print("\n\n🎉 All Demos Completed Successfully!")
        print("\n📚 Next Steps:")
        print("• Integrate with LLM providers: create_llm('openai', model='gpt-4o-mini')")
        print("• Try ONNX optimization: backend='onnx'")
        print("• Use Matryoshka truncation: output_dims=256")
        print("• Build production RAG applications")
        print("• See docs/embeddings.md for comprehensive documentation")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("Note: Some models may require internet connection for first-time download")


if __name__ == "__main__":
    main()