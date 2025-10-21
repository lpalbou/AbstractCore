#!/usr/bin/env python3
"""
Complete RAG Application Example
===============================

This example demonstrates a complete Retrieval-Augmented Generation (RAG)
application using AbstractCore Core embeddings with real LLMs.

This is a real, working example you can adapt for production use.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abstractcore.embeddings import EmbeddingManager
from abstractcore import create_llm


class SimpleRAGSystem:
    """A simple but complete RAG system."""

    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", llm_provider="openai"):
        """Initialize the RAG system.

        Args:
            embedding_model: Embedding model to use
            llm_provider: LLM provider ('openai', 'anthropic', 'ollama', etc.)
        """
        self.embedder = EmbeddingManager(model=embedding_model)
        self.llm = create_llm(llm_provider)
        self.knowledge_base = []
        self.embeddings_cache = []

    def add_document(self, text: str):
        """Add a document to the knowledge base."""
        self.knowledge_base.append(text)
        # Pre-compute embedding for faster retrieval
        embedding = self.embedder.embed(text)
        self.embeddings_cache.append(embedding)
        print(f"✓ Added document: {text[:50]}...")

    def add_documents(self, documents: list):
        """Add multiple documents efficiently."""
        self.knowledge_base.extend(documents)
        # Batch compute embeddings for efficiency
        new_embeddings = self.embedder.embed_batch(documents)
        self.embeddings_cache.extend(new_embeddings)
        print(f"✓ Added {len(documents)} documents to knowledge base")

    def search(self, query: str, top_k: int = 3):
        """Search for most relevant documents."""
        if not self.knowledge_base:
            return []

        # Get query embedding
        query_embedding = self.embedder.embed(query)

        # Calculate similarities with all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings_cache):
            # Manual cosine similarity calculation
            import numpy as np
            q_emb = np.array(query_embedding)
            d_emb = np.array(doc_embedding)
            similarity = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
            similarities.append((similarity, i, self.knowledge_base[i]))

        # Sort by similarity and return top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:top_k]

    def ask(self, question: str, top_k: int = 2):
        """Ask a question using RAG."""
        print(f"\n🤖 Question: {question}")

        # Step 1: Retrieve relevant context
        print("🔍 Searching knowledge base...")
        results = self.search(question, top_k)

        if not results:
            return "No relevant information found in knowledge base."

        # Step 2: Prepare context
        contexts = []
        print("📄 Retrieved contexts:")
        for score, idx, doc in results:
            print(f"   Score: {score:.3f} - {doc[:60]}...")
            contexts.append(doc)

        combined_context = "\n\n".join(contexts)

        # Step 3: Create RAG prompt
        rag_prompt = f"""Context:
{combined_context}

Question: {question}

Based on the provided context, please answer the question. If the context doesn't contain enough information, say so.

Answer:"""

        # Step 4: Generate answer
        print("🧠 Generating answer...")
        response = self.llm.generate(rag_prompt)

        return response.content

    def get_stats(self):
        """Get system statistics."""
        cache_stats = self.embedder.get_cache_stats()
        return {
            "documents": len(self.knowledge_base),
            "embeddings_cached": cache_stats["persistent_cache_size"],
            "memory_cache_hits": cache_stats["memory_cache_info"]["hits"],
            "embedding_dimension": cache_stats["embedding_dimension"]
        }


def demo_programming_rag():
    """Demo RAG system with programming knowledge."""
    print("🚀 Programming Knowledge RAG Demo")
    print("=" * 60)

    # Initialize RAG system
    rag = SimpleRAGSystem()

    # Add programming knowledge
    programming_docs = [
        "Python is a high-level programming language created by Guido van Rossum. It was first released in 1991 and emphasizes code readability with its notable use of significant whitespace.",

        "React is a JavaScript library for building user interfaces. It was created by Facebook and released in 2013. React allows developers to create reusable UI components and manage application state efficiently.",

        "Docker is a containerization platform that packages applications and their dependencies into lightweight, portable containers. It was first released in 2013 and revolutionized application deployment.",

        "PostgreSQL is an advanced open-source relational database management system. It supports both SQL and JSON querying and is known for its reliability, feature robustness, and performance.",

        "Kubernetes is an open-source container orchestration platform originally developed by Google. It automates deployment, scaling, and management of containerized applications across clusters.",

        "FastAPI is a modern Python web framework for building APIs. It provides automatic API documentation, type hints support, and high performance comparable to NodeJS and Go.",

        "Git is a distributed version control system created by Linus Torvalds in 2005. It tracks changes in source code during software development and enables collaboration among developers.",

        "TensorFlow is an open-source machine learning framework developed by Google. It provides tools for building and deploying machine learning models at scale."
    ]

    rag.add_documents(programming_docs)

    # Ask questions
    questions = [
        "Who created Python and when was it released?",
        "What is React used for?",
        "What are the main benefits of Docker?",
        "When was Git created and by whom?",
        "What company developed TensorFlow?"
    ]

    for question in questions:
        answer = rag.ask(question)
        print(f"💬 Answer: {answer}")
        print("-" * 60)

    # Show stats
    stats = rag.get_stats()
    print(f"\n📊 System Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


def demo_custom_knowledge_rag():
    """Demo with custom knowledge domain."""
    print("\n\n🌍 Custom Knowledge RAG Demo")
    print("=" * 60)

    # Initialize RAG system
    rag = SimpleRAGSystem()

    # Add custom knowledge about cities
    city_docs = [
        "Paris is the capital and largest city of France. It has a population of over 2 million people within the city limits and is known for landmarks like the Eiffel Tower and Louvre Museum.",

        "Tokyo is the capital of Japan and one of the world's most populous metropolitan areas. It is a major financial center and known for its technology, culture, and cuisine.",

        "New York City is the most populous city in the United States. It consists of five boroughs and is known as a global center for finance, arts, fashion, and commerce.",

        "London is the capital of the United Kingdom and England. It is situated on the River Thames and is known for its history, museums, theaters, and financial district.",

        "Sydney is the largest city in Australia and the capital of New South Wales. It is famous for its Opera House, Harbour Bridge, and beautiful beaches."
    ]

    rag.add_documents(city_docs)

    # Interactive demo
    questions = [
        "What is the population of Paris?",
        "What is Tokyo known for?",
        "Which boroughs make up New York City?",
        "What river is London situated on?",
        "What landmarks is Sydney famous for?"
    ]

    for question in questions:
        answer = rag.ask(question)
        print(f"💬 Answer: {answer}")
        print("-" * 60)


def main():
    """Run the complete RAG demonstration."""
    try:
        demo_programming_rag()
        demo_custom_knowledge_rag()

        print("\n✅ RAG Demo Complete!")
        print("\n🔧 To use with real LLMs:")
        print("• Configure your preferred provider: 'openai', 'anthropic', 'ollama'")
        print("• Add your API key: create_llm('openai', api_key='your-key')")
        print("• Consider using 'embeddinggemma' for better embeddings")
        print("• Scale up knowledge base for production use")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install sentence-transformers")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()