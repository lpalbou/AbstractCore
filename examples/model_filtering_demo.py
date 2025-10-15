"""
Demo: Model Filtering by Type

Demonstrates the /v1/models endpoint with type filtering for
text-generation and text-embedding models.
"""

import requests
import json


def list_models(base_url: str = "http://localhost:8000", provider: str = None, model_type: str = None):
    """
    List models from the server with optional filtering.
    
    Args:
        base_url: Server URL
        provider: Optional provider filter ('ollama', 'lmstudio', etc.)
        model_type: Optional type filter ('text-generation', 'text-embedding')
    """
    params = {}
    if provider:
        params['provider'] = provider
    if model_type:
        params['type'] = model_type
    
    response = requests.get(f"{base_url}/v1/models", params=params)
    
    if response.status_code == 200:
        data = response.json()
        models = data.get('data', [])
        return models
    else:
        print(f"Error: {response.status_code}")
        return []


def demo_no_filter():
    """Demo: List all models (no filtering)"""
    print("\n=== All Models (No Filter) ===")
    models = list_models()
    print(f"Total models: {len(models)}")
    for model in models[:5]:  # Show first 5
        print(f"  - {model['id']}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")


def demo_embedding_only():
    """Demo: List only embedding models"""
    print("\n=== Embedding Models Only ===")
    models = list_models(model_type="text-embedding")
    print(f"Total embedding models: {len(models)}")
    for model in models:
        print(f"  - {model['id']}")


def demo_text_generation_only():
    """Demo: List only text generation models"""
    print("\n=== Text Generation Models Only ===")
    models = list_models(model_type="text-generation")
    print(f"Total text generation models: {len(models)}")
    for model in models[:10]:  # Show first 10
        print(f"  - {model['id']}")
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more")


def demo_provider_and_type():
    """Demo: Filter by provider AND type"""
    print("\n=== Ollama Embedding Models ===")
    models = list_models(provider="ollama", model_type="text-embedding")
    print(f"Total Ollama embedding models: {len(models)}")
    for model in models:
        print(f"  - {model['id']}")
    
    print("\n=== Ollama Text Generation Models ===")
    models = list_models(provider="ollama", model_type="text-generation")
    print(f"Total Ollama text generation models: {len(models)}")
    for model in models[:5]:  # Show first 5
        print(f"  - {model['id']}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")


def demo_embedding_detection():
    """Demo: Show how models are classified"""
    print("\n=== Model Classification Examples ===")
    
    test_models = [
        "granite-embedding:278m",
        "all-MiniLM-L6-v2",
        "nomic-embed-text-v1.5",
        "bert-base-uncased",
        "nomic-bert-2048",
        "llama3:latest",
        "qwen3-coder:30b",
        "text-embedding-ada-002",
        "gpt-4",
        "claude-3-opus"
    ]
    
    # Simple heuristic matching the server logic
    def is_embedding(name):
        patterns = ["embed", "all-minilm", "all-mpnet", "nomic-embed", "bert-", "-bert",
                   "bge-", "gte-", "e5-", "instructor-", "granite-embedding"]
        return any(p in name.lower() for p in patterns)
    
    for model in test_models:
        model_type = "embedding" if is_embedding(model) else "text-generation"
        print(f"  {model:<35} → {model_type}")


if __name__ == "__main__":
    print("=" * 70)
    print("Model Filtering Demo")
    print("=" * 70)
    print("\nThis demo shows how to filter models by type using the /v1/models endpoint.")
    print("Make sure the AbstractCore server is running on http://localhost:8000")
    print("=" * 70)
    
    try:
        # Demo 1: No filtering
        demo_no_filter()
        
        # Demo 2: Embedding models only
        demo_embedding_only()
        
        # Demo 3: Text generation models only
        demo_text_generation_only()
        
        # Demo 4: Combined filters
        demo_provider_and_type()
        
        # Demo 5: Classification logic
        demo_embedding_detection()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server at http://localhost:8000")
        print("Please make sure the AbstractCore server is running:")
        print("  python -m abstractcore.server.app")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nAPI Usage Examples:")
    print("  # All models")
    print("  GET /v1/models")
    print()
    print("  # Only embedding models")
    print("  GET /v1/models?type=text-embedding")
    print()
    print("  # Only text generation models")
    print("  GET /v1/models?type=text-generation")
    print()
    print("  # Ollama embedding models")
    print("  GET /v1/models?provider=ollama&type=text-embedding")
    print("=" * 70)

