#!/usr/bin/env python3
"""
Test script to verify VLM token calculator compatibility with open source LLMs
from LMStudio, Ollama, and HuggingFace providers.

This script will:
1. List available models from each provider
2. Check which models have VLM capabilities in our model_capabilities.json
3. Identify missing VLM model configurations that need research
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set
from abstractcore.providers.registry import ProviderRegistry
from abstractcore.utils.vlm_token_calculator import VLMTokenCalculator

def load_model_capabilities() -> Dict:
    """Load model capabilities from JSON file."""
    capabilities_path = Path("abstractcore/assets/model_capabilities.json")
    with open(capabilities_path, 'r') as f:
        return json.load(f)

def check_vision_support(model_name: str, capabilities: Dict) -> Dict:
    """Check if a model has vision support in our capabilities database."""
    models = capabilities.get("models", {})
    
    # Direct match
    if model_name in models:
        model_info = models[model_name]
        return {
            "found": True,
            "vision_support": model_info.get("vision_support", False),
            "image_tokenization_method": model_info.get("image_tokenization_method"),
            "image_patch_size": model_info.get("image_patch_size"),
            "max_image_tokens": model_info.get("max_image_tokens"),
            "match_type": "exact"
        }
    
    # Fuzzy matching for common patterns
    model_lower = model_name.lower()
    for db_model, model_info in models.items():
        db_model_lower = db_model.lower()
        
        # Check for partial matches (e.g., "llama3.2-vision:11b" matches "llama3.2-vision")
        if any([
            model_lower in db_model_lower,
            db_model_lower in model_lower,
            # Handle version variations
            model_lower.replace(":", "-") == db_model_lower,
            model_lower.replace("-", ":") == db_model_lower,
            # Handle common aliases
            model_lower.replace("vl", "-vl") == db_model_lower,
            model_lower.replace("-vl", "vl") == db_model_lower
        ]):
            return {
                "found": True,
                "vision_support": model_info.get("vision_support", False),
                "image_tokenization_method": model_info.get("image_tokenization_method"),
                "image_patch_size": model_info.get("image_patch_size"),
                "max_image_tokens": model_info.get("max_image_tokens"),
                "match_type": "fuzzy",
                "matched_db_model": db_model
            }
    
    return {"found": False}

def test_vlm_calculator_with_model(calculator: VLMTokenCalculator, model_name: str, provider: str) -> Dict:
    """Test VLM token calculator with a specific model."""
    try:
        result = calculator.calculate_tokens_for_image(
            width=1024, height=768,
            provider=provider,
            model=model_name
        )
        return {
            "success": True,
            "tokens": result.get("tokens"),
            "method": result.get("method"),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "tokens": None,
            "method": None,
            "error": str(e)
        }

def main():
    print("🔍 Testing VLM Token Calculator compatibility with open source providers...")
    print("=" * 80)
    
    # Load model capabilities
    try:
        capabilities = load_model_capabilities()
        print(f"✅ Loaded model capabilities database with {len(capabilities.get('models', {}))} models")
    except Exception as e:
        print(f"❌ Failed to load model capabilities: {e}")
        return
    
    # Initialize registry and calculator
    registry = ProviderRegistry()
    calculator = VLMTokenCalculator()
    
    # Test each provider
    providers_to_test = ["ollama", "lmstudio", "huggingface"]
    
    all_vision_models = []
    missing_models = []
    
    for provider_name in providers_to_test:
        print(f"\n📡 Testing {provider_name.upper()} provider...")
        print("-" * 40)
        
        try:
            # Get available models
            models = registry.get_available_models(provider_name)
            print(f"Found {len(models)} models from {provider_name}")
            
            if not models:
                print(f"⚠️  No models available from {provider_name} (provider may be offline)")
                continue
            
            # Check each model for vision capabilities
            vision_models = []
            
            for model in models[:10]:  # Limit to first 10 for testing
                print(f"  Checking: {model}")
                
                # Check if model is in our database
                model_info = check_vision_support(model, capabilities)
                
                if model_info["found"]:
                    if model_info["vision_support"]:
                        vision_models.append({
                            "name": model,
                            "provider": provider_name,
                            "db_info": model_info
                        })
                        print(f"    ✅ Vision model found in database ({model_info['match_type']} match)")
                        
                        # Test VLM calculator
                        calc_result = test_vlm_calculator_with_model(calculator, model, provider_name)
                        if calc_result["success"]:
                            print(f"    🧮 Calculator works: {calc_result['tokens']} tokens ({calc_result['method']})")
                        else:
                            print(f"    ❌ Calculator failed: {calc_result['error']}")
                    else:
                        print(f"    ℹ️  Text-only model (no vision support)")
                else:
                    # Check if model name suggests vision capabilities
                    model_lower = model.lower()
                    if any(keyword in model_lower for keyword in ["vision", "vl", "multimodal", "mm"]):
                        missing_models.append({
                            "name": model,
                            "provider": provider_name,
                            "reason": "Vision model not in database"
                        })
                        print(f"    ⚠️  Potential vision model missing from database!")
                    else:
                        print(f"    ℹ️  Model not in database (likely text-only)")
            
            all_vision_models.extend(vision_models)
            print(f"\n📊 {provider_name}: {len(vision_models)} vision models found")
            
        except Exception as e:
            print(f"❌ Error testing {provider_name}: {e}")
    
    # Summary report
    print("\n" + "=" * 80)
    print("📋 SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\n✅ Vision models with database coverage: {len(all_vision_models)}")
    for model in all_vision_models:
        print(f"  • {model['provider']}: {model['name']}")
        if model['db_info'].get('image_tokenization_method'):
            print(f"    Method: {model['db_info']['image_tokenization_method']}")
    
    print(f"\n⚠️  Potential vision models missing from database: {len(missing_models)}")
    for model in missing_models:
        print(f"  • {model['provider']}: {model['name']} - {model['reason']}")
    
    if missing_models:
        print(f"\n🔍 RESEARCH NEEDED:")
        print("The following models appear to have vision capabilities but are missing from our database.")
        print("Research is required to find their VLM tokenization parameters:")
        
        unique_missing = set()
        for model in missing_models:
            unique_missing.add(model['name'])
        
        for model_name in sorted(unique_missing):
            print(f"  - {model_name}")
    
    print(f"\n🧮 VLM Token Calculator Status:")
    print(f"  • Supports {len(all_vision_models)} vision models across open source providers")
    print(f"  • Missing coverage for {len(missing_models)} potential vision models")
    
    if missing_models:
        print(f"\n⚡ Next Steps:")
        print("1. Research VLM tokenization parameters for missing models")
        print("2. Add missing models to model_capabilities.json")
        print("3. Update VLM token calculator with new model-specific configs")
        return 1
    else:
        print(f"\n🎉 All detected vision models have database coverage!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
