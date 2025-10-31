#!/usr/bin/env python3
"""
Comprehensive verification script for VLM token calculator compatibility
with open source LLMs from LMStudio, Ollama, and HuggingFace providers.
"""

import json
from pathlib import Path
from typing import Dict, List
from abstractcore.providers.registry import ProviderRegistry
from abstractcore.utils.vlm_token_calculator import VLMTokenCalculator

def load_model_capabilities() -> Dict:
    """Load model capabilities from JSON file."""
    capabilities_path = Path("abstractcore/assets/model_capabilities.json")
    with open(capabilities_path, 'r') as f:
        return json.load(f)

def is_likely_vision_model(model_name: str) -> bool:
    """
    Determine if a model name suggests vision capabilities.
    Uses more refined heuristics to avoid false positives.
    """
    model_lower = model_name.lower()
    
    # Exclude embedding models (these are not VLMs)
    if any(term in model_lower for term in ["embedding", "embed"]):
        return False
    
    # Exclude obvious text-only models
    if any(term in model_lower for term in ["coder", "code", "instruct-only", "text-only"]):
        return False
    
    # Look for vision-specific keywords
    vision_keywords = [
        "vision", "vl", "multimodal", "mm", "visual", "image", 
        "clip", "blip", "paligemma", "llava"
    ]
    
    return any(keyword in model_lower for keyword in vision_keywords)

def check_model_in_database(model_name: str, capabilities: Dict) -> Dict:
    """Check if model exists in our capabilities database with enhanced matching."""
    models = capabilities.get("models", {})
    
    # Direct match
    if model_name in models:
        model_info = models[model_name]
        return {
            "found": True,
            "vision_support": model_info.get("vision_support", False),
            "image_tokenization_method": model_info.get("image_tokenization_method"),
            "match_type": "exact",
            "db_model": model_name
        }
    
    # Enhanced fuzzy matching
    model_lower = model_name.lower()
    for db_model, model_info in models.items():
        db_model_lower = db_model.lower()
        
        # Handle various naming conventions
        if any([
            # Direct substring matches
            model_lower in db_model_lower,
            db_model_lower in model_lower,
            # Version variations (: vs -)
            model_lower.replace(":", "-") == db_model_lower,
            model_lower.replace("-", ":") == db_model_lower,
            # VL variations
            model_lower.replace("vl", "-vl") == db_model_lower,
            model_lower.replace("-vl", "vl") == db_model_lower,
            # Handle provider prefixes
            model_lower.split("/")[-1] == db_model_lower,
            db_model_lower.split("/")[-1] == model_lower,
            # Handle common aliases
            model_lower.replace("_", "-") == db_model_lower,
            model_lower.replace("-", "_") == db_model_lower
        ]):
            return {
                "found": True,
                "vision_support": model_info.get("vision_support", False),
                "image_tokenization_method": model_info.get("image_tokenization_method"),
                "match_type": "fuzzy",
                "db_model": db_model
            }
    
    return {"found": False}

def test_vlm_calculator(calculator: VLMTokenCalculator, model_name: str, provider: str) -> Dict:
    """Test VLM token calculator with a model."""
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
    print("🔍 COMPREHENSIVE VLM TOKEN CALCULATOR COMPATIBILITY TEST")
    print("=" * 80)
    
    # Load capabilities
    try:
        capabilities = load_model_capabilities()
        print(f"✅ Loaded {len(capabilities.get('models', {}))} models from database")
    except Exception as e:
        print(f"❌ Failed to load capabilities: {e}")
        return 1
    
    # Initialize components
    registry = ProviderRegistry()
    calculator = VLMTokenCalculator()
    
    # Test providers
    providers = ["ollama", "lmstudio", "huggingface"]
    
    summary = {
        "total_models_tested": 0,
        "vision_models_found": 0,
        "calculator_compatible": 0,
        "false_positives_filtered": 0,
        "providers": {}
    }
    
    for provider_name in providers:
        print(f"\n📡 TESTING {provider_name.upper()} PROVIDER")
        print("-" * 50)
        
        provider_summary = {
            "total_models": 0,
            "vision_models": 0,
            "calculator_working": 0,
            "models": []
        }
        
        try:
            models = registry.get_available_models(provider_name)
            provider_summary["total_models"] = len(models)
            summary["total_models_tested"] += len(models)
            
            if not models:
                print(f"⚠️  No models available (provider offline or no models)")
                continue
            
            print(f"Found {len(models)} models")
            
            # Test each model (limit to first 15 for performance)
            for model in models[:15]:
                print(f"\n  🔍 {model}")
                
                # Check if likely vision model
                if not is_likely_vision_model(model):
                    print(f"    ℹ️  Filtered out (not a vision model)")
                    summary["false_positives_filtered"] += 1
                    continue
                
                # Check database
                db_result = check_model_in_database(model, capabilities)
                
                if db_result["found"]:
                    if db_result["vision_support"]:
                        print(f"    ✅ Vision model in database ({db_result['match_type']} → {db_result['db_model']})")
                        provider_summary["vision_models"] += 1
                        summary["vision_models_found"] += 1
                        
                        # Test calculator
                        calc_result = test_vlm_calculator(calculator, model, provider_name)
                        if calc_result["success"]:
                            print(f"    🧮 Calculator: {calc_result['tokens']} tokens ({calc_result['method']})")
                            provider_summary["calculator_working"] += 1
                            summary["calculator_compatible"] += 1
                        else:
                            print(f"    ❌ Calculator failed: {calc_result['error']}")
                        
                        provider_summary["models"].append({
                            "name": model,
                            "vision_support": True,
                            "calculator_works": calc_result["success"],
                            "db_model": db_result["db_model"]
                        })
                    else:
                        print(f"    ℹ️  Text-only model in database")
                else:
                    print(f"    ❓ Vision model not in database (needs research)")
                    provider_summary["models"].append({
                        "name": model,
                        "vision_support": "unknown",
                        "calculator_works": False,
                        "needs_research": True
                    })
        
        except Exception as e:
            print(f"❌ Error testing {provider_name}: {e}")
        
        summary["providers"][provider_name] = provider_summary
        print(f"\n📊 {provider_name}: {provider_summary['vision_models']} vision models, {provider_summary['calculator_working']} calculator-compatible")
    
    # Final report
    print("\n" + "=" * 80)
    print("📋 FINAL COMPATIBILITY REPORT")
    print("=" * 80)
    
    print(f"\n📈 STATISTICS:")
    print(f"  • Total models tested: {summary['total_models_tested']}")
    print(f"  • Vision models found: {summary['vision_models_found']}")
    print(f"  • Calculator compatible: {summary['calculator_compatible']}")
    print(f"  • False positives filtered: {summary['false_positives_filtered']}")
    
    print(f"\n✅ WORKING VLM MODELS:")
    for provider_name, provider_data in summary["providers"].items():
        working_models = [m for m in provider_data["models"] if m.get("calculator_works")]
        if working_models:
            print(f"  {provider_name.upper()}:")
            for model in working_models:
                print(f"    • {model['name']} → {model['db_model']}")
    
    print(f"\n❓ MODELS NEEDING RESEARCH:")
    research_needed = []
    for provider_name, provider_data in summary["providers"].items():
        for model in provider_data["models"]:
            if model.get("needs_research"):
                research_needed.append(f"{provider_name}: {model['name']}")
    
    if research_needed:
        for item in research_needed:
            print(f"    • {item}")
    else:
        print("    None - all potential vision models are covered!")
    
    # Conclusion
    compatibility_rate = (summary['calculator_compatible'] / max(summary['vision_models_found'], 1)) * 100
    
    print(f"\n🎯 CONCLUSION:")
    print(f"  VLM Token Calculator Compatibility: {compatibility_rate:.1f}%")
    
    if compatibility_rate >= 90:
        print(f"  Status: ✅ EXCELLENT - Ready for production use")
        return 0
    elif compatibility_rate >= 70:
        print(f"  Status: ⚠️  GOOD - Minor research needed")
        return 0
    else:
        print(f"  Status: ❌ NEEDS WORK - Significant research required")
        return 1

if __name__ == "__main__":
    exit(main())
