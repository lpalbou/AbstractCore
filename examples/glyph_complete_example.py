#!/usr/bin/env python3
"""
Complete Glyph Visual-Text Compression Example

This example demonstrates all aspects of Glyph compression with AbstractCore:
- Basic usage with automatic compression
- Explicit compression control
- Custom configuration
- Performance benchmarking
- Multi-provider testing
- Error handling and debugging

Requirements:
- AbstractCore with Glyph compression support
- At least one vision-capable model (Ollama, LMStudio, etc.)
- Sample documents for testing
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from abstractcore import create_llm, GlyphConfig


def create_sample_document(content: str, filename: str) -> str:
    """Create a sample text document for testing."""
    filepath = Path(filename)
    filepath.write_text(content)
    print(f"📄 Created sample document: {filename}")
    return str(filepath)


def basic_glyph_example():
    """Demonstrate basic Glyph compression usage."""
    print("\n🎨 Basic Glyph Compression Example")
    print("=" * 50)
    
    # Create a sample document
    sample_content = """
    # Research Paper: Advanced AI Techniques
    
    ## Abstract
    This paper presents novel approaches to artificial intelligence that leverage 
    visual-text compression for improved efficiency. Our methodology demonstrates 
    significant improvements in processing speed while maintaining accuracy.
    
    ## Introduction
    The field of artificial intelligence has seen remarkable advances in recent years.
    Large language models have become increasingly powerful, but they face challenges
    with long-context processing and computational efficiency.
    
    ## Methodology
    Our approach uses visual rendering of textual content to achieve compression
    ratios of 3-4x while preserving semantic information. The key innovations include:
    
    1. Intelligent typography optimization
    2. Provider-specific rendering parameters
    3. Quality validation mechanisms
    4. Adaptive caching strategies
    
    ## Results
    Experimental results show:
    - 14% faster processing times
    - 79% reduction in memory usage
    - Maintained analytical accuracy
    - Scalable across different document types
    
    ## Conclusion
    Visual-text compression represents a paradigm shift in document processing,
    enabling more efficient analysis of large documents without sacrificing quality.
    """
    
    doc_path = create_sample_document(sample_content, "sample_research_paper.txt")
    
    try:
        # Create LLM with automatic Glyph compression
        llm = create_llm("ollama", model="llama3.2-vision:11b")
        
        # Process document - Glyph will automatically decide whether to compress
        print("🔄 Processing document with automatic compression...")
        start_time = time.time()
        
        response = llm.generate(
            "Analyze this research paper and provide a summary of the key findings and methodology.",
            media=[doc_path]
        )
        
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📝 Response length: {len(response.content)} characters")
        
        # Check if compression was used
        if response.metadata and response.metadata.get('compression_used'):
            stats = response.metadata.get('compression_stats', {})
            print(f"🎨 Glyph compression was used!")
            print(f"   Compression ratio: {stats.get('compression_ratio', 'N/A')}")
            print(f"   Original tokens: {stats.get('original_tokens', 'N/A')}")
            print(f"   Compressed tokens: {stats.get('compressed_tokens', 'N/A')}")
        else:
            print(f"📝 Standard text processing was used")
        
        print(f"\n📄 Analysis Summary:")
        print(f"{response.content[:300]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have a vision-capable model available")
    
    finally:
        # Clean up
        Path(doc_path).unlink(missing_ok=True)


def explicit_compression_control():
    """Demonstrate explicit compression control."""
    print("\n🎛️ Explicit Compression Control Example")
    print("=" * 50)
    
    # Create a longer document for better compression demonstration
    long_content = """
    # Comprehensive Technical Documentation
    
    ## Executive Summary
    """ + "This document contains extensive technical information. " * 50 + """
    
    ## Technical Specifications
    """ + "Detailed technical specifications follow. " * 100 + """
    
    ## Implementation Details
    """ + "Implementation requires careful consideration of multiple factors. " * 75 + """
    
    ## Performance Analysis
    """ + "Performance metrics indicate significant improvements across all benchmarks. " * 60 + """
    
    ## Conclusion
    """ + "The results demonstrate the effectiveness of the proposed approach. " * 25
    
    doc_path = create_sample_document(long_content, "long_technical_doc.txt")
    
    try:
        llm = create_llm("ollama", model="qwen2.5vl:7b")
        
        # Test with compression forced OFF
        print("🔄 Testing WITHOUT compression...")
        start_no_compression = time.time()
        
        response_no_compression = llm.generate(
            "Summarize the key points of this technical document.",
            media=[doc_path],
            glyph_compression="never"  # Explicitly disable compression
        )
        
        time_no_compression = time.time() - start_no_compression
        
        # Test with compression forced ON
        print("🔄 Testing WITH compression...")
        start_compression = time.time()
        
        response_compression = llm.generate(
            "Summarize the key points of this technical document.",
            media=[doc_path],
            glyph_compression="always"  # Force compression
        )
        
        time_compression = time.time() - start_compression
        
        # Compare results
        print(f"\n📊 Performance Comparison:")
        print(f"   Without compression: {time_no_compression:.2f}s")
        print(f"   With compression:    {time_compression:.2f}s")
        
        if time_compression > 0:
            speedup = time_no_compression / time_compression
            print(f"   Speedup factor:      {speedup:.2f}x")
        
        print(f"\n📝 Response Quality Comparison:")
        print(f"   No compression length:  {len(response_no_compression.content)} chars")
        print(f"   With compression length: {len(response_compression.content)} chars")
        
        # Check compression stats
        if response_compression.metadata and response_compression.metadata.get('compression_used'):
            stats = response_compression.metadata.get('compression_stats', {})
            print(f"\n🎨 Compression Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        Path(doc_path).unlink(missing_ok=True)


def custom_configuration_example():
    """Demonstrate custom Glyph configuration."""
    print("\n⚙️ Custom Configuration Example")
    print("=" * 50)
    
    # Create configuration for high-quality compression
    high_quality_config = GlyphConfig(
        enabled=True,
        global_default="auto",
        quality_threshold=0.98,          # Very high quality requirement
        target_compression_ratio=2.5,    # Conservative compression
        provider_optimization=True,
        cache_enabled=True,
        provider_profiles={
            "ollama": {
                "dpi": 150,              # High DPI for quality
                "font_size": 9,          # Smaller font for more content
                "quality_threshold": 0.98
            },
            "lmstudio": {
                "dpi": 120,
                "font_size": 10,
                "quality_threshold": 0.95
            }
        }
    )
    
    # Create configuration for performance-focused compression
    performance_config = GlyphConfig(
        enabled=True,
        global_default="always",
        quality_threshold=0.85,          # Lower quality for speed
        target_compression_ratio=4.0,    # Aggressive compression
        provider_optimization=True,
        cache_enabled=True,
        provider_profiles={
            "ollama": {
                "dpi": 72,               # Lower DPI for speed
                "font_size": 8,          # Smaller font for more compression
                "quality_threshold": 0.85
            }
        }
    )
    
    sample_doc = create_sample_document(
        "This is a test document for configuration comparison. " * 100,
        "config_test_doc.txt"
    )
    
    try:
        configs = [
            ("High Quality", high_quality_config),
            ("Performance Focused", performance_config)
        ]
        
        for config_name, config in configs:
            print(f"\n🧪 Testing {config_name} Configuration...")
            
            llm = create_llm("ollama", model="granite3.2-vision:latest", glyph_config=config)
            
            start_time = time.time()
            response = llm.generate(
                "Analyze this document content.",
                media=[sample_doc],
                glyph_compression="auto"
            )
            processing_time = time.time() - start_time
            
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Response length: {len(response.content)} chars")
            
            if response.metadata and response.metadata.get('compression_used'):
                stats = response.metadata.get('compression_stats', {})
                print(f"   Compression ratio: {stats.get('compression_ratio', 'N/A')}")
                print(f"   Quality score: {stats.get('quality_score', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        Path(sample_doc).unlink(missing_ok=True)


def multi_provider_testing():
    """Test Glyph compression across multiple providers."""
    print("\n🌐 Multi-Provider Testing Example")
    print("=" * 50)
    
    # Define available models to test
    models_to_test = [
        ("ollama", "llama3.2-vision:11b"),
        ("ollama", "qwen2.5vl:7b"),
        ("ollama", "granite3.2-vision:latest"),
        # Uncomment if you have LMStudio running
        # ("lmstudio", "your-vision-model"),
    ]
    
    test_doc = create_sample_document(
        "Multi-provider test document. " * 200 + 
        "This document tests Glyph compression across different providers and models.",
        "multi_provider_test.txt"
    )
    
    question = "What is the main content of this document?"
    results = []
    
    for provider, model in models_to_test:
        print(f"\n🧪 Testing {provider} - {model}")
        
        try:
            if provider == "lmstudio":
                llm = create_llm(provider, model=model, base_url="http://localhost:1234/v1")
            else:
                llm = create_llm(provider, model=model)
            
            start_time = time.time()
            response = llm.generate(
                question,
                media=[test_doc],
                glyph_compression="auto"
            )
            processing_time = time.time() - start_time
            
            result = {
                'provider': provider,
                'model': model,
                'success': True,
                'processing_time': processing_time,
                'response_length': len(response.content),
                'compression_used': response.metadata.get('compression_used', False) if response.metadata else False
            }
            
            if response.metadata and response.metadata.get('compression_used'):
                result['compression_stats'] = response.metadata.get('compression_stats', {})
            
            results.append(result)
            
            print(f"   ✅ Success - {processing_time:.2f}s")
            print(f"   Response: {response.content[:100]}...")
            
            if result['compression_used']:
                print(f"   🎨 Glyph compression was used")
            else:
                print(f"   📝 Standard processing was used")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'provider': provider,
                'model': model,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 Multi-Provider Test Summary:")
    successful_tests = [r for r in results if r['success']]
    
    if successful_tests:
        avg_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
        compression_count = sum(1 for r in successful_tests if r.get('compression_used', False))
        
        print(f"   Successful tests: {len(successful_tests)}/{len(results)}")
        print(f"   Average processing time: {avg_time:.2f}s")
        print(f"   Compression used: {compression_count}/{len(successful_tests)} tests")
    
    Path(test_doc).unlink(missing_ok=True)
    return results


def benchmark_compression(document_path: str, model_name: str = "llama3.2-vision:11b"):
    """Comprehensive benchmarking function."""
    print(f"\n📊 Benchmarking Glyph Compression")
    print(f"Document: {document_path}")
    print(f"Model: {model_name}")
    print("=" * 50)
    
    try:
        llm = create_llm("ollama", model=model_name)
        
        # Test without compression
        print("🔄 Testing without compression...")
        start = time.time()
        response_no_glyph = llm.generate(
            "Provide a detailed analysis of this document including key points, methodology, and conclusions.",
            media=[document_path],
            glyph_compression="never"
        )
        time_no_glyph = time.time() - start
        
        # Test with compression
        print("🔄 Testing with compression...")
        start = time.time()
        response_glyph = llm.generate(
            "Provide a detailed analysis of this document including key points, methodology, and conclusions.",
            media=[document_path],
            glyph_compression="always"
        )
        time_glyph = time.time() - start
        
        # Calculate metrics
        speedup = time_no_glyph / time_glyph if time_glyph > 0 else 0
        
        print(f"\n📈 Benchmark Results:")
        print(f"   Without Glyph: {time_no_glyph:.2f}s")
        print(f"   With Glyph:    {time_glyph:.2f}s")
        print(f"   Speedup:       {speedup:.2f}x")
        print(f"   Time saved:    {time_no_glyph - time_glyph:.2f}s")
        
        print(f"\n📝 Response Quality Comparison:")
        print(f"   No Glyph length:  {len(response_no_glyph.content):,} chars")
        print(f"   Glyph length:     {len(response_glyph.content):,} chars")
        print(f"   Length ratio:     {len(response_glyph.content) / len(response_no_glyph.content):.2f}")
        
        # Token usage comparison
        if hasattr(response_no_glyph, 'usage') and hasattr(response_glyph, 'usage'):
            print(f"\n🎯 Token Usage Comparison:")
            if response_no_glyph.usage and response_glyph.usage:
                no_glyph_tokens = response_no_glyph.usage.get('total_tokens', 0)
                glyph_tokens = response_glyph.usage.get('total_tokens', 0)
                print(f"   No Glyph tokens:  {no_glyph_tokens:,}")
                print(f"   Glyph tokens:     {glyph_tokens:,}")
                if no_glyph_tokens > 0:
                    token_ratio = glyph_tokens / no_glyph_tokens
                    print(f"   Token ratio:      {token_ratio:.2f}")
        
        # Compression statistics
        if response_glyph.metadata and response_glyph.metadata.get('compression_used'):
            stats = response_glyph.metadata.get('compression_stats', {})
            print(f"\n🎨 Compression Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        return {
            'time_no_glyph': time_no_glyph,
            'time_glyph': time_glyph,
            'speedup': speedup,
            'response_no_glyph': response_no_glyph,
            'response_glyph': response_glyph
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None


def debug_compression_issues():
    """Demonstrate debugging techniques for Glyph compression."""
    print("\n🔍 Debugging Glyph Compression Issues")
    print("=" * 50)
    
    # Create a problematic document (very short, might not trigger compression)
    short_doc = create_sample_document("Short document.", "short_debug_test.txt")
    
    # Enable debug mode
    debug_config = GlyphConfig(
        enabled=True,
        global_default="always",  # Force compression even for short docs
        quality_threshold=0.80,
        debug_mode=True  # Enable detailed logging
    )
    
    try:
        llm = create_llm("ollama", model="qwen2.5vl:7b", glyph_config=debug_config)
        
        print("🔄 Testing with debug configuration...")
        response = llm.generate(
            "Analyze this document.",
            media=[short_doc],
            glyph_compression="always"
        )
        
        print(f"✅ Response received")
        
        # Examine metadata for debugging information
        if response.metadata:
            print(f"\n🔍 Debug Information:")
            for key, value in response.metadata.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"     {sub_key}: {sub_value}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("⚠️  No metadata available")
        
        # Common troubleshooting checks
        print(f"\n🛠️ Troubleshooting Checklist:")
        print(f"   ✅ Vision model used: qwen2.5vl:7b")
        print(f"   ✅ Compression forced: always")
        print(f"   ✅ Debug mode enabled: True")
        
        compression_used = response.metadata.get('compression_used', False) if response.metadata else False
        print(f"   {'✅' if compression_used else '❌'} Compression activated: {compression_used}")
        
        if not compression_used:
            print(f"\n💡 Possible reasons compression wasn't used:")
            print(f"   - Document too short (< 5,000 tokens recommended)")
            print(f"   - Quality threshold too high")
            print(f"   - Provider doesn't support vision")
            print(f"   - Rendering failed")
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        print(f"\n💡 Common solutions:")
        print(f"   - Check if the model is available: ollama list")
        print(f"   - Verify model supports vision capabilities")
        print(f"   - Try with a different model")
        print(f"   - Check AbstractCore installation")
    
    finally:
        Path(short_doc).unlink(missing_ok=True)


def main():
    """Run all Glyph compression examples."""
    print("🎨 Glyph Visual-Text Compression - Complete Examples")
    print("=" * 60)
    print("This script demonstrates all aspects of Glyph compression with AbstractCore")
    print("Make sure you have at least one vision-capable model available!")
    print()
    
    # Check if we have any vision models available
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            vision_models = [line for line in result.stdout.split('\n') 
                           if any(keyword in line.lower() for keyword in ['vision', 'vl', 'qwen2.5vl', 'llama3.2-vision', 'granite3.2-vision'])]
            if vision_models:
                print(f"🎯 Found vision models:")
                for model in vision_models[:3]:  # Show first 3
                    print(f"   {model.split()[0]}")
            else:
                print("⚠️  No vision models found. Please install a vision model first:")
                print("   ollama pull llama3.2-vision:11b")
                print("   ollama pull qwen2.5vl:7b")
                return
        else:
            print("⚠️  Ollama not available. Make sure it's installed and running.")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  Could not check available models. Proceeding anyway...")
    
    print()
    
    # Run examples
    examples = [
        ("Basic Usage", basic_glyph_example),
        ("Explicit Control", explicit_compression_control),
        ("Custom Configuration", custom_configuration_example),
        ("Multi-Provider Testing", multi_provider_testing),
        ("Debugging", debug_compression_issues),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print(f"\n⏹️  Interrupted during {name}")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
        
        print("\n" + "─" * 60)
    
    # Offer to run benchmark if user has a document
    print(f"\n🏁 Examples completed!")
    print(f"💡 To run a benchmark with your own document:")
    print(f"   benchmark_compression('your_document.pdf', 'llama3.2-vision:11b')")


if __name__ == "__main__":
    main()
