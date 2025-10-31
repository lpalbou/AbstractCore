#!/usr/bin/env python3
"""
Simple Glyph Compression Test - toto.py

A quick test script to verify Glyph compression is working correctly.
"""

from abstractcore import create_llm, GlyphConfig
import time

def main():
    """Quick Glyph compression test."""
    print("🎨 Quick Glyph Test")
    print("=" * 30)
    
    # Create a simple test document
    test_content = """
    # Test Document for Glyph Compression
    
    This is a test document to verify that Glyph visual-text compression
    is working correctly with AbstractCore. The document contains enough
    text to potentially trigger compression while remaining simple enough
    for quick testing.
    
    ## Key Points
    - Glyph converts text to optimized images
    - Vision models process the compressed content  
    - 3-4x token compression is achieved
    - Processing speed is improved
    - Quality is preserved
    
    ## Testing Methodology
    This test will process the document with and without compression
    to verify that the system is functioning correctly and measure
    any performance differences.
    """ * 5  # Repeat to make it longer
    
    # Save test document
    with open("test_document.txt", "w") as f:
        f.write(test_content)
    
    try:
        # Test with available vision model
        print("🔄 Testing Glyph compression...")
        
        llm = create_llm("ollama", model="llama3.2-vision:11b")
        
        # Test with compression
        start = time.time()
        response = llm.generate(
            "Summarize this test document.",
            media=["test_document.txt"],
            glyph_compression="always"
        )
        processing_time = time.time() - start
        
        print(f"✅ Test completed in {processing_time:.2f} seconds")
        print(f"📝 Response: {response.content[:100]}...")
        
        # Check compression status
        if response.metadata and response.metadata.get('compression_used'):
            print("🎨 Glyph compression was used!")
            stats = response.metadata.get('compression_stats', {})
            if stats:
                print(f"   Compression ratio: {stats.get('compression_ratio', 'N/A')}")
        else:
            print("📝 Standard processing was used")
        
        print("\n✅ Glyph test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Make sure Ollama is running: ollama serve")
        print("   - Install a vision model: ollama pull llama3.2-vision:11b")
        print("   - Check AbstractCore installation")
    
    finally:
        # Clean up
        import os
        if os.path.exists("test_document.txt"):
            os.remove("test_document.txt")

if __name__ == "__main__":
    main()
