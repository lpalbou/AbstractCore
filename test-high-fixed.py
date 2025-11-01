#!/usr/bin/env python3
"""
Fixed test for hybrid compression with proper media handling.
"""

from abstractcore.compression.vision_compressor import HybridCompressionPipeline
from abstractcore.compression.exceptions import CompressionQualityError
from abstractcore import create_llm

# Initialize LLM
llm = create_llm("ollama", model="llama3.2-vision:11b")

# Initialize hybrid pipeline
pipeline = HybridCompressionPipeline(
    vision_provider="ollama",
    vision_model="llama3.2-vision"
)

# Read text
with open("test-file.md", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Text length: {len(text)} characters")

# Compress with hybrid pipeline
result = pipeline.compress(
    text,
    target_ratio=20.0,  # Target 20x compression
    min_quality=0.90    # Minimum 90% quality
)

print(f"Achieved: {result['total_compression_ratio']:.1f}x compression")
print(f"Quality: {result['total_quality_score']:.1%}")
print(f"Images created: {len(result['media'])}")

# Now use the media field from the result
prompt = "Summarize the following document:"

try:
    # Use the compressed images from result['media']
    response = llm.generate(prompt, media=result['media'])

    print(f"\nSummary:\n{response.content}")

except CompressionQualityError:
    print("Compression quality error, falling back to uncompressed")
    response = llm.generate(prompt + "\n\n" + text[:4000])
    print(response.content)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()