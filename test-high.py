from abstractcore.compression.vision_compressor import HybridCompressionPipeline
from abstractcore.compression.exceptions import CompressionQualityError
from abstractcore import create_llm

llm = create_llm("ollama", model="llama3.2-vision:11b")

# Initialize hybrid pipeline
pipeline = HybridCompressionPipeline(
    vision_provider="ollama",
    vision_model="llama3.2-vision"
)

filename = "test-file.md"
filename = "untracked/toto.md"
print("Filename: ", filename)

text = ""
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

print("Text length: ", len(text), "characters")

# Compress with target ratio
result = pipeline.compress(
    text,
    target_ratio=20.0,  # Target 20x compression
    min_quality=0.90    # Minimum 90% quality
)

print(f"Achieved: {result['total_compression_ratio']:.1f}x compression")
print(f"Quality: {result['total_quality_score']:.1%}")
print(f"Images created: {len(result['media'])}")

prompt = "Summarize the following text: "
try:
    # Use result['media'] which contains the actual compressed images
    response = llm.generate(prompt, media=result['media'])
    print(f"\nSummary:\n{response.content}")
except CompressionQualityError:
    # Fall back to uncompressed
    print("Compression quality error, falling back to uncompressed")
    response = llm.generate(prompt + "\n\n" + text[:4000])  # Truncate for context limit
    print(response.content)
except Exception as e:
    print("Error: ", e)
    import traceback
    traceback.print_exc()