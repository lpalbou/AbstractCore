#!/usr/bin/env python3
"""
Simple, clean vision compression example that just works.
"""

from abstractcore import create_llm
from abstractcore.compression import GlyphProcessor
from abstractcore.compression.config import GlyphConfig

# Setup
llm = create_llm("ollama", model="llama3.2-vision:11b")
processor = GlyphProcessor(GlyphConfig(enabled=True, min_token_threshold=1000))

# Read text
with open("test-file.md", "r") as f:
    text = f.read()

print(f"Compressing {len(text)} characters...")

# Compress text to images
compressed_images = processor.process_text(
    text,
    provider="ollama",
    model="llama3.2-vision",
    user_preference="always"
)

print(f"Created {len(compressed_images)} images")

# Use compressed images with LLM
response = llm.generate(
    "Summarize this document:",
    media=compressed_images
)

print(f"\nSummary:\n{response.content}")