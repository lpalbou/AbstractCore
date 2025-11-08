# Vision Model Comprehensive Test Results

**Test Date**: 2025-11-08 21:58:00
**Total Models Tested**: 11
**Successful Models**: 9
**Success Rate**: 81.8%
**Warnings Captured**: 9
**Test Prompt**: What do you see in this image? Describe the main objects, colors, and setting in 1-2 sentences.

## Performance Rankings

### Speed Ranking (Fastest First)
1. **anthropic/claude-3-haiku-20240307**: 1.30s avg
2. **huggingface/unsloth/Qwen2.5-VL-7B-Instruct-GGUF**: 2.02s avg
3. **lmstudio/google/gemma-3n-e4b**: 3.40s avg
4. **lmstudio/qwen/qwen3-vl-4b**: 4.55s avg
5. **openai/gpt-4o**: 4.79s avg
6. **lmstudio/qwen/qwen2.5-vl-7b**: 5.77s avg
7. **openai/gpt-4-turbo**: 6.33s avg
8. **ollama/qwen2.5vl:7b**: 25.39s avg
9. **ollama/llama3.2-vision:11b**: 46.50s avg

### Reliability Ranking (Most Successful First)
1. **ollama/qwen2.5vl:7b**: 100.0% success rate
2. **ollama/llama3.2-vision:11b**: 100.0% success rate
3. **lmstudio/qwen/qwen2.5-vl-7b**: 100.0% success rate
4. **lmstudio/qwen/qwen3-vl-4b**: 100.0% success rate
5. **lmstudio/google/gemma-3n-e4b**: 100.0% success rate
6. **openai/gpt-4o**: 100.0% success rate
7. **openai/gpt-4-turbo**: 100.0% success rate
8. **anthropic/claude-3-haiku-20240307**: 100.0% success rate
9. **huggingface/unsloth/Qwen2.5-VL-7B-Instruct-GGUF**: 100.0% success rate

## Provider Averages

### OLLAMA
- **Average Speed**: 35.95s
- **Average Success Rate**: 100.0%
- **Models**: qwen2.5vl:7b, llama3.2-vision:11b

### LMSTUDIO
- **Average Speed**: 4.57s
- **Average Success Rate**: 100.0%
- **Models**: qwen/qwen2.5-vl-7b, qwen/qwen3-vl-4b, google/gemma-3n-e4b

### OPENAI
- **Average Speed**: 5.56s
- **Average Success Rate**: 100.0%
- **Models**: gpt-4o, gpt-4-turbo

### ANTHROPIC
- **Average Speed**: 1.30s
- **Average Success Rate**: 100.0%
- **Models**: claude-3-haiku-20240307

### HUGGINGFACE
- **Average Speed**: 2.02s
- **Average Success Rate**: 100.0%
- **Models**: unsloth/Qwen2.5-VL-7B-Instruct-GGUF

## Warnings Captured

- **ResourceWarning**: unclosed <socket.socket fd=18, family=2, type=1, proto=6, laddr=('127.0.0.1', 62863), raddr=('127.0.0.1', 11434)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/h11/_readers.py:43
- **ResourceWarning**: unclosed <socket.socket fd=19, family=2, type=1, proto=6, laddr=('127.0.0.1', 62922), raddr=('127.0.0.1', 11434)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/h11/_readers.py:43
- **ResourceWarning**: unclosed <socket.socket fd=19, family=2, type=1, proto=6, laddr=('127.0.0.1', 63027), raddr=('127.0.0.1', 1234)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/h11/_headers.py:247
- **ResourceWarning**: unclosed <socket.socket fd=22, family=2, type=1, proto=6, laddr=('127.0.0.1', 63056), raddr=('127.0.0.1', 1234)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/openai/_compat.py:110
- **DeprecationWarning**: The model 'claude-3-5-sonnet-20241022' is deprecated and will reach end-of-life on October 22, 2025.
Please migrate to a newer model. Visit https://docs.anthropic.com/en/docs/resources/model-deprecations for more information.
  - File: /Users/albou/projects/abstractcore/abstractcore/providers/anthropic_provider.py:192
- **DeprecationWarning**: The model 'claude-3-5-sonnet-20241022' is deprecated and will reach end-of-life on October 22, 2025.
Please migrate to a newer model. Visit https://docs.anthropic.com/en/docs/resources/model-deprecations for more information.
  - File: /Users/albou/projects/abstractcore/abstractcore/providers/anthropic_provider.py:192
- **DeprecationWarning**: The model 'claude-3-5-sonnet-20241022' is deprecated and will reach end-of-life on October 22, 2025.
Please migrate to a newer model. Visit https://docs.anthropic.com/en/docs/resources/model-deprecations for more information.
  - File: /Users/albou/projects/abstractcore/abstractcore/providers/anthropic_provider.py:192
- **DeprecationWarning**: The model 'claude-3-5-sonnet-20241022' is deprecated and will reach end-of-life on October 22, 2025.
Please migrate to a newer model. Visit https://docs.anthropic.com/en/docs/resources/model-deprecations for more information.
  - File: /Users/albou/projects/abstractcore/abstractcore/providers/anthropic_provider.py:192
- **DeprecationWarning**: The model 'claude-3-5-sonnet-20241022' is deprecated and will reach end-of-life on October 22, 2025.
Please migrate to a newer model. Visit https://docs.anthropic.com/en/docs/resources/model-deprecations for more information.
  - File: /Users/albou/projects/abstractcore/abstractcore/providers/anthropic_provider.py:192

## Detailed Results

### ollama/qwen2.5vl:7b
- **Success Rate**: 5/5 images
- **Average Response Time**: 25.39s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 29.27s (49 words)
    - Response: "The image shows a close-up of a bowl filled with a creamy salad containing chunks of potatoes, carro..."
  - ✅ mystery2_sc.jpg: 24.29s (47 words)
    - Response: "The image features a cat with striking yellow eyes peeking out from inside a transparent plastic dom..."
  - ✅ mystery4_wh.jpg: 24.80s (59 words)
    - Response: "The image shows a humpback whale breaching the surface of the ocean, with its body partially above t..."
  - ✅ mystery1_mp.jpg: 24.31s (71 words)
    - Response: "The image shows a scenic mountain landscape with a dirt path leading into the distance. The foregrou..."
  - ✅ mystery3_us.jpg: 24.28s (49 words)
    - Response: "The image depicts a serene park pathway lined with ornate street lamps and bordered by low hedges an..."

### ollama/llama3.2-vision:11b
- **Success Rate**: 5/5 images
- **Average Response Time**: 46.50s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 44.27s (50 words)
    - Response: "The image depicts a black ceramic bowl filled with a salad, featuring diced carrots, green peas, and..."
  - ✅ mystery2_sc.jpg: 44.18s (62 words)
    - Response: "This image features a cat's head protruding from a clear plastic dome on top of a white egg-shaped o..."
  - ✅ mystery4_wh.jpg: 44.92s (42 words)
    - Response: "This image shows a large whale breaching the water with its tail outstretched. It is black with whit..."
  - ✅ mystery1_mp.jpg: 51.67s (94 words)
    - Response: "The image shows a dirt road leading to a wooden fence with a mountainous backdrop. The road is surro..."
  - ✅ mystery3_us.jpg: 47.46s (52 words)
    - Response: "The image depicts a serene and peaceful urban park setting, characterized by a well-maintained path ..."

### ollama/gemma3:4b
- **Success Rate**: 0/5 images
- **Average Response Time**: 0.00s
- **Per-Image Results**:
  - ❌ mystery5_so.jpg: ❌ Model 'gemma3:4b' not found for Ollama provider.

✅ Available models (28):
  • all-minilm:33m
  • all-minilm:l6-v2
  • cogito:3b
  • embeddinggemma:300m
  • gemma3:1b
  • gemma3:1b-it-qat
  • gemma3:270m
  • gemma3:270m-it-qat
  • gemma3:4b-it-qat
  • glm-4.6:cloud
  • gpt-oss:120b-cloud
  • gpt-oss:20b
  • gpt-oss:20b-cloud
  • granite-embedding:278m
  • granite-embedding:30m
  • granite3.2-vision:latest
  • granite3.3:2b
  • kimi-k2:1t-cloud
  • llama3.2-vision:11b
  • minimax-m2:cloud
  • nomic-embed-text:latest
  • nomic-embed-text:v1.5
  • qwen2.5vl:7b
  • qwen3-coder:30b
  • qwen3-coder:480b-cloud
  • qwen3-embedding:0.6b
  • qwen3-vl:235b-instruct-cloud
  • qwen3:4b-instruct-2507-q4_K_M
  - ❌ mystery2_sc.jpg: ❌ Model 'gemma3:4b' not found for Ollama provider.

✅ Available models (28):
  • all-minilm:33m
  • all-minilm:l6-v2
  • cogito:3b
  • embeddinggemma:300m
  • gemma3:1b
  • gemma3:1b-it-qat
  • gemma3:270m
  • gemma3:270m-it-qat
  • gemma3:4b-it-qat
  • glm-4.6:cloud
  • gpt-oss:120b-cloud
  • gpt-oss:20b
  • gpt-oss:20b-cloud
  • granite-embedding:278m
  • granite-embedding:30m
  • granite3.2-vision:latest
  • granite3.3:2b
  • kimi-k2:1t-cloud
  • llama3.2-vision:11b
  • minimax-m2:cloud
  • nomic-embed-text:latest
  • nomic-embed-text:v1.5
  • qwen2.5vl:7b
  • qwen3-coder:30b
  • qwen3-coder:480b-cloud
  • qwen3-embedding:0.6b
  • qwen3-vl:235b-instruct-cloud
  • qwen3:4b-instruct-2507-q4_K_M
  - ❌ mystery4_wh.jpg: ❌ Model 'gemma3:4b' not found for Ollama provider.

✅ Available models (28):
  • all-minilm:33m
  • all-minilm:l6-v2
  • cogito:3b
  • embeddinggemma:300m
  • gemma3:1b
  • gemma3:1b-it-qat
  • gemma3:270m
  • gemma3:270m-it-qat
  • gemma3:4b-it-qat
  • glm-4.6:cloud
  • gpt-oss:120b-cloud
  • gpt-oss:20b
  • gpt-oss:20b-cloud
  • granite-embedding:278m
  • granite-embedding:30m
  • granite3.2-vision:latest
  • granite3.3:2b
  • kimi-k2:1t-cloud
  • llama3.2-vision:11b
  • minimax-m2:cloud
  • nomic-embed-text:latest
  • nomic-embed-text:v1.5
  • qwen2.5vl:7b
  • qwen3-coder:30b
  • qwen3-coder:480b-cloud
  • qwen3-embedding:0.6b
  • qwen3-vl:235b-instruct-cloud
  • qwen3:4b-instruct-2507-q4_K_M
  - ❌ mystery1_mp.jpg: ❌ Model 'gemma3:4b' not found for Ollama provider.

✅ Available models (28):
  • all-minilm:33m
  • all-minilm:l6-v2
  • cogito:3b
  • embeddinggemma:300m
  • gemma3:1b
  • gemma3:1b-it-qat
  • gemma3:270m
  • gemma3:270m-it-qat
  • gemma3:4b-it-qat
  • glm-4.6:cloud
  • gpt-oss:120b-cloud
  • gpt-oss:20b
  • gpt-oss:20b-cloud
  • granite-embedding:278m
  • granite-embedding:30m
  • granite3.2-vision:latest
  • granite3.3:2b
  • kimi-k2:1t-cloud
  • llama3.2-vision:11b
  • minimax-m2:cloud
  • nomic-embed-text:latest
  • nomic-embed-text:v1.5
  • qwen2.5vl:7b
  • qwen3-coder:30b
  • qwen3-coder:480b-cloud
  • qwen3-embedding:0.6b
  • qwen3-vl:235b-instruct-cloud
  • qwen3:4b-instruct-2507-q4_K_M
  - ❌ mystery3_us.jpg: ❌ Model 'gemma3:4b' not found for Ollama provider.

✅ Available models (28):
  • all-minilm:33m
  • all-minilm:l6-v2
  • cogito:3b
  • embeddinggemma:300m
  • gemma3:1b
  • gemma3:1b-it-qat
  • gemma3:270m
  • gemma3:270m-it-qat
  • gemma3:4b-it-qat
  • glm-4.6:cloud
  • gpt-oss:120b-cloud
  • gpt-oss:20b
  • gpt-oss:20b-cloud
  • granite-embedding:278m
  • granite-embedding:30m
  • granite3.2-vision:latest
  • granite3.3:2b
  • kimi-k2:1t-cloud
  • llama3.2-vision:11b
  • minimax-m2:cloud
  • nomic-embed-text:latest
  • nomic-embed-text:v1.5
  • qwen2.5vl:7b
  • qwen3-coder:30b
  • qwen3-coder:480b-cloud
  • qwen3-embedding:0.6b
  • qwen3-vl:235b-instruct-cloud
  • qwen3:4b-instruct-2507-q4_K_M

### lmstudio/qwen/qwen2.5-vl-7b
- **Success Rate**: 5/5 images
- **Average Response Time**: 5.77s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 7.75s (42 words)
    - Response: "The image shows a bowl filled with potato salad, featuring visible chunks of potato, carrot, and pea..."
  - ✅ mystery2_sc.jpg: 5.33s (64 words)
    - Response: "The image features a cat peeking out from inside a transparent plastic dome attached to what appears..."
  - ✅ mystery4_wh.jpg: 4.77s (50 words)
    - Response: "The image captures a humpback whale breaching out of the water, with its tail visible above the surf..."
  - ✅ mystery1_mp.jpg: 5.40s (64 words)
    - Response: "The image depicts a serene rural scene with a dirt path winding through lush greenery under a bright..."
  - ✅ mystery3_us.jpg: 5.59s (58 words)
    - Response: "The image depicts a serene urban park pathway during sunset or sunrise, with soft pink and orange hu..."

### lmstudio/qwen/qwen3-vl-4b
- **Success Rate**: 5/5 images
- **Average Response Time**: 4.55s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 6.76s (59 words)
    - Response: "The image shows a freshly prepared chicken salad in a dark gray bowl, featuring chunks of chicken, c..."
  - ✅ mystery2_sc.jpg: 4.20s (52 words)
    - Response: "In this whimsical image, a tabby cat peers curiously out of a transparent dome atop a white, spacesh..."
  - ✅ mystery4_wh.jpg: 3.70s (58 words)
    - Response: "In this image, a massive humpback whale breaches the ocean surface, its dark, textured body and wide..."
  - ✅ mystery1_mp.jpg: 4.09s (55 words)
    - Response: "This image captures a sun-drenched mountain trail winding through a grassy hillside, bordered by a w..."
  - ✅ mystery3_us.jpg: 3.98s (66 words)
    - Response: "The image captures a serene, winding park pathway in Paris at dusk, with a glowing vintage green lam..."

### lmstudio/google/gemma-3n-e4b
- **Success Rate**: 5/5 images
- **Average Response Time**: 3.40s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 7.19s (44 words)
    - Response: "In the image, I see a black bowl filled with potato salad. The salad appears to contain chunks of po..."
  - ✅ mystery2_sc.jpg: 1.70s (63 words)
    - Response: "In this image, a cute tabby cat is peeking out from inside a white, dome-shaped pet carrier with por..."
  - ✅ mystery4_wh.jpg: 2.75s (68 words)
    - Response: "The image captures a majestic humpback whale breaching out of the ocean, its massive dark body emerg..."
  - ✅ mystery1_mp.jpg: 2.80s (77 words)
    - Response: "The image captures a scenic mountain landscape on a bright, sunny day. The main focus is a dirt path..."
  - ✅ mystery3_us.jpg: 2.57s (77 words)
    - Response: "The image shows a serene, tree-lined pathway in what appears to be a park or garden at dusk. A warm ..."

### openai/gpt-4o
- **Success Rate**: 5/5 images
- **Average Response Time**: 4.79s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 4.65s (42 words)
    - Response: "The image shows a bowl of mixed salad containing diced vegetables such as potatoes, carrots, and pea..."
  - ✅ mystery2_sc.jpg: 3.46s (30 words)
    - Response: "The image shows a cat inside a white carrier with a transparent dome, resembling a space capsule. Th..."
  - ✅ mystery4_wh.jpg: 6.25s (40 words)
    - Response: "The image shows a humpback whale breaching out of the water in an ocean setting. The whale is dark g..."
  - ✅ mystery1_mp.jpg: 4.30s (40 words)
    - Response: "The image shows a dirt path leading through a grassy landscape with a wooden fence on the left. The ..."
  - ✅ mystery3_us.jpg: 5.29s (38 words)
    - Response: "The image shows a pathway lined with street lamps and trees on both sides, with buildings visible in..."

### openai/gpt-4-turbo
- **Success Rate**: 5/5 images
- **Average Response Time**: 6.33s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 6.00s (57 words)
    - Response: "The image shows a bowl of salad containing a mix of diced vegetables and chunks of meat, possibly ch..."
  - ✅ mystery2_sc.jpg: 5.12s (46 words)
    - Response: "This image features a cat looking through the transparent dome of a white, space capsule-style pet c..."
  - ✅ mystery4_wh.jpg: 7.59s (45 words)
    - Response: "This image captures a humpback whale breaching the surface of the ocean. The whale's massive body is..."
  - ✅ mystery1_mp.jpg: 5.85s (54 words)
    - Response: "This image captures a mountainous landscape under a clear blue sky, featuring a dirt road leading to..."
  - ✅ mystery3_us.jpg: 7.11s (57 words)
    - Response: "This image captures a tranquil city park at dusk, featuring a pathway lined with street lamps that a..."

### anthropic/claude-3-5-sonnet-20241022
- **Success Rate**: 0/5 images
- **Average Response Time**: 0.00s
- **Per-Image Results**:
  - ❌ mystery5_so.jpg: ❌ Model 'claude-3-5-sonnet-20241022' not found for Anthropic provider.

✅ Available models (9):
  • claude-sonnet-4-5-20250929
  • claude-sonnet-4-20250514
  • claude-opus-4-20250514
  • claude-opus-4-1-20250805
  • claude-haiku-4-5-20251001
  • claude-3-opus-20240229
  • claude-3-haiku-20240307
  • claude-3-7-sonnet-20250219
  • claude-3-5-haiku-20241022
  - ❌ mystery2_sc.jpg: ❌ Model 'claude-3-5-sonnet-20241022' not found for Anthropic provider.

✅ Available models (9):
  • claude-sonnet-4-5-20250929
  • claude-sonnet-4-20250514
  • claude-opus-4-20250514
  • claude-opus-4-1-20250805
  • claude-haiku-4-5-20251001
  • claude-3-opus-20240229
  • claude-3-haiku-20240307
  • claude-3-7-sonnet-20250219
  • claude-3-5-haiku-20241022
  - ❌ mystery4_wh.jpg: ❌ Model 'claude-3-5-sonnet-20241022' not found for Anthropic provider.

✅ Available models (9):
  • claude-sonnet-4-5-20250929
  • claude-sonnet-4-20250514
  • claude-opus-4-20250514
  • claude-opus-4-1-20250805
  • claude-haiku-4-5-20251001
  • claude-3-opus-20240229
  • claude-3-haiku-20240307
  • claude-3-7-sonnet-20250219
  • claude-3-5-haiku-20241022
  - ❌ mystery1_mp.jpg: ❌ Model 'claude-3-5-sonnet-20241022' not found for Anthropic provider.

✅ Available models (9):
  • claude-sonnet-4-5-20250929
  • claude-sonnet-4-20250514
  • claude-opus-4-20250514
  • claude-opus-4-1-20250805
  • claude-haiku-4-5-20251001
  • claude-3-opus-20240229
  • claude-3-haiku-20240307
  • claude-3-7-sonnet-20250219
  • claude-3-5-haiku-20241022
  - ❌ mystery3_us.jpg: ❌ Model 'claude-3-5-sonnet-20241022' not found for Anthropic provider.

✅ Available models (9):
  • claude-sonnet-4-5-20250929
  • claude-sonnet-4-20250514
  • claude-opus-4-20250514
  • claude-opus-4-1-20250805
  • claude-haiku-4-5-20251001
  • claude-3-opus-20240229
  • claude-3-haiku-20240307
  • claude-3-7-sonnet-20250219
  • claude-3-5-haiku-20241022

### anthropic/claude-3-haiku-20240307
- **Success Rate**: 5/5 images
- **Average Response Time**: 1.30s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 1.37s (42 words)
    - Response: "The image shows a bowl containing a creamy, chunky mixture of cooked ingredients, including pieces o..."
  - ✅ mystery2_sc.jpg: 1.22s (27 words)
    - Response: "The image shows a cat's face peering through a transparent dome or container, with a patterned backg..."
  - ✅ mystery4_wh.jpg: 1.32s (34 words)
    - Response: "The image shows a large humpback whale breaching the surface of the ocean. The whale's dark, wrinkle..."
  - ✅ mystery1_mp.jpg: 1.09s (45 words)
    - Response: "The image shows a scenic mountain landscape with a dirt path leading through a grassy field, surroun..."
  - ✅ mystery3_us.jpg: 1.53s (34 words)
    - Response: "The image shows a scenic urban park setting at sunset. The main objects include a paved walking path..."

### huggingface/unsloth/Qwen2.5-VL-7B-Instruct-GGUF
- **Success Rate**: 5/5 images
- **Average Response Time**: 2.02s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 2.99s (70 words)
    - Response: "The image appears to be an abstract composition with a dominant blue color scheme. It features a ser..."
  - ✅ mystery2_sc.jpg: 1.89s (67 words)
    - Response: "The image appears to be an abstract artwork with various shades of blue and gray, creating a sense o..."
  - ✅ mystery4_wh.jpg: 1.92s (67 words)
    - Response: "The image appears to be an abstract composition with a predominantly white background and a central ..."
  - ✅ mystery1_mp.jpg: 1.50s (50 words)
    - Response: "The image appears to be an abstract artwork with a variety of shapes and colors. It features a mix o..."
  - ✅ mystery3_us.jpg: 1.81s (62 words)
    - Response: "The image appears to be an abstract painting with a mix of geometric shapes and organic forms. The c..."

