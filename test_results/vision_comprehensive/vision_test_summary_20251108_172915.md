# Vision Model Comprehensive Test Results

**Test Date**: 2025-11-08 17:29:15
**Total Models Tested**: 11
**Successful Models**: 9
**Success Rate**: 81.8%
**Warnings Captured**: 11
**Test Prompt**: What do you see in this image? Describe the main objects, colors, and setting in 1-2 sentences.

## Performance Rankings

### Speed Ranking (Fastest First)
1. **anthropic/claude-3-haiku-20240307**: 1.44s avg
2. **huggingface/unsloth/Qwen2.5-VL-7B-Instruct-GGUF**: 2.13s avg
3. **lmstudio/google/gemma-3n-e4b**: 3.61s avg
4. **lmstudio/qwen/qwen3-vl-4b**: 4.54s avg
5. **lmstudio/qwen/qwen2.5-vl-7b**: 5.50s avg
6. **openai/gpt-4o**: 6.34s avg
7. **openai/gpt-4-turbo**: 6.85s avg
8. **ollama/qwen2.5vl:7b**: 24.25s avg
9. **ollama/llama3.2-vision:11b**: 44.45s avg

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
- **Average Speed**: 34.35s
- **Average Success Rate**: 100.0%
- **Models**: qwen2.5vl:7b, llama3.2-vision:11b

### LMSTUDIO
- **Average Speed**: 4.55s
- **Average Success Rate**: 100.0%
- **Models**: qwen/qwen2.5-vl-7b, qwen/qwen3-vl-4b, google/gemma-3n-e4b

### OPENAI
- **Average Speed**: 6.59s
- **Average Success Rate**: 100.0%
- **Models**: gpt-4o, gpt-4-turbo

### ANTHROPIC
- **Average Speed**: 1.44s
- **Average Success Rate**: 100.0%
- **Models**: claude-3-haiku-20240307

### HUGGINGFACE
- **Average Speed**: 2.13s
- **Average Success Rate**: 100.0%
- **Models**: unsloth/Qwen2.5-VL-7B-Instruct-GGUF

## Warnings Captured

- **ResourceWarning**: unclosed <socket.socket fd=24, family=2, type=1, proto=6, laddr=('127.0.0.1', 54231), raddr=('127.0.0.1', 11434)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/PIL/TiffImagePlugin.py:779
- **ResourceWarning**: unclosed <socket.socket fd=18, family=2, type=1, proto=6, laddr=('127.0.0.1', 54181), raddr=('127.0.0.1', 11434)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/PIL/TiffImagePlugin.py:779
- **ResourceWarning**: unclosed <socket.socket fd=18, family=2, type=1, proto=6, laddr=('127.0.0.1', 54347), raddr=('127.0.0.1', 11434)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/httpx/_models.py:315
- **ResourceWarning**: unclosed <socket.socket fd=24, family=2, type=1, proto=6, laddr=('127.0.0.1', 54348), raddr=('127.0.0.1', 1234)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/site-packages/h11/_headers.py:247
- **ResourceWarning**: unclosed <socket.socket fd=18, family=2, type=1, proto=6, laddr=('127.0.0.1', 54356), raddr=('127.0.0.1', 1234)>
  - File: /Users/albou/.pyenv/versions/3.12.11/lib/python3.12/logging/__init__.py:200
- **ResourceWarning**: unclosed <socket.socket fd=24, family=2, type=1, proto=6, laddr=('127.0.0.1', 54372), raddr=('127.0.0.1', 1234)>
  - File: /Users/albou/projects/abstractcore/tests/media_handling/test_vision_comprehensive.py:309
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
- **Average Response Time**: 24.25s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 24.74s (43 words)
    - Response: "The image shows a bowl of potato salad, likely containing ingredients such as diced potatoes, carrot..."
  - ✅ mystery2_sc.jpg: 23.82s (40 words)
    - Response: "The image shows a cat peeking out from inside a transparent plastic dome, which is attached to a whi..."
  - ✅ mystery4_wh.jpg: 24.52s (62 words)
    - Response: "The image shows a humpback whale breaching the water, with its massive body partially out of the oce..."
  - ✅ mystery1_mp.jpg: 24.04s (59 words)
    - Response: "The image shows a serene rural scene with a dirt path leading towards distant mountains under a brig..."
  - ✅ mystery3_us.jpg: 24.15s (63 words)
    - Response: "The image depicts a serene urban scene at sunset, featuring a paved walkway lined with trees and str..."

### ollama/llama3.2-vision:11b
- **Success Rate**: 5/5 images
- **Average Response Time**: 44.45s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 42.67s (54 words)
    - Response: "The image features a bowl of potato salad, characterized by a mix of diced potatoes, carrots, and gr..."
  - ✅ mystery2_sc.jpg: 48.16s (150 words)
    - Response: "This image depicts a cat with light-brown fur and greenish-yellow eyes, gazing directly at the camer..."
  - ✅ mystery4_wh.jpg: 43.62s (59 words)
    - Response: "The image depicts a humpback whale breaching the surface of the water, its tail visible above the wa..."
  - ✅ mystery1_mp.jpg: 42.89s (53 words)
    - Response: "This image depicts a scenic dirt road winding through a mountainous region, with a wooden fence runn..."
  - ✅ mystery3_us.jpg: 44.93s (35 words)
    - Response: "The image shows a long, straight sidewalk lined with street lamps, trees, and bushes on either side,..."

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
- **Average Response Time**: 5.50s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 6.89s (53 words)
    - Response: "The image shows a close-up of a bowl containing what appears to be a creamy tuna or chicken salad wi..."
  - ✅ mystery2_sc.jpg: 4.92s (42 words)
    - Response: "The image shows a cat inside a transparent dome-like enclosure on top of what appears to be a white ..."
  - ✅ mystery4_wh.jpg: 4.38s (46 words)
    - Response: "The image captures an enormous whale breaching from the ocean's surface with its massive head and fl..."
  - ✅ mystery1_mp.jpg: 5.30s (59 words)
    - Response: "The image showcases a scenic rural landscape featuring a dirt path that winds through lush greenery ..."
  - ✅ mystery3_us.jpg: 6.00s (73 words)
    - Response: "The image captures a tranquil street scene at sunset with vibrant hues of pink and orange illuminati..."

### lmstudio/qwen/qwen3-vl-4b
- **Success Rate**: 5/5 images
- **Average Response Time**: 4.54s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 7.05s (73 words)
    - Response: "The image shows a freshly prepared chicken salad in a dark gray bowl, featuring chunks of chicken, c..."
  - ✅ mystery2_sc.jpg: 4.24s (50 words)
    - Response: "In this whimsical image, a tabby cat with wide, curious eyes peers out from a transparent dome atop ..."
  - ✅ mystery4_wh.jpg: 3.47s (37 words)
    - Response: "In this image, a massive humpback whale breaches the ocean's surface, its dark, textured body and en..."
  - ✅ mystery1_mp.jpg: 3.93s (59 words)
    - Response: "The image captures a sun-drenched mountain trail winding through a grassy hillside, bordered by a wo..."
  - ✅ mystery3_us.jpg: 4.00s (46 words)
    - Response: "This image captures a serene Parisian park pathway at dusk, with a vibrant orange and pink sunset sk..."

### lmstudio/google/gemma-3n-e4b
- **Success Rate**: 5/5 images
- **Average Response Time**: 3.61s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 8.95s (60 words)
    - Response: "In this image, a black bowl filled with a creamy potato salad is the main focus. The salad contains ..."
  - ✅ mystery2_sc.jpg: 1.52s (48 words)
    - Response: "A cute tabby cat is peeking out from inside a white, dome-shaped pet carrier with porthole-like open..."
  - ✅ mystery4_wh.jpg: 1.84s (53 words)
    - Response: "In this image, a massive humpback whale is breaching out of the ocean water, with its large pectoral..."
  - ✅ mystery1_mp.jpg: 3.78s (86 words)
    - Response: "The image shows a dirt path winding through a grassy, mountainous landscape under a bright, partly c..."
  - ✅ mystery3_us.jpg: 1.99s (35 words)
    - Response: "The image shows a quiet, tree-lined pathway in Paris at dusk. A warm, pink and orange sunset fills t..."

### openai/gpt-4o
- **Success Rate**: 5/5 images
- **Average Response Time**: 6.34s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 4.76s (47 words)
    - Response: "The image shows a bowl filled with a mixed salad containing diced vegetables such as potatoes, carro..."
  - ✅ mystery2_sc.jpg: 10.57s (40 words)
    - Response: "The image shows a cat peering out from a small, clear dome on a white, pod-like carrier, which has t..."
  - ✅ mystery4_wh.jpg: 6.02s (30 words)
    - Response: "The image shows a humpback whale breaching the ocean surface, with water splashing around it. The wh..."
  - ✅ mystery1_mp.jpg: 4.67s (34 words)
    - Response: "The image shows a dirt path leading through a grassy landscape with a wooden fence on the left. The ..."
  - ✅ mystery3_us.jpg: 5.65s (39 words)
    - Response: "The image depicts a serene pathway flanked by trees and illuminated by vintage-style street lamps, l..."

### openai/gpt-4-turbo
- **Success Rate**: 5/5 images
- **Average Response Time**: 6.85s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 6.33s (68 words)
    - Response: "The image shows a bowl of salad, which appears to contain ingredients like diced carrots, potatoes, ..."
  - ✅ mystery2_sc.jpg: 5.27s (41 words)
    - Response: "This image features a cat inside a clear dome attached to a white object with several circular vents..."
  - ✅ mystery4_wh.jpg: 7.84s (57 words)
    - Response: "The image captures a humpback whale breaching the surface of the ocean. The whale is predominantly d..."
  - ✅ mystery1_mp.jpg: 7.26s (63 words)
    - Response: "The image features a scenic mountain landscape with a dirt road leading into the distance, flanked b..."
  - ✅ mystery3_us.jpg: 7.54s (54 words)
    - Response: "This image captures a serene park pathway at dusk, lined with green lamp posts that are lit, adding ..."

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
- **Average Response Time**: 1.44s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 1.43s (45 words)
    - Response: "The image shows a bowl containing a creamy, chunky mixture of potatoes, carrots, and other ingredien..."
  - ✅ mystery2_sc.jpg: 1.34s (30 words)
    - Response: "The image shows a cat's face peering out from inside a transparent dome or bubble, with a patterned ..."
  - ✅ mystery4_wh.jpg: 1.84s (40 words)
    - Response: "The image shows a large humpback whale emerging from the ocean, with its distinctive black and white..."
  - ✅ mystery1_mp.jpg: 1.19s (36 words)
    - Response: "The image shows a dirt path surrounded by a wooden fence, with mountains and a hazy blue sky in the ..."
  - ✅ mystery3_us.jpg: 1.39s (34 words)
    - Response: "The image shows a picturesque urban park setting at sunset. The main elements are a paved walkway li..."

### huggingface/unsloth/Qwen2.5-VL-7B-Instruct-GGUF
- **Success Rate**: 5/5 images
- **Average Response Time**: 2.13s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 3.12s (70 words)
    - Response: "The image appears to be an abstract composition with a dominant blue color scheme. It features a ser..."
  - ✅ mystery2_sc.jpg: 2.02s (67 words)
    - Response: "The image appears to be an abstract artwork with various shades of blue and gray, creating a sense o..."
  - ✅ mystery4_wh.jpg: 2.00s (67 words)
    - Response: "The image appears to be an abstract composition with a predominantly white background and a central ..."
  - ✅ mystery1_mp.jpg: 1.60s (50 words)
    - Response: "The image appears to be an abstract artwork with a variety of shapes and colors. It features a mix o..."
  - ✅ mystery3_us.jpg: 1.91s (62 words)
    - Response: "The image appears to be an abstract painting with a mix of geometric shapes and organic forms. The c..."

