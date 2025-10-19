# Vision Model Comprehensive Test Results

**Test Date**: 2025-10-19 04:56:01
**Total Models Tested**: 9
**Successful Models**: 9
**Success Rate**: 100.0%
**Warnings Captured**: 11
**Test Prompt**: What do you see in this image? Describe the main objects, colors, and setting in 1-2 sentences.

## Performance Rankings

### Speed Ranking (Fastest First)
1. **anthropic/claude-3-haiku-20240307**: 1.39s avg
2. **openai/gpt-4o**: 4.28s avg
3. **lmstudio/google/gemma-3n-e4b**: 4.80s avg
4. **ollama/qwen2.5vl:7b**: 6.56s avg
5. **lmstudio/qwen/qwen2.5-vl-7b**: 7.31s avg
6. **lmstudio/qwen/qwen3-vl-4b**: 8.01s avg
7. **ollama/llama3.2-vision:11b**: 15.00s avg
8. **huggingface/unsloth/Qwen2.5-VL-7B-Instruct-GGUF**: 15.19s avg
9. **ollama/gemma3:4b**: 15.33s avg

### Reliability Ranking (Most Successful First)
1. **ollama/qwen2.5vl:7b**: 100.0% success rate
2. **ollama/llama3.2-vision:11b**: 100.0% success rate
3. **ollama/gemma3:4b**: 100.0% success rate
4. **lmstudio/qwen/qwen2.5-vl-7b**: 100.0% success rate
5. **lmstudio/qwen/qwen3-vl-4b**: 100.0% success rate
6. **lmstudio/google/gemma-3n-e4b**: 100.0% success rate
7. **openai/gpt-4o**: 100.0% success rate
8. **anthropic/claude-3-haiku-20240307**: 100.0% success rate
9. **huggingface/unsloth/Qwen2.5-VL-7B-Instruct-GGUF**: 100.0% success rate

## Provider Averages

### OLLAMA
- **Average Speed**: 12.30s
- **Average Success Rate**: 100.0%
- **Models**: qwen2.5vl:7b, llama3.2-vision:11b, gemma3:4b

### LMSTUDIO
- **Average Speed**: 6.71s
- **Average Success Rate**: 100.0%
- **Models**: qwen/qwen2.5-vl-7b, qwen/qwen3-vl-4b, google/gemma-3n-e4b

### OPENAI
- **Average Speed**: 4.28s
- **Average Success Rate**: 100.0%
- **Models**: gpt-4o

### ANTHROPIC
- **Average Speed**: 1.39s
- **Average Success Rate**: 100.0%
- **Models**: claude-3-haiku-20240307

### HUGGINGFACE
- **Average Speed**: 15.19s
- **Average Success Rate**: 100.0%
- **Models**: unsloth/Qwen2.5-VL-7B-Instruct-GGUF

## Warnings Captured

- **ResourceWarning**: unclosed <socket.socket fd=15, family=2, type=1, proto=6, laddr=('127.0.0.1', 52055), raddr=('127.0.0.1', 11434)>
  - File: /opt/anaconda3/lib/python3.12/site-packages/h11/_headers.py:173
- **ResourceWarning**: unclosed <socket.socket fd=16, family=2, type=1, proto=6, laddr=('127.0.0.1', 52075), raddr=('127.0.0.1', 11434)>
  - File: /opt/anaconda3/lib/python3.12/site-packages/PIL/TiffImagePlugin.py:837
- **ResourceWarning**: unclosed <socket.socket fd=15, family=2, type=1, proto=6, laddr=('127.0.0.1', 52230), raddr=('127.0.0.1', 11434)>
  - File: /opt/anaconda3/lib/python3.12/logging/__init__.py:1761
- **ResourceWarning**: unclosed <socket.socket fd=16, family=2, type=1, proto=6, laddr=('127.0.0.1', 52344), raddr=('127.0.0.1', 1234)>
  - File: <frozen _collections_abc>:868
- **ResourceWarning**: unclosed <socket.socket fd=16, family=2, type=1, proto=6, laddr=('127.0.0.1', 52374), raddr=('127.0.0.1', 1234)>
  - File: /opt/anaconda3/lib/python3.12/site-packages/structlog/_frames.py:63
- **ResourceWarning**: unclosed <socket.socket fd=21, family=2, type=1, proto=6, laddr=('127.0.0.1', 52382), raddr=('127.0.0.1', 1234)>
  - File: /opt/anaconda3/lib/python3.12/site-packages/structlog/_frames.py:63
- **ResourceWarning**: unclosed <socket.socket fd=16, family=2, type=1, proto=6, laddr=('127.0.0.1', 52397), raddr=('127.0.0.1', 1234)>
  - File: /opt/anaconda3/lib/python3.12/re/__init__.py:224
- **ResourceWarning**: unclosed <socket.socket fd=21, family=2, type=1, proto=6, laddr=('127.0.0.1', 52399), raddr=('127.0.0.1', 1234)>
  - File: /opt/anaconda3/lib/python3.12/re/__init__.py:224
- **ResourceWarning**: unclosed <socket.socket fd=15, family=2, type=1, proto=6, laddr=('127.0.0.1', 52412), raddr=('127.0.0.1', 1234)>
  - File: /opt/anaconda3/lib/python3.12/site-packages/PIL/TiffImagePlugin.py:837
- **ResourceWarning**: unclosed <socket.socket fd=16, family=2, type=1, proto=6, laddr=('127.0.0.1', 52413), raddr=('127.0.0.1', 1234)>
  - File: /opt/anaconda3/lib/python3.12/site-packages/PIL/TiffImagePlugin.py:837
- **ResourceWarning**: unclosed <socket.socket fd=21, family=2, type=1, proto=6, laddr=('127.0.0.1', 52414), raddr=('127.0.0.1', 1234)>
  - File: /Users/albou/projects/abstractcore/tests/test_vision_comprehensive.py:309

## Detailed Results

### ollama/qwen2.5vl:7b
- **Success Rate**: 5/5 images
- **Average Response Time**: 6.56s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 8.02s (73 words)
    - Response: "The image shows a close-up of a bowl containing a creamy, mixed salad with visible chunks of vegetab..."
  - ✅ mystery2_sc.jpg: 6.08s (56 words)
    - Response: "The image shows a cat inside a clear, dome-shaped enclosure, which appears to be a part of a white, ..."
  - ✅ mystery4_wh.jpg: 6.26s (49 words)
    - Response: "The image depicts a humpback whale breaching the ocean surface, with its large, dark body partially ..."
  - ✅ mystery1_mp.jpg: 6.19s (56 words)
    - Response: "The image shows a scenic mountain landscape with a dirt path leading into the distance. A wooden fen..."
  - ✅ mystery3_us.jpg: 6.27s (44 words)
    - Response: "The image depicts a serene urban park path at dusk, lined with bare trees and green lampposts. The s..."

### ollama/llama3.2-vision:11b
- **Success Rate**: 5/5 images
- **Average Response Time**: 15.00s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 16.61s (126 words)
    - Response: "This image shows a black bowl of salad, likely potato salad, on a black table or countertop. The bow..."
  - ✅ mystery2_sc.jpg: 11.36s (53 words)
    - Response: "This image features a cat with its head poking out of a white, egg-shaped carrier with three ventila..."
  - ✅ mystery4_wh.jpg: 11.90s (36 words)
    - Response: "The image features a humpback whale breaching the water's surface, its tail visible above the surfac..."
  - ✅ mystery1_mp.jpg: 13.20s (43 words)
    - Response: "The image shows a dirt road with a fence on the left side and a grassy field on the right. The road ..."
  - ✅ mystery3_us.jpg: 21.94s (49 words)
    - Response: "The image features a serene city park scene at sunset, with a paved walkway lined with trees and ben..."

### ollama/gemma3:4b
- **Success Rate**: 5/5 images
- **Average Response Time**: 15.33s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 12.52s (43 words)
    - Response: "Here's a description of the image:  The image shows a dark, round bowl filled with a creamy, texture..."
  - ✅ mystery2_sc.jpg: 13.09s (48 words)
    - Response: "Here's a description of the image:  The image shows a fluffy, gray tabby cat peering out from inside..."
  - ✅ mystery4_wh.jpg: 17.36s (47 words)
    - Response: "Here's a description of the image:  The image shows a massive gray whale breaching the surface of th..."
  - ✅ mystery1_mp.jpg: 18.58s (55 words)
    - Response: "Here's a description of the image:  The image shows a dirt path winding up a grassy hill under a bri..."
  - ✅ mystery3_us.jpg: 15.11s (54 words)
    - Response: "Here's a description of the image:  The image depicts a long, paved walkway lined with trees and ill..."

### lmstudio/qwen/qwen2.5-vl-7b
- **Success Rate**: 5/5 images
- **Average Response Time**: 7.31s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 12.64s (45 words)
    - Response: "The image shows a bowl of potato salad with visible chunks of potatoes, carrots, and peas mixed toge..."
  - ✅ mystery2_sc.jpg: 6.53s (57 words)
    - Response: "The image shows a curious tabby cat nestled inside what appears to be a dome-shaped, white device re..."
  - ✅ mystery4_wh.jpg: 6.91s (52 words)
    - Response: "The image depicts a humpback whale breaching out of the water near a shoreline. The whale's dark gra..."
  - ✅ mystery1_mp.jpg: 4.83s (47 words)
    - Response: "The image shows a dirt path leading towards distant mountain peaks under a bright blue sky with scat..."
  - ✅ mystery3_us.jpg: 5.66s (63 words)
    - Response: "The image showcases a picturesque urban park scene during what appears to be sunrise or sunset, as i..."

### lmstudio/qwen/qwen3-vl-4b
- **Success Rate**: 5/5 images
- **Average Response Time**: 8.01s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 10.41s (41 words)
    - Response: "A black bowl brimming with creamy potato salad sits on a dark surface, its contents rich with diced ..."
  - ✅ mystery2_sc.jpg: 8.39s (43 words)
    - Response: "A curious tabby cat peeks playfully through a clear dome of a whimsical white spaceship-shaped pet b..."
  - ✅ mystery4_wh.jpg: 7.51s (45 words)
    - Response: "A majestic humpback whale breaches the choppy ocean surface, its massive body arcing upward with flu..."
  - ✅ mystery1_mp.jpg: 6.78s (42 words)
    - Response: "The image showcases a sun-drenched mountain trail winding along a grassy ridge, flanked by a weather..."
  - ✅ mystery3_us.jpg: 6.97s (52 words)
    - Response: "A tranquil park pathway at dusk, flanked by elegant lampposts and bare trees, stretches into the dis..."

### lmstudio/google/gemma-3n-e4b
- **Success Rate**: 5/5 images
- **Average Response Time**: 4.80s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 4.51s (59 words)
    - Response: "The image shows a black bowl filled with a creamy potato salad. The salad contains chunks of potatoe..."
  - ✅ mystery2_sc.jpg: 6.39s (97 words)
    - Response: "In this image, a cat is peering out from inside a white, spaceship-shaped pet carrier. The carrier h..."
  - ✅ mystery4_wh.jpg: 5.10s (54 words)
    - Response: "In this image, I see a magnificent humpback whale breaching out of the ocean, its massive pectoral f..."
  - ✅ mystery1_mp.jpg: 3.72s (45 words)
    - Response: "The image shows a dirt path winding through a grassy, mountainous landscape under a bright, partly c..."
  - ✅ mystery3_us.jpg: 4.28s (61 words)
    - Response: "The image shows a long, straight pathway lined with lampposts and trees in what appears to be a park..."

### openai/gpt-4o
- **Success Rate**: 5/5 images
- **Average Response Time**: 4.28s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 4.41s (44 words)
    - Response: "The image shows a bowl filled with a mixed salad, likely containing ingredients like potatoes, carro..."
  - ✅ mystery2_sc.jpg: 3.08s (29 words)
    - Response: "The image shows a cat inside a white pet carrier with a transparent dome, resembling a space helmet...."
  - ✅ mystery4_wh.jpg: 6.01s (43 words)
    - Response: "The image shows a large whale breaching the surface of the ocean, with its body partially out of the..."
  - ✅ mystery1_mp.jpg: 3.35s (44 words)
    - Response: "The image depicts a dirt path leading through a grassy landscape with a wooden fence on the left, un..."
  - ✅ mystery3_us.jpg: 4.56s (38 words)
    - Response: "The image shows a pathway lined with street lamps and bare trees on either side, set in an urban env..."

### anthropic/claude-3-haiku-20240307
- **Success Rate**: 5/5 images
- **Average Response Time**: 1.39s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 1.24s (36 words)
    - Response: "The image shows a bowl of what appears to be a creamy potato salad or similar dish. The main ingredi..."
  - ✅ mystery2_sc.jpg: 0.98s (27 words)
    - Response: "The image shows a cat's face peering through a transparent dome or container, with a patterned backg..."
  - ✅ mystery4_wh.jpg: 1.67s (37 words)
    - Response: "The image shows a large humpback whale emerging from the ocean, with its distinctive black and gray ..."
  - ✅ mystery1_mp.jpg: 1.56s (32 words)
    - Response: "The image shows a dirt path or trail winding through a grassy field, with a wooden fence running alo..."
  - ✅ mystery3_us.jpg: 1.53s (30 words)
    - Response: "The image depicts a scenic urban park setting at sunset. The main elements include a tree-lined walk..."

### huggingface/unsloth/Qwen2.5-VL-7B-Instruct-GGUF
- **Success Rate**: 5/5 images
- **Average Response Time**: 15.19s
- **Per-Image Results**:
  - ✅ mystery5_so.jpg: 20.78s (70 words)
    - Response: "The image appears to be an abstract composition with a dominant blue color scheme. It features a ser..."
  - ✅ mystery2_sc.jpg: 16.86s (67 words)
    - Response: "The image appears to be an abstract artwork with various shades of blue and gray, creating a sense o..."
  - ✅ mystery4_wh.jpg: 13.61s (67 words)
    - Response: "The image appears to be an abstract composition with a predominantly white background and a central ..."
  - ✅ mystery1_mp.jpg: 12.48s (50 words)
    - Response: "The image appears to be an abstract artwork with a variety of shapes and colors. It features a mix o..."
  - ✅ mystery3_us.jpg: 12.22s (62 words)
    - Response: "The image appears to be an abstract painting with a mix of geometric shapes and organic forms. The c..."

