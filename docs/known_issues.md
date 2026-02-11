# Known Issues

This document tracks known issues and limitations in the AbstractCore vision testing framework and model integrations.

## Vision Model Issues

### Ollama Gemma3n Models - Vision Not Implemented

**Issue**: Ollama Gemma3n models (`gemma3n:e4b`, `gemma3n:e2b`) respond with "Please provide me with the image!" when provided with images, despite the base models supporting vision.

**Affected Models**:
- `gemma3n:e4b`
- `gemma3n:e2b`

**Root Cause**:
The Ollama implementations of Gemma3n models currently show "Text input" only in their specifications. While the underlying Gemma 3n models DO support multimodal capabilities (vision, audio, video), these capabilities are not implemented in the Ollama versions as of 2024/2025.

**Evidence**:
- Web search results confirm Ollama Gemma3n models currently show "Text input" only
- Gemma 3n models are designed as true multimodal models supporting image, audio, video, and text inputs
- The discrepancy between base model capabilities and Ollama implementation suggests incomplete feature porting

**Current Status**:
Active limitation in Ollama. The full multimodal capabilities may require using other frameworks like Hugging Face transformers or MLX.

**Workaround**:
- Removed from AbstractCore vision tests
- Use alternative vision models: `qwen2.5vl:7b`, `llama3.2-vision:11b`, or cloud providers

**References**:
- Gemma 3n technical documentation confirms multimodal capabilities
- Ollama model listings show text-only input for current implementations

### Ollama Qwen2.5-VL Repetition Bug

**Issue**: Qwen2.5-VL models in Ollama get stuck in infinite repetition loops, especially for keyword extraction tasks.

**Affected Models**:
- `qwen2.5vl:7b`
- `qwen2.5vl:32b`
- `qwen2.5vl:72b`

**Root Cause**:
Confirmed bug in Ollama's implementation where the `repeat_penalty` parameter has no effect on Qwen2.5-VL models.

**Evidence**:
- GitHub Issue: [Token repetition issue with Qwen2.5-VL #10767](https://github.com/ollama/ollama/issues/10767)
- Multiple user reports confirming `repeat_penalty` settings have no effect
- Issue persists across different image types and prompts
- Original HuggingFace demo does not show this behavior, confirming it's Ollama-specific

**Current Status**:
Known active bug in Ollama as of 2024/2025. No official fix timeline provided.

**Workaround**:
- Use alternative sampling parameters (lower temperature, adjusted top_p)
- Consider using HuggingFace implementation directly for critical applications
- Monitor outputs and implement length limits in testing frameworks

**References**:
- [Ollama GitHub Issue #10767](https://github.com/ollama/ollama/issues/10767)
- Community reports on repetition behavior across Qwen2.5-VL model sizes

### LMStudio GLM-4.1V Vision Support Missing

**Issue**: LMStudio does not properly support GLM-4.1V-9B-Thinking vision capabilities, despite the model being a multimodal vision model.

**Affected Models**:
- `glm-4.1v-9b-thinking`
- Potentially other GLM-4.1V variants

**Root Cause**:
LMStudio appears to lack proper vision configuration for GLM-4.1V models. The error logs show:
- "Ignoring image file part in user message due to missing image config"
- "Model does not support images. Please use a model that does."

**Evidence**:
- GLM-4.1V-9B-Thinking is confirmed as a vision model supporting up to 4K resolution and arbitrary aspect ratios
- [HuggingFace model page](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) confirms multimodal capabilities
- LMStudio correctly receives OpenAI-compatible vision API format but rejects images
- Error occurs at LMStudio's internal message processing level, not at the API level

**Current Status**:
Active limitation in LMStudio. The GLM-4.1V model family may not be properly configured for vision tasks in LMStudio's model definitions.

**Workaround**:
- Use alternative vision models in LMStudio: `qwen/qwen3-vl-30b`, `qwen/qwen3-vl-4b`
- Use GLM-4.1V models through other frameworks (HuggingFace, MLX, or cloud APIs)
- Consider using Qwen3-VL models which have confirmed LMStudio vision support

**Technical Details**:
```
Error: Model does not support images. Please use a model that does.
Debug: [createJinjaMessagesInput] Ignoring image file part in user message due to missing image config
```

**References**:
- [GLM-4.1V-9B-Thinking HuggingFace](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)
- [GLM-V GitHub Repository](https://github.com/zai-org/GLM-V)
- AbstractCore model capabilities updated to reflect vision support

### HuggingFace GLM4V Architecture Support

**Issue**: HuggingFace transformers library does not recognize the custom `glm4v` architecture used by Glyph and GLM-4.1V models, causing `KeyError: 'glm4v'` when trying to load with `AutoModelForCausalLM`.

**Affected Models**:
- `zai-org/Glyph`
- `zai-org/GLM-4.1V-9B-Thinking`
- `zai-org/GLM-4.1V-9B-Base`
- Other GLM4V-based models

**Root Cause**:
The GLM4V architecture is a custom vision-language model architecture that requires:
- `AutoModelForImageTextToText` instead of `AutoModelForCausalLM`
- `AutoProcessor` instead of `AutoTokenizer`
- Special multimodal message handling

**Solution**:
Enhanced HuggingFace provider with vision model detection and proper loading:
- Added `_is_vision_model()` method to detect GLM4V and other vision models
- Added `_load_vision_model()` method using correct AutoModel classes
- Added `_generate_vision_model()` method for proper multimodal generation
- Updated model capabilities with GLM4V architecture information
- **IMPORTANT**: Requires `transformers>=4.57.1` for GLM4V architecture support

**Installation Requirements**:
```bash
pip install transformers>=4.57.1
```

**Technical Implementation**:
```python
# Vision model detection
def _is_vision_model(self, model: str) -> bool:
    vision_models = ['glyph', 'glm-4.1v', 'glm4v', 'qwen-vl', ...]
    return any(keyword in model.lower() for keyword in vision_models)

# Proper loading for vision models
processor = AutoProcessor.from_pretrained(model)
model_instance = AutoModelForImageTextToText.from_pretrained(model)
```

**Status**: Resolved in AbstractCore HuggingFace provider

**References**:
- [Glyph HuggingFace Model](https://huggingface.co/zai-org/Glyph)
- [Glyph GitHub Repository](https://github.com/thu-coai/Glyph)
- HuggingFace transformers documentation on vision models

## Model Configuration Issues

### Granite Vision Model Confusion

**Issue**: Multiple Granite model variants with different vision capabilities cause configuration errors.

**Affected Models**:
- `granite3.2:2b` (text-only)
- `granite3.3:2b` (text-only)
- `granite3.2-vision:2b` (vision-capable)

**Root Cause**:
Similar naming patterns between text-only and vision-capable Granite models.

**Solution**:
- Only `granite3.2-vision:2b` supports vision
- Updated test configurations to use correct model
- Added clear documentation of vision vs text-only variants

**Status**: Resolved

## Media Handling Issues

### HuggingFace GGUF Vision Format

**Issue**: HuggingFace GGUF vision models (like Qwen2.5-VL) received text placeholders instead of actual images, causing hallucination.

**Root Cause**:
LocalMediaHandler was incorrectly applied to vision-capable GGUF models, converting images to text descriptions instead of using proper multimodal message format.

**Solution**:
Implemented provider-specific vision handling using correct HuggingFace format:
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "prompt"},
    {"type": "image", "image": "file:///path/to/image.jpg"}
  ]
}
```

**Status**: Resolved

## Testing Framework Issues

### Image Path Resolution

**Issue**: Complex path resolution logic failed to handle all test image filename suffixes.

**Root Cause**:
Path resolution only checked for `_mp.jpg` suffix, missing `_wh.jpg`, `_sc.jpg`, `_us.jpg` variants.

**Solution**:
Simplified to direct filepath specification: `--image tests/vision_examples/mystery4_wh.jpg`

**Status**: Resolved

---

## Reporting New Issues

When reporting new vision-related issues:

1. **Include Model Details**: Provider, exact model name, version
2. **Provide Test Examples**: Specific images and prompts that fail
3. **Expected vs Actual**: What should happen vs what actually happens
4. **Environment Info**: OS, provider versions, AbstractCore version
5. **Reproduction Steps**: Minimal steps to reproduce the issue

Submit issues to the AbstractCore repository with the `vision` label.