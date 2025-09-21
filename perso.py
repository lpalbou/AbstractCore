from abstractllm import create_llm
from abstractllm.core.types import GenerateResponse
from abstractllm.tools.common_tools import COMMON_TOOLS

#provider = create_llm("ollama", model="qwen3:4b", base_url="http://localhost:11434")
#provider = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234/v1")
provider = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")

# Test streaming
stream = provider.generate(
    "Say hello in three different languages.",
    stream=True
)

for chunk in stream:
    print(chunk.content, end="", flush=True)

