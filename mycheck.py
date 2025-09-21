from abstractllm import create_llm

# Create a provider
#llm = create_llm("openai", model="gpt-3.5-turbo")
llm = create_llm("anthropic", model="claude-3.5-haiku:latest") 
#llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
#llm = create_llm("ollama", model="qwen3-coder:30b")
#llm = create_llm("mlx", model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
#llm = create_llm("lmstudio", model="qwen/qwen3-coder-30b")
# too much messages llm = create_llm("huggingface", model="Qwen/Qwen3-4B")

# Generate a response
response = llm.generate("Hello, who are you ? identify yourself")
print(response.content)
