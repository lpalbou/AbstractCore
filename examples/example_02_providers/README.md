# AbstractCore Core: Provider Deep Dive

## What You'll Learn

- 🌐 Explore multiple AI providers
- 🔧 Configure advanced provider settings
- 🔄 Switch between different models and providers

### Learning Objectives

1. Understand provider diversity in AbstractCore
2. Compare and configure different providers
3. Handle provider-specific configurations
4. Implement dynamic provider selection

### Example Walkthrough

This example showcases the flexibility of AbstractCore Core by demonstrating:
- Multiple provider initialization
- Provider-specific parameter tuning
- Comparative model generation

### Key Code Snippet

```python
from abstractcore import create_llm

# Ollama Provider
ollama_llm = create_llm(
    provider='ollama',
    model='llama3',
    temperature=0.7
)

# OpenAI Provider
openai_llm = create_llm(
    provider='openai',
    model='gpt-4',
    max_tokens=150
)

# Dynamically select provider
def select_best_provider(task_complexity):
    return ollama_llm if task_complexity < 5 else openai_llm
```

### Next Steps

After understanding provider configurations, progress to `example_03_tools` to learn about function calling and tool integration.