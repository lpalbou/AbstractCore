from setuptools import setup, find_packages

setup(
    name="abstractcore",
    version="2.0.0",
    author="AbstractCore Team",
    description="Unified interface to all LLM providers with essential infrastructure",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",      # For data validation
        "httpx>=0.24.0",        # For HTTP requests
        "tiktoken>=0.5.0",      # For tokenization
    ],
    extras_require={
        "openai": ["openai>=1.0.0,<2.0.0"],
        "anthropic": ["anthropic>=0.25.0,<1.0.0"],
        "ollama": [],  # No additional dependencies - uses core httpx
        "huggingface": ["transformers>=4.30.0,<5.0.0", "torch>=1.12.0,<3.0.0", "llama-cpp-python>=0.2.0,<1.0.0"],
        "mlx": ["mlx>=0.15.0,<1.0.0", "mlx-lm>=0.15.0,<1.0.0"],
        "lmstudio": [],  # Uses OpenAI compatible API
        "all": [
            "openai>=1.0.0,<2.0.0",
            "anthropic>=0.25.0,<1.0.0",
            "transformers>=4.30.0,<5.0.0",
            "torch>=1.12.0,<3.0.0",
            "llama-cpp-python>=0.2.0,<1.0.0",
            "mlx>=0.15.0,<1.0.0",
            "mlx-lm>=0.15.0,<1.0.0",
        ],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "mypy>=1.5.0"],
    }
)