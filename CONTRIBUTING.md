# Contributing to AbstractCore

We welcome contributions to AbstractCore! This guide will help you get started.

## Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/lpalbou/AbstractCore.git
   cd AbstractCore
   ```
3. **Install in development mode**:
   ```bash
   pip install -e ".[all]"  # Install with all optional dependencies
   ```
4. **Run tests** to make sure everything works:
   ```bash
   pytest
   ```

## How to Contribute

### 🐛 Bug Reports
- Use GitHub Issues to report bugs
- Include your Python version, AbstractCore version, and provider details
- Provide a minimal code example that reproduces the issue
- Include the full error message and traceback

### 💡 Feature Requests
- Check existing issues to avoid duplicates
- Clearly describe the use case and expected behavior
- Consider if the feature aligns with AbstractCore's philosophy of being focused infrastructure

### 🔧 Code Contributions

#### Development Setup
```bash
# Clone the repo
git clone https://github.com/lpalbou/AbstractCore.git
cd AbstractCore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all]"

# Install development tools
pip install pytest black isort mypy
```

#### Code Style
- **Black** for code formatting: `black .`
- **isort** for import sorting: `isort .`
- **Type hints** are required for new code
- **Docstrings** for all public functions and classes

#### Testing
- **No mocking** - AbstractCore tests against real implementations
- **Cross-provider testing** - Ensure features work across all providers
- **Integration tests** - Test real-world scenarios
- **Run tests**: `pytest`

#### Pull Request Guidelines
1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Write tests** for your changes
3. **Update documentation** if needed
4. **Run the full test suite**: `pytest`
5. **Format code**: `black . && isort .`
6. **Commit with clear messages**:
   ```bash
   git commit -m "Add retry logic for structured output validation

   - Implement FeedbackRetry class with error feedback
   - Add automatic validation retry in StructuredOutputHandler
   - Include comprehensive tests for retry scenarios"
   ```
7. **Push and create PR** on GitHub

## Types of Contributions We Need

### 🔌 Provider Support
- **New providers**: Add support for additional LLM providers
- **Model updates**: Update model lists and capabilities
- **Provider bug fixes**: Fix provider-specific issues

### 🛠️ Tool System
- **New common tools**: Add useful tools to `abstractllm/tools/common_tools.py`
- **Tool improvements**: Better error handling, validation, examples
- **Architecture support**: Tool calling for new model architectures

### 📊 Structured Output
- **Validation improvements**: Better error messages and retry logic
- **Pydantic compatibility**: Support for new Pydantic features
- **Provider compatibility**: Ensure structured output works across providers

### 🔄 Reliability & Performance
- **Retry mechanisms**: Improve retry logic and error handling
- **Circuit breakers**: Enhance circuit breaker functionality
- **Performance**: Optimize hot code paths

### 📖 Documentation
- **Examples**: Real-world use cases and code examples
- **Guides**: Tutorials for specific use cases
- **API docs**: Improve API reference documentation
- **Troubleshooting**: Common issues and solutions

## Code Organization

```
abstractllm/
├── core/           # Core interfaces and base classes
├── providers/      # LLM provider implementations
├── tools/          # Tool calling system
├── structured/     # Structured output handling
├── events/         # Event system
├── embeddings/     # Vector embeddings
├── utils/          # Utilities and helpers
└── exceptions/     # Custom exceptions
```

## What We DON'T Want

AbstractCore is focused infrastructure. We generally **don't accept** contributions for:

- **Application-level features** (use AbstractAgent/AbstractMemory instead)
- **Complex workflows** (keep AbstractCore simple)
- **Heavy dependencies** (maintain lightweight core)
- **Provider-specific hacks** (prefer universal solutions)
- **Framework lock-in** (maintain provider agnosticism)

## Development Philosophy

### Keep It Simple
- AbstractCore should remain lightweight and focused
- Prefer simple, universal solutions over complex ones
- Each feature should work across all providers

### Production First
- All code should be production-ready
- Include comprehensive error handling
- Add proper logging and observability

### No Breaking Changes
- Maintain backward compatibility
- Use deprecation warnings for removed features
- Follow semantic versioning

## Getting Help

- **GitHub Discussions** for questions and ideas
- **GitHub Issues** for bug reports and feature requests
- **Look at existing code** to understand patterns and conventions

## Recognition

Contributors are recognized in:
- **Git commits** show your contributions
- **Release notes** highlight significant contributions
- **ACKNOWLEDGEMENTS.md** lists all contributors

## Questions?

Feel free to open a GitHub Discussion if you have questions about contributing, need help getting started, or want to discuss a potential contribution.

Thank you for contributing to AbstractCore! 🚀