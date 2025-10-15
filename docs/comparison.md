# Framework Comparison

This guide compares AbstractCore with other popular LLM frameworks to help you choose the right tool for your needs.

## Quick Comparison Table

| Feature | AbstractCore | LiteLLM | LangChain | LangGraph | LlamaIndex |
|---------|-------------|---------|-----------|-----------|-------------|
| **Primary Focus** | Clean LLM infrastructure | API compatibility | Full LLM framework | Agent workflows | RAG/search |
| **Size** | ~8k LOC | ~15k LOC | 100k+ LOC | ~30k LOC | ~80k LOC |
| **Learning Curve** | Minimal | Minimal | Steep | Medium | Medium |
| **Provider Support** | 6 providers | 100+ providers | Many via integrations | Via LangChain | Via LlamaHub |
| **Tool Calling** | ✅ Universal execution | ⚠️ Pass-through only | ✅ Via integrations | ✅ Native | ⚠️ Limited |
| **Streaming** | ✅ With tool support | ✅ Basic | ✅ Basic | ❌ Limited | ❌ Limited |
| **Structured Output** | ✅ With retry logic | ❌ None | ⚠️ Via parsers | ⚠️ Basic | ⚠️ Basic |
| **Production Ready** | ✅ Retry + circuit breakers | ⚠️ Basic | ✅ Via LangSmith | ✅ Via LangSmith | ⚠️ Depends |
| **Memory/Sessions** | ✅ Simple sessions | ❌ None | ✅ Advanced | ✅ Advanced | ✅ Advanced |
| **RAG Support** | ⚠️ Embeddings only | ❌ None | ✅ Full pipelines | ⚠️ Basic | ✅ Full pipelines |
| **Agent Support** | ❌ Single calls | ❌ None | ✅ Via chains | ✅ Native | ⚠️ Query engines |
| **Observability** | ✅ Built-in events | ⚠️ Basic logging | ✅ Via LangSmith | ✅ Via LangSmith | ⚠️ Basic |

## Detailed Comparisons

### AbstractCore vs LiteLLM

**LiteLLM** provides API compatibility across providers. **AbstractCore** provides production infrastructure.

#### When to choose LiteLLM:
```python
# LiteLLM is perfect for simple API compatibility
from litellm import completion

# Same API for all providers - that's it
response = completion(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])
```

#### When to choose AbstractCore:
```python
# AbstractCore for production features
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")

# Tool calling that actually executes functions
response = llm.generate("What's the weather?", tools=weather_tools)

# Structured output with automatic retry
user = llm.generate("Extract user info", response_model=UserModel)

# Production reliability built-in
# - Automatic retries with circuit breakers
# - Comprehensive event system
# - Streaming with tool support
```

**Summary**:
- **LiteLLM** = API compatibility layer
- **AbstractCore** = Production LLM infrastructure

### AbstractCore vs LangChain

**LangChain** is a comprehensive framework with many components. **AbstractCore** is focused infrastructure.

#### LangChain Strengths:
- **Ecosystem**: Massive ecosystem with pre-built components
- **RAG**: Complete RAG pipelines out of the box
- **Integrations**: Hundreds of integrations with external services
- **Prompt Management**: Advanced prompt templates and few-shot learning

#### AbstractCore Advantages:
- **Simplicity**: Learn in 5 minutes vs days for LangChain
- **Reliability**: Production-grade retry and error handling built-in
- **Performance**: Lightweight with minimal dependencies
- **Tool Execution**: Actually executes tools vs just formatting them

#### Code Comparison:

**LangChain approach:**
```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool

# More complex setup
llm = OpenAI(temperature=0)
tools = [CustomTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
response = agent.run("What's the weather?")
```

**AbstractCore approach:**
```python
from abstractcore import create_llm

# Simple and direct
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("What's the weather?", tools=weather_tools)
```

**Summary**:
- **LangChain** = Full framework with everything included
- **AbstractCore** = Clean infrastructure you build on

### AbstractCore vs LangGraph

**LangGraph** focuses on agent workflows and state management. **AbstractCore** focuses on reliable LLM calls.

#### LangGraph Strengths:
- **Agent Workflows**: Multi-step agent reasoning with state
- **Graph Structure**: DAG-based workflow definition
- **Human-in-the-loop**: Built-in human approval workflows
- **State Persistence**: Automatic state management

#### AbstractCore + AbstractAgent:
For complex agents, use [AbstractAgent](https://github.com/lpalbou/AbstractAgent) which builds on AbstractCore.

```python
# LangGraph - workflow definition
from langgraph import StateGraph

workflow = StateGraph()
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

# AbstractAgent - simpler agent creation
from abstract_agent import Agent
from abstractcore import create_llm

agent = Agent(
    llm=create_llm("openai", model="gpt-4o-mini"),
    tools=tools,
    memory=memory_system
)
```

**Summary**:
- **LangGraph** = Complex workflow orchestration
- **AbstractCore** = Foundation for building agents

### AbstractCore vs LlamaIndex

**LlamaIndex** specializes in RAG and document processing. **AbstractCore** provides general LLM infrastructure.

#### LlamaIndex Strengths:
- **RAG Focus**: Built specifically for retrieval-augmented generation
- **Document Processing**: Advanced chunking and parsing strategies
- **Vector Stores**: Integration with all major vector databases
- **Query Engines**: Sophisticated query processing and routing

#### AbstractCore Approach:
AbstractCore provides the embeddings foundation - you build the RAG pipeline:

```python
# LlamaIndex - full RAG out of the box
from llama_index import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")

# AbstractCore - you build the pipeline
from abstractcore import create_llm
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager()
llm = create_llm("openai", model="gpt-4o-mini")

# You implement: document chunking, vector storage, retrieval
relevant_docs = your_retrieval_system(query, embedder)
context = "\n".join(relevant_docs)
response = llm.generate(f"Context: {context}\nQuestion: {query}")
```

**Summary**:
- **LlamaIndex** = Full RAG framework
- **AbstractCore** = RAG building blocks

## Migration Guides

### From LiteLLM to AbstractCore

**Before (LiteLLM):**
```python
import litellm

response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

**After (AbstractCore):**
```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-3.5-turbo")
response = llm.generate("Hello")
print(response.content)

# Plus you get: tool calling, structured output, streaming, reliability
```

### From LangChain to AbstractCore

**Before (LangChain):**
```python
from langchain.llms import OpenAI
from langchain.schema import HumanMessage

llm = OpenAI()
response = llm([HumanMessage(content="Hello")])
```

**After (AbstractCore):**
```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-3.5-turbo")
response = llm.generate("Hello")

# Simpler API, same functionality
```

### From OpenAI SDK to AbstractCore

**Before (OpenAI SDK):**
```python
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

**After (AbstractCore):**
```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-3.5-turbo")
response = llm.generate("Hello")
print(response.content)

# Plus: works with any provider, built-in reliability, tool calling
```

## Performance Comparison

### Memory Usage

| Framework | Import Time | Memory (MB) | Dependencies |
|-----------|-------------|-------------|--------------|
| **AbstractCore** | ~0.3s | ~15MB | 3 core |
| **LiteLLM** | ~0.5s | ~25MB | 5+ |
| **LangChain** | ~2.0s | ~150MB | 50+ |
| **LangGraph** | ~1.0s | ~80MB | 20+ |
| **LlamaIndex** | ~1.5s | ~120MB | 40+ |

### Feature Completeness vs Complexity

```
                    ┌─────────────────────┐
                 High│  LangChain     ●   │
 Feature Completeness│                     │
                     │  LlamaIndex ●      │
                     │                     │
                     │  LangGraph     ●   │
                Medium│                     │
                     │                     │
                     │  AbstractCore  ●   │
                     │                     │
                  Low│  LiteLLM       ●   │
                     └─────────────────────┘
                      Low        Medium   High
                           Complexity

● = Sweet spot for different use cases
```

## Use Case Recommendations

### Simple LLM Integration
```
Your App → AbstractCore → LLM Provider
```
**Best choice**: AbstractCore
**Why**: Simple, reliable, production-ready

### API Compatibility Only
```
Your App → LiteLLM → Multiple Providers
```
**Best choice**: LiteLLM
**Why**: Massive provider support, minimal overhead

### Complex RAG System
```
Documents → LlamaIndex → Vector DB → Query Engine
```
**Best choice**: LlamaIndex
**Why**: Full RAG pipeline with document processing

### Agent Workflows
```
Task → LangGraph → Multi-step Agent → Tools
```
**Best choice**: LangGraph or AbstractAgent
**Why**: State management and workflow orchestration

### Full Application Framework
```
Complex App → LangChain → Everything
```
**Best choice**: LangChain
**Why**: Comprehensive ecosystem

## Decision Matrix

### Choose AbstractCore if you want:
- ✅ Production-ready LLM infrastructure
- ✅ Universal tool calling across all providers
- ✅ Structured output with automatic retry
- ✅ Streaming with tool support
- ✅ Clean, simple API
- ✅ Built-in reliability (retry, circuit breakers)
- ✅ Comprehensive observability

### Choose LiteLLM if you want:
- ✅ Simple API compatibility
- ✅ Maximum provider coverage (100+)
- ✅ Drop-in replacement for OpenAI SDK
- ❌ But no: tool execution, structured output, reliability features

### Choose LangChain if you want:
- ✅ Pre-built components for everything
- ✅ Massive ecosystem and integrations
- ✅ Complex chain orchestration
- ❌ But accept: complexity, learning curve, many dependencies

### Choose LangGraph if you want:
- ✅ Multi-step agent workflows
- ✅ State management and persistence
- ✅ Human-in-the-loop workflows
- ❌ But don't need: simple LLM calls, basic tool execution

### Choose LlamaIndex if you want:
- ✅ Full RAG pipelines out of the box
- ✅ Advanced document processing
- ✅ Vector database integrations
- ❌ But don't need: general-purpose LLM calls, other use cases

## Combination Strategies

Many projects benefit from combining tools:

### AbstractCore + LlamaIndex
```python
# Use LlamaIndex for RAG, AbstractCore for LLM calls
from llama_index import VectorStoreIndex
from abstractcore import create_llm

index = VectorStoreIndex.from_documents(docs)
relevant_docs = index.similarity_search(query)

llm = create_llm("anthropic", model="claude-3-5-sonnet-latest")
response = llm.generate(f"Context: {relevant_docs}\nQuestion: {query}")
```

### AbstractCore + Custom Agents
```python
# Use AbstractCore for reliable LLM calls in your agent
from abstractcore import create_llm

class MyAgent:
    def __init__(self):
        self.llm = create_llm("openai", model="gpt-4o-mini")

    def plan(self, task):
        return self.llm.generate(f"Plan steps for: {task}")

    def execute_step(self, step, tools):
        return self.llm.generate(step, tools=tools)
```

### AbstractCore + LangChain Components
```python
# Use LangChain for prompts, AbstractCore for execution
from langchain.prompts import PromptTemplate
from abstractcore import create_llm

template = PromptTemplate.from_template("Translate {text} to {language}")
llm = create_llm("openai", model="gpt-4o-mini")

prompt = template.format(text="Hello", language="French")
response = llm.generate(prompt)
```

## Summary

**AbstractCore** is designed to be the **reliable foundation** that other tools can build on. It excels at:

1. **Production reliability** - Built-in retry, circuit breakers, error handling
2. **Universal tool calling** - Works across all providers
3. **Structured output** - Type-safe responses with validation
4. **Clean architecture** - Simple to use and integrate

For specialized needs, combine AbstractCore with:
- **LlamaIndex** for RAG pipelines
- **AbstractAgent** for complex agents
- **AbstractMemory** for advanced memory
- **LangChain components** for specific features

Choose the right tool for your specific needs, and don't hesitate to combine them when it makes sense.

---

**The key insight**: Every framework has trade-offs. AbstractCore prioritizes reliability and simplicity over feature completeness, making it an excellent foundation for production LLM applications.