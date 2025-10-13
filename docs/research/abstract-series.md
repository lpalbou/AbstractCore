# Components of the Abstract Series

The Abstract Series is a framework to govern every aspect of LLM communication, memory, agent and swarm.
Below are the components and their characteristics.

## AbstractCore
- Unified access to various Providers and Models
- Detect model architecture and capabilities
- Unified media system (handling of file types)
- Unified communication with LLM
    - same API for all provider & LLM
    - support both prompted and native tools
        - provides tool registration + basic tools (eg filesystem manipulation + execute command + web_search)
        - realtime rewrite of a tool call syntax to any format (openai, qwen3, xml, llama3, gemma, custom)
    - support both prompted and native structured outputs
- Unified retry mechanisms (both connexion + validation)
- Unified Event System (so that any other code can listen to events)
- Unified Logging System (dual output both console and file)
- Unified Basic Session (with /compact /facts /judge and serialization)
- Basic apps : 
    - summarizer (use for /compact)
    - extractor (use for /facts)
    - judge (use for /judge)


## AbstractServer (still in AbstractCore at the moment)
- OpenAI compatible endpoints /v1/models, /v1/chat/completions, etc
    - Additional optional parameters to filter by provider and model type (eg text-generation, text-embeddings)
- Provide FastAPI endpoints & docs
- Tested with Codex agentic CLI
- Next : 
    - should work with crush, opencode, openinterpreter, potentially more
    - load balancing to distribute requests across multiple instances


## AbstractMemory
- Multilayered memory session with integrated knowledgegraph
    - Core : identity
    - Working : short term memory (a dynamic abstraction of the current context; to be coupled with chat history session)
    - Episodic : long term memory (events)
    - Semantic : knowledge
    - Relational : people we know, their profiles and preferences
    - Notes : personal subjective and experiential notes
    - Verbatims : original discussions with 100% accuracy
    - Archive (Library) : every experience, every text reads but living only in subconscious; require stimuli trigger or active reconstruction for access
- Triple Storage/Access layer:
    - File system with markdowns : it requires a proper indexing of the files and internal representation of where to find the information
    - Lancedb : it enables the semantic search with vector embeddings over all memory components AND it allows to filter the vector space based on important/grounding metadata : user (relational), time (temporal), location (spatial)
    - Knowledge graph with the main entities and relationships; each with short summaries and direct links to the file system actual memories (possibly lancedb ?)
- The memory system exposes tools to create agency over memory
    - reconstruct(user, query, time, location) : rebuild a context based on the query and the grounding components : user, time, location. the process is dynamic and mimic the way human actively remembers about concepts
    - remember(fact/event/people/idea/knowledge, user, query, time, location) : will actively store in memory a fact, an event, a person, an idea or a knowledge. user, query, time and location are used to provide further context on when and how this active remember was triggered
    - reflect(query, focus) : will actively reflect and search related memories with a special angle / focus
    - alignment(query) : is the query aligned with the core values and self model ? does it create tensions or resonances ?
    - plan(query) : leverage working, semantic and potentially episodic memory to plan a task
- Methods to trigger manually / to be scheduled :
    - consolidate(day/week/month) : will consolidate the working memory component and infuse potentially new insights to the other memory components. Dual role of removing redundancy but also enabling long term changes of the identity and experiences
    - free(day/week/month) : will look at all the memory components for useless information / recollections and deprecate them into a secondary tier of Archive
    - resolve(day/week/month) : take the unresolved questions and engage in self-reflection loops to see if those questions can now be resolved and if not, what experiences would need to accumulate to solve them
- Passive processes
     - at the start of each interaction
        - automatically reconstruct() memories based on user, query, time, location and inject that context in the active memory
        - automatically inject facts (or relevant facts based on user, query, time, location) in the active memory
     - at the end of each interaction
        - automatically log the verbatim and note of each interaction
            - automatically create the links and KG (verbatim <-> note)
        - automatically extract new facts from each interaction

## AbstractAgent
- Advanced reasoning prompting
- Leverage ReAct loops for deep cycle of think->act->observe and enabling self reflection
- Direct access to AbstractMemory tools and capabilities
- Tool set : validated toolset to provide agency over filesystem, processing units and internet
- AbstractToolForge : special sandbox to enable the dynamic creation, modification or deprecation of tools (self-evolution)
- Queue logic to execute several tasks
    - Task can originate from the user or from the AI
    - Task has a description, a goal, a priority and a way to test/evaluate its completion
- Start/Suspend/Stop (Task)


## AbstractToolForge
- Enable the dynamic creation and evolution of tools based on execution feedback and requirements
- create_tool(specifications, goal, test_criteria)
- update_tool(specifications, feedbacks, goal, test_criteria)
- deprecate_tool(tool_id, reason, optional:replacement_tool)
- list_tools() : list tools currently registered
- tool(tool_id) : retrieve the full tool definition and code
- stats(tool_id) : the statistics on the tool usage, success rate and quality assessments
- protect(tool_id) : make a tool read-only
- unprotect(tool_id) : make a tool read & write