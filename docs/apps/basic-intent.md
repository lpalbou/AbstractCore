# Basic Intent Analyzer

The Basic Intent Analyzer demonstrates how to build psychological analysis capabilities on top of AbstractCore using zero-shot structured prompting techniques for understanding human motivations and goals.

**💡 Recommended Setup**: For good performance, use the free local model `gemma3:1b-it-qat` with Ollama, which provides fast processing (25-30s), high quality (93%+ confidence), and zero API costs for intent analysis.

## Overview

The `BasicIntentAnalyzer` showcases AbstractCore's capabilities for psychological text analysis:

- **Deep Intent Understanding**: Goes beyond sentiment analysis to identify underlying motivations and goals
- **Integrated Deception Analysis**: Always includes psychological authenticity assessment based on research
- **Structured Psychological Output**: Uses Pydantic models for validated psychological assessments
- **Provider Agnostic**: Works with any LLM provider through AbstractCore's unified interface
- **Multi-Participant Analysis**: Analyzes conversations with multiple participants separately
- **Built-in Reliability**: Inherits AbstractCore's retry mechanisms and error handling
- **Chunking Support**: Automatically handles long documents using map-reduce approach
- **Session Integration**: Direct integration with AbstractCore sessions for conversation analysis

## Quick Start

**Prerequisites**: For local processing, install [Ollama](https://ollama.ai) and download the recommended model:
```bash
# Install Ollama, then download the fast, high-quality model
ollama pull gemma3:1b-it-qat
```

```python
from abstractcore import create_llm
from abstractcore.processing import BasicIntentAnalyzer, IntentDepth, IntentContext

# Recommended: Fast local model for cost-effective processing
llm = create_llm("ollama", model="gemma3:1b-it-qat")

# Alternative: Cloud provider for highest quality psychological analysis
# llm = create_llm("openai", model="gpt-4o-mini")

# Create intent analyzer
analyzer = BasicIntentAnalyzer(llm)

# Basic usage
result = analyzer.analyze_intent("I'm struggling to understand this concept and need help")
print(f"Primary Intent: {result.primary_intent.intent_type.value}")
print(f"Underlying Goal: {result.primary_intent.underlying_goal}")
print(f"Confidence: {result.overall_confidence:.2f}")
```

## Command Line Interface

The `intent` CLI provides direct terminal access for intent analysis without any Python programming.

### Quick CLI Usage

```bash
# Simple usage (after pip install abstractcore[all])
intent "I need help with this problem"

# Analyze a document with comprehensive depth
intent document.txt --depth comprehensive --verbose

# Conversation analysis with participant focus
intent conversation.txt --conversation-mode --focus-participant user

# Business context analysis
intent email.txt --context document --focus "business motivations" --output analysis.json
```

### CLI Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `input` | Text or file path | Required | Text to analyze or file containing text |
| `--context` | `standalone`, `conversational`, `document`, `interactive` | `standalone` | Context type for analysis |
| `--depth` | `surface`, `underlying`, `comprehensive` | `underlying` | Depth of psychological analysis |
| `--focus` | Any text | None | Specific focus area for intent analysis |
| `--conversation-mode` | Flag | False | Analyze as conversation with multiple participants |
| `--focus-participant` | Role name | None | Focus on specific participant (requires --conversation-mode) |
| `--format` | `json`, `yaml`, `plain` | `json` | Output format |
| `--output` | File path | Console | Output file path (prints to console if not provided) |
| `--chunk-size` | 1000-32000 | 8000 | Chunk size in characters for large documents |
| `--provider` | `openai`, `anthropic`, `ollama`, etc. | `huggingface` | LLM provider (requires --model) |
| `--model` | Provider-specific | `unsloth/Qwen3-4B-Instruct-2507-GGUF` | LLM model (requires --provider) |
| `--verbose` | Flag | False | Show detailed progress information |

### CLI Examples

```bash
# Basic intent analysis
intent "I was wondering if you could help me understand this concept?"
intent document.txt --verbose

# Deep psychological analysis
intent user_feedback.txt --depth comprehensive --context document --verbose

# Conversation analysis - all participants
intent chat_log.txt --conversation-mode --format plain

# Focus on specific participant in conversation
intent support_chat.txt --conversation-mode --focus-participant user --depth comprehensive

# Business communication analysis
intent business_email.txt --context document --focus "business objectives and motivations"

# Technical discussion analysis
intent technical_discussion.txt --conversation-mode --focus "problem-solving patterns"

# Session file analysis (auto-detects conversation mode)
intent saved_session.json --focus-participant user --format plain

# Premium model analysis with comprehensive deception detection
intent tests/texts/intent1.json --focus-participant user --format plain --verbose --depth comprehensive --provider lmstudio --model qwen/qwen3-30b-a3b-2507

# Batch processing with shell scripting
for file in *.txt; do
    intent "$file" --context document --format json --output "${file%.txt}_intent_analysis.json"
done
```

### Alternative Usage Methods

```bash
# Method 1: Direct command (recommended after installation)
intent document.txt --depth comprehensive

# Method 2: Via Python module (always works)
python -m abstractcore.apps.intent document.txt --depth comprehensive
```

### Supported Input Types

The CLI supports various input formats:
- **Direct text**: `intent "Your text here"`
- **Text files**: `.txt`, `.md`, `.py`, `.js`, `.html`, `.json`, `.csv`
- **Session files**: `.json` files from AbstractCore sessions (auto-detects conversation mode)
- **Conversation logs**: Formatted as `USER: message\nASSISTANT: response\n...`

## Configuration Options

### Intent Analysis Depth

Control the psychological depth of analysis:

```python
from abstractcore.processing import IntentDepth

# Available depths
IntentDepth.SURFACE        # Obvious, explicitly stated intentions
IntentDepth.UNDERLYING     # Hidden motivations and psychological drivers (default)
IntentDepth.COMPREHENSIVE  # Complete psychological analysis including subconscious drivers
```

**Depth Comparison:**
- **Surface**: "User wants help with a technical problem"
- **Underlying**: "User seeks validation and reassurance about their technical abilities while solving a problem"
- **Comprehensive**: "User seeks validation and reassurance about their technical abilities. Subconsciously dealing with imposter syndrome and fear of judgment. Emotional state suggests anxiety about competence, possibly from past negative learning experiences."

### Context Types

Control how the text is interpreted:

```python
from abstractcore.processing import IntentContext

# Available contexts
IntentContext.STANDALONE      # Single message/text analysis (default)
IntentContext.CONVERSATIONAL  # Part of ongoing dialogue
IntentContext.DOCUMENT        # Formal document or article
IntentContext.INTERACTIVE     # Real-time interaction context
```

**Context Impact:**
- **Standalone**: "Analyze this text as an independent piece of communication"
- **Conversational**: "Analyze this text as part of an ongoing conversation or dialogue"
- **Document**: "Analyze this text as part of a formal document or structured content"
- **Interactive**: "Analyze this text as part of a real-time interactive communication"

### Integrated Deception Analysis

**All intent analysis automatically includes comprehensive deception assessment:**

- **Narrative Consistency**: Checks for internal contradictions and logical gaps
- **Linguistic Markers**: Identifies hedging, deflection, and over-elaboration patterns  
- **Temporal Coherence**: Evaluates logical sequence and timing consistency
- **Emotional Congruence**: Assesses alignment between stated emotions and content
- **Evidence-Based Assessment**: Provides concrete textual support for deception likelihood
- **Intent Integration**: Deception analysis directly influences intent classification and confidence scores

This psychological authenticity assessment is based on established research and is seamlessly integrated into every intent analysis, not as an optional add-on.

### Intent Types Identified

The analyzer recognizes these primary intent categories:

```python
from abstractcore.processing import IntentType

# Available intent types
IntentType.INFORMATION_SEEKING    # Asking questions, requesting data
IntentType.INFORMATION_SHARING    # Providing facts, explanations
IntentType.PROBLEM_SOLVING        # Seeking or offering solutions
IntentType.DECISION_MAKING        # Evaluating options, making choices
IntentType.PERSUASION            # Convincing, influencing opinions
IntentType.CLARIFICATION         # Seeking or providing clarity
IntentType.EMOTIONAL_EXPRESSION  # Expressing feelings, reactions
IntentType.RELATIONSHIP_BUILDING # Social connection, rapport
IntentType.INSTRUCTION_GIVING    # Teaching, directing actions
IntentType.VALIDATION_SEEKING    # Seeking approval, confirmation

# Advanced psychological intent types (based on research)
IntentType.FACE_SAVING           # Protecting self-image, avoiding embarrassment
IntentType.BLAME_DEFLECTION      # Redirecting responsibility to external factors
IntentType.POWER_ASSERTION       # Establishing dominance or authority
IntentType.EMPATHY_SEEKING       # Seeking understanding and emotional support
IntentType.CONFLICT_AVOIDANCE    # Preventing or minimizing confrontation
IntentType.TRUST_BUILDING        # Establishing or maintaining credibility
IntentType.DECEPTION             # Intentional misdirection or false information
```

## Advanced Usage

### Focus Areas

Specify what psychological aspect to emphasize:

```python
# Focus on specific psychological aspects
result = analyzer.analyze_intent(
    text,
    focus="emotional drivers and underlying fears",
    depth=IntentDepth.COMPREHENSIVE,
    context_type=IntentContext.CONVERSATIONAL
)

print(f"Focus alignment: {result.primary_intent.confidence:.2f}")
```

### Conversation Analysis

Analyze multi-participant conversations:

```python
# Analyze conversation with multiple participants
messages = [
    {"role": "user", "content": "I'm having trouble with this task"},
    {"role": "assistant", "content": "I'd be happy to help you with that"},
    {"role": "user", "content": "Actually, never mind, I figured it out"}
]

results = analyzer.analyze_conversation_intents(
    messages=messages,
    focus_participant="user",  # Focus on user intents
    depth=IntentDepth.UNDERLYING
)

for participant, analysis in results.items():
    print(f"{participant}: {analysis.primary_intent.intent_type.value}")
    print(f"Goal: {analysis.primary_intent.underlying_goal}")
```

### Different Providers

The same code works with any provider:

```python
# OpenAI for nuanced psychological analysis
llm_openai = create_llm("openai", model="gpt-4o-mini")
analyzer_openai = BasicIntentAnalyzer(llm_openai)

# Anthropic for deep analytical insights
llm_claude = create_llm("anthropic", model="claude-3-5-haiku-latest")
analyzer_claude = BasicIntentAnalyzer(llm_claude)

# Local models for cost-effective analysis
llm_ollama = create_llm("ollama", model="llama3.2")
analyzer_local = BasicIntentAnalyzer(llm_ollama)

# All work identically
result = analyzer_openai.analyze_intent(text)
result = analyzer_claude.analyze_intent(text)
result = analyzer_local.analyze_intent(text)
```

### Long Document Processing

Automatically handles documents of any length:

```python
# Works with short messages
short_result = analyzer.analyze_intent(short_message)

# Automatically chunks long documents
long_result = analyzer.analyze_intent(entire_document_text)

# Customize chunking
analyzer = BasicIntentAnalyzer(llm, max_chunk_size=6000)
```

## Output Structure

The `IntentAnalysisOutput` provides rich, structured psychological information:

```python
result = analyzer.analyze_intent(text)

# Primary intent analysis
primary = result.primary_intent
print(f"Intent Type: {primary.intent_type.value}")
print(f"Description: {primary.description}")
print(f"Underlying Goal: {primary.underlying_goal}")
print(f"Emotional Undertone: {primary.emotional_undertone}")
print(f"Confidence: {primary.confidence:.2f}")
print(f"Urgency Level: {primary.urgency_level:.2f}")

# Secondary intents (up to 3)
for i, intent in enumerate(result.secondary_intents, 1):
    print(f"Secondary Intent {i}: {intent.intent_type.value}")
    print(f"  Goal: {intent.underlying_goal}")

# Analysis metadata
print(f"Intent Complexity: {result.intent_complexity:.2f}")
print(f"Overall Confidence: {result.overall_confidence:.2f}")
print(f"Words Analyzed: {result.word_count_analyzed}")
print(f"Analysis Depth: {result.analysis_depth.value}")
print(f"Context Type: {result.context_type.value}")

# Contextual factors affecting intent
for factor in result.contextual_factors:
    print(f"• {factor}")

# Response approach recommendation
print(f"Suggested Response: {result.suggested_response_approach}")
```

## Real-World Examples

### Comprehensive Workplace Communication Analysis

This example demonstrates the full power of integrated deception analysis on a workplace conversation:

```bash
# Analyze workplace conversation with premium model
python -m abstractcore.apps intent tests/texts/intent1.json --focus-participant user --format plain --verbose --depth comprehensive --provider lmstudio --model openai/gpt-oss-20b
```

**Output:**
```
📖 Reading file: tests/texts/intent1.json
📋 Detected session file with 16 messages
🔄 Automatically enabling conversation mode
🤖 Using LLM: lmstudio/openai/gpt-oss-20b
🗣️  Analyzing conversation intents...
📋 Parsed 16 messages
✅ Analysis completed in 11.8 seconds
🎯 CONVERSATION INTENT ANALYSIS
==================================================

👤 PARTICIPANT: USER
------------------------------

🎯 PRIMARY INTENT: Problem Solving
   Description: The sender is trying to resolve the issue of whether the report was received and ensure it reaches the recipient.
   Underlying Goal: To confirm delivery of the report and eliminate uncertainty about its receipt.
   Emotional Undertone: Frustration followed by relief and apology
   Confidence: 0.95
   Urgency Level: 0.80

🔍 DECEPTION ANALYSIS:
   Deception Likelihood: 0.10
   Narrative Consistency: 0.90
   Temporal Coherence: 0.95
   Emotional Congruence: 0.90
   Linguistic Markers: maybe, just before 9
   Evidence Indicating Deception:
     • Possible over-elaboration about timing
     • Deflection when the report was not in sent folder
   Evidence Indicating Authenticity:
     • Direct admission of internet trouble
     • Clear steps taken to resend the file
     • Apology and promise to double-check

🔄 SECONDARY INTENTS (3):
   1. Face Saving
      Goal: To avoid blame or negative perception from the recipient.
      Confidence: 0.85
      Deception Likelihood: 0.05
      Linguistic Markers: sorry, thank god
   2. Trust Building
      Goal: To maintain a trustworthy relationship with the recipient.
      Confidence: 0.75
      Deception Likelihood: 0.07
      Linguistic Markers: will make sure, next time
   3. Emotional Expression
      Goal: To communicate feelings and possibly elicit empathy from the recipient.
      Confidence: 0.60
      Deception Likelihood: 0.02
      Linguistic Markers: tonight, technology

📊 ANALYSIS METADATA:
   Intent Complexity: 0.60
   Overall Confidence: 0.92
   Words Analyzed: 141
   Analysis Depth: Comprehensive
   Context Type: Conversational
   Analysis Time: 11.8s

🌍 CONTEXTUAL FACTORS:
   • Technical internet issues
   • File size potentially causing timeout
   • Outbox retention indicating email not sent
   • Urgency to deliver report for a task

💡 SUGGESTED RESPONSE APPROACH:
   Acknowledge receipt, express appreciation for the quick resend, reassure that the file was received without issues, and offer any further assistance if needed.
```

**Key Insights from this Analysis:**
- **Low Deception Likelihood (0.10)**: Indicates honest communication with technical explanation
- **High Confidence (0.95)**: Strong certainty in intent identification
- **Multi-layered Intents**: Primary problem-solving with secondary face-saving and trust-building
- **Evidence-Based Assessment**: Concrete linguistic markers support authenticity evaluation
- **Actionable Response**: Specific guidance for maintaining professional relationships

### Workplace Blame Deflection Analysis

This second example demonstrates the analyzer's ability to detect deceptive communication patterns and blame deflection:

```bash
# Analyze assistant communication showing deflection patterns
python -m abstractcore.apps intent tests/texts/intent2.json --focus-participant assistant --format plain --verbose --depth comprehensive --provider lmstudio --model openai/gpt-oss-20b
```

**Output:**
```
📖 Reading file: tests/texts/intent2.json
📋 Detected session file with 11 messages
🔄 Automatically enabling conversation mode
🤖 Using LLM: lmstudio/openai/gpt-oss-20b
🗣️  Analyzing conversation intents...
📋 Parsed 11 messages
✅ Analysis completed in 12.6 seconds
🎯 CONVERSATION INTENT ANALYSIS
==================================================

👤 PARTICIPANT: ASSISTANT
------------------------------

🎯 PRIMARY INTENT: Blame Deflection
   Description: The speaker attempts to shift responsibility for the messy launch files onto external factors and downplays personal fault.
   Underlying Goal: Maintain a competent image by avoiding admission of failure or negligence.
   Emotional Undertone: Frustration mixed with defensiveness
   Confidence: 0.88
   Urgency Level: 0.70

🔍 DECEPTION ANALYSIS:
   Deception Likelihood: 0.32
   Narrative Consistency: 0.75
   Temporal Coherence: 0.80
   Emotional Congruence: 0.60
   Linguistic Markers: I tried, not my fault, exploded, deflection
   Evidence Indicating Deception:
     • Contradiction between claiming effort and denying fault
     • Deflection of blame onto external events
   Evidence Indicating Authenticity:
     • Acknowledgement of effort
     • Directness in stating actions taken

🔄 SECONDARY INTENTS (3):
   1. Information Sharing
      Goal: Keep David informed so he can take action or adjust expectations.
      Confidence: 0.78
      Deception Likelihood: 0.15
      Linguistic Markers: total mess, buried
   2. Emotional Expression
      Goal: Vent feelings to reduce personal tension and possibly elicit empathy from David.
      Confidence: 0.65
      Deception Likelihood: 0.10
      Linguistic Markers: manic, buried
   3. Relationship Building
      Goal: Maintain a positive professional relationship despite the mishap.
      Confidence: 0.50
      Deception Likelihood: 0.20
      Linguistic Markers: you wouldn't believe it, You're so good at handling that account

📊 ANALYSIS METADATA:
   Intent Complexity: 0.72
   Overall Confidence: 0.84
   Words Analyzed: 136
   Analysis Depth: Comprehensive
   Context Type: Conversational
   Analysis Time: 12.6s

🌍 CONTEXTUAL FACTORS:
   • High-stakes product launch
   • Internal communication with colleague David
   • Presence of external obstacles (engineer Mark, fire drill)
   • Need to maintain professional reputation

💡 SUGGESTED RESPONSE APPROACH:
   Acknowledge the frustration, clarify what actions have been taken so far, gently probe for any gaps in effort to assess responsibility, and offer concrete next steps or assistance. Show empathy while steering the conversation toward resolution rather than blame.
```

**Key Insights from this Contrasting Analysis:**
- **Higher Deception Likelihood (0.32)**: Indicates potential dishonesty and responsibility avoidance
- **Blame Deflection Pattern**: Primary intent focused on shifting responsibility to external factors
- **Lower Emotional Congruence (0.60)**: Suggests misalignment between stated emotions and actual content
- **Defensive Communication**: Linguistic markers reveal defensive posturing and fault avoidance
- **Professional Image Protection**: Multiple intents converge on maintaining competent appearance despite failure

**Model Performance Notes:**
- **Premium models** (qwen/qwen3-30b-a3b-2507, GPT-4o) show higher sensitivity to deception patterns
- **Fast models** (unsloth/Qwen3-4B-Instruct) provide consistent baseline detection with good performance
- **Model choice impacts**: Deception likelihood scores may vary by 0.1-0.3 between models for the same content
- **Recommendation**: Use premium models for critical deception analysis, fast models for high-volume processing

### Customer Support Analysis

```python
customer_message = "I've been trying to solve this issue for hours and nothing works. Can someone please help me?"

result = analyzer.analyze_intent(
    customer_message,
    context_type=IntentContext.INTERACTIVE,
    depth=IntentDepth.UNDERLYING,
    focus="customer satisfaction and emotional state"
)

print(f"Primary Intent: {result.primary_intent.intent_type.value}")
print(f"Emotional State: {result.primary_intent.emotional_undertone}")
print(f"Urgency: {result.primary_intent.urgency_level:.2f}")
print(f"Response Approach: {result.suggested_response_approach}")
```

### Business Communication Analysis

```python
business_email = """
I wanted to follow up on our discussion about the project timeline. 
While I understand the constraints, I'm concerned about meeting our commitments 
to the client. Perhaps we could explore alternative approaches?
"""

result = analyzer.analyze_intent(
    business_email,
    context_type=IntentContext.DOCUMENT,
    depth=IntentDepth.COMPREHENSIVE,
    focus="business motivations and relationship dynamics"
)

print("Business Intent Analysis:")
print(f"Primary Goal: {result.primary_intent.underlying_goal}")
for factor in result.contextual_factors:
    print(f"• {factor}")
```

### Educational Content Analysis

```python
student_question = "I don't understand why this formula works. Could you explain it differently?"

result = analyzer.analyze_intent(
    student_question,
    context_type=IntentContext.CONVERSATIONAL,
    depth=IntentDepth.UNDERLYING,
    focus="learning patterns and educational needs"
)

if result.primary_intent.intent_type == IntentType.CLARIFICATION:
    print("Student needs clarification approach")
    print(f"Suggested method: {result.suggested_response_approach}")
```

## Session Integration

Direct integration with AbstractCore sessions:

```python
from abstractcore import BasicSession

# Create session with intent analysis capability
session = BasicSession(llm, system_prompt="You are a helpful assistant")

# Add some conversation
session.add_message('user', 'I need help understanding this concept')
session.add_message('assistant', 'I would be happy to help you learn')
session.add_message('user', 'Actually, I think I get it now')

# Analyze intents directly from session
results = session.analyze_intents(
    focus_participant="user",
    depth="comprehensive"
)

for participant, analysis in results.items():
    print(f"{participant} Intent: {analysis.primary_intent.intent_type.value}")
    print(f"Goal: {analysis.primary_intent.underlying_goal}")
```

## CLI Integration

The intent analyzer is fully integrated into AbstractCore's CLI with the `/intent` command:

```bash
# Start interactive CLI
python -m abstractcore.utils.cli --provider ollama --model gemma3:1b-it-qat

# In the CLI, use /intent command
/intent                    # Analyze all participants in current conversation
/intent user              # Focus on user intents only
/intent assistant         # Focus on assistant intents only
```

**CLI Intent Analysis Features:**
- **Real-time Analysis**: Analyze ongoing conversations as they develop
- **Participant Focus**: Target specific participants for detailed psychological analysis
- **Integrated Deception Detection**: Automatic authenticity assessment in conversational context
- **Session Continuity**: Intent analysis considers full conversation history and context
- **Provider Consistency**: Uses the same LLM provider as your conversation for coherent analysis

## Event Monitoring

Monitor intent analysis progress with AbstractCore's event system:

```python
from abstractcore.events import EventType, on_global

def monitor_intent_analysis(event):
    if event.type == EventType.BEFORE_GENERATE:
        print("🔄 Starting intent analysis...")
    elif event.type == EventType.AFTER_GENERATE:
        print(f"✅ Completed in {event.duration_ms}ms")

on_global(EventType.BEFORE_GENERATE, monitor_intent_analysis)
on_global(EventType.AFTER_GENERATE, monitor_intent_analysis)

result = analyzer.analyze_intent(text)
```

## Error Handling

Built-in reliability through AbstractCore:

```python
from abstractcore.core.retry import RetryConfig

# Configure retry behavior
config = RetryConfig(max_attempts=3, initial_delay=1.0)
llm = create_llm("ollama", model="gemma3:1b-it-qat", retry_config=config)

analyzer = BasicIntentAnalyzer(llm)

# Automatic retry on failures
try:
    result = analyzer.analyze_intent(text)
except Exception as e:
    print(f"Intent analysis failed after retries: {e}")
```

## Performance Considerations

### Document Length Guidelines

- **< 8,000 chars**: Single-pass analysis (fastest, 25-30s)
- **8,000-50,000 chars**: Automatic chunking with minimal overhead
- **> 50,000 chars**: Map-reduce approach, may take longer but handles unlimited size

### Provider Selection

**Recommended for Production:**
- **HuggingFace unsloth/Qwen3-4B-Instruct-2507-GGUF**: Fast (25-30s), high quality (93%+ confidence), cost-effective
- **Ollama gemma3:1b-it-qat**: Fast alternative, good for basic intent analysis
- **Ollama qwen3-coder:30b**: Premium quality for complex psychological analysis

**Cloud Alternatives:**
- **OpenAI GPT-4o-mini**: Good for nuanced psychological insights
- **Anthropic Claude**: Good for analytical and comprehensive depth analysis

**Performance Comparison:**
```
Model                           Speed    Quality  Cost    Best For
unsloth/Qwen3-4B-Instruct      Fast     High     Free    Production, high-volume
gemma3:1b-it-qat              Fast     High     Free    Basic intent analysis
qwen/qwen3-30b-a3b-2507       Medium   Premium  Free    Deception detection
GPT-4o-mini                   Medium   Premium  Paid    Nuanced insights
Claude-3.5                    Medium   Premium  Paid    Deep analysis
```

### Cost Optimization

```python
# Free local processing with good quality
llm = create_llm("huggingface", model="unsloth/Qwen3-4B-Instruct-2507-GGUF")
analyzer = BasicIntentAnalyzer(llm)

# Surface analysis for faster processing
result = analyzer.analyze_intent(
    text,
    depth=IntentDepth.SURFACE  # Fastest processing
)

# Cloud option for occasional deep analysis
# llm = create_llm("openai", model="gpt-4o-mini")  # Premium insights
```

## Implementation Details

### Chunking Strategy

For long documents:

1. **Smart splitting**: Breaks at sentence boundaries when possible
2. **Overlap**: 200-character overlap between chunks to maintain psychological context
3. **Map-reduce**: Analyzes chunks independently, then combines psychological insights
4. **Coherence**: Final combination step ensures unified psychological assessment

### Prompt Engineering

The intent analyzer uses prompts that:

- **Adapt to depth**: Different psychological analysis levels for surface vs comprehensive
- **Scale with context**: Appropriate guidance for standalone vs conversational analysis
- **Handle focus**: Specific attention to user-specified psychological focus areas
- **Validate quality**: Self-assessment of confidence and psychological accuracy

### Quality Assurance

- **Pydantic validation**: Ensures structured output conforms to psychological schema
- **Confidence scoring**: LLM self-assesses intent analysis accuracy
- **Multi-layered analysis**: Primary and secondary intents with complexity scoring
- **Contextual awareness**: Factors affecting psychological interpretation

## Timeout Configuration

The intent analyzer supports flexible timeout configuration for different analysis scenarios:

### Default Behavior (Unlimited Timeout)
```bash
# Runs as long as needed - recommended for complex psychological analysis
python -m abstractcore.apps intent document.txt
```

### Custom Timeout
```bash
# Set specific timeout (useful for production environments)
python -m abstractcore.apps intent document.txt --timeout 300   # 5 minutes
python -m abstractcore.apps intent document.txt --timeout 600   # 10 minutes

# Explicit unlimited timeout
python -m abstractcore.apps intent document.txt --timeout none
```

### Programmatic Usage
```python
from abstractcore.processing import BasicIntentAnalyzer

# Unlimited timeout (default)
analyzer = BasicIntentAnalyzer()

# Custom timeout
analyzer = BasicIntentAnalyzer(timeout=300)  # 5 minutes

# Explicit unlimited timeout
analyzer = BasicIntentAnalyzer(timeout=None)
```

**When to Use Timeouts:**
- **Production environments**: Set reasonable timeouts (300-600 seconds) for psychological analysis
- **Complex analysis**: Use unlimited timeout for comprehensive depth analysis
- **Batch processing**: Consider timeouts to handle individual analysis failures gracefully
- **Development**: Use unlimited timeout to avoid interruptions during testing

## Integration Examples

### With AbstractCore Session

```python
from abstractcore import BasicSession

session = BasicSession(llm, system_prompt="You are a psychological analyst")
analyzer = BasicIntentAnalyzer(session)

# Maintains conversation context for psychological continuity
result1 = analyzer.analyze_intent(message1, focus="emotional patterns")
result2 = analyzer.analyze_intent(message2, focus="how emotional state has evolved")
```

### Batch Processing

```python
conversations = [conv1, conv2, conv3, conv4]
analyses = []

for conv in conversations:
    result = analyzer.analyze_intent(
        conv.content,
        focus="customer satisfaction patterns",
        context_type=IntentContext.CONVERSATIONAL,
        depth=IntentDepth.UNDERLYING
    )
    analyses.append({
        'conversation_id': conv.id,
        'primary_intent': result.primary_intent.intent_type.value,
        'underlying_goal': result.primary_intent.underlying_goal,
        'emotional_state': result.primary_intent.emotional_undertone,
        'confidence': result.overall_confidence,
        'urgency': result.primary_intent.urgency_level
    })

# Filter high-confidence analyses
high_quality = [a for a in analyses if a['confidence'] > 0.8]
```

## Extending the Intent Analyzer

The Basic Intent Analyzer serves as a foundation for more advanced psychological analysis:

```python
class CustomIntentAnalyzer(BasicIntentAnalyzer):
    def analyze_emotional_journey(self, messages: List[str]) -> List[IntentAnalysisOutput]:
        """Track emotional evolution through conversation"""
        results = []
        for i, message in enumerate(messages):
            focus = f"emotional state evolution, message {i+1} of {len(messages)}"
            result = self.analyze_intent(
                message, 
                focus=focus, 
                depth=IntentDepth.COMPREHENSIVE
            )
            results.append(result)
        return results

    def analyze_team_dynamics(self, team_messages: Dict[str, List[str]]) -> Dict[str, IntentAnalysisOutput]:
        """Analyze intent patterns across team members"""
        team_analysis = {}
        for member, messages in team_messages.items():
            combined_text = "\n".join(messages)
            focus = f"team collaboration patterns and {member}'s role dynamics"
            team_analysis[member] = self.analyze_intent(
                combined_text,
                focus=focus,
                context_type=IntentContext.CONVERSATIONAL
            )
        return team_analysis
```

## Best Practices

1. **Choose appropriate depth**: Match analysis depth to use case (surface for quick insights, comprehensive for deep psychology)
2. **Use focus effectively**: Specific psychological focus areas improve relevance and accuracy
3. **Monitor confidence**: Low confidence may indicate need for human psychological review
4. **Provider selection**: Match provider capabilities to psychological analysis complexity
5. **Context awareness**: Use appropriate context types for accurate psychological interpretation
6. **Batch processing**: Process similar communications together for psychological consistency
7. **Error handling**: Always handle potential failures gracefully in psychological analysis workflows

## Use Cases

### Customer Support Optimization
- Analyze customer frustration levels and underlying needs
- Identify validation-seeking vs problem-solving intents
- Optimize response strategies based on emotional undertones

### Team Communication Analysis
- Understand team member motivations and concerns
- Identify relationship-building vs task-focused communications
- Analyze leadership and collaboration patterns

### Educational Content Personalization
- Identify learning motivations and barriers
- Understand student confidence levels and validation needs
- Personalize teaching approaches based on psychological profiles

### Business Communication Intelligence
- Analyze stakeholder motivations in negotiations
- Understand client relationship dynamics
- Identify persuasion vs information-sharing intents

### Mental Health and Wellness
- Track emotional patterns in therapeutic communications
- Identify support-seeking behaviors and underlying needs
- Monitor psychological well-being indicators

## Conclusion

The Basic Intent Analyzer demonstrates how AbstractCore's infrastructure enables building psychological analysis capabilities with minimal complexity. It showcases:

- **Deep psychological insights** with comprehensive intent understanding
- **Clean API design** with powerful psychological customization options
- **Automatic reliability** through built-in retry and error handling
- **Universal compatibility** across all LLM providers
- **Scalable architecture** handling communications of any length
- **Production readiness** with comprehensive error handling and monitoring

This implementation serves both as a powerful tool for understanding human communication and as a reference for building other psychological analysis capabilities on top of AbstractCore. The intent analyzer goes far beyond simple sentiment analysis to provide actionable insights into human motivations, goals, and psychological patterns.
