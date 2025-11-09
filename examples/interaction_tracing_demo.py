"""
Demonstration of Interaction Tracing for LLM Observability.

This example shows how to use AbstractCore's interaction tracing to capture
and analyze LLM interactions for debugging and observability.
"""

from abstractcore import create_llm
from abstractcore.core.session import BasicSession
from abstractcore.utils import export_traces, summarize_traces


def demo_provider_tracing():
    """Demonstrate provider-level tracing."""
    print("=" * 80)
    print("DEMO 1: Provider-Level Tracing")
    print("=" * 80)

    # Create provider with tracing enabled
    llm = create_llm(
        'ollama',
        model='qwen3:4b-instruct-2507-q4_K_M',
        enable_tracing=True,
        max_traces=10
    )

    # Generate with custom metadata
    response = llm.generate(
        "Say 'Hello World' exactly",
        temperature=0,
        trace_metadata={
            'step': 'greeting',
            'attempt': 1,
            'user_id': 'demo_user'
        }
    )

    print(f"\nResponse: {response.content}")
    print(f"Trace ID: {response.metadata['trace_id']}")

    # Retrieve and display trace
    trace_id = response.metadata['trace_id']
    trace = llm.get_traces(trace_id=trace_id)

    print(f"\n--- Trace Details ---")
    print(f"Timestamp: {trace['timestamp']}")
    print(f"Provider: {trace['provider']}")
    print(f"Model: {trace['model']}")
    print(f"Prompt: {trace['prompt']}")
    print(f"Parameters: {trace['parameters']}")
    print(f"Usage: {trace['response']['usage']}")
    print(f"Generation Time: {trace['response']['generation_time_ms']:.2f}ms")
    print(f"Custom Metadata: {trace['metadata']}")


def demo_session_tracing():
    """Demonstrate session-level tracing."""
    print("\n" + "=" * 80)
    print("DEMO 2: Session-Level Tracing")
    print("=" * 80)

    # Create provider and session with tracing
    llm = create_llm(
        'ollama',
        model='qwen3:4b-instruct-2507-q4_K_M',
        enable_tracing=True
    )
    session = BasicSession(provider=llm, enable_tracing=True)

    # Simulate multi-step conversation
    print("\nStep 1: Ask a question")
    response1 = session.generate("What is 2+2?", temperature=0)
    print(f"Response: {response1.content}")

    print("\nStep 2: Follow-up question")
    response2 = session.generate("What about 3+3?", temperature=0)
    print(f"Response: {response2.content}")

    # Get all interaction traces
    traces = session.get_interaction_history()
    print(f"\n--- Session Summary ---")
    print(f"Session ID: {session.id}")
    print(f"Total Interactions: {len(traces)}")

    for i, trace in enumerate(traces, 1):
        print(f"\nInteraction {i}:")
        print(f"  Prompt: {trace['prompt']}")
        print(f"  Tokens: {trace['response']['usage']['total_tokens']}")
        print(f"  Time: {trace['response']['generation_time_ms']:.2f}ms")


def demo_multi_step_workflow():
    """Demonstrate tracing for multi-step code generation with retries."""
    print("\n" + "=" * 80)
    print("DEMO 3: Multi-Step Workflow (Code Generation with Retries)")
    print("=" * 80)

    llm = create_llm(
        'ollama',
        model='qwen3:4b-instruct-2507-q4_K_M',
        enable_tracing=True
    )
    session = BasicSession(provider=llm, enable_tracing=True)

    # Step 1: Generate code
    print("\nStep 1: Generate Python function")
    response = session.generate(
        "Write a Python function called 'add' that adds two numbers",
        system_prompt="You are a Python code generator. Only output Python code.",
        step_type='code_generation',
        attempt_number=1,
        temperature=0
    )
    print(f"Generated code:\n{response.content}")

    # Step 2: Simulate retry (if needed)
    print("\nStep 2: Generate methodology text")
    response = session.generate(
        "Write a brief scientific methodology description for an addition function",
        step_type='methodology_generation',
        temperature=0
    )
    print(f"Methodology:\n{response.content}")

    # Analyze workflow
    traces = session.get_interaction_history()
    print(f"\n--- Workflow Analysis ---")
    print(f"Total steps: {len(traces)}")

    for i, trace in enumerate(traces, 1):
        print(f"\nStep {i}:")
        print(f"  Type: {trace['metadata']['step_type']}")
        print(f"  Tokens: {trace['response']['usage']['total_tokens']}")
        print(f"  Time: {trace['response']['generation_time_ms']:.2f}ms")

    # Get summary statistics
    summary = summarize_traces(traces)
    print(f"\n--- Summary Statistics ---")
    print(f"Total interactions: {summary['total_interactions']}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Average tokens: {summary['avg_tokens_per_interaction']:.0f}")
    print(f"Total time: {summary['total_time_ms']:.2f}ms")
    print(f"Average time: {summary['avg_time_ms']:.2f}ms")


def demo_export():
    """Demonstrate trace export."""
    print("\n" + "=" * 80)
    print("DEMO 4: Trace Export")
    print("=" * 80)

    llm = create_llm(
        'ollama',
        model='qwen3:4b-instruct-2507-q4_K_M',
        enable_tracing=True
    )

    # Generate a few interactions
    for i in range(3):
        llm.generate(f"Count to {i+1}", temperature=0)

    traces = llm.get_traces()

    # Export to different formats
    print("\nExporting traces...")

    # JSONL
    jsonl_content = export_traces(traces, format='jsonl')
    print(f"\nJSONL export: {len(jsonl_content.split(chr(10)))} lines")

    # JSON
    json_content = export_traces(traces, format='json')
    print(f"JSON export: {len(json_content)} characters")

    # Markdown
    md_content = export_traces(traces, format='markdown')
    print(f"Markdown export: {len(md_content)} characters")
    print(f"\nFirst 200 characters of markdown report:")
    print(md_content[:200] + "...")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("AbstractCore Interaction Tracing Demo")
    print("=" * 80)

    try:
        # Run all demos
        demo_provider_tracing()
        demo_session_tracing()
        demo_multi_step_workflow()
        demo_export()

        print("\n" + "=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        print("\nFor more information, see: docs/interaction-tracing.md")

    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Make sure Ollama is running with qwen3:4b-instruct-2507-q4_K_M model")
