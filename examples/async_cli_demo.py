#!/usr/bin/env python3
"""
AsyncCLI Demo - Educational Reference Implementation

⚠️  THIS IS A DEMONSTRATION - NOT FOR PRODUCTION USE ⚠️

For production CLI, use: python -m abstractcore.utils.cli

This demo illustrates 5 core async/await patterns in AbstractCore:
1. Async/await patterns with LLM providers
2. Real-time progress events via GlobalEventBus
3. Parallel tool execution with asyncio.gather()
4. Async streaming with async for loops
5. Animated progress indicators (spinners)

Purpose: Educational reference for developers building async applications
Maintenance: Minimal - only update if async patterns change significantly

Usage (demo only):
    python examples/async_cli_demo.py --provider ollama --model qwen3:4b
    python examples/async_cli_demo.py --provider lmstudio --model qwen/qwen3-vl-30b --stream

Learn more about async in AbstractCore:
    - docs/acore-cli.md (CLI documentation)
    - docs/session.md (Session management)
"""

import asyncio
import sys
import argparse
from typing import List, Dict
from datetime import datetime

from abstractcore import create_llm, BasicSession
from abstractcore.events import EventType, GlobalEventBus
from abstractcore.tools.common_tools import list_files, read_file, search_files


class AsyncCLIDemo:
    """
    Minimal async CLI demonstrating AbstractCore's async capabilities.

    This is a reference implementation showing:
    - How to use agenerate() with async/await
    - How to handle async streaming correctly
    - How to execute tools in parallel
    - How to emit/handle progress events

    For production use: python -m abstractcore.utils.cli
    """

    def __init__(self, provider: str, model: str, stream: bool = False):
        self.provider_name = provider
        self.model_name = model
        self.stream_mode = stream

        # Initialize provider and session
        self.provider = create_llm(provider, model=model)
        self.session = BasicSession(
            self.provider,
            system_prompt="You are a helpful AI assistant with tool access.",
            tools=[list_files, read_file, search_files]
        )

        # Spinner animation state
        self._active_tools: Dict[str, Dict] = {}
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0

        # PATTERN 1: Register async event handlers for real-time feedback
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """
        PATTERN 1: Event-driven architecture with GlobalEventBus.

        Register async callbacks that execute when tools start/complete.
        This enables real-time progress updates without blocking.
        """
        GlobalEventBus.on_async(EventType.TOOL_STARTED, self._on_tool_started)
        GlobalEventBus.on_async(EventType.TOOL_COMPLETED, self._on_tool_completed)

    async def _on_tool_started(self, event):
        """
        PATTERN 2: Async event handler with spinner animation.

        When a tool starts, we:
        1. Track start time
        2. Display tool name with arguments
        3. Launch async spinner animation task
        """
        tool_name = event.data.get('tool_name', 'unknown')
        arguments = event.data.get('arguments', {})

        self._active_tools[tool_name] = {
            'start': datetime.now(),
            'completed': False
        }

        # Simple inline display for 0-2 params
        if len(arguments) <= 2:
            params = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
            display = f"{tool_name}({params})" if params else tool_name
        else:
            display = tool_name

        print(f"\n⏳ {display}: executing...")

        # PATTERN 3: Non-blocking animation with asyncio.create_task()
        asyncio.create_task(self._animate_spinner(tool_name))

    async def _animate_spinner(self, tool_name: str):
        """
        PATTERN 3: Async animation loop.

        Uses asyncio.sleep() for non-blocking delays.
        Runs until tool completes (checked via shared state).
        """
        await asyncio.sleep(0.05)  # Small delay before starting

        while tool_name in self._active_tools and not self._active_tools[tool_name].get('completed', True):
            self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
            spinner = self._spinner_chars[self._spinner_idx]
            elapsed = (datetime.now() - self._active_tools[tool_name]['start']).total_seconds()

            # Update spinner in place with \r
            print(f"\r   {spinner} Working... ({elapsed:.1f}s)", end="", flush=True)

            # PATTERN 4: Non-blocking sleep - event loop continues during this
            await asyncio.sleep(0.1)

    async def _on_tool_completed(self, event):
        """
        PATTERN 2: Async event handler for tool completion.

        Stop spinner and show final status.
        """
        tool_name = event.data.get('tool_name', 'unknown')
        success = event.data.get('success', True)
        duration = event.data.get('duration_ms', 0)

        # Mark as completed to stop spinner
        if tool_name in self._active_tools:
            self._active_tools[tool_name]['completed'] = True

        await asyncio.sleep(0.05)  # Let spinner stop

        icon = "✅" if success else "❌"
        print(f"\n{icon} {tool_name}: completed in {duration:.0f}ms")

    async def _execute_tools_parallel(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        PATTERN 5: Parallel execution with asyncio.gather().

        Key insight: Independent tools can run concurrently!
        asyncio.gather() waits for ALL to complete before returning.
        """
        if not tool_calls:
            return []

        async def execute_single_tool(tool_data: Dict) -> Dict:
            """Execute one tool with event emissions."""
            tool_name = tool_data.get('name')
            tool_args = tool_data.get('arguments', {})

            # Emit TOOL_STARTED event (triggers spinner)
            await GlobalEventBus.emit_async(EventType.TOOL_STARTED, {
                'tool_name': tool_name,
                'arguments': tool_args
            })

            start_time = datetime.now()

            try:
                # Map tool names to functions
                tool_map = {
                    'list_files': list_files,
                    'search_files': search_files,
                    'read_file': read_file,
                }

                tool_fn = tool_map.get(tool_name)
                if not tool_fn:
                    raise ValueError(f"Unknown tool: {tool_name}")

                # PATTERN 6: Run sync function in thread pool with asyncio.to_thread()
                # This is the SOTA pattern for sync tools in async context
                result = await asyncio.to_thread(tool_fn, **tool_args)

                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Emit TOOL_COMPLETED event (stops spinner)
                await GlobalEventBus.emit_async(EventType.TOOL_COMPLETED, {
                    'tool_name': tool_name,
                    'result': str(result)[:500],
                    'success': True,
                    'duration_ms': duration_ms
                })

                # Add to session history
                self.session.add_message('tool', str(result),
                                       tool_name=tool_name,
                                       tool_arguments=tool_args,
                                       status="ok",
                                       duration_ms=duration_ms)

                return {'tool_name': tool_name, 'result': result, 'success': True}

            except Exception as e:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                await GlobalEventBus.emit_async(EventType.TOOL_COMPLETED, {
                    'tool_name': tool_name,
                    'error': str(e),
                    'success': False,
                    'duration_ms': duration_ms
                })

                return {'tool_name': tool_name, 'error': str(e), 'success': False}

        # PATTERN 5: Execute ALL tools concurrently with asyncio.gather()
        # This is the key pattern for parallel execution!
        results = await asyncio.gather(*[
            execute_single_tool(tc) for tc in tool_calls
        ], return_exceptions=True)

        return results

    async def generate_with_streaming(self, user_input: str):
        """
        PATTERN 7: Async streaming with proper await pattern.

        Critical: Must await agenerate() FIRST to get async generator,
        THEN use 'async for' to iterate over chunks.

        Common mistake: async for chunk in self.session.agenerate(...)
        Correct pattern: stream_gen = await self.session.agenerate(...)
                        async for chunk in stream_gen:
        """
        print("🤖 Assistant: ", end="", flush=True)

        # PATTERN 7a: Await the coroutine to get async generator
        stream_gen = await self.session.agenerate(user_input, stream=True)

        collected_tool_calls = []
        full_content = ""

        # PATTERN 7b: Async iterate over streaming chunks
        async for chunk in stream_gen:
            # Stream text content to console
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                full_content += chunk.content

            # Collect tool calls from chunks
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    collected_tool_calls.append({
                        'name': tool_call.name,
                        'arguments': tool_call.arguments or {}
                    })

        print()  # Newline after streaming

        # Execute tools if any were requested
        if collected_tool_calls:
            await self._execute_tools_parallel(collected_tool_calls)

            # Let LLM process tool results and respond
            print("\n🤖 Assistant: ", end="", flush=True)
            stream_gen = await self.session.agenerate("", stream=True)

            async for chunk in stream_gen:
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)

            print()

    async def generate_without_streaming(self, user_input: str):
        """Non-streaming generation (simpler pattern)."""
        response = await self.session.agenerate(user_input)
        print(f"🤖 Assistant: {response.content}")

    async def run_demo(self):
        """
        Main async REPL loop demonstrating all patterns together.

        This shows how to build an async interactive CLI with:
        - Non-blocking user input (asyncio.to_thread)
        - Async generation with streaming
        - Parallel tool execution
        - Real-time progress feedback
        """
        print("=" * 70)
        print("🚀 AsyncCLI Demo - Educational Reference".center(70))
        print("=" * 70)
        print("⚠️  THIS IS A DEMO - For production use: abstractcore.utils.cli")
        print("=" * 70)
        print(f"🤖 Provider: {self.provider_name} | Model: {self.model_name}")
        print(f"🌊 Streaming: {'ON' if self.stream_mode else 'OFF'}")
        print("💡 Commands: /quit, /stream, /help")
        print("=" * 70)

        while True:
            try:
                # PATTERN 8: Non-blocking input with asyncio.to_thread()
                user_input = await asyncio.to_thread(input, "\n👤 You: ")
                user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith('/'):
                    if await self._handle_command(user_input):
                        continue

                # Generate response with appropriate mode
                if self.stream_mode:
                    await self.generate_with_streaming(user_input)
                else:
                    await self.generate_without_streaming(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Use /quit to exit.")
            except EOFError:
                print("\n👋 Goodbye!")
                break

    async def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = cmd[1:].lower()

        if cmd in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            sys.exit(0)
        elif cmd == 'stream':
            self.stream_mode = not self.stream_mode
            print(f"🌊 Streaming: {'ON' if self.stream_mode else 'OFF'}")
        elif cmd == 'help':
            print("\n📖 Async Patterns Demonstrated:")
            print("  1. Event-driven progress (GlobalEventBus)")
            print("  2. Async event handlers (on_async)")
            print("  3. Non-blocking animations (create_task)")
            print("  4. Async sleep for cooperative multitasking")
            print("  5. Parallel execution (asyncio.gather)")
            print("  6. Sync tools in async context (asyncio.to_thread)")
            print("  7. Async streaming (await + async for)")
            print("  8. Non-blocking input (asyncio.to_thread)")
            print("\n💡 Commands:")
            print("  /quit       Exit the demo")
            print("  /stream     Toggle streaming on/off")
            print("  /help       Show this help")
        else:
            print(f"❓ Unknown command: /{cmd}")

        return True


def main():
    """Entry point for async CLI demo."""
    parser = argparse.ArgumentParser(
        description="Async CLI Demo - Educational Reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  DEMO ONLY - For production CLI, use: python -m abstractcore.utils.cli

This demo illustrates async/await patterns in AbstractCore:
  • Real-time progress indicators during tool execution
  • Parallel execution of independent tools (asyncio.gather)
  • Async streaming with proper await pattern
  • Event-driven architecture (GlobalEventBus)
  • Non-blocking animations and user input

Examples:
  python examples/async_cli_demo.py --provider ollama --model qwen3:4b
  python examples/async_cli_demo.py --provider lmstudio --model qwen/qwen3-vl-30b --stream

Learn more:
  docs/acore-cli.md - CLI documentation
  docs/session.md - Session management
        """
    )
    parser.add_argument('--provider', default='ollama', help='LLM provider')
    parser.add_argument('--model', default='qwen3:4b', help='Model name')
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')

    args = parser.parse_args()

    demo = AsyncCLIDemo(
        provider=args.provider,
        model=args.model,
        stream=args.stream
    )

    # Run the async demo
    asyncio.run(demo.run_demo())


if __name__ == "__main__":
    main()
