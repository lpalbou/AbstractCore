#!/usr/bin/env python3
"""
AbstractCore Async CLI - Demonstrates async generation with real-time tool feedback.

This CLI uses async/await for:
- Non-blocking LLM generation
- Parallel tool execution (independent tools)
- Real-time progress indicators via EventBus
- Streaming responses with async iterators

Usage:
    python -m abstractcore.utils.cli_async --provider lmstudio --model qwen/qwen3-vl-30b
    python -m abstractcore.utils.cli_async --provider ollama --model qwen3:4b --stream
"""

import asyncio
import sys
import argparse
from typing import List, Dict, Any, Tuple
from datetime import datetime

from .. import create_llm, BasicSession
from ..events import EventType, GlobalEventBus
from ..tools.common_tools import list_files, read_file, write_file, execute_command, search_files


class AsyncCLI:
    """Async CLI with real-time tool execution feedback."""

    def __init__(self, provider: str, model: str, stream: bool = False, debug: bool = False, **kwargs):
        self.provider_name = provider
        self.model_name = model
        self.stream_mode = stream
        self.debug_mode = debug

        # Initialize provider and session
        self.provider = create_llm(provider, model=model, **kwargs)
        self.session = BasicSession(
            self.provider,
            system_prompt="You are a helpful AI assistant with vision capabilities. When users provide images or media files, analyze and describe them directly. You also have access to file operation tools.",
            tools=[list_files, read_file, write_file, execute_command, search_files]
        )

        # Progress tracking
        self._active_tools: Dict[str, Dict] = {}
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0

        # Register event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Register async event handlers for real-time feedback."""
        GlobalEventBus.on_async(EventType.TOOL_STARTED, self._on_tool_started)
        GlobalEventBus.on_async(EventType.TOOL_PROGRESS, self._on_tool_progress)
        GlobalEventBus.on_async(EventType.TOOL_COMPLETED, self._on_tool_completed)

    def _format_tool_display(self, tool_name: str, arguments: dict) -> str:
        """Format tool name with parameters for display."""
        if not arguments:
            return tool_name

        # Inline for simple cases (0-2 params)
        if len(arguments) <= 2:
            params = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{tool_name}({params})"

        # Just name for complex cases (params shown separately)
        return tool_name

    def _format_tool_params(self, arguments: dict) -> str:
        """Format parameters as separate lines for complex cases."""
        if not arguments or len(arguments) <= 2:
            return ""  # Already inline

        lines = []
        for key, value in arguments.items():
            # Truncate long values
            value_str = repr(value)
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            lines.append(f"   │ {key}: {value_str}")
        return "\n".join(lines)

    async def _on_tool_started(self, event):
        """Handle tool start - show spinner and start animation."""
        tool_name = event.data.get('tool_name', 'unknown')
        arguments = event.data.get('arguments', {})

        self._active_tools[tool_name] = {
            'start': datetime.now(),
            'status': 'starting...',
            'completed': False
        }

        # Format display with smart parameter handling
        display_name = self._format_tool_display(tool_name, arguments)
        param_lines = self._format_tool_params(arguments)

        # Print starting message with parameters on its own line
        print(f"\n⏳ {display_name}: executing...")
        if param_lines:
            print(param_lines)

        # Start spinner animation task
        asyncio.create_task(self._animate_spinner(tool_name))

    async def _animate_spinner(self, tool_name: str):
        """Animate spinner until tool completes (on separate line from params)."""
        # Small delay before starting spinner
        await asyncio.sleep(0.05)

        while tool_name in self._active_tools and not self._active_tools[tool_name].get('completed', True):
            self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
            spinner = self._spinner_chars[self._spinner_idx]
            elapsed = (datetime.now() - self._active_tools[tool_name]['start']).total_seconds()
            # Spinner on its own line, updates in place
            print(f"\r   {spinner} Working... ({elapsed:.1f}s)", end="", flush=True)
            await asyncio.sleep(0.1)  # Update every 100ms

    async def _on_tool_progress(self, event):
        """Handle tool progress - update status message."""
        tool_name = event.data.get('tool_name', 'unknown')
        progress = event.data.get('progress', '')
        if tool_name in self._active_tools:
            self._active_tools[tool_name]['status'] = progress

    async def _on_tool_completed(self, event):
        """Handle tool completion - stop spinner and show result."""
        tool_name = event.data.get('tool_name', 'unknown')
        success = event.data.get('success', True)
        duration = event.data.get('duration_ms', 0)
        result = event.data.get('result', '')

        # Mark as completed to stop spinner
        if tool_name in self._active_tools:
            self._active_tools[tool_name]['completed'] = True

        # Small delay to let spinner stop
        await asyncio.sleep(0.05)

        icon = "✅" if success else "❌"

        # Print on new line to preserve parameter display
        status_line = f"{icon} {tool_name}: completed in {duration:.0f}ms"

        # In debug mode, show result preview
        if self.debug_mode and result:
            result_preview = str(result)[:100]
            if len(str(result)) > 100:
                result_preview += "..."
            print(f"\n{status_line}\n   └─ Result: {result_preview}")
        else:
            print(f"\n{status_line}")

    def _parse_file_attachments(self, user_input: str) -> Tuple[str, List[str]]:
        """Parse @filename references using AbstractCore's message preprocessor."""
        from ..utils.message_preprocessor import MessagePreprocessor
        import os

        # Use AbstractCore's centralized file parsing logic
        clean_input, media_files = MessagePreprocessor.parse_file_attachments(
            user_input,
            validate_existence=True,
            verbose=self.debug_mode
        )

        # Show user-friendly status messages for CLI
        if media_files:
            print(f"📎 Attaching {len(media_files)} file(s): {', '.join(media_files)}")

            # Check for vision capabilities if images are attached
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            image_files = [f for f in media_files if os.path.splitext(f.lower())[1] in image_extensions]

            if image_files:
                try:
                    from ..media.capabilities import is_vision_model
                    if is_vision_model(self.model_name):
                        print(f"👁️  Vision model detected - will analyze {len(image_files)} image(s)")
                    else:
                        print(f"📷 Text model - will use vision fallback for {len(image_files)} image(s)")
                except:
                    print(f"📷 Processing {len(image_files)} image(s)")

        return clean_input, media_files

    def _parse_tool_calls(self, content: str) -> Tuple[str, List[Dict]]:
        """
        Parse tool calls from content and return (clean_content, tool_calls).

        Returns:
            Tuple of (content_without_tool_calls, list_of_tool_call_dicts)
        """
        import re
        import json

        if not content:
            return content, []

        # Use the universal parser from tools.parser for better compatibility
        try:
            from ..tools.parser import _parse_any_format
            detected_calls = _parse_any_format(content)

            if not detected_calls:
                return content, []

            # Convert to simple dicts for execution
            tool_calls = []
            for call in detected_calls:
                tool_calls.append({
                    'name': call.name,
                    'arguments': call.arguments if isinstance(call.arguments, dict) else {}
                })

            # Strip tool call syntax from content using syntax rewriter
            from ..tools.syntax_rewriter import ToolCallSyntaxRewriter, SyntaxFormat
            rewriter = ToolCallSyntaxRewriter(SyntaxFormat.PASSTHROUGH, model_name=self.model_name)
            clean_content = rewriter.remove_tool_call_patterns(content)

            return clean_content, tool_calls

        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Tool parsing fallback to regex: {e}")

            # Fallback to regex parsing
            tool_calls = []
            clean_content = content

            patterns = [
                r'<\|tool_call\|>(.*?)</\|tool_call\|>',  # Qwen3
                r'<function_call>(.*?)</function_call>',   # LLaMA3
                r'<tool_call>(.*?)</tool_call>',           # XML
                r'```tool_code\s*\n(.*?)\n```',            # Gemma
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        tool_data = json.loads(match.strip())
                        if 'name' in tool_data:
                            tool_calls.append({
                                'name': tool_data['name'],
                                'arguments': tool_data.get('arguments', {})
                            })
                            clean_content = re.sub(pattern, '', clean_content, count=1, flags=re.DOTALL)
                    except json.JSONDecodeError:
                        continue

            clean_content = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_content).strip()

            return clean_content, tool_calls

    async def _execute_tools_async(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute tools with async + events.

        Independent tools run in parallel via asyncio.gather().
        All tools complete before returning (ReAct-safe).
        """
        if not tool_calls:
            return []

        async def execute_single_tool(tool_data: Dict) -> Dict:
            """Execute one tool with event emissions."""
            tool_name = tool_data.get('name')
            tool_args = tool_data.get('arguments', {})

            # Emit TOOL_STARTED
            await GlobalEventBus.emit_async(EventType.TOOL_STARTED, {
                'tool_name': tool_name,
                'arguments': tool_args
            })

            start_time = datetime.now()

            try:
                # Get tool function
                tool_map = {
                    'list_files': list_files,
                    'search_files': search_files,
                    'read_file': read_file,
                    'write_file': write_file,
                    'execute_command': execute_command
                }

                tool_fn = tool_map.get(tool_name)
                if not tool_fn:
                    raise ValueError(f"Unknown tool: {tool_name}")

                # Execute in thread pool (non-blocking)
                result = await asyncio.to_thread(tool_fn, **tool_args)

                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Emit TOOL_COMPLETED
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

                # Add error to session history
                self.session.add_message('tool', f"Error: {str(e)}",
                                       tool_name=tool_name,
                                       tool_arguments=tool_args,
                                       status="error",
                                       duration_ms=duration_ms,
                                       stderr=str(e))

                return {'tool_name': tool_name, 'error': str(e), 'success': False}

        # Execute ALL tools in parallel (independent)
        # asyncio.gather waits for all to complete - ReAct safe!
        results = await asyncio.gather(*[
            execute_single_tool(tc) for tc in tool_calls
        ], return_exceptions=True)

        return results

    async def generate_response_async(self, user_input: str):
        """Generate response with async tools and streaming."""
        # Parse file attachments (sync, fast)
        clean_input, media_files = self._parse_file_attachments(user_input)

        if self.stream_mode:
            full_content = ""
            has_visible_content = False

            # Await to get async generator, then iterate
            stream_gen = await self.session.agenerate(
                clean_input,
                stream=True,
                media=media_files if media_files else None
            )

            collected_tool_calls = []
            seen_tool_calls = set()  # For deduplication

            async for chunk in stream_gen:
                # Collect text content
                if hasattr(chunk, 'content') and chunk.content:
                    if not has_visible_content:
                        print("🤖 Assistant: ", end="", flush=True)
                        has_visible_content = True
                    print(chunk.content, end="", flush=True)
                    full_content += chunk.content

                # Collect tool calls from chunks (with deduplication)
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        # Create unique key for deduplication
                        tool_key = (tool_call.name, frozenset(
                            (k, str(v)) for k, v in (tool_call.arguments or {}).items()
                        ))

                        # Only add if not seen before
                        if tool_key not in seen_tool_calls:
                            seen_tool_calls.add(tool_key)
                            collected_tool_calls.append({
                                'name': tool_call.name,
                                'arguments': tool_call.arguments or {}
                            })

            if has_visible_content:
                print()  # Newline after streaming

            # Limit tool calls to prevent LLM hallucination issues
            MAX_TOOLS_PER_RESPONSE = 10
            if len(collected_tool_calls) > MAX_TOOLS_PER_RESPONSE:
                print(f"\n⚠️  Warning: LLM requested {len(collected_tool_calls)} tool calls, limiting to {MAX_TOOLS_PER_RESPONSE}")
                collected_tool_calls = collected_tool_calls[:MAX_TOOLS_PER_RESPONSE]

            if collected_tool_calls:
                await self._execute_tools_async(collected_tool_calls)

                # ReAct loop: Let LLM process tool results and respond
                print("\n🤖 Assistant: ", end="", flush=True)
                full_content = ""

                stream_gen = await self.session.agenerate(
                    "",  # Empty prompt - LLM sees tool results in history
                    stream=True
                )

                async for chunk in stream_gen:
                    if hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_content += chunk.content

                print()
        else:
            # Non-streaming
            response = await self.session.agenerate(
                clean_input,
                media=media_files if media_files else None
            )

            clean_content, tool_calls = self._parse_tool_calls(response.content)

            if clean_content.strip():
                print(f"🤖 Assistant: {clean_content}")

            if tool_calls:
                await self._execute_tools_async(tool_calls)

                # ReAct loop: Let LLM process tool results and respond
                response = await self.session.agenerate("")
                if response.content.strip():
                    print(f"🤖 Assistant: {response.content}")

    async def run_interactive_async(self):
        """Async interactive REPL loop."""
        print("=" * 70)
        print("🚀 AbstractCore Async CLI - Real-time Tool Feedback".center(70))
        print("=" * 70)
        print(f"🤖 Provider: {self.provider_name} | Model: {self.model_name}")
        print(f"🌊 Streaming: {'ON' if self.stream_mode else 'OFF'}")
        print("💡 Commands: /quit, /help, /stream")
        print("=" * 70)

        while True:
            try:
                # Use asyncio-compatible input
                user_input = await asyncio.to_thread(input, "\n👤 You: ")
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.startswith('/'):
                    if await self._handle_command_async(user_input):
                        continue

                await self.generate_response_async(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Use /quit to exit.")
            except EOFError:
                print("\n👋 Goodbye!")
                break

    async def _handle_command_async(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = cmd[1:].lower()

        if cmd in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            sys.exit(0)
        elif cmd == 'stream':
            self.stream_mode = not self.stream_mode
            print(f"🌊 Streaming: {'ON' if self.stream_mode else 'OFF'}")
        elif cmd == 'help':
            print("\n📖 Commands:")
            print("  /quit, /exit, /q    Exit the CLI")
            print("  /stream             Toggle streaming on/off")
            print("  /help               Show this help")
            print("\n💡 Features:")
            print("  • Real-time tool execution feedback with spinners")
            print("  • Parallel execution of independent tools")
            print("  • Async streaming for faster responses")
            print("  • File attachments with @filename syntax")
        else:
            print(f"❓ Unknown command: /{cmd}")
            print("💡 Type /help for available commands")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Async CLI with real-time tool feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m abstractcore.utils.cli_async --provider lmstudio --model qwen/qwen3-vl-30b
  python -m abstractcore.utils.cli_async --provider ollama --model qwen3:4b --stream
  python -m abstractcore.utils.cli_async --provider openai --model gpt-4o-mini

Features:
  • Real-time progress indicators during tool execution
  • Parallel execution of independent tools
  • Async streaming for non-blocking responses
  • Full file attachment support (@filename)
        """
    )
    parser.add_argument('--provider', default='lmstudio', help='LLM provider')
    parser.add_argument('--model', default='qwen/qwen3-next-80b', help='Model name')
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--base-url', help='Base URL for provider')
    parser.add_argument('--api-key', help='API key for provider')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')

    args = parser.parse_args()

    kwargs = {'temperature': args.temperature}
    if args.base_url:
        kwargs['base_url'] = args.base_url
    if args.api_key:
        kwargs['api_key'] = args.api_key

    cli = AsyncCLI(
        provider=args.provider,
        model=args.model,
        stream=args.stream,
        debug=args.debug,
        **kwargs
    )
    asyncio.run(cli.run_interactive_async())


if __name__ == "__main__":
    main()
