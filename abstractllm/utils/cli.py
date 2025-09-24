#!/usr/bin/env python3
"""
AbstractCore CLI - Basic demonstrator for AbstractLLM capabilities.

This is a simple CLI tool that demonstrates basic AbstractCore functionality.
It provides chat, file operations, and command execution but has limitations:
- Simple chat interactions only
- Basic single tool execution
- No ReAct pattern or complex reasoning chains
- No adaptive actions or advanced reasoning patterns
- Limited to basic demonstration purposes

For production use cases requiring advanced reasoning, multi-step tool chains,
or complex agent behaviors, consider building custom solutions using the
AbstractCore framework directly.

Usage:
    python -m abstractllm.utils.cli --provider ollama --model qwen3-coder:30b
    python -m abstractllm.utils.cli --provider openai --model gpt-4o-mini --stream
    python -m abstractllm.utils.cli --provider anthropic --model claude-3-5-haiku-20241022 --prompt "What is Python?"
"""

import argparse
import sys
import time

from .. import create_llm, BasicSession
from ..tools.common_tools import list_files, read_file, execute_command, web_search


class SimpleCLI:
    """Simplified CLI REPL for AbstractLLM"""

    def __init__(self, provider: str, model: str, stream: bool = False,
                 max_tokens: int = 4000, debug: bool = False, **kwargs):
        self.provider_name = provider
        self.model_name = model
        self.stream_mode = stream
        self.debug_mode = debug
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Initialize provider and session with tools
        self.provider = create_llm(provider, model=model, max_tokens=max_tokens, **kwargs)
        self.session = BasicSession(
            self.provider,
            system_prompt="You are a helpful AI assistant.",
            tools=[list_files, read_file, execute_command, web_search]
        )

        print(f"🚀 AbstractLLM CLI - {provider}:{model}")
        print(f"Stream: {'ON' if stream else 'OFF'} | Debug: {'ON' if debug else 'OFF'}")
        print("Commands: /help /quit /clear /stream /debug /history /model <spec>")
        print("Tools: list_files, read_file, execute_command, web_search")
        print("=" * 60)

    def handle_command(self, user_input: str) -> bool:
        """Handle commands. Returns True if command processed, False otherwise."""
        if not user_input.startswith('/'):
            return False

        cmd = user_input[1:].strip()

        if cmd in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            sys.exit(0)

        elif cmd == 'help':
            print("\n📖 Commands:")
            print("  /help - Show this help")
            print("  /quit - Exit")
            print("  /clear - Clear history")
            print("  /stream - Toggle streaming")
            print("  /debug - Toggle debug mode")
            print("  /history - Show conversation history")
            print("  /model <provider:model> - Change model")
            print("\n🛠️ Tools: list_files, read_file, execute_command, web_search, web_search\n")

        elif cmd == 'clear':
            self.session.clear_history(keep_system=True)
            print("🧹 History cleared")

        elif cmd == 'stream':
            self.stream_mode = not self.stream_mode
            print(f"🌊 Stream mode: {'ON' if self.stream_mode else 'OFF'}")

        elif cmd == 'debug':
            self.debug_mode = not self.debug_mode
            print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")

        elif cmd == 'history':
            messages = self.session.get_history(include_system=True)
            if not messages:
                print("📝 No history")
            else:
                print(f"\n📚 History ({len(messages)} messages):")
                for i, msg in enumerate(messages[-10:], 1):  # Show last 10
                    role = {"system": "⚙️", "user": "👤", "assistant": "🤖"}.get(msg["role"], "❓")
                    content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                    print(f"{i:2d}. {role} {content}")
                print()

        elif cmd.startswith('model '):
            try:
                model_spec = cmd[6:]
                if ':' in model_spec:
                    self.provider_name, self.model_name = model_spec.split(':', 1)
                else:
                    self.model_name = model_spec

                print(f"🔄 Switching to {self.provider_name}:{self.model_name}...")
                self.provider = create_llm(self.provider_name, model=self.model_name,
                                         max_tokens=self.max_tokens, **self.kwargs)
                self.session = BasicSession(
                    self.provider,
                    system_prompt="You are a helpful AI assistant.",
                    tools=[list_files, read_file, execute_command, web_search]
                )
                print("✅ Model switched")
            except Exception as e:
                print(f"❌ Failed to switch: {e}")

        else:
            print(f"❓ Unknown command: /{cmd}. Type /help for help.")

        return True

    def generate_response(self, user_input: str):
        """Generate and display response."""
        start_time = time.time()

        try:
            if self.debug_mode:
                print(f"🔍 Sending to {self.provider_name}:{self.model_name}")

            response = self.session.generate(user_input, stream=self.stream_mode)

            if self.stream_mode:
                print("🤖 Assistant: ", end="", flush=True)
                for chunk in response:
                    if hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end="", flush=True)
                print()  # New line
            else:
                print(f"🤖 Assistant: {response.content}")

            if self.debug_mode:
                latency = (time.time() - start_time) * 1000
                print(f"⏱️ Response in {latency:.0f}ms")

        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def run_interactive(self):
        """Run the interactive REPL."""
        try:
            while True:
                try:
                    user_input = input("\n👤 You: ").strip()
                    if not user_input:
                        continue

                    # Handle commands
                    if self.handle_command(user_input):
                        continue

                    # Generate response
                    self.generate_response(user_input)

                except KeyboardInterrupt:
                    print("\n\n👋 Use /quit to exit.")
                    continue
                except EOFError:
                    print("\n👋 Goodbye!")
                    break

        except Exception as e:
            print(f"❌ Fatal error: {e}")

    def run_single_prompt(self, prompt: str):
        """Execute single prompt and exit."""
        try:
            response = self.session.generate(prompt, stream=False)
            print(response.content)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Simplified CLI REPL for AbstractLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m abstractllm.utils.cli --provider ollama --model qwen3-coder:30b
  python -m abstractllm.utils.cli --provider openai --model gpt-4o-mini --stream
  python -m abstractllm.utils.cli --provider anthropic --model claude-3-5-haiku-20241022
  python -m abstractllm.utils.cli --provider ollama --model qwen3-coder:30b --prompt "What is Python?"

Commands:
  /help - Show help
  /quit - Exit
  /clear - Clear history
  /stream - Toggle streaming
  /debug - Toggle debug mode
  /history - Show conversation history
  /model <provider:model> - Change model

Tools: list_files, read_file, execute_command, web_search

Note: This is a basic demonstrator with limited capabilities. For production
use cases requiring advanced reasoning, ReAct patterns, or complex tool chains,
build custom solutions using the AbstractCore framework directly.
        """
    )

    # Required arguments
    parser.add_argument('--provider', required=True,
                       choices=['openai', 'anthropic', 'ollama', 'huggingface', 'mlx', 'lmstudio'],
                       help='LLM provider to use')
    parser.add_argument('--model', required=True, help='Model name to use')

    # Optional arguments
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--max-tokens', type=int, default=4000, help='Maximum tokens (default: 4000)')
    parser.add_argument('--prompt', help='Execute single prompt and exit')

    # Provider-specific
    parser.add_argument('--base-url', help='Base URL (ollama, lmstudio)')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature (default: 0.7)')

    args = parser.parse_args()

    # Build kwargs
    kwargs = {'temperature': args.temperature}
    if args.base_url:
        kwargs['base_url'] = args.base_url
    if args.api_key:
        kwargs['api_key'] = args.api_key

    # Create CLI
    cli = SimpleCLI(
        provider=args.provider,
        model=args.model,
        stream=args.stream,
        max_tokens=args.max_tokens,
        debug=args.debug,
        **kwargs
    )

    # Run
    if args.prompt:
        cli.run_single_prompt(args.prompt)
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()