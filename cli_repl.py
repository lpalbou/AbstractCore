#!/usr/bin/env python3
"""
CLI REPL for AbstractLLM with tool support.

This tool provides an interactive command-line interface to communicate with any
supported LLM provider while maintaining conversation history and supporting tools.

Features:
- Support for all AbstractLLM providers
- Conversation history maintenance via BasicSession
- Built-in file system tools (list_files, read_file)
- Streaming and non-streaming modes
- Interactive commands for switching providers/models
- Token limit configuration

Usage:
    python cli_repl.py --provider ollama --model qwen3:4b
    python cli_repl.py --provider openai --model gpt-4o-mini --stream
"""

import argparse
import sys
import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractllm import create_llm, BasicSession
from abstractllm.tools.common_tools import list_files, read_file
from abstractllm.tools import ToolDefinition, register_tool
from abstractllm.core.types import GenerateResponse
from abstractllm.utils.structured_logging import configure_logging, get_logger


class CLIRepl:
    """Interactive CLI REPL for AbstractLLM"""
    
    def __init__(self, provider: str, model: str, stream: bool = False,
                 max_tokens: int = 32000, max_input_tokens: int = 28000,
                 max_output_tokens: int = 4000, debug: bool = False, **kwargs):
        """Initialize the REPL with provider and model configuration"""

        self.provider_name = provider
        self.model_name = model
        self.stream_mode = stream
        self.debug_mode = debug
        self.provider_kwargs = kwargs

        # Token configuration
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        # Configure logging based on debug mode
        self._setup_logging()

        # Get logger instance
        self.logger = get_logger("cli_repl")

        # Register tools properly
        self._register_tools()

        # Initialize provider and session
        self.provider = None
        self.session = None

        # Initialize connection
        self._initialize_provider()

        print(f"🚀 AbstractLLM CLI REPL")
        print(f"Provider: {self.provider_name}")
        print(f"Model: {self.model_name}")
        print(f"Stream mode: {'ON' if self.stream_mode else 'OFF'}")
        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        print(f"Token limits: {self.max_tokens} total, {self.max_input_tokens} input, {self.max_output_tokens} output")
        print(f"Available commands: /stream, /model, /clear, /help, /quit, /debug, /history, /logs")
        print(f"Available tools: list_files, read_file")
        if self.debug_mode:
            print(f"📝 Logging enabled: ~/.abstractllm/logs/")
        print("=" * 60)

    def _register_tools(self):
        """Register the available tools"""
        # Register list_files tool
        self._list_files_tool = ToolDefinition.from_function(list_files)
        register_tool(self._list_files_tool)
        
        # Register read_file tool
        self._read_file_tool = ToolDefinition.from_function(read_file)
        register_tool(self._read_file_tool)

    def _setup_logging(self):
        """Setup logging configuration"""
        if self.debug_mode:
            # Enable debug-level logging to console and files
            configure_logging(
                console_level=logging.DEBUG,
                file_level=logging.DEBUG,
                log_dir=os.path.expanduser("~/.abstractllm/logs"),
                verbatim_enabled=True,
                console_json=False,  # Human-readable for console
                file_json=True       # Machine-readable for files
            )
        else:
            # Standard logging: warnings to console, info to file
            configure_logging(
                console_level=logging.WARNING,
                file_level=logging.INFO,
                log_dir=os.path.expanduser("~/.abstractllm/logs"),
                verbatim_enabled=True,
                console_json=False,
                file_json=True
            )

    def _initialize_provider(self):
        """Initialize the provider and session"""
        try:
            self.provider = create_llm(
                self.provider_name, 
                model=self.model_name,
                max_tokens=self.max_tokens,
                max_input_tokens=self.max_input_tokens,
                max_output_tokens=self.max_output_tokens,
                **self.provider_kwargs
            )
            
            self.session = BasicSession(
                provider=self.provider,
                system_prompt="You are a helpful AI assistant."
            )

            # Log successful connection
            self.logger.info(
                "Provider initialized successfully",
                provider=self.provider_name,
                model=self.model_name,
                session_id=self.session.id
            )

            print(f"✅ Connected to {self.provider_name} with model {self.model_name}")

        except Exception as e:
            self.logger.error(
                "Failed to initialize provider",
                provider=self.provider_name,
                model=self.model_name,
                error=str(e)
            )
            print(f"❌ Failed to initialize provider: {str(e)}")
            print(f"Make sure {self.provider_name} is running and accessible")
            sys.exit(1)

    def _handle_command(self, user_input: str) -> str:
        """Handle special commands. Returns 'quit' to exit, 'handled' if processed, 'continue' if not a command"""
        
        if user_input.startswith('/'):
            command = user_input[1:].strip()
            
            if command == 'quit' or command == 'exit' or command == 'q':
                print("👋 Goodbye!")
                return 'quit'
                
            elif command == 'help':
                self._show_help()
                return 'handled'  # Command handled, don't pass to LLM
                
            elif command == 'clear':
                self.session.clear_history(keep_system=True)
                print("🧹 Conversation history cleared")
                return 'handled'  # Command handled, don't pass to LLM
                
            elif command == 'stream':
                self.stream_mode = not self.stream_mode
                print(f"🌊 Stream mode: {'ON' if self.stream_mode else 'OFF'}")
                return 'handled'  # Command handled, don't pass to LLM

            elif command == 'debug':
                self.debug_mode = not self.debug_mode
                self._setup_logging()  # Reconfigure logging
                print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                if self.debug_mode:
                    print(f"📝 Detailed logging enabled: ~/.abstractllm/logs/")
                return 'handled'

            elif command == 'history':
                self._show_history()
                return 'handled'

            elif command == 'logs':
                self._show_logs()
                return 'handled'

            elif command.startswith('model '):
                self._handle_model_change(command[6:])
                return 'handled'  # Command handled, don't pass to LLM

            else:
                print(f"❓ Unknown command: /{command}")
                self._show_help()
                return 'handled'  # Even unknown commands shouldn't go to LLM
        
        return 'continue'

    def _handle_model_change(self, model_spec: str):
        """Handle model change command"""
        try:
            # Parse model specification: provider:model
            if ':' in model_spec:
                self.provider_name, self.model_name = model_spec.split(':', 1)
            else:
                self.model_name = model_spec
            
            print(f"🔄 Switching to {self.provider_name}:{self.model_name}...")
            self._initialize_provider()
                
        except Exception as e:
            print(f"❌ Failed to switch model: {str(e)}")
            print("Usage: /model provider:model or /model model_name")

    def _show_help(self):
        """Show help information"""
        print("\n📖 Available Commands:")
        print("  /help          - Show this help")
        print("  /quit, /exit   - Exit the REPL")
        print("  /clear         - Clear conversation history")
        print("  /stream        - Toggle streaming mode")
        print("  /debug         - Toggle debug/logging mode")
        print("  /history       - Show conversation history")
        print("  /logs          - Show recent log files location")
        print("  /model <spec>  - Change model (e.g., /model ollama:qwen3:4b)")
        print("\n🛠️  Available Tools:")
        print("  list_files - List files in a directory with pattern matching")
        print("  read_file - Read contents of a file")
        print()

    def _show_history(self):
        """Show conversation history"""
        if not self.session:
            print("❌ No active session")
            return

        messages = self.session.get_history(include_system=True)
        if not messages:
            print("📝 No conversation history")
            return

        print(f"\n📚 Conversation History (Session: {self.session.id[:8]}...):")
        print("-" * 60)

        for i, msg in enumerate(messages, 1):
            role_emoji = {"system": "⚙️", "user": "👤", "assistant": "🤖"}.get(msg["role"], "❓")
            role = msg["role"].upper()
            content = msg["content"]

            # Truncate long messages for display
            if len(content) > 100:
                content = content[:97] + "..."

            print(f"{i:2d}. {role_emoji} {role}: {content}")

        print("-" * 60)
        print(f"Total messages: {len(messages)}")

    def _show_logs(self):
        """Show information about log files"""
        log_dir = os.path.expanduser("~/.abstractllm/logs")
        print(f"\n📁 Log Directory: {log_dir}")

        if not os.path.exists(log_dir):
            print("❌ Log directory doesn't exist yet")
            return

        try:
            log_files = []
            for file in os.listdir(log_dir):
                if file.endswith(('.log', '.jsonl')):
                    file_path = os.path.join(log_dir, file)
                    stat = os.stat(file_path)
                    log_files.append((file, stat.st_mtime, stat.st_size))

            if not log_files:
                print("📝 No log files found")
                return

            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: x[1], reverse=True)

            print("\n📋 Recent Log Files:")
            for file, mtime, size in log_files[:5]:  # Show only latest 5
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                print(f"  {file:<30} {mtime_str} ({size_str})")

            print(f"\nTo view logs: tail -f {log_dir}/abstractllm.log")
            if self.debug_mode:
                print(f"Verbatim logs: ls {log_dir}/verbatim_*.jsonl")

        except Exception as e:
            print(f"❌ Error reading log directory: {e}")

    def _generate_response(self, user_input: str):
        """Generate response from the LLM"""
        import time

        start_time = time.time()

        try:
            # Log the user input
            self.logger.info(
                "User input received",
                user_input=user_input,
                session_id=self.session.id,
                message_count=len(self.session.messages)
            )

            # Log what's being sent to the model
            messages_for_model = self.session._format_messages_for_provider()
            self.logger.debug(
                "Messages sent to model",
                provider=self.provider_name,
                model=self.model_name,
                message_count=len(messages_for_model),
                system_prompt=self.session.system_prompt,
                messages=messages_for_model if self.debug_mode else f"[{len(messages_for_model)} messages]"
            )

            # Generate response
            response = self.session.generate(
                user_input,
                stream=self.stream_mode,
                tools=[self._list_files_tool, self._read_file_tool]
            )

            if self.stream_mode:
                # Handle streaming response - session now handles collecting content for history
                print("🤖 Assistant: ", end="", flush=True)

                collected_content = ""
                for chunk in response:
                    if hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end="", flush=True)
                        collected_content += chunk.content

                print()  # New line after streaming

                # Log the complete response (still needed for telemetry)
                latency_ms = (time.time() - start_time) * 1000
                self.logger.log_generation(
                    provider=self.provider_name,
                    model=self.model_name,
                    prompt=user_input,
                    response=collected_content,
                    latency_ms=latency_ms,
                    success=True
                )

            else:
                # Handle non-streaming response
                print(f"🤖 Assistant: {response.content}")

                # Log the complete response
                latency_ms = (time.time() - start_time) * 1000
                self.logger.log_generation(
                    provider=self.provider_name,
                    model=self.model_name,
                    prompt=user_input,
                    response=response.content,
                    tokens=getattr(response, 'token_usage', None),
                    latency_ms=latency_ms,
                    success=True
                )

            # Log session state after response
            self.logger.debug(
                "Response generated successfully",
                session_id=self.session.id,
                total_messages=len(self.session.messages),
                latency_ms=(time.time() - start_time) * 1000
            )

        except KeyboardInterrupt:
            print("\n⏸️  Generation interrupted")
            self.logger.warning("Generation interrupted by user", session_id=self.session.id)
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Log the error
            self.logger.error(
                "Error generating response",
                provider=self.provider_name,
                model=self.model_name,
                user_input=user_input,
                error=str(e),
                error_type=type(e).__name__,
                session_id=self.session.id,
                latency_ms=latency_ms
            )

            print(f"❌ Error generating response: {str(e)}")
            if self.debug_mode and hasattr(e, '__traceback__'):
                print("📝 Debug info:")
                traceback.print_exc()

    def run(self):
        """Run the REPL loop"""
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("\n👤 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    command_result = self._handle_command(user_input)
                    if command_result == 'quit':
                        break
                    elif command_result == 'handled':
                        continue  # Command was processed, get next input
                    
                    # Generate response (only if command_result == 'continue')
                    self._generate_response(user_input)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted. Use /quit to exit.")
                    continue
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
                    
        except Exception as e:
            print(f"❌ Fatal error: {str(e)}")
            traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CLI REPL for AbstractLLM with tool support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_repl.py --provider ollama --model qwen3:4b
  python cli_repl.py --provider openai --model gpt-4o-mini --stream
  python cli_repl.py --provider anthropic --model claude-3-5-haiku-20241022
  python cli_repl.py --provider lmstudio --model qwen/qwen3-coder-30b --base-url http://localhost:1234/v1

  # Enable detailed logging for debugging session issues:
  python cli_repl.py --provider ollama --model qwen3-coder:30b --debug

Interactive Commands:
  /help          - Show help
  /quit          - Exit
  /clear         - Clear history
  /stream        - Toggle streaming
  /debug         - Toggle debug/logging mode
  /history       - Show conversation history
  /logs          - Show log files location
  /model <spec>  - Change model (e.g., /model ollama:qwen3:4b)

Debugging Session Issues:
  1. Use --debug flag to enable detailed logging
  2. Use /history to see what the LLM received in each conversation turn
  3. Use /logs to locate log files for analysis
  4. Check ~/.abstractllm/logs/abstractllm.log for structured logs
  5. Check ~/.abstractllm/logs/verbatim_*.jsonl for complete prompt/response pairs
        """
    )
    
    # Required arguments
    parser.add_argument('--provider', required=True,
                       choices=['openai', 'anthropic', 'ollama', 'huggingface', 'mlx', 'lmstudio', 'mock'],
                       help='LLM provider to use')
    parser.add_argument('--model', required=True,
                       help='Model name to use')
    
    # Optional arguments
    parser.add_argument('--stream', action='store_true',
                       help='Enable streaming mode by default')
    parser.add_argument('--max-tokens', type=int, default=32000,
                       help='Maximum total tokens (default: 32000)')
    parser.add_argument('--max-input-tokens', type=int, default=28000,
                       help='Maximum input tokens (default: 28000)')
    parser.add_argument('--max-output-tokens', type=int, default=4000,
                       help='Maximum output tokens (default: 4000)')
    
    # Provider-specific arguments
    parser.add_argument('--base-url', 
                       help='Base URL for providers that support it (ollama, lmstudio)')
    parser.add_argument('--api-key',
                       help='API key for providers that require it')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for generation (default: 0.7)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Build provider kwargs
    provider_kwargs = {
        'temperature': args.temperature
    }
    
    if args.base_url:
        provider_kwargs['base_url'] = args.base_url
    if args.api_key:
        provider_kwargs['api_key'] = args.api_key
    
    # Create and run REPL
    repl = CLIRepl(
        provider=args.provider,
        model=args.model,
        stream=args.stream,
        debug=args.debug,
        max_tokens=args.max_tokens,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        **provider_kwargs
    )

    repl.run()


if __name__ == "__main__":
    main()

