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
from typing import Optional, Dict, Any, List
import traceback

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractllm import create_llm, BasicSession
from abstractllm.tools.common_tools import list_files, read_file
from abstractllm.tools import ToolDefinition, register_tool
from abstractllm.core.types import GenerateResponse


class CLIRepl:
    """Interactive CLI REPL for AbstractLLM"""
    
    def __init__(self, provider: str, model: str, stream: bool = False,
                 max_tokens: int = 32000, max_input_tokens: int = 28000, 
                 max_output_tokens: int = 4000, **kwargs):
        """Initialize the REPL with provider and model configuration"""
        
        self.provider_name = provider
        self.model_name = model
        self.stream_mode = stream
        self.provider_kwargs = kwargs
        
        # Token configuration
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        
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
        print(f"Token limits: {self.max_tokens} total, {self.max_input_tokens} input, {self.max_output_tokens} output")
        print(f"Available commands: /stream, /model, /clear, /help, /quit")
        print(f"Available tools: list_files, read_file")
        print("=" * 60)

    def _register_tools(self):
        """Register the available tools"""
        # Register list_files tool
        list_files_tool = ToolDefinition.from_function(list_files)
        register_tool(list_files_tool)
        
        # Register read_file tool  
        read_file_tool = ToolDefinition.from_function(read_file)
        register_tool(read_file_tool)

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
            
            print(f"✅ Connected to {self.provider_name} with model {self.model_name}")
            
        except Exception as e:
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
        print("  /model <spec>  - Change model (e.g., /model ollama:qwen3:4b)")
        print("\n🛠️  Available Tools:")
        print("  list_files - List files in a directory with pattern matching")
        print("  read_file - Read contents of a file")
        print()

    def _generate_response(self, user_input: str):
        """Generate response from the LLM"""
        try:
            # Simple generation - tools are registered globally
            response = self.session.generate(
                user_input,
                stream=self.stream_mode
            )
            
            if self.stream_mode:
                # Handle streaming response
                print("🤖 Assistant: ", end="", flush=True)
                
                for chunk in response:
                    if hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end="", flush=True)
                        
                print()  # New line after streaming
                
            else:
                # Handle non-streaming response
                print(f"🤖 Assistant: {response.content}")
                    
        except KeyboardInterrupt:
            print("\n⏸️  Generation interrupted")
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            if hasattr(e, '__traceback__'):
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

Interactive Commands:
  /help          - Show help
  /quit          - Exit
  /clear         - Clear history  
  /stream        - Toggle streaming
  /model <spec>  - Change model (e.g., /model ollama:qwen3:4b)
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
        'temperature': args.temperature,
        'debug': args.debug
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
        max_tokens=args.max_tokens,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        **provider_kwargs
    )
    
    repl.run()


if __name__ == "__main__":
    main()

