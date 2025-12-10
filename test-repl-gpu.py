#!/usr/bin/env python3
"""
vLLM Provider REPL - Direct Interactive Chat

A minimal REPL (Read-Eval-Print Loop) to chat directly with vLLM provider
without the AbstractCore server. This is useful for quick testing and debugging.

Prerequisites:
- vLLM server running with the default model
- AbstractCore installed: pip install -e .

Usage:
    python test-repl-gpu.py

Commands:
    /help       - Show this help
    /stream     - Toggle streaming mode
    /temp X     - Set temperature (0.0-2.0)
    /tokens X   - Set max tokens
    /clear      - Clear conversation history
    /history    - Show conversation history
    /quit       - Exit the REPL

----------------------------------------------------------------------
HOW TO DOWNLOAD THE MODEL (if not already on GPU server)
----------------------------------------------------------------------

# Option 1: vLLM will download automatically on first run
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000

# Option 2: Pre-download using HuggingFace CLI
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct

# Option 3: Download with Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Coder-30B-A3B-Instruct')
"

# The model will be cached in: ~/.cache/huggingface/hub/
# Default size: ~60GB for the 30B model

# For gated models (like Llama), you need a HuggingFace token:
huggingface-cli login
# Then download:
huggingface-cli download meta-llama/Llama-3.2-70B-Instruct

----------------------------------------------------------------------
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class VLLMRepl:
    """Minimal REPL for vLLM provider."""

    def __init__(self):
        self.llm = None
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = "You are a helpful AI assistant."
        self.streaming = False
        self.temperature = 0.7
        self.max_tokens = 500
        self.model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

    def initialize(self):
        """Initialize the vLLM provider."""
        from abstractcore import create_llm

        print("Initializing vLLM provider...")
        print(f"Model: {self.model}")

        vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        print(f"vLLM Server: {vllm_url}")

        try:
            self.llm = create_llm('vllm', model=self.model)
            print(f"✅ Provider initialized")
            print(f"   Provider: {self.llm.provider}")
            print(f"   Base URL: {self.llm.base_url}")

            # Test connectivity
            models = self.llm.list_available_models()
            print(f"✅ Connected to vLLM server")
            print(f"   Available models: {', '.join(models)}")

            return True

        except Exception as e:
            print(f"❌ Failed to initialize vLLM provider: {e}")
            print("\nMake sure vLLM server is running:")
            print(f"  vllm serve {self.model} --port 8000")
            return False

    def print_settings(self):
        """Print current settings."""
        print(f"\n{'='*60}")
        print("Current Settings:")
        print(f"  Model: {self.model}")
        print(f"  Streaming: {'ON' if self.streaming else 'OFF'}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Tokens: {self.max_tokens}")
        print(f"  History: {len(self.conversation_history)} messages")
        print(f"{'='*60}\n")

    def print_help(self):
        """Print help message."""
        print(f"\n{'='*60}")
        print("Available Commands:")
        print("  /help       - Show this help")
        print("  /stream     - Toggle streaming mode")
        print("  /temp X     - Set temperature (0.0-2.0)")
        print("  /tokens X   - Set max tokens")
        print("  /clear      - Clear conversation history")
        print("  /history    - Show conversation history")
        print("  /settings   - Show current settings")
        print("  /quit       - Exit the REPL")
        print(f"{'='*60}\n")

    def handle_command(self, user_input: str) -> bool:
        """Handle slash commands. Returns True to continue, False to quit."""
        if user_input == "/help":
            self.print_help()
            return True

        elif user_input == "/stream":
            self.streaming = not self.streaming
            print(f"Streaming mode: {'ON' if self.streaming else 'OFF'}")
            return True

        elif user_input.startswith("/temp "):
            try:
                temp = float(user_input.split()[1])
                if 0.0 <= temp <= 2.0:
                    self.temperature = temp
                    print(f"Temperature set to: {self.temperature}")
                else:
                    print("Temperature must be between 0.0 and 2.0")
            except (IndexError, ValueError):
                print("Usage: /temp 0.7")
            return True

        elif user_input.startswith("/tokens "):
            try:
                tokens = int(user_input.split()[1])
                if tokens > 0:
                    self.max_tokens = tokens
                    print(f"Max tokens set to: {self.max_tokens}")
                else:
                    print("Max tokens must be positive")
            except (IndexError, ValueError):
                print("Usage: /tokens 500")
            return True

        elif user_input == "/clear":
            self.conversation_history.clear()
            print("✅ Conversation history cleared")
            return True

        elif user_input == "/history":
            if not self.conversation_history:
                print("No conversation history")
            else:
                print(f"\n{'='*60}")
                print("Conversation History:")
                for i, msg in enumerate(self.conversation_history, 1):
                    role = msg['role'].upper()
                    content = msg['content']
                    # Truncate long messages
                    if len(content) > 100:
                        content = content[:97] + "..."
                    print(f"{i}. [{role}] {content}")
                print(f"{'='*60}\n")
            return True

        elif user_input == "/settings":
            self.print_settings()
            return True

        elif user_input in ["/quit", "/exit", "/q"]:
            print("\n👋 Goodbye!")
            return False

        else:
            print(f"Unknown command: {user_input}")
            print("Type /help for available commands")
            return True

    def generate_response(self, user_message: str):
        """Generate response from vLLM."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            if self.streaming:
                # Streaming response
                print("Assistant: ", end="", flush=True)
                full_response = ""

                for chunk in self.llm.generate(
                    prompt=user_message,
                    messages=self.conversation_history[:-1],  # Exclude current message
                    system_prompt=self.system_prompt,
                    stream=True,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                ):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content

                print("\n")  # New line after streaming

                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })

            else:
                # Non-streaming response
                response = self.llm.generate(
                    prompt=user_message,
                    messages=self.conversation_history[:-1],  # Exclude current message
                    system_prompt=self.system_prompt,
                    stream=False,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                print(f"Assistant: {response.content}\n")

                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })

        except KeyboardInterrupt:
            print("\n\n⚠️  Generation interrupted by user")
            # Remove the user message since we didn't get a response
            self.conversation_history.pop()

        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            # Remove the user message since we didn't get a response
            self.conversation_history.pop()

    def run(self):
        """Run the REPL."""
        print("\n" + "="*60)
        print("  vLLM Provider REPL")
        print("  Direct chat with vLLM (no OpenAI endpoint)")
        print("="*60)

        if not self.initialize():
            return

        self.print_settings()
        print("Type /help for commands, /quit to exit\n")

        try:
            while True:
                try:
                    # Get user input
                    user_input = input("You: ").strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        should_continue = self.handle_command(user_input)
                        if not should_continue:
                            break
                        continue

                    # Generate response
                    self.generate_response(user_input)

                except KeyboardInterrupt:
                    print("\n\nUse /quit to exit")
                    continue

                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            raise


def main():
    """Main entry point."""
    repl = VLLMRepl()
    repl.run()


if __name__ == "__main__":
    main()
