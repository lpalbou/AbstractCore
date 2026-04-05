#!/usr/bin/env python3
"""
AbstractCore CLI - Basic demonstrator for AbstractCore capabilities.

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
    python -m abstractcore.utils.cli --provider ollama --model qwen3-coder:30b
    python -m abstractcore.utils.cli --provider openai --model gpt-5-mini --stream
    python -m abstractcore.utils.cli --provider anthropic --model claude-haiku-4-5 --prompt "What is Python?"
    python -m abstractcore.utils.cli --provider lmstudio --model qwen/qwen3-4b-2507 --base-url http://localhost:1234/v1
    python -m abstractcore.utils.cli --provider openrouter --model openai/gpt-4o-mini
    python -m abstractcore.utils.cli --provider portkey --model gpt-4o-mini --base-url https://api.portkey.ai/v1
"""

import argparse
import os
import sys
import time
import uuid
import locale
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, Iterator, List, Union

from .truncation import preview_text

# Enable command history and arrow key navigation
try:
    import readline
    # Configure readline for better history behavior
    readline.set_startup_hook(lambda: readline.insert_text(''))
    readline.parse_and_bind("tab: complete")
    # Set a reasonable history length
    readline.set_history_length(1000)
except ImportError:
    # readline not available (typically on Windows)
    readline = None

from .. import create_llm, BasicSession
from ..tools.common_tools import list_files, read_file, write_file, execute_command, search_files
from ..processing import BasicExtractor, BasicJudge, BasicIntentAnalyzer


class _NoPromptCacheProvider:
    """Proxy that forces `prompt_cache_key=None` for every call (to avoid polluting KV caches)."""

    def __init__(self, provider: Any):
        self._provider = provider

    def generate(self, *args: Any, **kwargs: Any):
        kwargs["prompt_cache_key"] = None
        return self._provider.generate(*args, **kwargs)

    async def agenerate(self, *args: Any, **kwargs: Any):
        kwargs["prompt_cache_key"] = None
        return await self._provider.agenerate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


class SimpleCLI:
    """Simplified CLI REPL for AbstractCore."""

    def __init__(
        self,
        provider: str,
        model: str,
        stream: bool = False,
        max_tokens: int = None,
        max_output_tokens: int = None,
        debug: bool = False,
        show_banner: bool = True,
        audio_policy: Optional[str] = None,
        audio_language: Optional[str] = None,
        **kwargs,
    ):
        self.provider_name = provider
        self.model_name = model
        self.stream_mode = stream
        self.debug_mode = debug
        self.single_prompt_mode = not show_banner  # Clean output for single-prompt mode
        self.kwargs = kwargs
        self.audio_policy: Optional[str] = None
        if audio_policy is not None:
            normalized = str(audio_policy or "").strip().lower()
            if normalized in {"native"}:
                normalized = "native_only"
            if normalized in {"stt"}:
                normalized = "speech_to_text"
            if normalized:
                self.audio_policy = normalized
        self.audio_language: Optional[str] = None
        if isinstance(audio_language, str) and audio_language.strip():
            self.audio_language = audio_language.strip()

        # Auto-detect max_tokens from model capabilities if not specified
        self.max_tokens_auto = max_tokens is None
        if max_tokens is None:
            try:
                from ..architectures.detection import get_model_capabilities
                capabilities = get_model_capabilities(model)
                max_tokens = capabilities.get('max_tokens', 16384)  # Fallback to 16K if not found
                if debug:
                    print(f"🔍 Auto-detected max_tokens: {max_tokens} (from model capabilities)")
            except Exception as e:
                max_tokens = 16384  # Safe fallback
                if debug:
                    print(f"⚠️ Failed to auto-detect max_tokens, using fallback: {max_tokens} ({e})")

        self.max_tokens = max_tokens
        self.max_output_tokens_auto = max_output_tokens is None
        # Unified thinking/reasoning control (best-effort, provider/model dependent).
        # - None: auto (provider/model default)
        # - bool: on/off
        # - str: "low"|"medium"|"high" when supported
        self.thinking: Optional[Union[bool, str]] = None
        # Whether to display model-supplied reasoning/thinking separately.
        # - None: auto (show when thinking != off)
        # - bool: force on/off
        self.show_reasoning: Optional[bool] = None

        # Initialize command history with persistent storage
        self._setup_command_history()

        # Initialize provider and session with tools
        provider_kwargs = dict(kwargs)
        provider_kwargs["max_tokens"] = max_tokens
        if max_output_tokens is not None:
            provider_kwargs["max_output_tokens"] = max_output_tokens
        self.provider = create_llm(provider, model=model, **provider_kwargs)
        # Store the effective max_output_tokens (provider may auto-select based on model capabilities).
        self.max_output_tokens = getattr(self.provider, "max_output_tokens", max_output_tokens or 2048)
        self.session = BasicSession(
            self.provider,
            system_prompt="You are a helpful AI assistant with vision capabilities. When users provide images or media files, analyze and describe them directly. You also have access to file operation tools.",
            tools=[list_files, read_file, write_file, execute_command, search_files]
        )

        # Prompt caching (best-effort; provider-dependent).
        self.country_code = self._get_country_code()
        self.prompt_cache_mode = "off"  # off | key | kv
        self.prompt_cache_key: Optional[str] = None
        self.prompt_cache_file: Optional[str] = None
        self._init_prompt_caching(show_banner=show_banner)

        # Only show banner in interactive mode
        if show_banner:
            print("=" * 70)
            print("🚀 AbstractCore CLI - Interactive LLM Interface".center(70))
            print("=" * 70)
            print(f"🤖 Provider: {provider}")
            print(f"📝 Model: {model}")
            print(f"🌊 Streaming: {'ON' if stream else 'OFF'} | 🐛 Debug: {'ON' if debug else 'OFF'}")
            print()
            print("💬 Quick Commands: /help /session /cache /status /history /quit")
            print("🛠️  Available Tools: list_files, search_files, read_file, write_file, execute_command")
            print()
            print("💡 Type '/help' for comprehensive command guide")
            print("💡 Ask questions naturally or use tools: 'What files are here?'")
            print("=" * 70)

    def _setup_command_history(self):
        """Setup command history with persistent storage."""
        if readline is None:
            return  # No readline support available

        # Store history in user's home directory
        import os
        import pathlib

        # Create .abstractcore directory if it doesn't exist
        history_dir = pathlib.Path.home() / '.abstractcore'
        history_dir.mkdir(exist_ok=True)

        # Define history file path
        self.history_file = history_dir / 'cli_history.txt'

        try:
            # Load existing history if file exists
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
                if self.debug_mode:
                    history_size = readline.get_current_history_length()
                    print(f"🔍 Loaded {history_size} command(s) from history")
        except (FileNotFoundError, PermissionError) as e:
            if self.debug_mode:
                print(f"⚠️ Could not load command history: {e}")

    def _save_command_history(self):
        """Save current command history to disk."""
        if readline is None or not hasattr(self, 'history_file'):
            return

        try:
            # Ensure the directory exists
            self.history_file.parent.mkdir(exist_ok=True)
            # Save history to file
            readline.write_history_file(str(self.history_file))
        except (PermissionError, OSError) as e:
            if self.debug_mode:
                print(f"⚠️ Could not save command history: {e}")

    def handle_command(self, user_input: str) -> bool:
        """Handle commands. Returns True if command processed, False otherwise."""
        if not user_input.startswith('/'):
            return False

        cmd = user_input[1:].strip()

        if cmd in ['quit', 'exit', 'q']:
            self._save_command_history()
            print("👋 Goodbye!")
            sys.exit(0)

        elif cmd == 'help':
            print("\n" + "=" * 70)
            print("🚀 AbstractCore CLI - Interactive LLM Interface".center(70))
            print("=" * 70)
            
            print("\n📖 CORE COMMANDS")
            print("─" * 50)
            print("  /help                    Show this comprehensive help")
            print("  /quit                    Exit the CLI")
            print("  /clear                   Clear prompt cache + context (like mlx-chat)")
            print("  /cls                     Clear the screen (like unix terminal)")
            print("  /reset                   Reset conversation history")
            print("  /status                  Show system status and capabilities")
            
            print("\n💬 CONVERSATION MANAGEMENT")
            print("─" * 50)
            print("  /history [n]             Show conversation history")
            print("                           • /history        - Show all messages")
            print("                           • /history 5      - Show last 5 interactions")
            print("  /compact [focus]         Compress chat history using local model")
            print("                           • /compact                    - General compaction")
            print("                           • /compact technical details - Focus on technical aspects")
            print("                           • /compact key decisions     - Focus on decisions made")
            print("  /system [prompt]         View or change system prompt")
            print("                           • /system         - Show current prompt")
            print("                           • /system <text>  - Set new prompt")
            
            print("\n💾 SESSION & CACHE")
            print("─" * 50)
            print("  /session save <name> [options]  Save session to <name>.json with optional analytics")
            print("                           • /session save chat")
            print("                           • /session save analyzed --summary --assessment --facts")
            print("                           Options:")
            print("                             --summary     Generate conversation summary")
            print("                             --assessment  Evaluate conversation quality")
            print("                             --facts       Extract knowledge as facts")
            print("  /session load <name>            Load session from <name>.json (replaces current)")
            print("                           • /session load chat")
            print("  /session clear                  Clear session + cache (same as /clear)")
            print("  /save /load                     Aliases for /session save|load (sessions only)")
            print("  /cache save <name>              Save prompt/KV cache to <name>.safetensors (MLX only, model-locked)")
            print("                           • /cache save chat_cache")
            print("                             --q8          Quantize cache before saving (smaller, lossy)")
            print("  /cache load <name>              Load prompt/KV cache from <name>.safetensors (MLX only, model-locked)")
            print("                           • /cache load chat_cache")
            print("  /cache clear                    Clear prompt cache only (KV mode rebuilds from transcript)")
            
            print("\n📊 ANALYTICS & INSIGHTS")
            print("─" * 50)
            print("  /facts [file]            Extract facts from conversation")
            print("                           • /facts          - Display in chat")
            print("                           • /facts data     - Save as data.jsonld")
            print("  /judge                   Evaluate conversation quality")
            print("  /intent [participant]    Analyze intents behind conversation")
            print("                           • /intent         - Analyze all participants")
            print("                           • /intent user    - Focus on user intents")
            print("                           • /intent assistant - Focus on assistant intents")
            
            print("\n⚙️  CONFIGURATION")
            print("─" * 50)
            print("  /model <provider:model>  Switch LLM provider/model")
            print("                           • /model openai:gpt-5-mini")
            print("                           • /model anthropic:claude-haiku-4-5")
            print("                           • /model openrouter:openai/gpt-4o-mini")
            print("  /max-tokens <n|auto>     Set context token budget")
            print("  /max-output-tokens <n|auto> Set max output tokens per response")
            print("  /thinking <mode>         Set thinking/reasoning mode (best-effort)")
            print("                           • /thinking auto|on|off|none|low|medium|high|xhigh")
            print("  /show-reasoning <mode>   Display reasoning separately (auto/on/off)")
            print("                           • /show-reasoning auto|on|off")
            print("  /stream                  Toggle streaming mode on/off")
            print("  /debug                   Toggle debug info (timing, detection)")
            
            print("\n🛠️ AVAILABLE TOOLS")
            print("─" * 50)
            print("  The assistant can use these tools automatically:")
            print("  • list_files             List directory contents")
            print("  • search_files           Search for text patterns inside files")
            print("  • read_file              Read file contents")
            print("  • write_file             Create or modify files")
            print("  • execute_command        Run shell commands")
            
            print("\n📎 FILE ATTACHMENTS")
            print("─" * 50)
            print("  Use @filename syntax to attach files to your message:")
            print("  • Images: 'Analyze this screenshot @screenshot.png'")
            print("  • Documents: 'Summarize @report.pdf and @data.csv'")
            print("  • Multiple files: 'Compare @image1.jpg @image2.jpg @notes.txt'")
            print("  • Vision analysis: Works with vision models (GPT-4o, Claude, qwen2.5vl)")
            print("  • Auto-fallback: Text-only models use vision captioning for images")
            print("  • Supported formats: Images (jpg, png, gif), PDFs, Office docs, text files")

            print("\n💡 TIPS & EXAMPLES")
            print("─" * 50)
            print("  • Ask questions naturally: 'What files are in this directory?'")
            print("  • Search inside files: 'Find all TODO comments in Python files'")
            print("  • Request file operations: 'Read the README.md file'")
            print("  • Attach files: 'What's in this image? @photo.jpg'")
            print("  • Save important conversations: '/session save project_discussion --summary'")
            print("  • Switch models for different tasks: '/model ollama:qwen3-coder:30b'")
            print("  • Use /status to check token usage and model capabilities")
            
            print("\n" + "=" * 70)
            print("Type any message to start chatting, or use commands above".center(70))
            print("=" * 70 + "\n")

        elif cmd == 'clear':
            self.handle_clear()

        elif cmd == 'cls':
            self._clear_screen()

        elif cmd == 'reset':
            if self.prompt_cache_mode == "kv":
                self.handle_clear()
            else:
                self.session.clear_history(keep_system=True)
                print("🧹 Chat history reset")

        elif cmd == 'stream':
            self.stream_mode = not self.stream_mode
            print(f"🌊 Stream mode: {'ON' if self.stream_mode else 'OFF'}")

        elif cmd == 'debug':
            self.debug_mode = not self.debug_mode
            print(f"🐛 CLI Debug mode: {'ON' if self.debug_mode else 'OFF'} (controls timing & auto-detection info)")
            print("💡 Note: System debug logs are controlled by logging level, not CLI debug mode")

        elif cmd == 'status':
            self.handle_status()

        elif cmd.startswith('thinking'):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 1:
                current = "auto" if self.thinking is None else ("on" if self.thinking is True else "off" if self.thinking is False else str(self.thinking))
                print(f"🧠 thinking: {current}")
                print("❓ Usage: /thinking <auto|on|off|none|low|medium|high|xhigh>")
                return True

            raw = parts[1].strip().lower()
            if raw in {"auto", "null"}:
                self.thinking = None
            elif raw in {"on", "true", "1", "yes"}:
                self.thinking = True
            elif raw in {"off", "false", "0", "no", "none"}:
                self.thinking = False
            elif raw in {"low", "medium", "high", "xhigh"}:
                self.thinking = raw
            else:
                print("❓ Usage: /thinking <auto|on|off|none|low|medium|high|xhigh>")
                return True

            current = "auto" if self.thinking is None else ("on" if self.thinking is True else "off" if self.thinking is False else str(self.thinking))
            print(f"✅ thinking set to: {current}")
            return True

        elif cmd.startswith('show-reasoning') or cmd.startswith('reasoning'):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 1:
                current = "auto" if self.show_reasoning is None else ("on" if self.show_reasoning else "off")
                print(f"🧠 show-reasoning: {current}")
                print("❓ Usage: /show-reasoning <auto|on|off>")
                return True

            raw = parts[1].strip().lower()
            if raw in {"auto", "none", "null"}:
                self.show_reasoning = None
            elif raw in {"on", "true", "1", "yes"}:
                self.show_reasoning = True
            elif raw in {"off", "false", "0", "no"}:
                self.show_reasoning = False
            else:
                print("❓ Usage: /show-reasoning <auto|on|off>")
                return True

            current = "auto" if self.show_reasoning is None else ("on" if self.show_reasoning else "off")
            print(f"✅ show-reasoning set to: {current}")
            return True

        elif cmd.startswith('max-tokens'):
            parts = cmd.split()
            if len(parts) == 1:
                print(f"💾 max_tokens (context budget): {self.max_tokens:,} ({'auto' if self.max_tokens_auto else 'manual'})")
                print("❓ Usage: /max-tokens <n|auto>")
            else:
                raw_value = parts[1].strip().lower()
                if raw_value in {"auto", "-1"}:
                    try:
                        from ..architectures.detection import get_model_capabilities
                        capabilities = get_model_capabilities(self.model_name)
                        detected = capabilities.get('max_tokens', 16384)
                    except Exception:
                        detected = 16384
                    self.max_tokens = int(detected)
                    self.max_tokens_auto = True
                else:
                    try:
                        new_max = int(raw_value)
                        if new_max <= 0:
                            raise ValueError
                        self.max_tokens = new_max
                        self.max_tokens_auto = False
                    except ValueError:
                        print("❓ Usage: /max-tokens <n|auto> (n must be a positive integer)")
                        return True

                # Apply to current provider (best-effort; mostly used for token budgeting/compaction).
                try:
                    setattr(self.provider, "max_tokens", self.max_tokens)
                except Exception:
                    pass

                # Safety clamp: output should not exceed total budget.
                if isinstance(self.max_output_tokens, int) and self.max_output_tokens > int(self.max_tokens):
                    self.max_output_tokens = int(self.max_tokens)
                    try:
                        setattr(self.provider, "max_output_tokens", self.max_output_tokens)
                    except Exception:
                        pass

                print(f"✅ max_tokens set to {self.max_tokens:,}")

        elif cmd.startswith('max-output-tokens'):
            parts = cmd.split()
            if len(parts) == 1:
                print(f"✍️ max_output_tokens (per response): {self.max_output_tokens:,} ({'auto' if self.max_output_tokens_auto else 'manual'})")
                print("❓ Usage: /max-output-tokens <n|auto>")
            else:
                raw_value = parts[1].strip().lower()
                if raw_value in {"auto", "-1"}:
                    try:
                        from ..architectures.detection import get_model_capabilities
                        capabilities = get_model_capabilities(self.model_name)
                        detected = capabilities.get('max_output_tokens', getattr(self.provider, "max_output_tokens", 2048))
                    except Exception:
                        detected = getattr(self.provider, "max_output_tokens", 2048)
                    self.max_output_tokens = int(detected)
                    self.max_output_tokens_auto = True
                else:
                    try:
                        new_max = int(raw_value)
                        if new_max <= 0:
                            raise ValueError
                        self.max_output_tokens = new_max
                        self.max_output_tokens_auto = False
                    except ValueError:
                        print("❓ Usage: /max-output-tokens <n|auto> (n must be a positive integer)")
                        return True

                # Safety clamp: output should not exceed total budget.
                if isinstance(self.max_tokens, int) and self.max_output_tokens > int(self.max_tokens):
                    self.max_output_tokens = int(self.max_tokens)

                try:
                    setattr(self.provider, "max_output_tokens", self.max_output_tokens)
                except Exception:
                    pass
                print(f"✅ max_output_tokens set to {self.max_output_tokens:,}")

        elif cmd.startswith('history'):
            # Parse /history [n] command
            parts = cmd.split()
            if len(parts) == 1:
                # Show all history
                self.handle_history(None)
            else:
                try:
                    n = int(parts[1])
                    self.handle_history(n)
                except (ValueError, IndexError):
                    print("❓ Usage: /history [n] where n is number of interactions")

        elif cmd.startswith('model '):
            try:
                model_spec = cmd[6:]
                if ':' in model_spec:
                    self.provider_name, self.model_name = model_spec.split(':', 1)
                else:
                    self.model_name = model_spec

                print(f"🔄 Switching to {self.provider_name}:{self.model_name}...")
                # If token limits were auto-detected, re-detect them for the new model.
                next_max_tokens = self.max_tokens
                if self.max_tokens_auto:
                    try:
                        from ..architectures.detection import get_model_capabilities
                        capabilities = get_model_capabilities(self.model_name)
                        next_max_tokens = int(capabilities.get('max_tokens', 16384))
                    except Exception:
                        next_max_tokens = 16384

                next_max_output_tokens = self.max_output_tokens
                if self.max_output_tokens_auto:
                    try:
                        from ..architectures.detection import get_model_capabilities
                        capabilities = get_model_capabilities(self.model_name)
                        next_max_output_tokens = int(capabilities.get('max_output_tokens', self.max_output_tokens))
                    except Exception:
                        next_max_output_tokens = self.max_output_tokens

                # Safety clamp: output should not exceed total budget.
                if isinstance(next_max_tokens, int) and isinstance(next_max_output_tokens, int):
                    if next_max_output_tokens > next_max_tokens:
                        next_max_output_tokens = next_max_tokens

                self.provider = create_llm(self.provider_name, model=self.model_name,
                                         max_tokens=next_max_tokens,
                                         max_output_tokens=next_max_output_tokens,
                                         **self.kwargs)
                self.max_tokens = next_max_tokens
                self.max_output_tokens = getattr(self.provider, "max_output_tokens", next_max_output_tokens)
                self.session = BasicSession(
                    self.provider,
                    system_prompt="You are a helpful AI assistant with vision capabilities. When users provide images or media files, analyze and describe them directly. You also have access to file operation tools.",
                    tools=[list_files, read_file, write_file, execute_command, search_files]
                )
                # Reset caching state for the new provider+model.
                self.prompt_cache_key = None
                self.prompt_cache_file = None
                self.prompt_cache_mode = "off"
                self._init_prompt_caching(show_banner=False)
                print("✅ Model switched")
            except Exception as e:
                print(f"❌ Failed to switch: {e}")

        elif cmd.startswith('compact'):
            # Parse /compact [focus] command
            parts = cmd.split(maxsplit=1)
            if len(parts) == 1:
                # No focus specified - use default
                self.handle_compact(None)
            else:
                # Focus specified - extract everything after "compact "
                focus = user_input[9:].strip()  # Remove "/compact " prefix
                if focus:
                    self.handle_compact(focus)
                else:
                    self.handle_compact(None)

        elif cmd.startswith('facts'):
            # Parse /facts [file] command
            parts = cmd.split()
            if len(parts) == 1:
                # No file specified - display facts in chat
                self.handle_facts(None)
            else:
                # File specified - save as JSON-LD
                filename = parts[1]
                self.handle_facts(filename)

        elif cmd == 'judge':
            self.handle_judge()

        elif cmd.startswith('intent'):
            # Parse /intent [participant] command
            parts = cmd.split()
            if len(parts) == 1:
                # No participant specified - analyze all
                self.handle_intent(None)
            else:
                # Participant specified
                participant = parts[1]
                self.handle_intent(participant)

        elif cmd.startswith('system'):
            # Parse /system [prompt] command
            if cmd == 'system':
                # Show current system prompt
                self.handle_system_show()
            else:
                # Change system prompt - extract everything after "system "
                new_prompt = user_input[8:].strip()  # Remove "/system " prefix
                if new_prompt:
                    self.handle_system_change(new_prompt)
                else:
                    self.handle_system_show()

        elif cmd.startswith('session'):
            # /session save|load|clear ...
            parts = cmd.split()
            if len(parts) < 2:
                print("❓ Usage: /session <save|load|clear> ...")
                print("   Examples:")
                print("     /session save my_conversation")
                print("     /session save analyzed_session --summary --assessment --facts")
                print("     /session load my_conversation")
                print("     /session clear")
                return True

            action = parts[1].strip().lower()
            if action == "save":
                if len(parts) < 3:
                    print("❓ Usage: /session save <name> [--summary] [--assessment] [--facts]")
                    return True
                filename = parts[2]
                options = {
                    'summary': '--summary' in parts[3:],
                    'assessment': '--assessment' in parts[3:],
                    'facts': '--facts' in parts[3:],
                }
                self.handle_save(filename, **options)
                return True

            if action == "load":
                if len(parts) != 3:
                    print("❓ Usage: /session load <name>")
                    return True
                self.handle_load(parts[2])
                return True

            if action == "clear":
                self.handle_clear()
                return True

            print("❓ Usage: /session <save|load|clear> ...")
            return True

        elif cmd.startswith('cache'):
            # /cache save|load|clear ...
            parts = cmd.split()
            if len(parts) < 2:
                print("❓ Usage: /cache <save|load|clear> ...")
                print("   Examples:")
                print("     /cache save chat_cache")
                print("     /cache load chat_cache")
                print("     /cache clear")
                return True

            action = parts[1].strip().lower()
            if action == "save":
                if len(parts) < 3:
                    print("❓ Usage: /cache save <name> [--q8]")
                    return True
                filename = parts[2]
                self.handle_save_prompt_cache(filename, q8=("--q8" in parts[3:]))
                return True

            if action == "load":
                if len(parts) != 3:
                    print("❓ Usage: /cache load <name>")
                    return True
                self.handle_load_prompt_cache(parts[2])
                return True

            if action == "clear":
                self.handle_cache_clear()
                return True

            print("❓ Usage: /cache <save|load|clear> ...")
            return True

        elif cmd.startswith('save'):
            # Parse /save <file> [--summary] [--assessment] [--facts] command
            parts = cmd.split()
            if len(parts) < 2:
                print("❓ Usage: /save <filename> [--summary] [--assessment] [--facts]")
                print("   Example: /save my_conversation")
                print("   Hint: use /cache save <name> for prompt caches")
                print("   Example: /save analyzed_session --summary --assessment --facts")
            else:
                filename = parts[1]
                options = {
                    'summary': '--summary' in parts,
                    'assessment': '--assessment' in parts,
                    'facts': '--facts' in parts
                }
                self.handle_save(filename, **options)

        elif cmd.startswith('load'):
            # Parse /load <file> command
            parts = cmd.split()
            if len(parts) != 2:
                print("❓ Usage: /load <filename>")
                print("   Example: /load my_conversation")
                print("   Hint: use /cache load <name> for prompt caches")
            else:
                filename = parts[1]
                self.handle_load(filename)

        elif cmd.startswith('tooltag'):
            # Parse /tooltag <opening_tag> <closing_tag> command
            parts = cmd.split()
            if len(parts) != 3:
                print("❓ Usage: /tooltag <opening_tag> <closing_tag>")
                print("   Example: /tooltag '<|tool_call|>' '</|tool_call|>'")
                print("   Example: /tooltag '<function_call>' '</function_call>'")
                print("   Example: /tooltag '<tool_call>' '</tool_call>'")
            else:
                # Strip quotes from the tags if present
                opening_tag = parts[1].strip("'\"")
                closing_tag = parts[2].strip("'\"")
                self.handle_tooltag_test(opening_tag, closing_tag)

        else:
            print(f"❓ Unknown command: /{cmd}. Type /help for help.")

        return True

    def _clear_screen(self) -> None:
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_error(self, msg: str) -> None:
        red = "\033[31m"
        reset = "\033[0m"
        print(f"{red}{msg}{reset}")

    def _print_warn(self, msg: str) -> None:
        yellow = "\033[33m"
        reset = "\033[0m"
        print(f"{yellow}{msg}{reset}")

    def _force_extension(self, filename: str, ext: str) -> str:
        """Ensure `filename` ends with `ext` by replacing any existing suffix (best-effort)."""
        ext = str(ext or "").strip()
        if not ext:
            return filename
        if not ext.startswith("."):
            ext = f".{ext}"
        try:
            p = Path(filename)
        except Exception:
            return f"{filename}{ext}"
        if p.suffix:
            return str(p.with_suffix(ext))
        return f"{p}{ext}"

    def _resolve_session_path(self, filename: str) -> Optional[str]:
        """Resolve a session file path (prefers exact match, then `.json`)."""
        if not isinstance(filename, str) or not filename.strip():
            return None
        raw = filename.strip()
        candidates = [raw]
        forced = self._force_extension(raw, ".json")
        if forced != raw:
            candidates.append(forced)
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _resolve_cache_path(self, filename: str) -> Optional[str]:
        """Resolve a cache file path (prefers exact match, then `.safetensors` / `.safetensor`)."""
        if not isinstance(filename, str) or not filename.strip():
            return None
        raw = filename.strip()
        candidates = [raw]
        forced = self._force_extension(raw, ".safetensors")
        if forced != raw:
            candidates.append(forced)
        forced_alt = self._force_extension(raw, ".safetensor")
        if forced_alt not in candidates:
            candidates.append(forced_alt)
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _kv_cache_token_count(self, key: str) -> Optional[int]:
        """Best-effort token count for the active KV cache key (MLX)."""
        if not isinstance(key, str) or not key.strip():
            return None
        try:
            cache_obj = getattr(self.provider, "_prompt_cache_store").get(key.strip())
        except Exception:
            cache_obj = None
        if cache_obj is None:
            return None
        try:
            tok = getattr(self.provider, "_prompt_cache_backend_token_count")(cache_obj)
            return int(tok) if isinstance(tok, int) else None
        except Exception:
            return None

    def _kv_refresh_tools_if_needed(self, *, reason: str, force: bool = False) -> bool:
        """Re-inject tool specs into the active KV cache when recency or origin requires it."""
        if self.prompt_cache_mode != "kv":
            return False
        if not self._is_mlx_provider():
            return False
        if not self._supports_prompt_cache():
            return False
        if not getattr(self.session, "tools", None):
            return False

        key = self.prompt_cache_key
        if not isinstance(key, str) or not key.strip():
            return False

        # Long-context models can “forget” early tool specs; re-inject near the end when the cache is very large.
        threshold_default = 50_000
        try:
            threshold = int(os.getenv("ABSTRACTCORE_CLI_KV_REFRESH_TOOLS_AT", str(threshold_default)))
        except Exception:
            threshold = threshold_default
        if threshold < 0:
            threshold = threshold_default

        tok = self._kv_cache_token_count(key)
        should = bool(force) or (isinstance(tok, int) and tok >= threshold)
        if not should:
            return False

        try:
            getattr(self.provider, "prompt_cache_update")(
                key,
                system_prompt=None,  # tools-only system message for recency
                tools=self.session.tools,
                add_generation_prompt=False,
            )
        except Exception as e:
            self._print_warn(f"⚠️ Could not refresh tools into KV cache ({reason}): {e}")
            return False

        if not self.single_prompt_mode:
            extra = f" (~{tok:,} tokens)" if isinstance(tok, int) and tok > 0 else ""
            print(f"🧰 Tools refreshed into KV cache ({reason}){extra}")
        return True

    def _get_country_code(self) -> str:
        val = os.getenv("ABSTRACTCORE_CLI_COUNTRY")
        if isinstance(val, str) and val.strip():
            cc = val.strip().upper()
            return cc if len(cc) == 2 else cc[:2]

        # Best-effort locale fallback (e.g. "en_US" -> "US")
        try:
            loc = locale.getlocale()[0] or ""
        except Exception:
            loc = ""
        if isinstance(loc, str) and "_" in loc:
            cc = loc.split("_", 1)[1].strip().upper()
            if cc:
                return cc[:2]

        return "FR"

    def _timestamp_user_message(self, text: str) -> str:
        ts = datetime.now().strftime("%Y/%m/%d %H:%M")
        return f"[{ts} {self.country_code}] {text}"

    def _supports_prompt_cache(self) -> bool:
        try:
            fn = getattr(self.provider, "supports_prompt_cache", None)
            return bool(fn and fn())
        except Exception:
            return False

    def _is_mlx_provider(self) -> bool:
        return str(self.provider_name or "").strip().lower() == "mlx"

    def _analysis_provider(self) -> Any:
        """Provider to use for internal CLI analytics (never mutates KV prompt cache)."""
        if self.prompt_cache_mode != "kv":
            return self.provider
        return _NoPromptCacheProvider(self.provider)

    def _init_prompt_caching(self, *, show_banner: bool) -> None:
        if not self._supports_prompt_cache():
            self.prompt_cache_mode = "off"
            return

        # Default policy:
        # - MLX: local KV cache (append-only) with explicit prefill (system+tools).
        # - Other providers: key-only hint (pass-through / best-effort).
        if self._is_mlx_provider():
            self.prompt_cache_mode = "kv"
        else:
            self.prompt_cache_mode = "key"

        self.prompt_cache_key = f"cli:{uuid.uuid4().hex[:12]}"
        try:
            ok = bool(getattr(self.provider, "prompt_cache_set")(self.prompt_cache_key, make_default=True))
        except Exception:
            ok = False

        if not ok:
            self.prompt_cache_mode = "off"
            self.prompt_cache_key = None
            return

        if self.prompt_cache_mode == "kv":
            # Prefill stable modules once so each turn can be appended safely.
            try:
                getattr(self.provider, "prompt_cache_update")(
                    self.prompt_cache_key,
                    system_prompt=self.session.system_prompt,
                    tools=self.session.tools,
                    add_generation_prompt=False,
                )
            except Exception as e:
                self._print_warn(f"⚠️ Prompt cache prefill failed; falling back to key-only mode: {e}")
                self.prompt_cache_mode = "key"

        if show_banner:
            if self.prompt_cache_mode == "kv":
                print(f"🧠 Prompt caching: ON (KV local)  key={self.prompt_cache_key}")
            elif self.prompt_cache_mode == "key":
                print(f"🧠 Prompt caching: ON (key hint)  key={self.prompt_cache_key}")

    def handle_clear(self) -> None:
        """Clear prompt cache and context (best-effort)."""
        # Clear session transcript (keep system prompt for user visibility).
        self.session.clear_history(keep_system=True)

        if not self._supports_prompt_cache():
            print("🧹 Context cleared (prompt caching unsupported)")
            return

        # Clear provider-side in-process caches (best-effort).
        try:
            getattr(self.provider, "prompt_cache_clear")(None)
        except Exception:
            pass

        # Re-init caching for this run.
        self.prompt_cache_key = None
        self.prompt_cache_file = None
        self._init_prompt_caching(show_banner=False)

        if self.prompt_cache_mode == "off":
            print("🧹 Context cleared (prompt caching disabled)")
        else:
            print("🧹 Context + prompt cache cleared")

    def handle_cache_clear(self) -> None:
        """Clear prompt cache only (best-effort)."""
        if not self._supports_prompt_cache():
            print("🧹 Prompt cache cleared (prompt caching unsupported)")
            return

        # In KV mode the cache is the source-of-truth for model context; clearing it without clearing
        # or resending history would desync the model and the transcript. Rebuild from transcript.
        if self.prompt_cache_mode == "kv":
            self._print_warn("⚠️ KV cache cleared; rebuilding from current session transcript")
            try:
                self._rebuild_kv_cache_from_session()
                return
            except Exception as e:
                self._print_error(f"❌ KV cache rebuild failed: {e}")
                self._print_warn("⚠️ Falling back to session-managed mode (no KV)")
                self.prompt_cache_mode = "key"

        # Key-only / remote mode: clear provider-side caches (best-effort) and rotate key.
        try:
            getattr(self.provider, "prompt_cache_clear")(None)
        except Exception:
            pass

        self.prompt_cache_key = None
        self.prompt_cache_file = None
        self._init_prompt_caching(show_banner=False)

        if self.prompt_cache_mode == "off":
            print("🧹 Prompt cache cleared (prompt caching disabled)")
        else:
            print("🧹 Prompt cache cleared")

    def handle_save_prompt_cache(self, filename: str, *, q8: bool = False) -> None:
        """Save MLX prompt cache to disk (writes a `.safetensors` file; model-locked)."""
        if not self._is_mlx_provider():
            self._print_error("❌ KV cache save is only supported for provider 'mlx'")
            return
        if not self._supports_prompt_cache():
            self._print_error("❌ This provider does not support prompt caching")
            return
        filename = self._force_extension(filename, ".safetensors")

        key = self.prompt_cache_key
        if not isinstance(key, str) or not key.strip():
            self._print_error("❌ No active prompt cache key; start chatting first or /clear to re-init caching")
            return

        meta: Dict[str, str] = {
            "format": "abstractcore-cli-prompt-cache/v1",
            "provider": str(self.provider_name),
            "model": str(getattr(self.provider, "model", self.model_name)),
            "saved_at": datetime.now().isoformat(),
        }

        try:
            result = getattr(self.provider, "prompt_cache_save")(key, filename, q8=bool(q8), meta=meta)
            self.prompt_cache_file = filename
            extra = ""
            if isinstance(result, dict):
                out_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
                tok = out_meta.get("token_count")
                if isinstance(tok, str) and tok.strip().isdigit():
                    tok = int(tok.strip())
                if isinstance(tok, int) and tok > 0:
                    extra = f" ({tok} tokens)"
            print(f"💾 Cache saved to {filename}{extra}")
        except Exception as e:
            self._print_error(f"❌ Failed to save prompt cache: {e}")

    def handle_load_prompt_cache(self, filename: str) -> None:
        """Load MLX prompt cache from disk (reads a `.safetensors` file; model-locked)."""
        if not self._is_mlx_provider():
            self._print_error("❌ KV cache load is only supported for provider 'mlx'")
            return
        if not self._supports_prompt_cache():
            self._print_error("❌ This provider does not support prompt caching")
            return
        resolved = self._resolve_cache_path(filename)
        if not resolved:
            self._print_error(f"❌ File not found: {self._force_extension(filename, '.safetensors')}")
            return

        # Clear existing caches and install the loaded cache under a fresh key.
        try:
            getattr(self.provider, "prompt_cache_clear")(None)
        except Exception:
            pass

        try:
            result = getattr(self.provider, "prompt_cache_load")(resolved, make_default=True)
        except Exception as e:
            msg = str(e)
            if "Prompt cache model mismatch" in msg:
                current_model = str(getattr(self.provider, "model", self.model_name))
                self._print_error(
                    "❌ Prompt cache model mismatch:\n"
                    f"   current model: {current_model}\n"
                    f"   detail: {msg}\n"
                    f"   hint: run `/model mlx:<model>` then `/cache load {self._force_extension(filename, '.safetensors')}`"
                )
            else:
                self._print_error(f"❌ Failed to load prompt cache: {e}")
            return

        self.prompt_cache_mode = "kv"
        self.prompt_cache_key = result.get("key") if isinstance(result, dict) else None
        self.prompt_cache_file = resolved

        # Reset transcript; the cache becomes the source of truth for context.
        self.session.clear_history(keep_system=False)
        out_meta = result.get("meta") if isinstance(result, dict) and isinstance(result.get("meta"), dict) else {}
        token_count = out_meta.get("token_count")
        if isinstance(token_count, str) and token_count.strip().isdigit():
            token_count = int(token_count.strip())
        token_note = f" ({token_count} tokens)" if isinstance(token_count, int) and token_count > 0 else ""
        print(f"📂 Cache loaded from {resolved}{token_note} (key={self.prompt_cache_key})")

        cache_format = out_meta.get("format") if isinstance(out_meta, dict) else None
        force_refresh = cache_format != "abstractcore-cli-prompt-cache/v1"
        if force_refresh and not self.single_prompt_mode:
            self._print_warn(
                "⚠️ Loaded cache has no AbstractCore CLI metadata; it may not include tool specs.\n"
                "   Injecting current CLI tool definitions into the KV cache for recency."
            )
        self._kv_refresh_tools_if_needed(reason="cache load", force=force_refresh)

    def handle_compact(self, focus: Optional[str] = None):
        """Handle /compact [focus] command - compact chat history with optional focus"""
        messages = self.session.get_messages()

        if len(messages) <= 3:  # System + minimal conversation
            print("📝 Not enough history to compact (need at least 2 exchanges)")
            return

        try:
            # Display what we're doing
            if focus:
                print(f"🗜️  Compacting chat history with focus: '{focus}'...")
            else:
                print("🗜️  Compacting chat history...")
            print(f"   Before: {len(messages)} messages (~{self.session.get_token_estimate()} tokens)")

            # Create compact provider using gemma3:1b-it-qat for fast, local processing
            try:
                from .. import create_llm
                compact_provider = create_llm("ollama", model="gemma3:1b-it-qat")
                print("   Using gemma3:1b-it-qat for compaction...")
            except Exception as e:
                print(f"⚠️  Could not create gemma3:1b-it-qat provider: {e}")
                print("   Using current provider instead...")
                compact_provider = None

            start_time = time.time()

            # Perform in-place compaction with optional focus
            compacted = self.session.compact(
                preserve_recent=4,  # Keep last 4 messages (2 exchanges)
                focus=focus or "key information and ongoing context",
                compact_provider=compact_provider,
                reason="user_requested",
            )
            # Replace current session with compacted version (in-place).
            try:
                self.session._replace_with_compacted(compacted)
            except Exception:
                self.session = compacted

            duration = time.time() - start_time

            print(f"✅ Compaction completed in {duration:.1f}s")
            print(f"   After: {len(self.session.get_messages())} messages (~{self.session.get_token_estimate()} tokens)")

            # Show compacted structure
            messages_after = self.session.get_messages()
            print("   Structure:")
            for i, msg in enumerate(messages_after):
                if msg.role == 'system':
                    if '[CONVERSATION HISTORY]' in msg.content:
                        print(f"   {i+1}. 📚 Conversation summary ({len(msg.content)} chars)")
                    else:
                        print(f"   {i+1}. ⚙️  System prompt")
                elif msg.role == 'user':
                    preview = preview_text(msg.content, max_chars=50)
                    print(f"   {i+1}. 👤 {preview}")
                elif msg.role == 'assistant':
                    preview = preview_text(msg.content, max_chars=50)
                    print(f"   {i+1}. 🤖 {preview}")

            print("   💡 Note: Token count may increase initially due to detailed summary")
            print("       but will decrease significantly as conversation continues")

        except Exception as e:
            print(f"❌ Compaction failed: {e}")

    def handle_facts(self, filename: str = None):
        """Handle /facts [file] command - extract facts from conversation history"""
        messages = self.session.get_messages()

        if len(messages) <= 1:  # Only system message
            print("📝 No conversation history to extract facts from")
            return

        try:
            print("🔍 Extracting facts from conversation history...")

            # Create fact extractor using current provider for consistency
            extractor = BasicExtractor(self._analysis_provider())

            # Format conversation history as text
            conversation_text = self._format_conversation_for_extraction(messages)

            if not conversation_text.strip():
                print("📝 No substantive conversation content found")
                return

            print(f"   Processing {len(conversation_text)} characters of conversation...")

            start_time = time.time()

            if filename is None:
                # Display facts as triples in chat
                result = extractor.extract(conversation_text, output_format="triples")

                duration = time.time() - start_time
                print(f"✅ Fact extraction completed in {duration:.1f}s")

                if result and result.get("simple_triples"):
                    print("\n📋 Facts extracted from conversation:")
                    print("=" * 50)
                    for i, triple in enumerate(result["simple_triples"], 1):
                        print(f"{i:2d}. {triple}")
                    print("=" * 50)

                    stats = result.get("statistics", {})
                    entities_count = stats.get("entities_count", 0)
                    relationships_count = stats.get("relationships_count", 0)
                    print(f"📊 Found {entities_count} entities and {relationships_count} relationships")
                else:
                    print("❌ No facts could be extracted from the conversation")

            else:
                # Save as JSON-LD file
                result = extractor.extract(conversation_text, output_format="jsonld")

                duration = time.time() - start_time
                print(f"✅ Fact extraction completed in {duration:.1f}s")

                if result and result.get("@graph"):
                    # Ensure filename has .jsonld extension
                    if not filename.endswith('.jsonld'):
                        filename = f"{filename}.jsonld"

                    import json
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    entities = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('e:')]
                    relationships = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('r:')]

                    print(f"💾 Facts saved to {filename}")
                    print(f"📊 Saved {len(entities)} entities and {len(relationships)} relationships as JSON-LD")
                else:
                    print("❌ No facts could be extracted from the conversation")

        except Exception as e:
            print(f"❌ Fact extraction failed: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_judge(self):
        """Handle /judge command - evaluate conversation quality and provide feedback"""
        messages = self.session.get_messages()

        if len(messages) <= 1:  # Only system message
            print("📝 No conversation history to evaluate")
            return

        try:
            print("⚖️  Evaluating conversation quality...")

            # Create judge using current provider for consistency
            judge = BasicJudge(self._analysis_provider())

            # Format conversation history as text
            conversation_text = self._format_conversation_for_extraction(messages)

            if not conversation_text.strip():
                print("📝 No substantive conversation content found")
                return

            print(f"   Analyzing {len(conversation_text)} characters of conversation...")

            start_time = time.time()

            # Evaluate the conversation with focus on discussion quality
            from ..processing.basic_judge import JudgmentCriteria
            criteria = JudgmentCriteria(
                is_clear=True,       # How clear is the discussion
                is_coherent=True,    # How well does it flow
                is_actionable=True,  # Does it provide useful insights
                is_relevant=True,    # Is the discussion focused
                is_complete=True,    # Does it address the topics thoroughly
                is_innovative=False, # Not focused on innovation for general chat
                is_working=False,    # Not applicable to conversation
                is_sound=True,       # Are the arguments/explanations sound
                is_simple=True       # Is the communication clear and accessible
            )

            assessment = judge.evaluate(
                content=conversation_text,
                context="conversational discussion quality",
                criteria=criteria
            )

            duration = time.time() - start_time
            print(f"✅ Evaluation completed in {duration:.1f}s")

            # Display judge's summary first (most important)
            judge_summary = assessment.get('judge_summary', '')
            if judge_summary:
                print(f"\n📝 Judge's Assessment:")
                print(f"   {judge_summary}")

            # Source reference
            source_ref = assessment.get('source_reference', '')
            if source_ref:
                print(f"\n📄 Source: {source_ref}")

            # Display assessment in a conversational format
            overall_score = assessment.get('overall_score', 0)
            print(f"\n📊 Overall Discussion Quality: {overall_score}/5")

            # Show key dimension scores
            key_scores = [
                ('clarity_score', 'Clarity'),
                ('coherence_score', 'Coherence'),
                ('actionability_score', 'Actionability'),
                ('relevance_score', 'Relevance'),
                ('completeness_score', 'Completeness'),
                ('soundness_score', 'Soundness'),
                ('simplicity_score', 'Simplicity')
            ]

            print("\n📈 Quality Dimensions:")
            for field, label in key_scores:
                score = assessment.get(field)
                if score is not None:
                    print(f"   {label:13}: {score}/5")

            # Show strengths
            strengths = assessment.get('strengths', [])
            if strengths:
                print(f"\n✅ Conversation Strengths:")
                for strength in strengths[:3]:  # Show top 3
                    print(f"   • {strength}")

            # Show improvement suggestions
            feedback = assessment.get('actionable_feedback', [])
            if feedback:
                print(f"\n💡 Suggestions for Better Discussions:")
                for suggestion in feedback[:3]:  # Show top 3
                    print(f"   • {suggestion}")

            # Show brief reasoning (shortened for chat)
            reasoning = assessment.get('reasoning', '')
            if reasoning:
                # Extract first few sentences of reasoning
                sentences = reasoning.split('. ')
                brief_reasoning = '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else reasoning
                print(f"\n🤔 Assessment Summary:")
                print(f"   {brief_reasoning}")

            print(f"\n📌 Note: This is a demonstrator showing LLM-as-a-judge capabilities for objective assessment.")

        except Exception as e:
            print(f"❌ Conversation evaluation failed: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_intent(self, focus_participant: str = None):
        """Handle /intent [participant] command - analyze intents behind conversation"""
        messages = self.session.get_messages()

        if len(messages) <= 1:  # Only system message
            print("📝 No conversation history to analyze intents from")
            return

        try:
            if focus_participant:
                print(f"🎯 Analyzing {focus_participant} intents in conversation...")
            else:
                print("🎯 Analyzing conversation intents for all participants...")

            # Create intent analyzer using current provider for consistency
            analyzer = BasicIntentAnalyzer(self._analysis_provider())

            # Convert session messages to the format expected by intent analyzer
            conversation_messages = [msg for msg in messages if msg.role != 'system']
            message_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_messages]

            if not message_dicts:
                print("📝 No substantive conversation content found")
                return

            print(f"   Processing {len(message_dicts)} messages...")

            start_time = time.time()

            # Analyze conversation intents
            from ..processing.basic_intent import IntentDepth
            results = analyzer.analyze_conversation_intents(
                messages=message_dicts,
                focus_participant=focus_participant,
                depth=IntentDepth.UNDERLYING
            )

            duration = time.time() - start_time
            print(f"✅ Intent analysis completed in {duration:.1f}s")

            if not results:
                print("❌ No intents could be analyzed from the conversation")
                return

            # Display results in a conversational format
            print("\n🎯 CONVERSATION INTENT ANALYSIS")
            print("=" * 60)

            for participant, analysis in results.items():
                print(f"\n👤 {participant.upper()} INTENTS:")
                print("─" * 40)
                
                # Primary Intent
                primary = analysis.primary_intent
                print(f"🎯 Primary Intent: {primary.intent_type.value.replace('_', ' ').title()}")
                print(f"   Description: {primary.description}")
                print(f"   Underlying Goal: {primary.underlying_goal}")
                print(f"   Emotional Undertone: {primary.emotional_undertone}")
                print(f"   Confidence: {primary.confidence:.2f} | Urgency: {primary.urgency_level:.2f}")
                
                # Secondary Intents (show top 2 for brevity)
                if analysis.secondary_intents:
                    print(f"\n🔄 Secondary Intents:")
                    for i, intent in enumerate(analysis.secondary_intents[:2], 1):
                        print(f"   {i}. {intent.intent_type.value.replace('_', ' ').title()}")
                        print(f"      Goal: {intent.underlying_goal}")
                        print(f"      Confidence: {intent.confidence:.2f}")
                
                # Key contextual factors (show top 3)
                if analysis.contextual_factors:
                    print(f"\n🌍 Key Context Factors:")
                    for factor in analysis.contextual_factors[:3]:
                        print(f"   • {factor}")
                
                # Response approach
                print(f"\n💡 Suggested Response Approach:")
                # Truncate long response approaches for readability
                response_approach = analysis.suggested_response_approach
                if len(response_approach) > 200:
                    response_approach = preview_text(response_approach, max_chars=200)
                print(f"   {response_approach}")
                
                # Analysis metadata
                print(f"\n📊 Analysis: {analysis.word_count_analyzed} words | "
                      f"Complexity: {analysis.intent_complexity:.2f} | "
                      f"Confidence: {analysis.overall_confidence:.2f} | "
                      f"Time: {duration:.1f}s")

            print("\n" + "=" * 60)
            print("💡 Note: This analysis identifies underlying motivations and goals behind communication")

        except Exception as e:
            print(f"❌ Intent analysis failed: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def _format_conversation_for_extraction(self, messages):
        """Format conversation messages for fact extraction"""
        formatted_lines = []

        for msg in messages:
            # Skip system messages for fact extraction
            if msg.role == 'system':
                continue

            content = msg.content.strip()
            if not content:
                continue

            if msg.role == 'user':
                formatted_lines.append(f"User: {content}")
            elif msg.role == 'assistant':
                formatted_lines.append(f"Assistant: {content}")

        return "\n\n".join(formatted_lines)

    def handle_history(self, n_interactions: int = None):
        """Handle /history [n] command - show conversation history verbatim"""
        messages = self.session.get_messages()

        if not messages:
            print("📝 No conversation history")
            return

        # Check for conversation summary (from compaction)
        summary_message = None
        for msg in messages:
            if msg.role == 'system' and '[CONVERSATION HISTORY]' in msg.content:
                summary_message = msg
                break

        # Filter out system messages for interaction counting
        conversation_messages = [msg for msg in messages if msg.role != 'system']

        if not conversation_messages and not summary_message:
            print("📝 No conversation history")
            return

        if n_interactions is None:
            # Show all conversation
            print("📜 Conversation History:\n")
            display_messages = conversation_messages
        else:
            # Show last n interactions (each interaction = user + assistant)
            # Calculate how many messages that represents
            messages_needed = n_interactions * 2  # user + assistant per interaction
            display_messages = conversation_messages[-messages_needed:] if messages_needed <= len(conversation_messages) else conversation_messages
            print(f"📜 Last {n_interactions} interactions:\n")

        # Show conversation summary if it exists (from compaction)
        if summary_message:
            summary_content = summary_message.content.replace('[CONVERSATION HISTORY]: ', '')
            print("📚 Earlier Conversation Summary:")
            print("─" * 50)
            print(summary_content)
            print("─" * 50)
            print()

        # Display the recent messages verbatim without numbers
        if display_messages:
            if summary_message:
                print("💬 Recent Conversation:")
                print()

            for msg in display_messages:
                if msg.role == 'user':
                    print("👤 You:")
                    print(msg.content)
                    print()  # Empty line after user message
                elif msg.role == 'assistant':
                    print("🤖 Assistant:")
                    print(msg.content)
                    print()  # Empty line after assistant message
        elif summary_message:
            print("💡 Only summary available - recent messages were preserved but may have been cleared")

        print(f"📊 Total tokens estimate: ~{self.session.get_token_estimate()}")

    def handle_system_show(self):
        """Show current system prompt - both fixed part and full prompt with tools"""
        # Get the original system prompt (fixed part)
        fixed_prompt = self.session.system_prompt or "No system prompt set"

        print("⚙️  Current System Prompt:")
        print("=" * 50)
        print(f"📝 Fixed Part:\n{fixed_prompt}\n")

        # Show full prompt as it appears to the LLM (including tool descriptions)
        messages = self.session.get_messages()
        system_messages = [msg for msg in messages if msg.role == 'system']

        if system_messages:
            print("🔧 Full Prompt (as seen by LLM):")
            for i, sys_msg in enumerate(system_messages, 1):
                if i == 1:
                    print(f"System Message {i} (Base):")
                else:
                    print(f"System Message {i}:")
                print(f"{sys_msg.content}")
                if i < len(system_messages):
                    print()  # Separator between system messages
        else:
            print("⚠️  No system messages found in session")

        print("=" * 50)

    def handle_system_change(self, new_prompt: str):
        """Change the system prompt (fixed part only, preserves tools)"""
        old_prompt = self.session.system_prompt or "No previous prompt"

        # Update the session's system prompt
        self.session.system_prompt = new_prompt

        # Update the first system message in the session if it exists
        messages = self.session.get_messages()
        for msg in messages:
            if msg.role == 'system' and not msg.content.startswith('[CONVERSATION HISTORY]'):
                # This is the original system message, update it
                msg.content = new_prompt
                break
        else:
            # No existing system message, add one at the beginning
            created = self.session.add_message('system', new_prompt)
            # add_message appends; move the created system message to the front for correct ordering.
            try:
                self.session.messages.remove(created)
            except Exception:
                pass
            self.session.messages.insert(0, created)

        print("✅ System prompt updated!")
        print(f"📝 Old: {old_prompt[:100]}{'...' if len(old_prompt) > 100 else ''}")
        print(f"📝 New: {new_prompt[:100]}{'...' if len(new_prompt) > 100 else ''}")

        if self.prompt_cache_mode == "kv":
            self._print_warn("⚠️ KV prompt cache invalidated by system prompt change; clearing cache and context")
            self.handle_clear()

    def handle_save(self, filename: str, summary: bool = False, assessment: bool = False, facts: bool = False):
        """Handle /save <file> command - save current session to file with optional analytics"""
        try:
            filename = self._force_extension(filename, ".json")
            
            print(f"💾 Saving session to {filename}...")
            
            # Get session info before saving
            messages = self.session.get_messages()
            tokens = self.session.get_token_estimate()
            
            # Generate optional analytics if requested
            analytics_generated = []
            analysis_provider = self._analysis_provider()
            
            if summary:
                print("   🔄 Generating summary...")
                try:
                    self.session.generate_summary(focus="key discussion points", compact_provider=analysis_provider)
                    analytics_generated.append("summary")
                    print("   ✅ Summary generated")
                except Exception as e:
                    print(f"   ⚠️  Summary generation failed: {e}")
            
            if assessment:
                print("   🔄 Generating assessment...")
                original_provider = None
                try:
                    original_provider = self.session.provider
                    self.session.provider = analysis_provider
                    self.session.generate_assessment()
                    self.session.provider = original_provider
                    analytics_generated.append("assessment")
                    print("   ✅ Assessment generated")
                except Exception as e:
                    try:
                        if original_provider is not None:
                            self.session.provider = original_provider
                    except Exception:
                        pass
                    print(f"   ⚠️  Assessment generation failed: {e}")
            
            if facts:
                print("   🔄 Extracting facts...")
                original_provider = None
                try:
                    original_provider = self.session.provider
                    self.session.provider = analysis_provider
                    self.session.extract_facts()
                    self.session.provider = original_provider
                    analytics_generated.append("facts")
                    print("   ✅ Facts extracted")
                except Exception as e:
                    try:
                        if original_provider is not None:
                            self.session.provider = original_provider
                    except Exception:
                        pass
                    print(f"   ⚠️  Fact extraction failed: {e}")
            
            # Save using enhanced serialization
            self.session.save(filename)
            
            print(f"✅ Session saved successfully!")
            print(f"   📁 File: {filename}")
            print(f"   📝 Messages: {len(messages)}")
            print(f"   🔢 Tokens: ~{tokens:,}")
            print(f"   🤖 Provider: {self.provider_name}:{self.model_name}")
            print(f"   ⚙️  Settings: auto_compact={self.session.auto_compact}")
            
            if analytics_generated:
                print(f"   📊 Analytics: {', '.join(analytics_generated)}")
            
            # Note about provider restoration
            print(f"   💡 Note: Provider and tools will need to be specified when loading")
            
        except Exception as e:
            print(f"❌ Failed to save session: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def handle_load(self, filename: str):
        """Handle /load <file> command - load session from file"""
        try:
            resolved = self._resolve_session_path(filename) or self._force_extension(filename, ".json")
            
            # Check if file exists
            import os
            if not os.path.exists(resolved):
                print(f"❌ File not found: {resolved}")
                return
            
            print(f"📂 Loading session from {resolved}...")
            
            # Store current session info for comparison
            old_messages = len(self.session.get_messages())
            old_tokens = self.session.get_token_estimate()
            
            # Load session with current provider and tools
            from ..tools.common_tools import list_files, read_file, write_file, execute_command, search_files
            tools = [list_files, read_file, write_file, execute_command, search_files]
            
            loaded_session = BasicSession.load(resolved, provider=self.provider, tools=tools)
            
            # Replace current session
            self.session = loaded_session

            # If we're in local KV cache mode (MLX), rebuild the cache from the loaded transcript so
            # the model context matches what the user sees.
            if self._is_mlx_provider() and self._supports_prompt_cache():
                try:
                    self.prompt_cache_mode = "kv"
                    self._rebuild_kv_cache_from_session()
                except Exception as e:
                    self._print_warn(f"⚠️ KV cache rebuild from session failed; continuing without KV mode: {e}")
                    self.prompt_cache_mode = "key"

            # Get new session info
            new_messages = len(self.session.get_messages())
            new_tokens = self.session.get_token_estimate()
            
            print(f"✅ Session loaded successfully!")
            print(f"   📁 File: {resolved}")
            print(f"   📝 Messages: {old_messages} → {new_messages}")
            print(f"   🔢 Tokens: ~{old_tokens:,} → ~{new_tokens:,}")
            print(f"   🤖 Provider: {self.provider_name}:{self.model_name} (current)")
            print(f"   ⚙️  Settings: auto_compact={self.session.auto_compact}")
            
            # Show session structure
            messages = self.session.get_messages()
            conversation_messages = [msg for msg in messages if msg.role != 'system']
            interactions = len(conversation_messages) // 2
            
            has_summary = any(msg.role == 'system' and '[CONVERSATION HISTORY]' in msg.content for msg in messages)
            if has_summary:
                print(f"   📚 History: Compacted conversation with {interactions} recent interactions")
            else:
                print(f"   💬 History: Full conversation with {interactions} interactions")
            
            # Show timestamps if available
            if messages:
                first_msg = next((msg for msg in messages if msg.role != 'system'), None)
                if first_msg and hasattr(first_msg, 'timestamp') and first_msg.timestamp:
                    print(f"   📅 Created: {first_msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Failed to load session: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def _rebuild_kv_cache_from_session(self) -> None:
        """Best-effort rebuild of the local KV prompt cache from the current session transcript."""
        if not self._is_mlx_provider():
            return
        if not self._supports_prompt_cache():
            return

        # Fresh cache key for the rebuilt state.
        try:
            getattr(self.provider, "prompt_cache_clear")(None)
        except Exception:
            pass

        key = f"cli:{uuid.uuid4().hex[:12]}"
        ok = False
        try:
            ok = bool(getattr(self.provider, "prompt_cache_set")(key, make_default=True))
        except Exception:
            ok = False

        if not ok:
            self.prompt_cache_mode = "off"
            self.prompt_cache_key = None
            raise RuntimeError("provider failed to create a prompt cache")

        # Prefill stable modules.
        try:
            getattr(self.provider, "prompt_cache_update")(
                key,
                system_prompt=self.session.system_prompt,
                tools=self.session.tools,
                add_generation_prompt=False,
            )
        except Exception as e:
            raise RuntimeError(f"failed to prefill system/tools: {e}") from e

        # Append any additional transcript messages (excluding the main system prompt we just prefixed).
        messages_to_append: List[Dict[str, Any]] = []
        for msg in self.session.get_messages():
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            if role == "system":
                if isinstance(self.session.system_prompt, str) and content == self.session.system_prompt and not str(content).startswith("[CONVERSATION HISTORY]"):
                    continue
            if role and content is not None:
                messages_to_append.append({"role": role, "content": content})

        if messages_to_append:
            try:
                getattr(self.provider, "prompt_cache_update")(
                    key,
                    messages=messages_to_append,
                    add_generation_prompt=False,
                )
            except Exception as e:
                raise RuntimeError(f"failed to append transcript messages: {e}") from e

        self.prompt_cache_key = key
        self.prompt_cache_file = None
        self.prompt_cache_mode = "kv"
        print(f"🧠 KV prompt cache rebuilt from session (key={key}, messages={len(messages_to_append)})")
        self._kv_refresh_tools_if_needed(reason="session rebuild", force=False)

    def handle_tooltag_test(self, opening_tag: str, closing_tag: str):
        """Handle /tooltag command - demonstrate tool call format handling"""
        print(f"🏷️ Tool call format testing: {opening_tag}...{closing_tag}")
        print("💡 Note: CLI now uses universal tool call parser that handles multiple formats automatically")
        print("   Supported formats: Qwen3, LLaMA3, XML, Gemma, and plain JSON")
        print("   No configuration needed - detection is automatic!")

    def handle_status(self):
        """Handle /status command - show comprehensive system status"""
        print("📊 AbstractCore CLI Status")
        print("=" * 60)

        # Provider and Model info
        print(f"🔧 Provider: {self.provider_name}")
        print(f"🤖 Model: {self.model_name}")
        print(f"🌊 Streaming: {'Enabled' if self.stream_mode else 'Disabled'}")
        thinking_label = "auto" if self.thinking is None else ("on" if self.thinking is True else "off" if self.thinking is False else str(self.thinking))
        print(f"🧠 Thinking: {thinking_label}")
        show_reasoning_label = "auto" if self.show_reasoning is None else ("on" if self.show_reasoning else "off")
        print(f"🧠 Show reasoning: {show_reasoning_label}")
        if self.prompt_cache_mode != "off":
            cache_details = f"mode={self.prompt_cache_mode}"
            if self.prompt_cache_key:
                cache_details += f" key={self.prompt_cache_key}"
            if self.prompt_cache_file:
                cache_details += f" file={self.prompt_cache_file}"
            print(f"🧠 Prompt caching: {cache_details}")
            try:
                if hasattr(self.provider, "get_prompt_cache_stats"):
                    stats = self.provider.get_prompt_cache_stats()
                    if isinstance(stats, dict):
                        entries = stats.get("entries")
                        max_entries = stats.get("max_entries")
                        if entries is not None and max_entries is not None:
                            print(f"   Cache store: {entries}/{max_entries} entries")
            except Exception:
                pass
        else:
            print("🧠 Prompt caching: off")

        # Debug status - show both CLI and system logging
        print(f"🐛 CLI Debug: {'Enabled' if self.debug_mode else 'Disabled'}")

        # Try to detect system logging level
        try:
            import logging
            logger = logging.getLogger()
            current_level = logger.getEffectiveLevel()
            level_name = logging.getLevelName(current_level)

            # Check if debug messages would be shown
            if current_level <= logging.DEBUG:
                system_debug = "Enabled (DEBUG level)"
            elif current_level <= logging.INFO:
                system_debug = "Info level"
            else:
                system_debug = "Warning+ only"

            print(f"📊 System Logging: {system_debug}")
        except:
            print(f"📊 System Logging: Unknown")

        # Token usage
        current_tokens = self.session.get_token_estimate()
        print(f"💾 Context Usage: {current_tokens:,} / {self.max_tokens:,} tokens ({(current_tokens/self.max_tokens*100):.1f}%)")
        print(f"✍️ Max Output Tokens: {self.max_output_tokens:,}")

        # Model capabilities
        try:
            from ..architectures.detection import get_model_capabilities
            capabilities = get_model_capabilities(self.model_name)

            print("\n🎯 Model Capabilities:")
            print(f"   Max Input Tokens: {capabilities.get('max_tokens', 'Unknown'):,}")
            print(f"   Max Output Tokens: {capabilities.get('max_output_tokens', 'Unknown'):,}")
            print(f"   Tool Support: {capabilities.get('tool_support', 'Unknown')}")
            print(f"   Structured Output: {capabilities.get('structured_output', 'Unknown')}")
            print(f"   Vision Support: {'Yes' if capabilities.get('vision_support', False) else 'No'}")
            print(f"   Audio Support: {'Yes' if capabilities.get('audio_support', False) else 'No'}")
            print(f"   Thinking Support: {'Yes' if capabilities.get('thinking_support', False) else 'No'}")
            reasoning_levels = capabilities.get("reasoning_levels")
            if isinstance(reasoning_levels, list) and reasoning_levels:
                levels_str = ", ".join([str(x) for x in reasoning_levels if isinstance(x, str) and x.strip()])
                if levels_str:
                    print(f"   Reasoning Levels: {levels_str}")

            # Show aliases if any
            aliases = capabilities.get('aliases', [])
            if aliases:
                print(f"   Model Aliases: {', '.join(aliases)}")

        except Exception as e:
            print(f"\n⚠️ Could not retrieve model capabilities: {e}")

        # Available tools
        print("\n🛠️ Available Tools:")
        tools = ["list_files", "search_files", "read_file", "write_file", "execute_command"]
        for i, tool in enumerate(tools, 1):
            print(f"   {i}. {tool}")

        # Session info
        messages = self.session.get_messages()
        conversation_messages = [msg for msg in messages if msg.role != 'system']
        interactions = len(conversation_messages) // 2  # user + assistant = 1 interaction

        print(f"\n📝 Session Info:")
        print(f"   Total Messages: {len(messages)}")
        print(f"   Interactions: {interactions}")
        print(f"   System Prompt: {'Set' if self.session.system_prompt else 'Default'}")

        # Check for compaction
        has_summary = any(msg.role == 'system' and '[CONVERSATION HISTORY]' in msg.content for msg in messages)
        if has_summary:
            print(f"   History: Compacted (summary available)")
        else:
            print(f"   History: Full conversation")

        print("=" * 60)

    def _parse_file_attachments(self, user_input: str):
        """Parse @filename references using AbstractCore's message preprocessor."""
        from ..utils.message_preprocessor import MessagePreprocessor
        import os

        # Use AbstractCore's centralized file parsing logic
        clean_input, media_files = MessagePreprocessor.parse_file_attachments(
            user_input,
            validate_existence=True,
            verbose=self.debug_mode
        )

        # Show user-friendly status messages for CLI (only in interactive mode)
        if media_files and not self.single_prompt_mode:
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

            # Check for audio capabilities if audio is attached
            audio_extensions = {
                ".wav",
                ".mp3",
                ".m4a",
                ".aac",
                ".flac",
                ".ogg",
                ".opus",
                ".wma",
                ".aiff",
                ".aif",
                ".caf",
            }
            audio_files = [
                f for f in media_files if os.path.splitext(f.lower())[1] in audio_extensions
            ]
            if audio_files:
                try:
                    from ..media.capabilities import get_media_capabilities

                    supports_audio = bool(get_media_capabilities(self.model_name).audio_support)
                except Exception:
                    supports_audio = False

                effective_policy = self.audio_policy
                if effective_policy is None:
                    try:
                        from ..config.manager import get_config_manager

                        effective_policy = getattr(getattr(get_config_manager().config, "audio", None), "strategy", None)
                    except Exception:
                        effective_policy = "auto"
                effective_policy = str(effective_policy or "auto").strip().lower()
                if supports_audio:
                    print(f"🔊 Audio-capable model detected - will process {len(audio_files)} audio file(s)")
                elif effective_policy in {"native_only", "native", "disabled"}:
                    print(
                        "🎧 Text model - audio input is not supported (native_only). "
                        "Use --audio-policy auto or speech_to_text (requires `pip install abstractvoice`)."
                    )
                else:
                    stt_available = False
                    try:
                        stt_available = bool(
                            self.provider.capabilities.status()
                            .get("capabilities", {})
                            .get("audio", {})
                            .get("available", False)
                        )
                    except Exception:
                        stt_available = False

                    if stt_available:
                        print(
                            f"🎧 Text model - will use speech-to-text fallback for {len(audio_files)} audio file(s)"
                        )
                    else:
                        print(
                            "🎧 Text model - speech-to-text fallback requires installing `abstractvoice`: "
                            'pip install abstractvoice (or pass --audio-policy native_only)'
                        )

        return clean_input, media_files

    def generate_response(self, user_input: str):
        """Generate and display response with tool execution and file attachment support."""
        import os
        import re
        start_time = time.time()

        try:
            # Parse @filename attachments
            clean_input, media_files = self._parse_file_attachments(user_input)

            # If no text remains after removing file references, provide default prompt
            if not clean_input and media_files:
                clean_input = "Please analyze the attached file(s)."

            clean_input = self._timestamp_user_message(clean_input)

            if self.debug_mode:
                print(f"🔍 Sending to {self.provider_name}:{self.model_name}")
                if media_files:
                    print(f"🔍 Media files: {media_files}")

            if self.prompt_cache_mode == "kv":
                response = self._generate_response_kv(
                    clean_input,
                    media=media_files if media_files else None,
                )
            else:
                # Generate response with media support (session-managed history)
                gen_kwargs: Dict[str, Any] = {
                    "stream": self.stream_mode,
                    "media": media_files if media_files else None,
                    "max_output_tokens": self.max_output_tokens,
                }
                if media_files:
                    audio_exts = {
                        ".wav",
                        ".mp3",
                        ".m4a",
                        ".aac",
                        ".flac",
                        ".ogg",
                        ".opus",
                        ".wma",
                        ".aiff",
                        ".aif",
                        ".caf",
                    }
                    has_audio = any(
                        os.path.splitext(str(f).lower())[1] in audio_exts for f in (media_files or [])
                    )
                else:
                    has_audio = False

                if self.audio_policy is not None and has_audio:
                    gen_kwargs["audio_policy"] = self.audio_policy
                if self.audio_language is not None and has_audio:
                    gen_kwargs["audio_language"] = self.audio_language
                if self.thinking is not None:
                    gen_kwargs["thinking"] = self.thinking
                response = self.session.generate(clean_input, **gen_kwargs)

            if self.stream_mode:
                show_reasoning = self._should_show_reasoning() and not self.single_prompt_mode
                buffer_for_reasoning_first = self._should_buffer_stream_for_reasoning_first()
                if not self.single_prompt_mode and not buffer_for_reasoning_first:
                    print("🤖 Assistant: ", end="", flush=True)
                full_content = ""
                display_buffer = ""  # Buffer for cleaned display content
                reasoning_parts: List[str] = []
                
                for chunk in response:
                    if hasattr(chunk, 'content') and chunk.content:
                        full_content += chunk.content
                        
                        # Filter out internal model tags that shouldn't appear
                        # These tags indicate model formatting issues
                        chunk_text = chunk.content
                        
                        # Remove internal conversation tags
                        chunk_text = re.sub(r'<\|assistant\|>', '', chunk_text)
                        chunk_text = re.sub(r'<\|user\|>', '', chunk_text)
                        chunk_text = re.sub(r'<\|system\|>', '', chunk_text)
                        
                        # For now, don't display tool calls during streaming
                        # We'll show them after execution
                        # Check if this chunk contains tool call markers
                        has_tool_marker = any(marker in chunk_text for marker in [
                            '<|tool_call|>', '</|tool_call|>',
                            '<function_call>', '</function_call>',
                            '<tool_call>', '</tool_call>',
                            '```tool_code'
                        ])
                        
                        # If we want reasoning-first display, buffer output (no live streaming).
                        if buffer_for_reasoning_first:
                            display_buffer += chunk_text
                        else:
                            if not has_tool_marker:
                                print(chunk_text, end="", flush=True)
                                display_buffer += chunk_text
                            else:
                                # Buffer the chunk, we'll process after streaming
                                display_buffer += chunk_text

                    # Best-effort: capture streamed reasoning metadata (OpenAI-compatible deltas, etc.).
                    r = getattr(chunk, "reasoning", None)
                    if isinstance(r, str) and r.strip():
                        reasoning_parts.append(r.strip())
                
                if not buffer_for_reasoning_first:
                    print()  # New line after streaming
                
                # Parse and execute tool calls from full content
                clean_content, tool_calls = self._parse_and_strip_tool_calls(full_content)
                if self.prompt_cache_mode == "kv":
                    # Maintain transcript for UX; model context lives in KV cache.
                    try:
                        self.session.add_message("assistant", clean_content.strip() or full_content)
                    except Exception:
                        pass
                
                # If we buffered tool call content, we should have shown clean content
                # For now, if there's significant difference, show the clean version
                if tool_calls and clean_content.strip() and clean_content.strip() != display_buffer.strip():
                    # We had tool calls that weren't displayed cleanly
                    # This happens when tool calls appear mid-stream
                    if self.debug_mode:
                        print(f"\n🔍 Cleaned content differs from streamed content")

                combined = "\n\n".join(reasoning_parts).strip() if reasoning_parts else ""
                if show_reasoning and combined:
                    self._print_reasoning_block(combined)

                # Reasoning-first UX: show the final answer after reasoning (buffered).
                if buffer_for_reasoning_first:
                    if clean_content.strip():
                        print(f"🤖 Assistant: {clean_content}")
                    elif tool_calls and not self.single_prompt_mode:
                        print("🤖 Assistant: ", end="")
                    elif self.single_prompt_mode:
                        print(clean_content or full_content)
                    else:
                        print(f"🤖 Assistant: {clean_content or full_content}")

                self._execute_tool_calls(tool_calls)
            else:
                # Non-streaming: parse content, display clean version, execute tools
                clean_content, tool_calls = self._parse_and_strip_tool_calls(response.content)
                if self.prompt_cache_mode == "kv":
                    try:
                        self.session.add_message("assistant", clean_content.strip() or response.content)
                    except Exception:
                        pass
                
                meta = getattr(response, "metadata", None)
                if self._should_show_reasoning() and not self.single_prompt_mode and isinstance(meta, dict):
                    r = meta.get("reasoning")
                    if isinstance(r, str) and r.strip():
                        self._print_reasoning_block(r.strip())

                # Display only the clean content (without tool call syntax)
                if clean_content.strip():
                    if self.single_prompt_mode:
                        print(clean_content)
                    else:
                        print(f"🤖 Assistant: {clean_content}")
                elif tool_calls:
                    # Only tool calls, no text response
                    if not self.single_prompt_mode:
                        print("🤖 Assistant: ", end="")
                else:
                    # Empty response
                    if self.single_prompt_mode:
                        print(response.content)
                    else:
                        print(f"🤖 Assistant: {response.content}")

                # Execute tool calls
                self._execute_tool_calls(tool_calls)

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

    def _should_show_reasoning(self) -> bool:
        """Decide whether to display reasoning in the CLI output."""
        if self.show_reasoning is not None:
            return bool(self.show_reasoning)
        # Auto: show when present unless explicitly disabled.
        if self.thinking is False:
            return False
        return True

    def _should_buffer_stream_for_reasoning_first(self) -> bool:
        """Decide whether to buffer streaming output to show reasoning before the answer."""
        if self.single_prompt_mode:
            return False
        if not self._should_show_reasoning():
            return False

        # If the user explicitly enabled reasoning display or requested thinking, honor reasoning-first UX.
        if self.show_reasoning is True:
            return True
        if self.thinking is not None and self.thinking is not False:
            return True

        # Auto mode: only buffer when the model is expected to emit a separate reasoning channel.
        try:
            from ..architectures.detection import detect_architecture, get_architecture_format, get_model_capabilities

            caps = get_model_capabilities(self.model_name)
            arch = detect_architecture(self.model_name)
            arch_fmt = get_architecture_format(arch)
        except Exception:
            caps = {}
            arch_fmt = {}

        resp_fmt = str((caps or {}).get("response_format") or "").strip().lower()
        if resp_fmt == "harmony":
            return True

        for src in (caps, arch_fmt):
            if isinstance(src, dict):
                f = src.get("thinking_output_field")
                if isinstance(f, str) and f.strip():
                    return True

        return False

    def _print_reasoning_block(self, reasoning: str) -> None:
        """Print reasoning in a visually distinct style (best-effort)."""
        import sys

        text = reasoning.strip()
        if not text:
            return

        print("🧠 Reasoning:")
        if sys.stdout.isatty():
            # Grey + italic (best-effort; not all terminals support italics).
            print(f"\x1b[90m\x1b[3m{text}\x1b[0m")
        else:
            print(text)

    def _generate_response_kv(self, prompt: str, *, media: Optional[list] = None):
        """Generate response using append-only KV cache mode (local providers only)."""
        import os

        # Maintain a local transcript for UX, but do not send it to the model; the KV cache is source-of-truth.
        try:
            self.session.add_message("user", prompt)
        except Exception:
            pass

        gen_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "messages": None,
            "system_prompt": None,
            "tools": None,  # tools were prefixed into the cache during prefill
            "media": media,
            "stream": bool(self.stream_mode),
            "max_output_tokens": self.max_output_tokens,
        }
        if media:
            audio_exts = {
                ".wav",
                ".mp3",
                ".m4a",
                ".aac",
                ".flac",
                ".ogg",
                ".opus",
                ".wma",
                ".aiff",
                ".aif",
                ".caf",
            }
            has_audio = any(
                os.path.splitext(str(f).lower())[1] in audio_exts for f in (media or [])
            )
        else:
            has_audio = False

        if self.audio_policy is not None and has_audio:
            gen_kwargs["audio_policy"] = self.audio_policy
        if self.audio_language is not None and has_audio:
            gen_kwargs["audio_language"] = self.audio_language
        if self.thinking is not None:
            gen_kwargs["thinking"] = self.thinking
        # Preserve session-level generation parameters for consistency.
        try:
            if getattr(self.session, "temperature", None) is not None:
                gen_kwargs["temperature"] = self.session.temperature
            if isinstance(getattr(self.session, "seed", None), int) and self.session.seed >= 0:
                gen_kwargs["seed"] = self.session.seed
        except Exception:
            pass

        return self.provider.generate(**gen_kwargs)

    def _parse_and_strip_tool_calls(self, content: str):
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
        # IMPORTANT: Use format-agnostic parsing in CLI since models can generate
        # different formats regardless of their architecture
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
            
            # Fallback to regex parsing for multiple formats
            tool_calls = []
            clean_content = content
            
            # Support multiple tool call formats
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
                            # Remove this tool call from content
                            clean_content = re.sub(pattern, '', clean_content, count=1, flags=re.DOTALL)
                    except json.JSONDecodeError:
                        continue
            
            # Clean up extra whitespace
            clean_content = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_content).strip()
            
            return clean_content, tool_calls

    def _execute_tool_calls(self, tool_calls):
        """Execute a list of tool call dictionaries and add results to session history."""
        if not tool_calls:
            return
        
        if not self.single_prompt_mode:
            print("\n🔧 Tool Results:")
        
        # Available tools mapping
        available_tools = {
            "list_files": list_files,
            "search_files": search_files,
            "read_file": read_file,
            "write_file": write_file,
            "execute_command": execute_command
        }
        
        for tool_data in tool_calls:
            try:
                tool_name = tool_data.get("name")
                tool_args = tool_data.get("arguments", {})
                
                if tool_name not in available_tools:
                    error_msg = f"❌ Unknown tool: {tool_name}"
                    print(error_msg)
                    # Add error as tool message to session
                    self.session.add_message('tool', error_msg, 
                                           call_id=tool_data.get("call_id"),
                                           status="error",
                                           tool_name=tool_name)
                    continue
                
                # Display tool call for transparency (only in interactive mode)
                if not self.single_prompt_mode:
                    args_str = str(tool_args) if tool_args else "{}"
                    if len(args_str) > 100:
                        args_str = preview_text(args_str, max_chars=100)
                    print(f"**{tool_name}({args_str})**")
                
                # Execute the tool
                tool_function = available_tools[tool_name]
                
                start_time = time.time()
                try:
                    if tool_args:
                        result = tool_function(**tool_args)
                    else:
                        result = tool_function()
                    
                    execution_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Add successful tool result to session history
                    self.session.add_message('tool', str(result),
                                           call_id=tool_data.get("call_id"),
                                           status="ok",
                                           duration_ms=execution_time,
                                           tool_name=tool_name,
                                           tool_arguments=tool_args)
                    
                    # In single-prompt mode, just print the result cleanly
                    if self.single_prompt_mode:
                        print(result)
                    else:
                        print(f"✅ {result}")
                        
                except Exception as tool_error:
                    execution_time = (time.time() - start_time) * 1000
                    error_msg = f"Tool execution failed: {str(tool_error)}"
                    
                    # Add failed tool result to session history
                    self.session.add_message('tool', error_msg,
                                           call_id=tool_data.get("call_id"),
                                           status="error",
                                           duration_ms=execution_time,
                                           tool_name=tool_name,
                                           tool_arguments=tool_args,
                                           stderr=str(tool_error))
                    
                    print(f"❌ {error_msg}")
                    if self.debug_mode:
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"❌ Tool execution failed: {e}")
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
                    self._save_command_history()
                    print("\n👋 Goodbye!")
                    break

        except Exception as e:
            self._save_command_history()
            print(f"❌ Fatal error: {e}")

    def run_single_prompt(self, prompt: str):
        """Execute single prompt and exit."""
        try:
            # Use generate_response for consistent tool handling
            self.generate_response(prompt)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Simplified CLI REPL for AbstractCore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m abstractcore.utils.cli --provider ollama --model qwen3-coder:30b
  python -m abstractcore.utils.cli --provider openai --model gpt-5-mini --stream
  python -m abstractcore.utils.cli --provider anthropic --model claude-haiku-4-5
  python -m abstractcore.utils.cli --provider lmstudio --model qwen/qwen3-4b-2507 --base-url http://localhost:1234/v1
  python -m abstractcore.utils.cli --provider openrouter --model openai/gpt-4o-mini
  python -m abstractcore.utils.cli --provider portkey --model gpt-4o-mini --base-url https://api.portkey.ai/v1
  python -m abstractcore.utils.cli --prompt "What is Python?"  # Uses configured defaults

Key Commands:
  /help                           Show comprehensive command guide
  /session save <name> [--summary --assessment --facts]  Save session JSON (writes .json)
  /session load <name>            Load saved session JSON (reads .json)
  /cache save <name>              Save MLX prompt/KV cache (writes .safetensors)
  /cache load <name>              Load MLX prompt/KV cache (reads .safetensors)
  /status                         Show system status and capabilities
  /history [n]                    Show conversation history
  /model <provider:model>         Switch LLM provider/model
  /compact [focus]                Compress chat history with optional focus
  /facts [file]                   Extract knowledge facts
  /judge                          Evaluate conversation quality
  /intent [participant]           Analyze conversation intents and motivations
  /system [prompt]                View/change system prompt

Tools: list_files, search_files, read_file, write_file, execute_command

File Attachments:
  Use @filename syntax to attach files: "Analyze @image.jpg and @doc.pdf"
  Supports images, audio, PDFs, Office docs, and text files with automatic processing
  Vision models analyze images directly; text models use vision fallback (configure via `abstractcore --set-vision-provider ...`)
  For audio attachments, use --audio-policy auto|speech_to_text (speech-to-text requires `pip install abstractvoice`)

Configuration:
  Set defaults with: abstractcore --set-app-default cli <provider> <model>
  Check status with: abstractcore --status

Note: This is a basic demonstrator with limited capabilities. For production
use cases requiring advanced reasoning, ReAct patterns, or complex tool chains,
build custom solutions using the AbstractCore framework directly.
        """
    )

    # Optional arguments (no longer required - will use configured defaults)
    parser.add_argument('--provider',
                       choices=['openai', 'anthropic', 'openrouter', 'portkey', 'openai-compatible', 'vllm', 'ollama', 'huggingface', 'mlx', 'lmstudio'],
                       help='LLM provider to use (optional - uses configured default)')
    parser.add_argument('--model', help='Model name to use (optional - uses configured default)')

    # Optional arguments
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--max-tokens', type=int, default=None, help='Maximum total context tokens (default: auto-detect from model capabilities)')
    parser.add_argument('--max-output-tokens', type=int, default=None, help='Maximum output tokens per response (default: provider/model default)')
    parser.add_argument('--prompt', help='Execute single prompt and exit')

    # Provider-specific
    parser.add_argument('--base-url', help='Base URL override (OpenAI-compatible /v1 servers, proxies, Ollama)')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature (default: 0.7)')
    parser.add_argument(
        "--audio-policy",
        choices=["native_only", "speech_to_text", "auto"],
        default=None,
        help="Audio attachment policy (default: auto when audio is attached).",
    )
    parser.add_argument(
        "--audio-language",
        default=None,
        help="Optional language hint for speech-to-text (e.g. 'en', 'fr').",
    )

    args = parser.parse_args()

    # Load configuration manager for defaults
    try:
        from ..config import get_config_manager
        config_manager = get_config_manager()
    except Exception as e:
        config_manager = None
        if not args.provider or not args.model:
            print(f"❌ Error loading configuration: {e}")
            print("💡 Please specify --provider and --model explicitly")
            sys.exit(1)

    # Get provider and model from configuration if not specified
    if not args.provider or not args.model:
        if config_manager:
            default_provider, default_model = config_manager.get_app_default('cli')

            # Use configured defaults if available
            provider = args.provider or default_provider
            model = args.model or default_model

            if not provider or not model:
                print("❌ Error: No provider/model specified and no defaults configured")
                print()
                print("💡 Solutions:")
                print("   1. Specify explicitly: --provider ollama --model gemma3:1b-it-qat")
                print("   2. Configure defaults: abstractcore --set-app-default cli ollama gemma3:1b-it-qat")
                print("   3. Check current config: abstractcore --status")
                sys.exit(1)

            # Show what we're using if defaults were applied
            if not args.provider or not args.model:
                if not args.prompt:  # Only show in interactive mode
                    print(f"🔧 Using configured defaults: {provider}/{model}")
                    print("   (Configure with: abstractcore --set-app-default cli <provider> <model>)")
                    print()
        else:
            print("❌ Error: No provider/model specified and configuration unavailable")
            sys.exit(1)
    else:
        # Use explicit arguments
        provider = args.provider
        model = args.model

    # Get streaming default from configuration (only if --stream not explicitly provided)
    if not args.stream and config_manager:
        try:
            default_streaming = config_manager.get_streaming_default('cli')
            stream_mode = default_streaming
        except Exception:
            stream_mode = False  # Safe fallback
    else:
        stream_mode = args.stream

    # Build kwargs
    kwargs = {'temperature': args.temperature}
    if args.base_url:
        kwargs['base_url'] = args.base_url
    if args.api_key:
        kwargs['api_key'] = args.api_key

    # Create CLI (suppress banner for single-prompt mode)
    cli = SimpleCLI(
        provider=provider,
        model=model,
        stream=stream_mode,
        max_tokens=args.max_tokens,
        max_output_tokens=args.max_output_tokens,
        debug=args.debug,
        show_banner=not args.prompt,  # Hide banner in single-prompt mode
        audio_policy=args.audio_policy,
        audio_language=args.audio_language,
        **kwargs
    )

    # Run
    if args.prompt:
        cli.run_single_prompt(args.prompt)
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()
