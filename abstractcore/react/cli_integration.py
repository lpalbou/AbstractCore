"""
ReAct CLI Integration

Extends the existing AbstractCore CLI with ReAct agent capabilities.
Provides commands to run autonomous agents and manage ReAct sessions.
"""

import argparse
import sys
import time
from typing import Optional, Dict, Any

from ..utils.cli import SimpleCLI
from .agent import ReActAgent, ReActResult


class ReActCLI(SimpleCLI):
    """
    Extended CLI with ReAct agent capabilities
    
    Adds ReAct-specific commands while maintaining all existing CLI functionality.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.react_agent: Optional[ReActAgent] = None
        self._setup_react_agent()
    
    def _setup_react_agent(self):
        """Initialize the ReAct agent"""
        try:
            self.react_agent = ReActAgent(
                llm=self.provider,
                verbose=self.debug_mode,
                max_iterations=10
            )
            if self.debug_mode:
                print(f"🤖 ReAct agent initialized with {len(self.react_agent.list_tools())} tools")
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Failed to initialize ReAct agent: {e}")
            self.react_agent = None
    
    def handle_command(self, user_input: str) -> bool:
        """Extended command handler with ReAct commands"""
        if not user_input.startswith('/'):
            return False
        
        cmd = user_input[1:].strip()
        
        # Handle ReAct-specific commands
        if cmd.startswith('react '):
            self.handle_react_command(cmd[6:])  # Remove 'react ' prefix
            return True
        elif cmd == 'react':
            self.show_react_help()
            return True
        elif cmd.startswith('agent '):
            self.handle_agent_command(cmd[6:])  # Remove 'agent ' prefix
            return True
        elif cmd == 'scratchpad':
            self.handle_scratchpad_command()
            return True
        else:
            # Fall back to parent command handler
            return super().handle_command(user_input)
    
    def handle_react_command(self, args: str):
        """Handle /react <goal> command - run ReAct agent"""
        if not self.react_agent:
            print("❌ ReAct agent not available")
            return
        
        if not args.strip():
            print("❓ Usage: /react <goal>")
            print("   Example: /react Find all Python files and count their total lines")
            return
        
        goal = args.strip()
        
        try:
            print(f"\n🚀 Starting ReAct Agent")
            print(f"📋 Goal: {goal}")
            print("=" * 60)
            
            start_time = time.time()
            result = self.react_agent.run(goal)
            duration = time.time() - start_time
            
            print("\n" + "=" * 60)
            print(f"🏁 ReAct Agent Completed in {duration:.1f}s")
            
            if result.success:
                print(f"✅ Success: {result.final_answer}")
            else:
                print(f"❌ Failed: {result.final_answer}")
                if result.error:
                    print(f"   Error: {result.error}")
            
            print(f"📊 Iterations: {result.iterations}")
            print(f"⏱️  Total Time: {result.total_time_ms:.0f}ms")
            
            if result.scratchpad.has_content():
                print(f"\n📝 Scratchpad Summary:")
                print(result.scratchpad.get_summary())
            
        except Exception as e:
            print(f"❌ ReAct execution failed: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
    
    def handle_agent_command(self, args: str):
        """Handle /agent <subcommand> commands"""
        if not self.react_agent:
            print("❌ ReAct agent not available")
            return
        
        parts = args.split()
        if not parts:
            print("❓ Usage: /agent <subcommand>")
            print("   Subcommands: status, reset, tools, save, load")
            return
        
        subcommand = parts[0]
        
        if subcommand == "status":
            self.show_agent_status()
        elif subcommand == "reset":
            self.react_agent.reset()
            print("🔄 ReAct agent reset")
        elif subcommand == "tools":
            self.show_agent_tools()
        elif subcommand == "save" and len(parts) > 1:
            filename = parts[1]
            if not filename.endswith('.json'):
                filename += '.json'
            self.react_agent.save_session(filename)
        elif subcommand == "load" and len(parts) > 1:
            filename = parts[1]
            if not filename.endswith('.json'):
                filename += '.json'
            try:
                self.react_agent.load_session(filename)
            except Exception as e:
                print(f"❌ Failed to load session: {e}")
        else:
            print(f"❓ Unknown agent subcommand: {subcommand}")
    
    def handle_scratchpad_command(self):
        """Handle /scratchpad command - show current scratchpad"""
        if not self.react_agent:
            print("❌ ReAct agent not available")
            return
        
        if not self.react_agent.scratchpad.has_content():
            print("📝 Scratchpad is empty")
            return
        
        print("📝 Current Scratchpad:")
        print("=" * 50)
        print(self.react_agent.get_scratchpad_content())
        print("=" * 50)
        print(f"📊 {len(self.react_agent.scratchpad)} entries")
    
    def show_react_help(self):
        """Show ReAct-specific help"""
        print("\n" + "=" * 70)
        print("🤖 ReAct Agent Commands".center(70))
        print("=" * 70)
        
        print("\n🚀 REACT EXECUTION")
        print("─" * 50)
        print("  /react <goal>            Run ReAct agent with specified goal")
        print("                           • /react Find all Python files in this directory")
        print("                           • /react Analyze the codebase structure")
        print("                           • /react Count lines of code by file type")
        
        print("\n🤖 AGENT MANAGEMENT")
        print("─" * 50)
        print("  /agent status            Show agent status and configuration")
        print("  /agent reset             Reset agent state and scratchpad")
        print("  /agent tools             List available tools")
        print("  /agent save <file>       Save agent session and scratchpad")
        print("  /agent load <file>       Load agent session and scratchpad")
        
        print("\n📝 SCRATCHPAD")
        print("─" * 50)
        print("  /scratchpad              Show current scratchpad content")
        
        print("\n💡 REACT PATTERN")
        print("─" * 50)
        print("  The ReAct agent follows a Think-Act-Observe loop:")
        print("  • THINK: Analyze situation and plan next action")
        print("  • ACT: Execute planned action using available tools")
        print("  • OBSERVE: Analyze results and update understanding")
        print("  • Repeat until goal is achieved or max iterations reached")
        
        print("\n🔧 AVAILABLE TOOLS")
        if self.react_agent:
            tools = self.react_agent.list_tools()
            print("─" * 50)
            for i, tool in enumerate(tools, 1):
                print(f"  {i:2d}. {tool}")
        else:
            print("  ❌ ReAct agent not available")
        
        print("\n💡 TIPS")
        print("─" * 50)
        print("  • Be specific with your goals for better results")
        print("  • The agent will show its reasoning process in verbose mode")
        print("  • Use /agent status to monitor progress")
        print("  • Save sessions to preserve complex reasoning chains")
        print("  • The scratchpad tracks all thoughts and observations")
        
        print("\n" + "=" * 70)
    
    def show_agent_status(self):
        """Show detailed agent status"""
        if not self.react_agent:
            print("❌ ReAct agent not available")
            return
        
        print("🤖 ReAct Agent Status")
        print("=" * 50)
        print(f"🔧 Available Tools: {len(self.react_agent.list_tools())}")
        print(f"🔄 Max Iterations: {self.react_agent.max_iterations}")
        print(f"📝 Verbose Mode: {'ON' if self.react_agent.verbose else 'OFF'}")
        print(f"📋 Scratchpad Entries: {len(self.react_agent.scratchpad)}")
        
        if self.react_agent.scratchpad.has_content():
            print(f"📊 Scratchpad Summary:")
            print(f"   {self.react_agent.scratchpad.get_summary(max_length=200)}")
        
        print("=" * 50)
    
    def show_agent_tools(self):
        """Show available agent tools"""
        if not self.react_agent:
            print("❌ ReAct agent not available")
            return
        
        tools = self.react_agent.list_tools()
        print(f"🔧 Available Tools ({len(tools)}):")
        print("=" * 40)
        
        # Group tools by category
        file_tools = [t for t in tools if any(keyword in t for keyword in ['file', 'read', 'write', 'list'])]
        system_tools = [t for t in tools if any(keyword in t for keyword in ['execute', 'command', 'search'])]
        planning_tools = [t for t in tools if any(keyword in t for keyword in ['plan', 'reflect', 'assess', 'prioritize'])]
        other_tools = [t for t in tools if t not in file_tools + system_tools + planning_tools]
        
        if file_tools:
            print("📁 File Operations:")
            for tool in file_tools:
                print(f"   • {tool}")
        
        if system_tools:
            print("⚙️  System Operations:")
            for tool in system_tools:
                print(f"   • {tool}")
        
        if planning_tools:
            print("🧠 Planning & Reflection:")
            for tool in planning_tools:
                print(f"   • {tool}")
        
        if other_tools:
            print("🔧 Other Tools:")
            for tool in other_tools:
                print(f"   • {tool}")
        
        print("=" * 40)
    
    def generate_response(self, user_input: str):
        """Override to add ReAct suggestions"""
        # Check if this looks like a complex task that might benefit from ReAct
        task_indicators = [
            "find all", "analyze", "count", "list", "search for", "identify",
            "examine", "investigate", "discover", "determine", "figure out",
            "check", "verify", "compare", "summarize", "extract"
        ]
        
        is_complex_task = any(indicator in user_input.lower() for indicator in task_indicators)
        
        # Call parent method
        super().generate_response(user_input)
        
        # Suggest ReAct for complex tasks (only in interactive mode)
        if is_complex_task and not self.single_prompt_mode and self.react_agent:
            print(f"\n💡 Tip: This looks like a complex task that might benefit from autonomous reasoning.")
            print(f"   Try: /react {user_input}")


def create_react_cli(*args, **kwargs) -> ReActCLI:
    """Create a ReAct-enabled CLI instance"""
    return ReActCLI(*args, **kwargs)


def main():
    """Main entry point for ReAct CLI"""
    parser = argparse.ArgumentParser(
        description="AbstractCore CLI with ReAct Agent Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ReAct Agent Examples:
  /react Find all Python files in this directory and count their lines
  /react Analyze the codebase structure and identify main components  
  /react Search for TODO comments and categorize them by priority
  /react Examine test coverage and identify untested modules

ReAct Commands:
  /react <goal>           Run autonomous ReAct agent
  /agent status           Show agent status and configuration
  /agent reset            Reset agent state
  /scratchpad            Show agent's reasoning scratchpad

The ReAct agent uses Think-Act-Observe loops for complex reasoning tasks.
It maintains a scratchpad of thoughts and observations, adapting its strategy
based on discoveries. Perfect for multi-step analysis and investigation tasks.
        """
    )
    
    # Import the main CLI argument setup
    from ..utils.cli import main as cli_main
    
    # Get the original parser arguments
    original_parser = argparse.ArgumentParser()
    cli_main.__code__.co_consts  # This is a hack to get the parser setup
    
    # For now, let's just use the same arguments as the original CLI
    # but create a ReActCLI instance instead
    
    # Optional arguments (no longer required - will use configured defaults)
    parser.add_argument('--provider',
                       choices=['openai', 'anthropic', 'ollama', 'huggingface', 'mlx', 'lmstudio'],
                       help='LLM provider to use (optional - uses configured default)')
    parser.add_argument('--model', help='Model name to use (optional - uses configured default)')

    # Optional arguments
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--max-tokens', type=int, default=None, help='Maximum tokens (default: auto-detect from model capabilities)')
    parser.add_argument('--prompt', help='Execute single prompt and exit')
    parser.add_argument('--react', help='Execute single ReAct goal and exit')

    # Provider-specific
    parser.add_argument('--base-url', help='Base URL (ollama, lmstudio)')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature (default: 0.7)')

    args = parser.parse_args()

    # Load configuration manager for defaults (same as original CLI)
    try:
        from ..config import get_config_manager
        config_manager = get_config_manager()
    except Exception as e:
        config_manager = None
        if not args.provider or not args.model:
            print(f"❌ Error loading configuration: {e}")
            print("💡 Please specify --provider and --model explicitly")
            sys.exit(1)

    # Get provider and model from configuration if not specified (same logic as original)
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
                if not args.prompt and not args.react:  # Only show in interactive mode
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

    # Get streaming default from configuration (same as original)
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

    # Create ReAct CLI (suppress banner for single-prompt mode)
    cli = ReActCLI(
        provider=provider,
        model=model,
        stream=stream_mode,
        max_tokens=args.max_tokens,
        debug=args.debug,
        show_banner=not (args.prompt or args.react),  # Hide banner in single-prompt mode
        **kwargs
    )

    # Run
    if args.react:
        # Single ReAct execution
        if cli.react_agent:
            result = cli.react_agent.run(args.react)
            if result.success:
                print(result.final_answer)
            else:
                print(f"Error: {result.final_answer}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: ReAct agent not available", file=sys.stderr)
            sys.exit(1)
    elif args.prompt:
        cli.run_single_prompt(args.prompt)
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()
