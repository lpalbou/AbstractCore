"""
AbstractCore ReAct Agent System

A clean, efficient implementation of the ReAct (Reasoning and Acting) pattern
for autonomous agents that can think, act, and observe iteratively.

Key Features:
- Clear state separation (Think, Act, Observe)
- Scratchpad for tracking thoughts and observations
- Adaptive planning based on discoveries
- Integration with AbstractCore's tool system
- Verbose logging of agent reasoning process

Usage:
    from abstractcore.react import ReActAgent
    from abstractcore import create_llm
    
    llm = create_llm("openai", model="gpt-4o-mini")
    agent = ReActAgent(llm, verbose=True)
    
    result = agent.run("Find all Python files in the current directory and count their lines")
"""

from .agent import ReActAgent
from .states import ReActState, ThinkState, ActState, ObserveState
from .scratchpad import Scratchpad
from .tools import planning_tools
from .cli_integration import ReActCLI, create_react_cli

__all__ = [
    "ReActAgent",
    "ReActState", 
    "ThinkState",
    "ActState", 
    "ObserveState",
    "Scratchpad",
    "planning_tools",
    "ReActCLI",
    "create_react_cli"
]
