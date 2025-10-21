"""
ReAct Agent Implementation

The main ReAct agent that orchestrates the Think-Act-Observe loop
with scratchpad management and tool integration.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import time
import logging
from dataclasses import dataclass

from ..core.interface import AbstractCoreInterface
from ..core.session import BasicSession
from ..tools.common_tools import list_files, read_file, write_file, execute_command, search_files
from .states import ReActState, ThinkState, ActState, ObserveState, StateType, StateResult
from .scratchpad import Scratchpad
from .tools import planning_tools

logger = logging.getLogger(__name__)


@dataclass
class ReActResult:
    """Result of a ReAct agent execution"""
    success: bool
    final_answer: str
    iterations: int
    total_time_ms: float
    scratchpad: Scratchpad
    error: Optional[str] = None
    
    def __str__(self) -> str:
        status = "✅ Success" if self.success else "❌ Failed"
        return f"{status} - {self.iterations} iterations in {self.total_time_ms:.0f}ms: {self.final_answer}"


class ReActAgent:
    """
    ReAct (Reasoning and Acting) Agent
    
    Implements the ReAct pattern with:
    - Clear state separation (Think, Act, Observe)
    - Scratchpad for tracking reasoning
    - Tool integration with AbstractCore
    - Verbose logging of reasoning process
    - Adaptive planning based on discoveries
    """
    
    def __init__(self, 
                 llm: AbstractCoreInterface,
                 tools: Optional[List[Callable]] = None,
                 max_iterations: int = 75,  # Increased for thorough multi-file analysis
                 verbose: bool = True,
                 include_planning_tools: bool = True,
                 scratchpad_max_entries: int = 50):
        """
        Initialize ReAct agent
        
        Args:
            llm: AbstractCore LLM interface
            tools: Additional tools beyond the default set
            max_iterations: Maximum number of Think-Act-Observe cycles
            verbose: Whether to print detailed reasoning process
            include_planning_tools: Whether to include ReAct planning tools
            scratchpad_max_entries: Maximum entries in scratchpad
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize scratchpad
        self.scratchpad = Scratchpad(max_entries=scratchpad_max_entries)
        
        # Setup tools
        self.available_tools = self._setup_tools(tools, include_planning_tools)
        
        # Initialize states
        self.states = {
            StateType.THINK: ThinkState(),
            StateType.ACT: ActState(),
            StateType.OBSERVE: ObserveState()
        }
        
        # Initialize session with tools
        tool_list = list(self.available_tools.values())
        self.session = BasicSession(
            provider=llm,
            system_prompt=self._build_system_prompt(),
            tools=tool_list
        )
        
        logger.info(f"ReAct agent initialized with {len(self.available_tools)} tools")
    
    def _setup_tools(self, additional_tools: Optional[List[Callable]], 
                    include_planning_tools: bool) -> Dict[str, Callable]:
        """Setup the available tools for the agent"""
        tools = {
            # Core file/system tools
            "list_files": list_files,
            "read_file": read_file, 
            "write_file": write_file,
            "execute_command": execute_command,
            "search_files": search_files
        }
        
        # Add planning tools if requested
        if include_planning_tools:
            for tool in planning_tools:
                tools[tool.__name__] = tool
        
        # Add any additional tools
        if additional_tools:
            for tool in additional_tools:
                tool_name = getattr(tool, '__name__', str(tool))
                tools[tool_name] = tool
        
        return tools
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the ReAct agent"""
        return """You are an autonomous ReAct agent that thinks, acts, and observes iteratively to solve complex tasks.

Your process:
1. THINK: Analyze the situation, review observations, plan next action
2. ACT: Execute the planned action using available tools  
3. OBSERVE: Analyze results and update understanding

Key principles:
- Be methodical and thorough in your reasoning
- Use the scratchpad to track discoveries and insights
- Adapt your strategy based on what you learn
- Be explicit about your reasoning process
- When you determine the task is complete, respond with "COMPLETE: [your final answer]"

Available tools: """ + ", ".join(self.available_tools.keys())
    
    def run(self, goal: str, context: Optional[Dict[str, Any]] = None) -> ReActResult:
        """
        Run the ReAct agent to achieve the given goal
        
        Args:
            goal: The objective to achieve
            context: Optional additional context
            
        Returns:
            ReActResult with outcome and details
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"🚀 ReAct Agent Starting")
            print(f"📋 Goal: {goal}")
            print(f"🔧 Available Tools: {', '.join(self.available_tools.keys())}")
            print(f"🔄 Max Iterations: {self.max_iterations}")
            print("=" * 60)
        
        # Initialize execution context
        execution_context = {
            "goal": goal,
            "scratchpad": self.scratchpad,
            "session": self.session,
            "available_tools": self.available_tools,
            "max_iterations": self.max_iterations,
            "iteration": 0,
            "debug": self.verbose,  # Pass debug flag to states
            **(context or {})
        }
        
        # Clear scratchpad for fresh start
        self.scratchpad.clear()
        self.scratchpad.add_plan(f"Goal: {goal}")
        
        current_state = StateType.THINK
        iteration = 0  # Total state executions
        cycle_number = 1  # Actual ReAct cycles (Think→Act→Observe)
        final_answer = ""
        error = None
        
        try:
            while iteration < self.max_iterations:
                execution_context["iteration"] = iteration
                execution_context["cycle_number"] = cycle_number
                
                if self.verbose:
                    if current_state == StateType.THINK:
                        print(f"\n🔄 Cycle {cycle_number} - Iteration {iteration + 1}/{self.max_iterations}")
                    print(f"🎯 Current State: {current_state.value.upper()}")
                
                # Execute current state
                state_instance = self.states[current_state]
                state_instance.enter()
                
                try:
                    result = state_instance.execute(execution_context)
                    state_instance.exit()
                    
                    if self.verbose:
                        duration = state_instance.duration_ms or 0
                        print(f"⏱️  {current_state.value.upper()} completed in {duration:.0f}ms")
                        print(f"📝 {result.message}")
                    
                    # Handle state result
                    if result.next_state == StateType.COMPLETE:
                        final_answer = result.data.get("final_answer", result.message)
                        if self.verbose:
                            print(f"\n✅ Task Completed!")
                            print(f"📋 Final Answer: {final_answer}")
                        break
                    
                    elif result.next_state == StateType.ERROR:
                        error = result.error or result.message
                        if self.verbose:
                            print(f"\n❌ Error: {error}")
                        
                        # For certain errors, try to guide the agent back to correct format
                        if "Premature final answer" in error or "Malformed ReAct response" in error:
                            if iteration < self.max_iterations - 1:  # Not the last iteration
                                if self.verbose:
                                    print("🔄 Attempting to guide agent back to correct format...")
                                
                                # Add guidance to scratchpad
                                self.scratchpad.add_observation(f"ERROR: {error}. You must follow the exact format: Thought: [reasoning] Action: [tool_name] Action Input: [JSON]")
                                
                                # Continue to next iteration instead of breaking
                                current_state = StateType.THINK
                                continue
                        
                        break
                    
                    else:
                        # Update context with result data
                        execution_context.update(result.data)
                        current_state = result.next_state
                        
                        # Increment cycle number when completing Think→Act→Observe cycle
                        if current_state == StateType.THINK and iteration > 0:
                            cycle_number += 1
                            execution_context["cycle_number"] = cycle_number
                        
                        # Show additional details for certain states
                        if current_state == StateType.ACT and self.verbose:
                            planned_action = result.data.get("planned_action", "unknown")
                            print(f"🔧 Next Action: {planned_action}")
                        
                        elif current_state == StateType.OBSERVE and self.verbose:
                            action = result.data.get("action", "unknown")
                            success = result.data.get("success", False)
                            status = "✅" if success else "❌"
                            print(f"👁️  Observing result of {action} {status}")
                
                except Exception as e:
                    state_instance.exit()
                    error = f"State execution error: {str(e)}"
                    logger.error(f"Error in {current_state.value} state: {e}")
                    if self.verbose:
                        print(f"❌ State Error: {error}")
                    break
                
                iteration += 1
            
            # Check if we hit max iterations
            if iteration >= self.max_iterations and not final_answer and not error:
                if self.verbose:
                    print(f"\n⏰ Reached maximum iterations ({self.max_iterations})")
                    print("🔄 Attempting to get final answer with available information...")
                
                # Try one final attempt to get an answer with current information
                final_prompt = f"""You have reached the maximum number of iterations. You must now complete the task: {goal}

Based on the information you have gathered, you need to:
1. If the task requires writing a file, use the write_file action to create it
2. If the task requires analysis, provide your analysis
3. Complete the task with the information you have

Previous work:
{self.scratchpad.get_formatted_content()}

Complete the task now. If you need to write a file, use:
Action: write_file
Action Input: {{"file_path": "filename", "content": "your content"}}

Or provide your final answer:
Final Answer: [your complete answer]"""
                
                try:
                    final_response = self.llm.generate(final_prompt)
                    final_content = final_response.content.strip()
                    
                    # Check if it's an action (write_file) that needs to be executed
                    if "Action:" in final_content and "write_file" in final_content:
                        if self.verbose:
                            print("🔧 Executing final write_file action...")
                        
                        # Parse and execute the write_file action
                        from .states import ThinkState
                        think_state = ThinkState()
                        parsed = think_state._parse_react_response(final_content)
                        
                        if parsed.get("action") == "write_file" and parsed.get("action_input"):
                            write_tool = self.available_tools.get("write_file")
                            if write_tool:
                                try:
                                    result = write_tool(**parsed["action_input"])
                                    final_answer = f"Task completed successfully. {result}"
                                    if self.verbose:
                                        print(f"✅ File written successfully: {result}")
                                except Exception as write_error:
                                    final_answer = f"Attempted to write file but encountered error: {write_error}"
                                    if self.verbose:
                                        print(f"❌ Write error: {write_error}")
                            else:
                                final_answer = "Task completion attempted but write_file tool not available"
                        else:
                            final_answer = "Task completion attempted but could not parse write action"
                    
                    elif "Final Answer:" in final_content:
                        final_answer = final_content.split("Final Answer:")[-1].strip()
                        if self.verbose:
                            print(f"✅ Got final answer: {final_answer[:100]}...")
                    else:
                        error = f"Reached maximum iterations ({self.max_iterations}) without completion"
                        if self.verbose:
                            print(f"❌ Could not extract final answer from response")
                except Exception as e:
                    error = f"Reached maximum iterations ({self.max_iterations}) without completion"
                    if self.verbose:
                        print(f"❌ Failed to get final answer: {e}")
        
        except Exception as e:
            error = f"Agent execution error: {str(e)}"
            logger.error(f"ReAct agent error: {e}")
            if self.verbose:
                print(f"❌ Agent Error: {error}")
        
        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000
        
        # Create result
        result = ReActResult(
            success=bool(final_answer and not error),
            final_answer=final_answer or error or "No result",
            iterations=iteration,
            total_time_ms=total_time_ms,
            scratchpad=self.scratchpad,
            error=error
        )
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"🏁 ReAct Agent Finished")
            print(f"📊 {result}")
            if self.scratchpad.has_content():
                print(f"📝 Scratchpad Summary:")
                print(self.scratchpad.get_summary())
            print("=" * 60)
        
        return result
    
    def get_scratchpad_content(self, formatted: bool = True) -> str:
        """Get current scratchpad content"""
        if formatted:
            return self.scratchpad.get_formatted_content()
        else:
            return str(self.scratchpad)
    
    def save_session(self, filepath: str) -> None:
        """Save the current session and scratchpad"""
        # Save session
        self.session.save(filepath)
        
        # Save scratchpad separately
        scratchpad_path = filepath.replace('.json', '_scratchpad.json')
        self.scratchpad.save_to_file(scratchpad_path)
        
        if self.verbose:
            print(f"💾 Session saved to {filepath}")
            print(f"💾 Scratchpad saved to {scratchpad_path}")
    
    def load_session(self, filepath: str) -> None:
        """Load a previous session and scratchpad"""
        # Load session
        self.session = BasicSession.load(filepath, provider=self.llm, tools=list(self.available_tools.values()))
        
        # Load scratchpad if it exists
        scratchpad_path = filepath.replace('.json', '_scratchpad.json')
        try:
            self.scratchpad = Scratchpad.load_from_file(scratchpad_path)
            if self.verbose:
                print(f"📂 Session loaded from {filepath}")
                print(f"📂 Scratchpad loaded from {scratchpad_path}")
        except FileNotFoundError:
            if self.verbose:
                print(f"📂 Session loaded from {filepath}")
                print(f"⚠️  No scratchpad file found at {scratchpad_path}")
    
    def reset(self) -> None:
        """Reset the agent state"""
        self.scratchpad.clear()
        self.session.clear_history(keep_system=True)
        if self.verbose:
            print("🔄 ReAct agent reset")
    
    def add_tool(self, tool: Callable, name: Optional[str] = None) -> None:
        """Add a new tool to the agent"""
        tool_name = name or getattr(tool, '__name__', str(tool))
        self.available_tools[tool_name] = tool
        
        # Update session tools
        self.session.tools.append(tool)
        
        if self.verbose:
            print(f"🔧 Added tool: {tool_name}")
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the agent"""
        if name in self.available_tools:
            del self.available_tools[name]
            
            # Remove from session tools
            self.session.tools = [t for t in self.session.tools if getattr(t, '__name__', str(t)) != name]
            
            if self.verbose:
                print(f"🗑️  Removed tool: {name}")
            return True
        return False
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.available_tools.keys())
