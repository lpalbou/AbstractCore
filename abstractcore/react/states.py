"""
ReAct State Machine Implementation

Defines the core states of the ReAct pattern: Think, Act, Observe.
Each state has clear responsibilities and transition logic.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import time


class StateType(Enum):
    """ReAct state types"""
    THINK = "think"
    ACT = "act" 
    OBSERVE = "observe"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StateResult:
    """Result of a state execution"""
    next_state: StateType
    data: Dict[str, Any]
    message: str
    success: bool = True
    error: Optional[str] = None
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} {self.message}"


class ReActState(ABC):
    """Base class for all ReAct states"""
    
    def __init__(self, state_type: StateType):
        self.state_type = state_type
        self.entry_time: Optional[float] = None
        self.exit_time: Optional[float] = None
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> StateResult:
        """Execute the state logic"""
        pass
    
    def enter(self) -> None:
        """Called when entering this state"""
        self.entry_time = time.time()
    
    def exit(self) -> None:
        """Called when exiting this state"""
        self.exit_time = time.time()
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Duration spent in this state in milliseconds"""
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time) * 1000
        return None


class ThinkState(ReActState):
    """
    THINK State: Generate reasoning and decide next action OR final answer
    
    Following canonical ReAct pattern:
    - Analyze current situation and goal
    - Review previous observations
    - Either plan next action OR provide final answer
    """
    
    def __init__(self):
        super().__init__(StateType.THINK)
    
    def execute(self, context: Dict[str, Any]) -> StateResult:
        """
        Execute thinking phase following canonical ReAct format
        """
        goal = context.get("goal", "")
        scratchpad = context.get("scratchpad")
        session = context.get("session")
        iteration = context.get("iteration", 0)
        max_iterations = context.get("max_iterations", 10)
        debug = context.get("debug", False)
        
        if debug:
            print(f"\n🔍 DEBUG ThinkState.execute:")
            print(f"   Goal: {goal}")
            print(f"   Iteration: {iteration}/{max_iterations}")
            print(f"   Scratchpad entries: {len(scratchpad.entries) if scratchpad else 0}")
        
        if not session:
            return StateResult(
                next_state=StateType.ERROR,
                data={},
                message="No LLM session available for thinking",
                success=False,
                error="Missing session"
            )
        
        # Check if we've reached max iterations
        if iteration >= max_iterations:
            if debug:
                print(f"🔍 DEBUG: Reached max iterations {iteration}/{max_iterations}")
            return StateResult(
                next_state=StateType.COMPLETE,
                data={"final_answer": f"Reached maximum iterations ({max_iterations}) without completing the task."},
                message=f"Reached maximum iterations ({max_iterations})",
                success=False
            )
        
        # Build canonical ReAct thinking prompt
        thinking_prompt = self._build_canonical_prompt(goal, scratchpad, iteration)
        
        if debug:
            print(f"\n🔍 DEBUG: Sending prompt to LLM:")
            print("=" * 60)
            print(thinking_prompt)
            print("=" * 60)
        
        try:
            # Generate reasoning following ReAct format
            response = session.generate(thinking_prompt)
            content = response.content.strip()
            
            if debug:
                print(f"\n🔍 DEBUG: LLM Response:")
                print("=" * 60)
                print(content)
                print("=" * 60)
            
            # Parse response for Thought/Action/Final Answer
            parsed = self._parse_react_response(content)
            
            if debug:
                print(f"\n🔍 DEBUG: Parsed response:")
                for key, value in parsed.items():
                    print(f"   {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            
            # Add context size info and thought to scratchpad
            if scratchpad and parsed.get("thought"):
                cycle_number = context.get("cycle_number", 1)
                
                # Get context size information
                context_info = ""
                if session:
                    try:
                        # Get estimated tokens from session
                        current_tokens = session.get_token_estimate()
                        
                        # Get max tokens from provider if available
                        max_tokens = None
                        if hasattr(session.provider, 'max_tokens') and session.provider.max_tokens:
                            max_tokens = session.provider.max_tokens
                        elif hasattr(session.provider, '_calculate_effective_token_limits'):
                            effective_max, _, _ = session.provider._calculate_effective_token_limits()
                            max_tokens = effective_max
                        
                        if max_tokens:
                            context_info = f" [Context: {current_tokens}/{max_tokens}]"
                        else:
                            context_info = f" [Context: {current_tokens}]"
                    except Exception:
                        # If we can't get context info, just continue without it
                        pass
                
                scratchpad.add_thought(f"Cycle {cycle_number}{context_info}: {parsed['thought']}")
            
            # Validate response and determine next state
            if parsed.get("final_answer"):
                # Check if agent has actually taken any actions
                has_observations = scratchpad and any("observation" in entry.entry_type.lower() 
                                                    for entry in scratchpad.entries)
                
                if debug:
                    print(f"🔍 DEBUG: Final answer detected. Has observations: {has_observations}")
                
                if not has_observations and iteration == 0:
                    # Force the agent to take at least one action first
                    if debug:
                        print("🔍 DEBUG: Rejecting premature final answer")
                    return StateResult(
                        next_state=StateType.ERROR,
                        data={"content": content},
                        message="Cannot provide final answer without taking any actions first",
                        success=False,
                        error="Premature final answer - must take actions first"
                    )
                
                if debug:
                    print("🔍 DEBUG: Accepting final answer")
                return StateResult(
                    next_state=StateType.COMPLETE,
                    data={"final_answer": parsed["final_answer"]},
                    message=f"Task completed: {parsed['final_answer'][:100]}..."
                )
            elif parsed.get("action") and parsed.get("action_input"):
                if debug:
                    print(f"🔍 DEBUG: Action planned: {parsed['action']} with input: {parsed['action_input']}")
                return StateResult(
                    next_state=StateType.ACT,
                    data={
                        "thought": parsed["thought"],
                        "action": parsed["action"],
                        "action_input": parsed["action_input"]
                    },
                    message=f"Planned action: {parsed['action']}"
                )
            else:
                # Malformed response - try to continue
                if debug:
                    print("🔍 DEBUG: Malformed response - no valid action or final answer found")
                return StateResult(
                    next_state=StateType.ERROR,
                    data={"content": content},
                    message="Could not parse valid action or final answer from response",
                    success=False,
                    error="Malformed ReAct response"
                )
                
        except Exception as e:
            if debug:
                print(f"🔍 DEBUG: Exception in ThinkState: {e}")
                import traceback
                traceback.print_exc()
            return StateResult(
                next_state=StateType.ERROR,
                data={},
                message=f"Error during thinking: {str(e)}",
                success=False,
                error=str(e)
            )
    
    def _build_canonical_prompt(self, goal: str, scratchpad, iteration: int) -> str:
        """Build canonical ReAct prompt with few-shot examples"""
        prompt_parts = [
            "You are a ReAct agent that solves tasks by alternating between Thought, Action, and Observation.",
            "",
            f"Task: {goal}",
            "",
            "Available tools:",
            "- list_files: List files in a directory. Input: {\"directory\": \"path\"}",
            "- read_file: Read contents of a file. Input: {\"file_path\": \"path\"}", 
            "- write_file: Write content to a file. Input: {\"file_path\": \"path\", \"content\": \"text\"}",
            "- search_files: Search for pattern in files. Input: {\"pattern\": \"text\", \"directory\": \"path\"}",
            "- execute_command: Execute a shell command. Input: {\"command\": \"cmd\"}",
            "",
            "EXAMPLES of correct format:",
            "",
            "Example 1 - Taking an action:",
            "Thought: I need to see what files are in the current directory first.",
            "Action: list_files",
            "Action Input: {\"directory\": \".\"}",
            "",
            "Example 2 - Providing final answer after gathering information:",
            "Thought: I have read the files and gathered all the information needed. I can now provide the final answer.",
            "Final Answer: Based on my analysis of the files, here is the summary: [detailed answer]",
            "",
            "CRITICAL RULES:",
            "1. ALWAYS start with 'Thought:' to explain your reasoning",
            "2. If you need information, use 'Action:' and 'Action Input:' with proper JSON",
            "3. Take ONLY ONE action at a time - wait for the observation before the next action",
            "4. Read files completely and thoroughly - do not skip content",
            "5. Only use 'Final Answer:' when you have completed all necessary actions",
            "6. Use exact tool names from the list above",
            "",
        ]
        
        # Add conversation history if available
        if scratchpad and scratchpad.has_content():
            prompt_parts.extend([
                "Previous conversation:",
                scratchpad.get_formatted_content(last_n=6),  # Last 6 entries
                ""
            ])
        else:
            prompt_parts.extend([
                "This is your first step. You must start by taking an action to gather information.",
                ""
            ])
        
        prompt_parts.extend([
            f"Your turn (Step {iteration + 1}):",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_react_response(self, content: str) -> Dict[str, Any]:
        """Parse ReAct response following canonical format"""
        result = {}
        lines = content.strip().split('\n')
        
        current_section = None
        current_content = []
        action_count = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = "thought"
                current_content = [line[8:].strip()]
            elif line.startswith("Action:"):
                action_count += 1
                if action_count > 1:
                    # Multiple actions detected - warn and use only the first one
                    print(f"⚠️  WARNING: Multiple actions detected in response. Using only the first action.")
                    continue  # Skip additional actions
                    
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = "action"
                current_content = [line[7:].strip()]
            elif line.startswith("Action Input:"):
                if action_count > 1:
                    continue  # Skip additional action inputs
                    
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                current_section = "action_input"
                action_input_str = line[13:].strip()
                # Try to parse as JSON
                try:
                    import json
                    result["action_input"] = json.loads(action_input_str)
                except:
                    result["action_input"] = {"input": action_input_str}
                current_section = None
                current_content = []
            elif line.startswith("Final Answer:"):
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                result["final_answer"] = line[13:].strip()
                current_section = None
                current_content = []
            elif current_section and line:
                current_content.append(line)
        
        # Handle last section
        if current_section and current_content:
            result[current_section] = '\n'.join(current_content).strip()
        
        return result


class ActState(ReActState):
    """
    ACT State: Execute the planned action using available tools
    
    Canonical ReAct pattern:
    - Execute the action specified in the Think phase
    - Always proceed to Observe phase with results
    """
    
    def __init__(self):
        super().__init__(StateType.ACT)
    
    def execute(self, context: Dict[str, Any]) -> StateResult:
        """
        Execute action phase following canonical ReAct pattern
        """
        action = context.get("action", "")
        action_input = context.get("action_input", {})
        available_tools = context.get("available_tools", {})
        debug = context.get("debug", False)
        
        if debug:
            print(f"\n🔍 DEBUG ActState.execute:")
            print(f"   Action: {action}")
            print(f"   Action Input: {action_input}")
            print(f"   Available tools: {list(available_tools.keys())}")
        
        if not action:
            if debug:
                print("🔍 DEBUG: No action specified")
            return StateResult(
                next_state=StateType.OBSERVE,
                data={
                    "action": "none",
                    "observation": "Error: No action specified",
                    "success": False
                },
                message="No action to execute",
                success=False
            )
        
        # Map action to available tools
        tool_function = available_tools.get(action)
        if not tool_function:
            if debug:
                print(f"🔍 DEBUG: Tool '{action}' not found in available tools")
            return StateResult(
                next_state=StateType.OBSERVE,
                data={
                    "action": action,
                    "observation": f"Error: Tool '{action}' not available. Available tools: {list(available_tools.keys())}",
                    "success": False
                },
                message=f"Tool '{action}' not found",
                success=False
            )
        
        try:
            # Execute the tool with provided input
            start_time = time.time()
            
            if debug:
                print(f"🔍 DEBUG: Executing tool '{action}' with input: {action_input}")
            
            # Handle different input formats
            if isinstance(action_input, dict):
                if len(action_input) == 1 and "input" in action_input:
                    # Single input parameter
                    result = tool_function(action_input["input"])
                else:
                    # Multiple parameters
                    result = tool_function(**action_input)
            else:
                # Direct input
                result = tool_function(action_input)
            
            execution_time = (time.time() - start_time) * 1000
            
            if debug:
                result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                print(f"🔍 DEBUG: Tool execution successful. Result: {result_preview}")
                print(f"🔍 DEBUG: Execution time: {execution_time:.1f}ms")
            
            return StateResult(
                next_state=StateType.OBSERVE,
                data={
                    "action": action,
                    "action_input": action_input,
                    "observation": str(result),
                    "success": True,
                    "execution_time_ms": execution_time
                },
                message=f"Executed {action} successfully"
            )
            
        except Exception as e:
            if debug:
                print(f"🔍 DEBUG: Tool execution failed: {e}")
                import traceback
                traceback.print_exc()
            return StateResult(
                next_state=StateType.OBSERVE,
                data={
                    "action": action,
                    "action_input": action_input,
                    "observation": f"Error executing {action}: {str(e)}",
                    "success": False
                },
                message=f"Tool execution failed: {str(e)}",
                success=False
            )


class ObserveState(ReActState):
    """
    OBSERVE State: Record action results and return to Think
    
    Canonical ReAct pattern:
    - Record the observation from the action
    - Always return to Think phase to continue the cycle
    """
    
    def __init__(self):
        super().__init__(StateType.OBSERVE)
    
    def execute(self, context: Dict[str, Any]) -> StateResult:
        """
        Execute observation phase following canonical ReAct pattern
        """
        action = context.get("action", "")
        action_input = context.get("action_input", {})
        observation = context.get("observation", "")
        success = context.get("success", True)
        scratchpad = context.get("scratchpad")
        debug = context.get("debug", False)
        
        if debug:
            print(f"\n🔍 DEBUG ObserveState.execute:")
            print(f"   Action: {action}")
            print(f"   Action Input: {action_input}")
            print(f"   Success: {success}")
            observation_preview = observation[:200] + "..." if len(observation) > 200 else observation
            print(f"   Observation: {observation_preview}")
        
        # Format observation for scratchpad
        formatted_observation = self._format_observation(action, action_input, observation, success)
        
        # Add to scratchpad
        if scratchpad:
            scratchpad.add_observation(formatted_observation)
            if debug:
                print(f"🔍 DEBUG: Added observation to scratchpad. Total entries: {len(scratchpad.entries)}")
        
        # Always return to THINK phase to continue the cycle
        # The Think phase will decide whether to continue or provide final answer
        if debug:
            print("🔍 DEBUG: Returning to THINK state")
        
        return StateResult(
            next_state=StateType.THINK,
            data={"last_observation": formatted_observation},
            message=f"Observed result from {action}, returning to think"
        )
    
    def _format_observation(self, action: str, action_input: dict, observation: str, success: bool) -> str:
        """Format an observation entry for the scratchpad with exact action arguments"""
        status = "✅" if success else "❌"
        timestamp = time.strftime("%H:%M:%S")
        
        # Truncate very long observations
        if len(observation) > 1000:
            observation_summary = observation[:997] + "..."
        else:
            observation_summary = observation
        
        return f"{status} [{timestamp}] Action: {action}\nAction Input: {action_input}\nObservation: {observation_summary}"
