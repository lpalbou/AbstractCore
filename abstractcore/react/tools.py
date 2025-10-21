"""
ReAct-specific tools for planning, reflection, and meta-reasoning.

These tools enhance the agent's capabilities beyond basic file/system operations
by providing structured ways to plan, reflect, and reason about progress.
"""

from typing import Dict, Any, List, Optional
from ..tools import tool


@tool(
    description="Create or update a structured plan with goals, steps, and success criteria",
    tags=["planning", "strategy", "goals"],
    when_to_use="When you need to break down a complex task into manageable steps or update your approach",
    examples=[
        {
            "description": "Create a plan to analyze a codebase",
            "arguments": {
                "goal": "Analyze Python codebase structure and identify key components",
                "steps": [
                    "List all Python files in the directory",
                    "Identify main modules and their purposes", 
                    "Check for tests and documentation",
                    "Summarize architecture and key findings"
                ],
                "success_criteria": "Complete understanding of codebase structure with documented findings"
            }
        }
    ]
)
def create_plan(goal: str, steps: List[str], success_criteria: str, 
                priority: str = "medium", estimated_time: str = "unknown") -> str:
    """
    Create a structured plan for achieving a goal.
    
    Args:
        goal: The main objective to achieve
        steps: List of specific steps to take
        success_criteria: How to know when the goal is achieved
        priority: Priority level (low, medium, high, critical)
        estimated_time: Estimated time to complete
        
    Returns:
        Formatted plan summary
    """
    plan_summary = [
        f"📋 PLAN CREATED",
        f"Goal: {goal}",
        f"Priority: {priority.upper()}",
        f"Estimated Time: {estimated_time}",
        "",
        "Steps:",
    ]
    
    for i, step in enumerate(steps, 1):
        plan_summary.append(f"  {i}. {step}")
    
    plan_summary.extend([
        "",
        f"Success Criteria: {success_criteria}",
        "",
        "✅ Plan is ready for execution"
    ])
    
    return "\n".join(plan_summary)


@tool(
    description="Reflect on progress, identify what's working, and adjust strategy",
    tags=["reflection", "analysis", "strategy"],
    when_to_use="When you need to assess progress, learn from results, or adjust your approach",
    examples=[
        {
            "description": "Reflect on file analysis progress",
            "arguments": {
                "current_progress": "Found 15 Python files, analyzed 3 main modules",
                "what_worked": "File listing and initial analysis went smoothly",
                "challenges": "Some files are very large and complex to analyze quickly",
                "insights": "The codebase follows a clear modular structure",
                "next_steps": "Focus on the core modules first, then supporting utilities"
            }
        }
    ]
)
def reflect_on_progress(current_progress: str, what_worked: str, challenges: str,
                       insights: str, next_steps: str) -> str:
    """
    Structured reflection on current progress and strategy.
    
    Args:
        current_progress: What has been accomplished so far
        what_worked: What approaches/actions were successful
        challenges: What difficulties or obstacles were encountered
        insights: Key learnings or discoveries made
        next_steps: Recommended actions moving forward
        
    Returns:
        Formatted reflection summary
    """
    reflection = [
        f"🤔 PROGRESS REFLECTION",
        "",
        f"Current Progress:",
        f"  {current_progress}",
        "",
        f"✅ What Worked:",
        f"  {what_worked}",
        "",
        f"⚠️  Challenges:",
        f"  {challenges}",
        "",
        f"💡 Key Insights:",
        f"  {insights}",
        "",
        f"🎯 Recommended Next Steps:",
        f"  {next_steps}",
        "",
        "📈 Reflection complete - strategy updated"
    ]
    
    return "\n".join(reflection)


@tool(
    description="Assess whether a goal or task has been completed successfully",
    tags=["assessment", "completion", "validation"],
    when_to_use="When you think a task might be complete and need to validate against original requirements",
    examples=[
        {
            "description": "Check if codebase analysis is complete",
            "arguments": {
                "original_goal": "Analyze Python codebase structure and identify key components",
                "current_status": "Analyzed 15 files, documented 5 main modules, identified test structure",
                "success_criteria": "Complete understanding of codebase structure with documented findings",
                "confidence_level": "high"
            }
        }
    ]
)
def assess_completion(original_goal: str, current_status: str, success_criteria: str,
                     confidence_level: str = "medium") -> str:
    """
    Assess whether a goal has been successfully completed.
    
    Args:
        original_goal: The original objective that was set
        current_status: Current state of progress/completion
        success_criteria: The criteria that define success
        confidence_level: Confidence in completion (low, medium, high)
        
    Returns:
        Completion assessment with recommendation
    """
    # Simple heuristic assessment
    goal_words = set(original_goal.lower().split())
    status_words = set(current_status.lower().split())
    criteria_words = set(success_criteria.lower().split())
    
    # Check overlap between goal/criteria and current status
    goal_overlap = len(goal_words.intersection(status_words)) / len(goal_words) if goal_words else 0
    criteria_overlap = len(criteria_words.intersection(status_words)) / len(criteria_words) if criteria_words else 0
    
    # Determine completion likelihood
    avg_overlap = (goal_overlap + criteria_overlap) / 2
    
    if avg_overlap > 0.6 and confidence_level in ["high", "very high"]:
        completion_status = "✅ LIKELY COMPLETE"
        recommendation = "Task appears to be successfully completed based on criteria match and high confidence."
    elif avg_overlap > 0.4 and confidence_level in ["medium", "high"]:
        completion_status = "🔄 MOSTLY COMPLETE"
        recommendation = "Task is largely complete but may benefit from final validation or minor additions."
    else:
        completion_status = "⏳ IN PROGRESS"
        recommendation = "Task is not yet complete. Continue working toward the success criteria."
    
    assessment = [
        f"📊 COMPLETION ASSESSMENT",
        "",
        f"Original Goal: {original_goal}",
        f"Current Status: {current_status}",
        f"Success Criteria: {success_criteria}",
        f"Confidence Level: {confidence_level.upper()}",
        "",
        f"Assessment: {completion_status}",
        f"Goal-Status Alignment: {goal_overlap:.1%}",
        f"Criteria-Status Alignment: {criteria_overlap:.1%}",
        "",
        f"Recommendation: {recommendation}"
    ]
    
    return "\n".join(assessment)


@tool(
    description="Identify and prioritize the most important next actions",
    tags=["prioritization", "planning", "decision-making"],
    when_to_use="When you have multiple possible actions and need to decide what to do next",
    examples=[
        {
            "description": "Prioritize next steps in code analysis",
            "arguments": {
                "possible_actions": [
                    "Analyze remaining Python files",
                    "Check for configuration files",
                    "Look for documentation",
                    "Examine test coverage"
                ],
                "current_context": "Have analyzed main modules, need to complete the picture",
                "constraints": "Limited time, want most impactful insights first"
            }
        }
    ]
)
def prioritize_actions(possible_actions: List[str], current_context: str, 
                      constraints: str = "none") -> str:
    """
    Analyze and prioritize possible next actions.
    
    Args:
        possible_actions: List of possible actions to take
        current_context: Current situation and what's been done
        constraints: Any limitations or constraints to consider
        
    Returns:
        Prioritized action recommendations
    """
    if not possible_actions:
        return "❌ No actions provided to prioritize"
    
    # Simple prioritization heuristics
    priority_keywords = {
        "high": ["critical", "urgent", "main", "core", "essential", "key", "primary"],
        "medium": ["important", "significant", "useful", "helpful", "check", "analyze"],
        "low": ["optional", "nice", "additional", "extra", "minor", "cleanup"]
    }
    
    prioritized = []
    
    for action in possible_actions:
        action_lower = action.lower()
        priority = "medium"  # default
        
        # Check for high priority keywords
        if any(keyword in action_lower for keyword in priority_keywords["high"]):
            priority = "high"
        # Check for low priority keywords
        elif any(keyword in action_lower for keyword in priority_keywords["low"]):
            priority = "low"
        
        prioritized.append((priority, action))
    
    # Sort by priority (high, medium, low)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    prioritized.sort(key=lambda x: priority_order[x[0]])
    
    result = [
        f"🎯 ACTION PRIORITIZATION",
        "",
        f"Context: {current_context}",
        f"Constraints: {constraints}",
        "",
        "Prioritized Actions:"
    ]
    
    for i, (priority, action) in enumerate(prioritized, 1):
        priority_emoji = {"high": "🔥", "medium": "📋", "low": "💡"}[priority]
        result.append(f"  {i}. {priority_emoji} [{priority.upper()}] {action}")
    
    result.extend([
        "",
        f"💡 Recommendation: Start with action #{1} ({prioritized[0][1]})"
    ])
    
    return "\n".join(result)


# Export the planning tools as a list for easy registration
planning_tools = [
    create_plan,
    reflect_on_progress, 
    assess_completion,
    prioritize_actions
]
