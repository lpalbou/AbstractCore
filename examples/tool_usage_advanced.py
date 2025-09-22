#!/usr/bin/env python3
"""
Advanced Tool Usage Examples for AbstractLLM Core

This example demonstrates advanced tool usage patterns including:
- Custom tool implementations
- Error handling and validation
- Tool chaining and workflows
- Integration with external systems
- Performance optimization
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from abstractllm import AbstractLLM
from abstractllm.events import EventType
from abstractllm.tools.core import ToolCall, ToolResult

# Custom tool implementations
class WeatherAPI:
    """Mock weather API for demonstration"""

    @staticmethod
    def get_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
        """Simulate weather API call"""
        # In a real implementation, this would call an actual weather API
        mock_data = {
            "Paris": {"temp": 72 if unit == "fahrenheit" else 22, "condition": "Sunny"},
            "Tokyo": {"temp": 68 if unit == "fahrenheit" else 20, "condition": "Cloudy"},
            "New York": {"temp": 75 if unit == "fahrenheit" else 24, "condition": "Partly Cloudy"},
            "London": {"temp": 60 if unit == "fahrenheit" else 15, "condition": "Rainy"}
        }

        # Simulate API delay
        time.sleep(0.1)

        for city in mock_data:
            if city.lower() in location.lower():
                return {
                    "location": location,
                    "temperature": mock_data[city]["temp"],
                    "unit": unit,
                    "condition": mock_data[city]["condition"],
                    "timestamp": time.time()
                }

        return {
            "location": location,
            "temperature": 70 if unit == "fahrenheit" else 21,
            "unit": unit,
            "condition": "Unknown",
            "timestamp": time.time()
        }

class DatabaseManager:
    """Mock database for demonstration"""

    def __init__(self):
        self.data = {
            "users": [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ],
            "products": [
                {"id": 1, "name": "Laptop", "price": 999.99, "stock": 10},
                {"id": 2, "name": "Mouse", "price": 29.99, "stock": 50}
            ]
        }

    def query(self, table: str, conditions: Optional[Dict] = None) -> List[Dict]:
        """Query the mock database"""
        if table not in self.data:
            raise ValueError(f"Table '{table}' not found")

        results = self.data[table]

        if conditions:
            filtered_results = []
            for item in results:
                match = True
                for key, value in conditions.items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append(item)
            return filtered_results

        return results

# Global instances for tools
weather_api = WeatherAPI()
db_manager = DatabaseManager()

@dataclass
class ToolExecutionMetrics:
    """Track tool execution metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    tool_usage: Dict[str, int] = None

    def __post_init__(self):
        if self.tool_usage is None:
            self.tool_usage = {}

# Advanced tool definitions with custom handlers
ADVANCED_TOOLS = [
    {
        "name": "get_weather_advanced",
        "description": "Get detailed weather information with error handling and validation",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country/state (e.g., 'Paris, France')"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "fahrenheit"
                },
                "include_forecast": {
                    "type": "boolean",
                    "description": "Include 3-day forecast",
                    "default": False
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "database_query",
        "description": "Query the application database",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "enum": ["users", "products"],
                    "description": "Database table to query"
                },
                "conditions": {
                    "type": "object",
                    "description": "Query conditions as key-value pairs"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10
                }
            },
            "required": ["table"]
        }
    },
    {
        "name": "calculate_advanced",
        "description": "Advanced calculator with mathematical functions",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression (supports +, -, *, /, **, sqrt, sin, cos, etc.)"
                },
                "precision": {
                    "type": "integer",
                    "description": "Decimal places for result",
                    "default": 2
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "send_notification",
        "description": "Send a notification to users (simulated)",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Recipient identifier or email"
                },
                "message": {
                    "type": "string",
                    "description": "Notification message"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium"
                }
            },
            "required": ["recipient", "message"]
        }
    },
    {
        "name": "file_operations",
        "description": "Perform file system operations (simulated for safety)",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "read", "write", "delete"],
                    "description": "File operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content for write operations"
                }
            },
            "required": ["operation", "path"]
        }
    }
]

class ToolExecutionTracker:
    """Track and monitor tool executions"""

    def __init__(self):
        self.metrics = ToolExecutionMetrics()
        self.call_history = []

    def on_before_execution(self, event):
        """Handle before tool execution event"""
        for call in event.data['tool_calls']:
            self.metrics.total_calls += 1
            self.metrics.tool_usage[call.name] = self.metrics.tool_usage.get(call.name, 0) + 1

            # Log the call
            self.call_history.append({
                'tool': call.name,
                'arguments': call.arguments,
                'timestamp': time.time(),
                'status': 'started'
            })

    def on_after_execution(self, event):
        """Handle after tool execution event"""
        for i, result in enumerate(event.data['results']):
            if result.success:
                self.metrics.successful_calls += 1
            else:
                self.metrics.failed_calls += 1

            # Update call history
            if self.call_history:
                self.call_history[-len(event.data['results']) + i]['status'] = 'completed'
                self.call_history[-len(event.data['results']) + i]['success'] = result.success
                self.call_history[-len(event.data['results']) + i]['result'] = result.output if result.success else result.error

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        success_rate = (self.metrics.successful_calls / max(self.metrics.total_calls, 1)) * 100

        return {
            'total_calls': self.metrics.total_calls,
            'successful_calls': self.metrics.successful_calls,
            'failed_calls': self.metrics.failed_calls,
            'success_rate': f"{success_rate:.1f}%",
            'tool_usage': self.metrics.tool_usage,
            'recent_calls': self.call_history[-5:]  # Last 5 calls
        }

def custom_tool_handler(tool_call: ToolCall) -> ToolResult:
    """Custom tool execution handler with advanced logic"""
    start_time = time.time()

    try:
        if tool_call.name == "get_weather_advanced":
            args = tool_call.arguments
            location = args.get("location", "")
            unit = args.get("unit", "fahrenheit")
            include_forecast = args.get("include_forecast", False)

            if not location:
                return ToolResult(
                    success=False,
                    error="Location is required",
                    execution_time=time.time() - start_time
                )

            # Validate location format
            if len(location) < 2:
                return ToolResult(
                    success=False,
                    error="Location must be at least 2 characters",
                    execution_time=time.time() - start_time
                )

            weather_data = weather_api.get_weather(location, unit)

            if include_forecast:
                # Add mock forecast data
                weather_data["forecast"] = [
                    {"day": "Tomorrow", "temp": weather_data["temperature"] + 2, "condition": "Sunny"},
                    {"day": "Day 2", "temp": weather_data["temperature"] - 1, "condition": "Cloudy"},
                    {"day": "Day 3", "temp": weather_data["temperature"] + 3, "condition": "Partly Cloudy"}
                ]

            return ToolResult(
                success=True,
                output=json.dumps(weather_data, indent=2),
                execution_time=time.time() - start_time
            )

        elif tool_call.name == "database_query":
            args = tool_call.arguments
            table = args.get("table")
            conditions = args.get("conditions")
            limit = args.get("limit", 10)

            try:
                results = db_manager.query(table, conditions)
                limited_results = results[:limit]

                return ToolResult(
                    success=True,
                    output=json.dumps({
                        "table": table,
                        "conditions": conditions,
                        "count": len(limited_results),
                        "results": limited_results
                    }, indent=2),
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Database query failed: {str(e)}",
                    execution_time=time.time() - start_time
                )

        elif tool_call.name == "calculate_advanced":
            args = tool_call.arguments
            expression = args.get("expression", "")
            precision = args.get("precision", 2)

            if not expression:
                return ToolResult(
                    success=False,
                    error="Expression is required",
                    execution_time=time.time() - start_time
                )

            try:
                # Safe evaluation with limited scope
                import math
                allowed_names = {
                    k: v for k, v in math.__dict__.items() if not k.startswith("__")
                }
                allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

                result = eval(expression, {"__builtins__": {}}, allowed_names)
                formatted_result = round(float(result), precision)

                return ToolResult(
                    success=True,
                    output=f"{expression} = {formatted_result}",
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Calculation error: {str(e)}",
                    execution_time=time.time() - start_time
                )

        elif tool_call.name == "send_notification":
            args = tool_call.arguments
            recipient = args.get("recipient", "")
            message = args.get("message", "")
            priority = args.get("priority", "medium")

            if not recipient or not message:
                return ToolResult(
                    success=False,
                    error="Both recipient and message are required",
                    execution_time=time.time() - start_time
                )

            # Simulate notification sending
            notification_id = f"notif_{int(time.time())}"

            return ToolResult(
                success=True,
                output=f"Notification sent successfully!\nID: {notification_id}\nRecipient: {recipient}\nPriority: {priority}\nMessage: {message}",
                execution_time=time.time() - start_time
            )

        elif tool_call.name == "file_operations":
            args = tool_call.arguments
            operation = args.get("operation")
            path = args.get("path", "")
            content = args.get("content", "")

            # Simulate file operations (for safety, don't actually perform them)
            if operation == "list":
                mock_files = ["file1.txt", "file2.py", "data.json", "readme.md"]
                return ToolResult(
                    success=True,
                    output=f"Files in {path}:\n" + "\n".join(f"  - {f}" for f in mock_files),
                    execution_time=time.time() - start_time
                )
            elif operation == "read":
                return ToolResult(
                    success=True,
                    output=f"Contents of {path}:\n[Simulated file content - this is a demo]",
                    execution_time=time.time() - start_time
                )
            elif operation == "write":
                return ToolResult(
                    success=True,
                    output=f"Successfully wrote {len(content)} characters to {path}",
                    execution_time=time.time() - start_time
                )
            elif operation == "delete":
                return ToolResult(
                    success=True,
                    output=f"Successfully deleted {path}",
                    execution_time=time.time() - start_time
                )

        return ToolResult(
            success=False,
            error=f"Unknown tool: {tool_call.name}",
            execution_time=time.time() - start_time
        )

    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Tool execution error: {str(e)}",
            execution_time=time.time() - start_time
        )

def demonstrate_tool_chaining():
    """Demonstrate chaining multiple tools together"""
    print("\n" + "="*60)
    print(" Advanced Tool Chaining Example")
    print("="*60)

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        # Set up execution tracking
        tracker = ToolExecutionTracker()
        llm.add_event_listener(EventType.BEFORE_TOOL_EXECUTION, tracker.on_before_execution)
        llm.add_event_listener(EventType.AFTER_TOOL_EXECUTION, tracker.on_after_execution)

        # Complex workflow request
        response = llm.generate(
            prompt="""Please help me with this workflow:
            1. Get the weather for Paris with forecast
            2. Find all users in the database
            3. Calculate the total price of all products
            4. Send a weather notification to john@example.com
            5. List files in the '/data' directory

            Provide a summary of all operations.""",
            tools=ADVANCED_TOOLS,
            system_prompt="""You are an advanced AI assistant that can chain multiple tools together.
            Execute the requested workflow step by step and provide a comprehensive summary."""
        )

        print(f"Workflow Response:\n{response.content}")

        # Show execution metrics
        summary = tracker.get_summary()
        print(f"\nExecution Summary:")
        print(f"  Total tool calls: {summary['total_calls']}")
        print(f"  Success rate: {summary['success_rate']}")
        print(f"  Tool usage: {summary['tool_usage']}")

    except Exception as e:
        print(f"Error in tool chaining: {str(e)}")

def demonstrate_error_handling():
    """Demonstrate robust error handling in tool execution"""
    print("\n" + "="*60)
    print(" Error Handling and Validation")
    print("="*60)

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        # Request with intentional errors
        response = llm.generate(
            prompt="""Please try these operations that might fail:
            1. Get weather for an empty location
            2. Query a non-existent database table called 'invalid'
            3. Calculate an invalid mathematical expression: '1 / 0'
            4. Send a notification without specifying a recipient

            Handle any errors gracefully and report what happened.""",
            tools=ADVANCED_TOOLS,
            system_prompt="""You are an AI assistant that handles errors gracefully.
            When tools fail, explain what went wrong and suggest alternatives."""
        )

        print(f"Error Handling Response:\n{response.content}")

    except Exception as e:
        print(f"Error in error handling demo: {str(e)}")

def demonstrate_performance_optimization():
    """Demonstrate performance considerations and optimization"""
    print("\n" + "="*60)
    print(" Performance Optimization Example")
    print("="*60)

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        start_time = time.time()

        # Batch operations for efficiency
        response = llm.generate(
            prompt="""Perform these operations efficiently:
            1. Get weather for multiple cities: Paris, Tokyo, New York
            2. Calculate multiple expressions: 15*8, sqrt(144), 2**10
            3. Query both users and products tables

            Process everything in batch for maximum efficiency.""",
            tools=ADVANCED_TOOLS,
            system_prompt="""You are an efficiency-focused AI assistant.
            When possible, batch similar operations together for better performance."""
        )

        execution_time = time.time() - start_time

        print(f"Batch Operations Response:\n{response.content}")
        print(f"\nTotal execution time: {execution_time:.2f} seconds")

    except Exception as e:
        print(f"Error in performance demo: {str(e)}")

def demonstrate_streaming_with_complex_tools():
    """Demonstrate streaming with complex tool operations"""
    print("\n" + "="*60)
    print(" Streaming with Complex Tools")
    print("="*60)

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        print("Streaming complex tool operations:")
        print("Response: ", end="", flush=True)

        total_chunks = 0
        tool_chunks = 0

        for chunk in llm.generate(
            prompt="""Create a comprehensive report by:
            1. Getting detailed weather for London with forecast
            2. Analyzing our product database for inventory
            3. Calculating key business metrics
            4. Sending status notifications to relevant users

            Provide real-time updates as you complete each step.""",
            tools=ADVANCED_TOOLS,
            stream=True,
            system_prompt="""You are a business intelligence assistant.
            Provide real-time updates as you gather and process information."""
        ):
            total_chunks += 1

            if chunk.content:
                print(chunk.content, end="", flush=True)

            if chunk.tool_calls:
                tool_chunks += 1
                print(f"\n[Processing {len(chunk.tool_calls)} tool(s)...]", end="", flush=True)

        print(f"\n\nStreaming completed:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Tool chunks: {tool_chunks}")

    except Exception as e:
        print(f"Error in streaming demo: {str(e)}")

def demonstrate_conditional_tool_execution():
    """Demonstrate conditional tool execution based on context"""
    print("\n" + "="*60)
    print(" Conditional Tool Execution")
    print("="*60)

    try:
        llm = AbstractLLM(provider="ollama", model="llama3.2:3b")

        # Add conditional logic in event handler
        def conditional_handler(event):
            # Only allow certain tools during business hours
            current_hour = time.localtime().tm_hour

            for call in event.data['tool_calls']:
                if call.name == "send_notification" and (current_hour < 9 or current_hour > 17):
                    print(f"🕒 Preventing notification outside business hours (current hour: {current_hour})")
                    event.prevent()
                    break

        llm.add_event_listener(EventType.BEFORE_TOOL_EXECUTION, conditional_handler)

        response = llm.generate(
            prompt="Send a notification to all users about system maintenance and get the current weather for our office location (New York).",
            tools=ADVANCED_TOOLS,
            system_prompt="You are a system administrator assistant."
        )

        print(f"Conditional Execution Response:\n{response.content}")

    except Exception as e:
        print(f"Error in conditional execution demo: {str(e)}")

def main():
    """Run all advanced examples"""
    print("AbstractLLM Core - Advanced Tool Usage Examples")
    print("=" * 60)
    print("This script demonstrates advanced tool usage patterns.")
    print("Note: Examples may fail if Ollama is not running locally.")

    # Register custom tool handler globally
    from abstractllm.tools import register_tool_handler
    register_tool_handler(custom_tool_handler)

    # Run advanced examples
    demonstrate_tool_chaining()
    demonstrate_error_handling()
    demonstrate_performance_optimization()
    demonstrate_streaming_with_complex_tools()
    demonstrate_conditional_tool_execution()

    print("\n" + "="*60)
    print(" Advanced Examples Complete")
    print("="*60)
    print("These examples show the full power of the AbstractLLM Core tool system!")

if __name__ == "__main__":
    main()