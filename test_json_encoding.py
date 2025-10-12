"""
Quick test to understand JSON encoding behavior for OpenAI format.
"""

import json

# Test data
arguments = {"command": ["ls", "-la"], "workdir": "/tmp"}

# Method 1: json.dumps (CORRECT)
method1 = json.dumps(arguments)
print("Method 1 - json.dumps():")
print(f"  Result: {repr(method1)}")
print(f"  Type: {type(method1)}")
print(f"  Value: {method1}")
print()

# Method 2: str() (WRONG - produces Python repr, not JSON)
method2 = str(arguments)
print("Method 2 - str():")
print(f"  Result: {repr(method2)}")
print(f"  Type: {type(method2)}")
print(f"  Value: {method2}")
print()

# Create full OpenAI format with each method
print("=" * 80)
print("Full OpenAI format comparison:")
print("=" * 80)

# Correct way
correct_format = {
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "shell",
        "arguments": json.dumps(arguments)  # CORRECT
    }
}
print("\nCORRECT (using json.dumps):")
correct_json = json.dumps(correct_format, indent=2)
print(correct_json)

# Try to parse it back
parsed_correct = json.loads(correct_json)
print(f"\n✅ Can parse back: {parsed_correct['function']['name']}")
print(f"✅ Arguments is string: {isinstance(parsed_correct['function']['arguments'], str)}")
parsed_args = json.loads(parsed_correct['function']['arguments'])
print(f"✅ Can parse arguments: {parsed_args}")

print("\n" + "=" * 80)

# Wrong way (if using str())
wrong_format = {
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "shell",
        "arguments": str(arguments)  # WRONG
    }
}
print("\nWRONG (using str()):")
wrong_json = json.dumps(wrong_format, indent=2)
print(wrong_json)

# Try to parse it back
parsed_wrong = json.loads(wrong_json)
print(f"\n✅ Can parse outer JSON: {parsed_wrong['function']['name']}")
print(f"⚠️  Arguments is string: {isinstance(parsed_wrong['function']['arguments'], str)}")
print(f"⚠️  Arguments value: {repr(parsed_wrong['function']['arguments'])}")

try:
    parsed_args_wrong = json.loads(parsed_wrong['function']['arguments'])
    print(f"❌ SHOULD FAIL but parsed: {parsed_args_wrong}")
except json.JSONDecodeError as e:
    print(f"❌ FAILS to parse arguments: {e}")

print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)
print(f"Correct arguments: {repr(correct_format['function']['arguments'])}")
print(f"Wrong arguments:   {repr(wrong_format['function']['arguments'])}")
print()
print("The key difference:")
print("- json.dumps() produces: '{\"command\": [\"ls\", \"-la\"], \"workdir\": \"/tmp\"}'")
print("- str() produces:        \"{'command': ['ls', '-la'], 'workdir': '/tmp'}\"")
print("  (Note: Python uses single quotes, JSON requires double quotes!)")
