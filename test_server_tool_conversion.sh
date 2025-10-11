#!/bin/bash
# Quick test for server tool call conversion

echo "🧪 Testing Server Tool Call Conversion"
echo "========================================"

# Test with qwen3-coder
echo "Testing qwen3-coder model..."
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lmstudio/qwen/qwen3-coder-30b",
    "messages": [{"role": "user", "content": "list files in current directory"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "shell",
        "description": "Execute shell command",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {"type": "array", "items": {"type": "string"}}
          }
        }
      }
    }]
  }' | python -c "
import json
import sys
try:
    data = json.loads(sys.stdin.read())
    message = data['choices'][0]['message']

    if 'tool_calls' in message:
        print('✅ TOOL CALLS DETECTED - Proper OpenAI format!')
        for tool_call in message['tool_calls']:
            print(f'   Tool: {tool_call[\"function\"][\"name\"]}')
            print(f'   Args: {tool_call[\"function\"][\"arguments\"]}')
    elif '<|tool_call|>' in str(message.get('content', '')):
        print('❌ RAW TOOL TAGS DETECTED - Conversion failed!')
        print(f'   Raw content: {message.get(\"content\", \"\")}')
    else:
        print('ℹ️  No tool calls detected')
        print(f'   Content: {message.get(\"content\", \"\")}')
except Exception as e:
    print(f'❌ Error parsing response: {e}')
    print(sys.stdin.read())
"

echo ""
echo "If you see '✅ TOOL CALLS DETECTED', the conversion is working!"
echo "If you see '❌ RAW TOOL TAGS DETECTED', there's still an issue."