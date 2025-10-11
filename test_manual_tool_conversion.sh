#!/bin/bash

# Manual test script for universal tool call conversion
# This script tests that the server properly converts tool calls to OpenAI format

echo "====================================="
echo "Universal Tool Call Conversion Test"
echo "====================================="

# Test configuration
SERVER="http://localhost:9090"
MODEL="lmstudio/qwen/qwen3-next-80b"  # Model that outputs <function_call> format

echo ""
echo "Testing model: $MODEL"
echo "Server: $SERVER"
echo ""

# Create a temporary file for the request
cat > /tmp/test_request.json << 'EOF'
{
  "model": "MODEL_PLACEHOLDER",
  "messages": [
    {
      "role": "user",
      "content": "List files in current directory"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "shell",
        "description": "Execute shell commands",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {
              "type": "array",
              "items": {"type": "string"},
              "description": "Command to execute as array"
            }
          },
          "required": ["command"]
        }
      }
    }
  ],
  "stream": true
}
EOF

# Replace model placeholder
sed -i.bak "s/MODEL_PLACEHOLDER/$MODEL/" /tmp/test_request.json

echo "Sending request to server..."
echo "======================================="
echo ""

# Make the request and parse output
curl -X POST "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @/tmp/test_request.json 2>/dev/null | while IFS= read -r line; do

  # Skip empty lines
  if [ -z "$line" ]; then
    continue
  fi

  # Parse SSE data
  if [[ $line == data:* ]]; then
    data="${line#data: }"

    # Skip [DONE] marker
    if [ "$data" = "[DONE]" ]; then
      echo ""
      echo "======================================="
      echo "Stream completed"
      continue
    fi

    # Check for tool_calls in the JSON
    if echo "$data" | grep -q '"tool_calls"'; then
      echo "✅ TOOL CALL DETECTED - Proper OpenAI format!"
      echo "$data" | python3 -m json.tool | head -20
      echo ""
    elif echo "$data" | grep -q '<|tool_call|>\|<function_call>\|<tool_call>'; then
      echo "❌ RAW TOOL TAGS DETECTED - Not converted!"
      echo "$data"
      echo ""
    elif echo "$data" | grep -q '"content"'; then
      # Extract and display content
      content=$(echo "$data" | python3 -c "import sys, json; d=json.load(sys.stdin); c=d.get('choices',[{}])[0].get('delta',{}).get('content',''); print(c) if c else None" 2>/dev/null)
      if [ -n "$content" ]; then
        echo -n "$content"
      fi
    fi
  fi
done

echo ""
echo ""
echo "Test complete!"
echo ""
echo "Expected result:"
echo "  ✅ Tool calls should appear in OpenAI JSON format with 'tool_calls' field"
echo ""
echo "If you see:"
echo "  ❌ Raw tags like <|tool_call|> or <function_call> in content"
echo "     Then the conversion is NOT working"
echo ""

# Cleanup
rm -f /tmp/test_request.json /tmp/test_request.json.bak