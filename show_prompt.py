#!/usr/bin/env python3
"""Show the exact prompt being sent to the LLM for custom criteria"""

from abstractcore import create_llm
from abstractcore.processing import BasicJudge
from abstractcore.processing.basic_judge import JudgmentCriteria

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
judge = BasicJudge(llm)

content = "The machine learning model achieved 95% accuracy on the test set."
custom_criteria = {
    "technical_accuracy": "Is the technical information accurate and precise?",
    "statistical_validity": "Are the statistics presented in a valid way?"
}

# Build the prompt (call the internal method directly)
prompt = judge._build_evaluation_prompt(
    content=content,
    context="technical validation",
    criteria=JudgmentCriteria(),
    focus=None,
    reference=None,
    include_criteria=False,
    custom_criteria=custom_criteria
)

print("="*80)
print("EXACT PROMPT SENT TO LLM:")
print("="*80)
print(prompt)
print("="*80)
print()
print("PROMPT LENGTH:", len(prompt), "characters")
print()
print("="*80)
print("KEY SECTIONS TO CHECK:")
print("="*80)
print()

# Check 1: Custom criteria in EVALUATION CRITERIA section
if "technical_accuracy" in prompt.lower():
    print("✅ 'technical_accuracy' found in prompt")
else:
    print("❌ 'technical_accuracy' NOT found in prompt!")

if "Is the technical information accurate and precise?" in prompt:
    print("✅ Custom description for technical_accuracy found")
else:
    print("❌ Custom description NOT found!")

# Check 2: custom_scores in RESPONSE FORMAT
if '"custom_scores"' in prompt:
    print("✅ 'custom_scores' field in response format")

    # Extract the custom_scores section
    import re
    match = re.search(r'"custom_scores":\s*\{[^}]*\}', prompt, re.DOTALL)
    if match:
        print("\nCustom scores format in prompt:")
        print("  " + match.group(0))
else:
    print("❌ 'custom_scores' NOT in response format!")

# Check 3: Instructions about scoring
if "MUST" in prompt and "custom_scores" in prompt:
    print("✅ Instructions to populate custom_scores found")
else:
    print("❌ No clear instructions to populate custom_scores!")

print()
print("="*80)
print("To verify yourself, copy the prompt above and test it directly with:")
print("  curl http://localhost:11434/api/generate -d '{\"model\":\"qwen3:4b-instruct-2507-q4_K_M\",\"prompt\":\"<paste prompt here>\",\"stream\":false}'")
print("="*80)
