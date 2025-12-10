#!/usr/bin/env python3
"""
Verify that custom criteria are actually being included in the prompts sent to the LLM.
This proves the implementation is working correctly at the prompt level.
"""

from abstractcore import create_llm
from abstractcore.processing import BasicJudge
from abstractcore.processing.basic_judge import JudgmentCriteria

print("=" * 80)
print("PROMPT VERIFICATION: Custom Criteria in LLM Prompts")
print("=" * 80)

# Create real Ollama LLM
llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
judge = BasicJudge(llm)

# Test content
content = "The API returns JSON with user_id and email fields."

# Custom criteria
custom_criteria = {
    "api_design_quality": "Is the API design following best practices?",
    "data_format_clarity": "Is the data format clearly specified and appropriate?"
}

print("\nContent:", content)
print("\nCustom Criteria:")
for name, desc in custom_criteria.items():
    print(f"  • {name}: {desc}")

# Build the prompt (we'll call the internal method to see what gets built)
print("\n" + "=" * 80)
print("BUILDING EVALUATION PROMPT...")
print("=" * 80)

# Access the internal prompt building method
prompt = judge._build_evaluation_prompt(
    content=content,
    context="API documentation review",
    criteria=JudgmentCriteria(),
    focus=None,
    reference=None,
    include_criteria=False,
    custom_criteria=custom_criteria
)

print("\nGENERATED PROMPT:")
print("-" * 80)
print(prompt)
print("-" * 80)

# Verify custom criteria are in the prompt
if "api_design_quality" in prompt.lower() or "Api Design Quality" in prompt:
    print("\n✅ VERIFIED: 'api_design_quality' found in prompt!")
else:
    print("\n❌ FAILED: 'api_design_quality' NOT found in prompt!")

if "data_format_clarity" in prompt.lower() or "Data Format Clarity" in prompt:
    print("✅ VERIFIED: 'data_format_clarity' found in prompt!")
else:
    print("❌ FAILED: 'data_format_clarity' NOT found in prompt!")

if "Is the API design following best practices?" in prompt:
    print("✅ VERIFIED: Custom description for api_design_quality found in prompt!")
else:
    print("❌ FAILED: Custom description NOT found in prompt!")

if "Is the data format clearly specified and appropriate?" in prompt:
    print("✅ VERIFIED: Custom description for data_format_clarity found in prompt!")
else:
    print("❌ FAILED: Custom description NOT found in prompt!")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
