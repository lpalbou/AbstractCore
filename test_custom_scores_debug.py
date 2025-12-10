#!/usr/bin/env python3
"""Debug script to see if custom_scores are returned"""

from abstractcore import create_llm
from abstractcore.processing import BasicJudge

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
judge = BasicJudge(llm, debug=True)  # Enable debug to see raw response

content = "The machine learning model achieved 95% accuracy on the test set."
custom_criteria = {
    "technical_accuracy": "Is the technical information accurate and precise?",
    "statistical_validity": "Are the statistics presented in a valid way?"
}

print("Making API call with custom criteria...")
result = judge.evaluate(
    content=content,
    context="technical validation",
    custom_criteria=custom_criteria
)

print("\n" + "="*80)
print("RESULT:")
print("="*80)
import json
print(json.dumps(result, indent=2))

print("\n" + "="*80)
print("CUSTOM SCORES:")
print("="*80)
if "custom_scores" in result:
    print(f"custom_scores present: {result['custom_scores']}")
else:
    print("custom_scores NOT in result!")
