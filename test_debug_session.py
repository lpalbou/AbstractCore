#!/usr/bin/env python3
"""Debug session custom scores"""

from abstractcore import create_llm
from abstractcore.processing import BasicJudge

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
judge = BasicJudge(llm)

content = "Statistical analysis: Mean=30, Variance=125. The results show normal distribution."
custom_criteria = {
    "logical_coherence": "Are the results logically consistent?",
    "result_plausibility": "Are the findings plausible given the data?",
    "assumption_validity": "Were statistical assumptions checked?"
}

result = judge.evaluate(
    content=content,
    context="data analysis validation",
    custom_criteria=custom_criteria
)

print("="*80)
print("JUDGE RESULT:")
print("="*80)
import json
print(json.dumps(result, indent=2))

print("\n" + "="*80)
print("CUSTOM SCORES:")
print("="*80)
for criterion, score in result.get('custom_scores', {}).items():
    print(f"   {criterion}: {score}/5")
