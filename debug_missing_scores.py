#!/usr/bin/env python3
"""Debug why only some custom scores are returned"""

from abstractcore import create_llm
from abstractcore.processing import BasicJudge
from abstractcore.processing.basic_judge import JudgmentCriteria

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
judge = BasicJudge(llm, debug=True)

content = "Mean is 30, variance is 200. The calculations are correct."

custom_criteria = {
    "logical_coherence": "Are the results logically consistent?",
    "result_plausibility": "Are the findings plausible given the data?",
    "assumption_validity": "Were statistical assumptions checked?"
}

print("="*80)
print("CALLING EVALUATE WITH 3 CUSTOM CRITERIA")
print("="*80)

result = judge.evaluate(
    content=content,
    context="data analysis validation",
    custom_criteria=custom_criteria
)

print("\n" + "="*80)
print("CUSTOM SCORES IN RESULT:")
print("="*80)
print(result.get('custom_scores', {}))

print("\n" + "="*80)
print("HOW MANY SCORES RETURNED?")
print("="*80)
print(f"Expected: 3 ({list(custom_criteria.keys())})")
print(f"Got: {len(result.get('custom_scores', {}))} ({list(result.get('custom_scores', {}).keys())})")

if len(result.get('custom_scores', {})) < len(custom_criteria):
    print("\n⚠️  MISSING SCORES!")
    missing = set(custom_criteria.keys()) - set(result.get('custom_scores', {}).keys())
    print(f"Missing: {list(missing)}")
