#!/usr/bin/env python3
"""Test the user's original example"""

from abstractcore import create_llm
from abstractcore.core.session import BasicSession

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
session = BasicSession(provider=llm)

# Data analysis conversation
session.generate("What is the mean of [10, 20, 30, 40, 50]?")
session.generate("Calculate the variance")

# NEW: Custom criteria assessment
assessment = session.generate_assessment(
    custom_criteria={
        "logical_coherence": "Are the results logically consistent?",
        "result_plausibility": "Are the findings plausible given the data?",
        "assumption_validity": "Were statistical assumptions checked?"
    }
)

print(f"Overall Score: {assessment['overall_score']}/5")
print(f"Custom Criteria: {list(assessment['custom_criteria'].keys())}")
print(f"\n✅ CUSTOM SCORES:")
for criterion, score in assessment['custom_scores'].items():
    print(f"   {criterion}: {score}/5")
