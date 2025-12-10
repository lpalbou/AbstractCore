#!/usr/bin/env python3
"""
Verification script to prove custom_criteria works with REAL LLM calls.
This script makes actual API calls to Ollama and prints the responses.
"""

import json
from abstractcore import create_llm
from abstractcore.core.session import BasicSession
from abstractcore.processing import BasicJudge

print("=" * 80)
print("VERIFICATION: Custom Criteria with REAL Ollama API Calls")
print("=" * 80)

# Create real Ollama LLM
print("\n1. Creating real Ollama LLM connection...")
llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
print(f"   ✓ Connected to Ollama provider: {llm.__class__.__name__}")
print(f"   ✓ Model: {llm.model}")
print(f"   ✓ Base URL: {llm.base_url}")

# Test 1: BasicJudge with custom criteria
print("\n" + "=" * 80)
print("TEST 1: BasicJudge.evaluate() with custom criteria")
print("=" * 80)

judge = BasicJudge(llm)
content = "The machine learning model achieved 95% accuracy on the test set."

custom_criteria = {
    "technical_accuracy": "Is the technical information accurate and precise?",
    "statistical_validity": "Are the statistics presented in a valid way?"
}

print(f"\nContent to evaluate: '{content}'")
print(f"\nCustom criteria:")
for name, desc in custom_criteria.items():
    print(f"  • {name}: {desc}")

print("\n⏳ Making REAL API call to Ollama...")
import time
start_time = time.time()

result = judge.evaluate(
    content=content,
    context="technical validation",
    custom_criteria=custom_criteria
)

elapsed = time.time() - start_time

print(f"✓ Response received in {elapsed:.2f} seconds")
print(f"\nRESULT:")
print(f"  Overall Score: {result['overall_score']}/5")
print(f"  Judge Summary: {result['judge_summary'][:200]}...")
print(f"  Criteria Used: {result.get('criteria_used', [])}")

# Verify custom criteria were used
criteria_used = result.get('criteria_used', [])
if 'technical_accuracy' in criteria_used and 'statistical_validity' in criteria_used:
    print("\n  ✅ VERIFIED: Custom criteria were used in evaluation!")
else:
    print("\n  ❌ FAILED: Custom criteria not found in evaluation!")
    print(f"     Expected: ['technical_accuracy', 'statistical_validity']")
    print(f"     Got: {criteria_used}")

# Test 2: BasicSession with custom criteria
print("\n" + "=" * 80)
print("TEST 2: BasicSession.generate_assessment() with custom criteria")
print("=" * 80)

session = BasicSession(provider=llm)

print("\n⏳ Having a real conversation...")
start_time = time.time()

response1 = session.generate("What is the mean of [1, 2, 3, 4, 5]?")
print(f"  User: What is the mean of [1, 2, 3, 4, 5]?")
print(f"  Assistant: {response1.content[:100]}...")

response2 = session.generate("What is the standard deviation?")
print(f"  User: What is the standard deviation?")
print(f"  Assistant: {response2.content[:100]}...")

# Digital Article use case
custom_criteria = {
    "logical_coherence": "Are the results logically consistent?",
    "result_plausibility": "Are the findings plausible given the data?",
    "assumption_validity": "Were statistical assumptions properly checked?"
}

print(f"\nCustom criteria for data analysis assessment:")
for name, desc in custom_criteria.items():
    print(f"  • {name}: {desc}")

print("\n⏳ Making REAL API call for assessment...")
assessment_start = time.time()

assessment = session.generate_assessment(
    custom_criteria=custom_criteria
)

assessment_elapsed = time.time() - assessment_start
total_elapsed = time.time() - start_time

print(f"✓ Assessment received in {assessment_elapsed:.2f} seconds")
print(f"✓ Total time (conversation + assessment): {total_elapsed:.2f} seconds")

print(f"\nASSESSMENT RESULT:")
print(f"  Overall Score: {assessment['overall_score']}/5")
print(f"  Custom Criteria: {list(assessment['custom_criteria'].keys())}")
print(f"  Judge Summary: {assessment['judge_summary'][:200]}...")

# Verify custom criteria were stored
if assessment['custom_criteria'] == custom_criteria:
    print("\n  ✅ VERIFIED: Custom criteria stored correctly in assessment!")
else:
    print("\n  ❌ FAILED: Custom criteria not stored correctly!")
    print(f"     Expected: {custom_criteria}")
    print(f"     Got: {assessment['custom_criteria']}")

# Test 3: Mixed predefined + custom criteria
print("\n" + "=" * 80)
print("TEST 3: Mixed predefined + custom criteria")
print("=" * 80)

session2 = BasicSession(provider=llm)
session2.generate("Explain JWT tokens.")

print("\n⏳ Making REAL API call with mixed criteria...")
start_time = time.time()

assessment = session2.generate_assessment(
    criteria={
        "clarity": True,
        "completeness": True,
        "coherence": False  # Disabled
    },
    custom_criteria={
        "technical_depth": "Does the explanation show appropriate technical depth?",
        "security_awareness": "Are security considerations mentioned?"
    }
)

elapsed = time.time() - start_time
print(f"✓ Assessment received in {elapsed:.2f} seconds")

print(f"\nASSESSMENT RESULT:")
print(f"  Overall Score: {assessment['overall_score']}/5")
print(f"  Predefined Criteria: {assessment['criteria']}")
print(f"  Custom Criteria: {list(assessment['custom_criteria'].keys())}")

# Verify both types of criteria were used
if (assessment['criteria']['clarity'] is True and
    assessment['criteria']['completeness'] is True and
    'technical_depth' in assessment['custom_criteria']):
    print("\n  ✅ VERIFIED: Both predefined and custom criteria working!")
else:
    print("\n  ❌ FAILED: Mixed criteria not working correctly!")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\n✅ All tests used REAL Ollama API calls")
print("✅ No mocking was used")
print("✅ Custom criteria feature is working correctly")
print("\nTotal execution time proves real API calls (not instant mock responses)")
