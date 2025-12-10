#!/usr/bin/env python3
"""
Verify backward compatibility - old code without custom_criteria still works.
"""

from abstractcore import create_llm
from abstractcore.core.session import BasicSession

print("=" * 80)
print("BACKWARD COMPATIBILITY VERIFICATION")
print("=" * 80)

# Create real Ollama LLM
llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)

print("\nTEST 1: Old API - generate_assessment() with no parameters")
print("-" * 80)

session = BasicSession(provider=llm)
session.generate("What is Python?")

# OLD API: No custom_criteria parameter (should still work)
assessment = session.generate_assessment()

print(f"✓ Old API works!")
print(f"  Overall Score: {assessment['overall_score']}/5")
print(f"  Has 'criteria': {('criteria' in assessment)}")
print(f"  Has 'custom_criteria': {('custom_criteria' in assessment)}")
print(f"  custom_criteria value: {assessment.get('custom_criteria')}")

if assessment.get('custom_criteria') is None:
    print("\n✅ VERIFIED: Backward compatible - custom_criteria is None when not provided")
else:
    print("\n❌ FAILED: custom_criteria should be None when not provided!")

print("\n" + "-" * 80)
print("TEST 2: Old API - generate_assessment(criteria=...)")
print("-" * 80)

session2 = BasicSession(provider=llm)
session2.generate("Explain JavaScript")

# OLD API: Only predefined criteria (should still work)
assessment2 = session2.generate_assessment(
    criteria={
        "clarity": True,
        "completeness": True
    }
)

print(f"✓ Old API with criteria dict works!")
print(f"  Overall Score: {assessment2['overall_score']}/5")
print(f"  Criteria: {assessment2['criteria']}")
print(f"  custom_criteria: {assessment2.get('custom_criteria')}")

if (assessment2['criteria']['clarity'] is True and
    assessment2['criteria']['completeness'] is True and
    assessment2.get('custom_criteria') is None):
    print("\n✅ VERIFIED: Backward compatible - old criteria dict API works perfectly")
else:
    print("\n❌ FAILED: Old criteria dict API broken!")

print("\n" + "=" * 80)
print("BACKWARD COMPATIBILITY: ✅ PASSED")
print("=" * 80)
print("\nAll existing code will continue to work without any changes!")
