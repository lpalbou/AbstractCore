#!/usr/bin/env python3
"""
Test script: Custom criteria assessment should FAIL on logical_coherence

This script creates a conversation where the assistant gives a response with
LOGICAL CONTRADICTIONS and INCONSISTENT reasoning - should score LOW on logical_coherence.
"""

from abstractcore import create_llm
from abstractcore.core.session import BasicSession

def main():
    # Create LLM provider
    llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)

    # Create session
    session = BasicSession(provider=llm)

    # User asks about data analysis
    user_query = "Calculate the mean and variance of these numbers: [10, 20, 30, 40, 50]"

    # Generate a BAD response with LOGICAL CONTRADICTIONS
    # We'll manually inject this to simulate a flawed assistant response
    assistant_bad_response = """
I'll calculate the mean and variance for you.

First, the mean:
Mean = (10 + 20 + 30 + 40 + 50) / 5 = 150 / 5 = 30

Now for the variance:
1. The mean is 30
2. Each deviation squared:
   - (10-30)² = 400
   - (20-30)² = 100
   - (30-30)² = 0
   - (40-30)² = 100
   - (50-30)² = 400

3. Sum of squared deviations = 1000

4. Wait, actually the mean should be 35, not 30, because the middle value is 30.
   Let me recalculate with mean=35:
   - (10-35)² = 625
   - (20-35)² = 225
   - (30-35)² = 25
   - (40-35)² = 25
   - (50-35)² = 225

5. But using the original mean of 30, the variance is 1000/5 = 200.

6. However, since the numbers are evenly spaced, the variance must be zero because
   they follow a perfect pattern.

So the variance is both 200 and 0, depending on how you look at it.

CONTRADICTIONS:
- Claims mean is 30, then says it should be 35
- Calculates variance as 200
- Then claims variance must be 0 because of "perfect pattern"
- Uses incorrect logic: evenly spaced ≠ zero variance
"""

    # Add messages to session manually
    session.add_message("user", user_query)
    session.add_message("assistant", assistant_bad_response)

    # Generate assessment with custom criteria
    print("Generating assessment with custom criteria...")
    print("=" * 80)

    assessment = session.generate_assessment(
        custom_criteria={
            'logical_coherence': 'Are the results logically consistent throughout the explanation?',
            'result_plausibility': 'Are the findings plausible given the data?',
            'assumption_validity': 'Were statistical assumptions properly checked and valid?'
        }
    )

    # Display results
    print(f"\nOverall Score: {assessment['overall_score']}/5")
    print(f"\nCustom Scores:")
    for criterion, score in assessment.get('custom_scores', {}).items():
        print(f"  {criterion}: {score}/5")

    print(f"\nJudge Summary:")
    print(assessment['judge_summary'])

    print(f"\nStrengths:")
    for strength in assessment.get('strengths', []):
        print(f"  - {strength}")

    print(f"\nActionable Feedback:")
    for feedback in assessment.get('actionable_feedback', []):
        print(f"  - {feedback}")

    print("\n" + "=" * 80)
    print("EXPECTED: logical_coherence should score LOW (1-2) due to contradictions")
    print("EXPECTED: result_plausibility might score medium (variance 200 is correct calculation)")
    print("EXPECTED: assumption_validity might score low-medium (wrong assumption about evenly spaced)")

if __name__ == "__main__":
    main()
