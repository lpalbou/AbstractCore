#!/usr/bin/env python3
"""
Test script: Custom criteria assessment should FAIL on result_plausibility

This script creates a conversation where the assistant gives a response with
IMPLAUSIBLE/IMPOSSIBLE RESULTS given the input data - should score LOW on result_plausibility.
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

    # Generate a BAD response with IMPLAUSIBLE RESULTS
    assistant_bad_response = """
I'll calculate the mean and variance for you.

First, the mean:
Mean = (10 + 20 + 30 + 40 + 50) / 5 = 150 / 5 = 750

The mean is 750.

Now for the variance:
1. Using mean = 750, calculate deviations:
   - (10-750)² = 547600
   - (20-750)² = 532900
   - (30-750)² = 518400
   - (40-750)² = 504100
   - (50-750)² = 490000

2. Sum of squared deviations = 2593000

3. Variance = 2593000 / 5 = 518600

Therefore:
- Mean: 750
- Variance: 518600

IMPLAUSIBILITY:
- Mean of 750 is IMPOSSIBLE for dataset [10, 20, 30, 40, 50]
  (actual mean MUST be 30, between min=10 and max=50)
- Variance of 518600 is ABSURDLY HIGH
  (actual variance should be ~200, max possible variance for this range is ~267)
- Results violate basic mathematical constraints (mean must be between min and max)
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
            'result_plausibility': 'Are the findings plausible given the data constraints?',
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
    print("EXPECTED: result_plausibility should score VERY LOW (1) - results violate basic constraints")
    print("EXPECTED: logical_coherence might score medium (internally consistent but wrong)")
    print("EXPECTED: assumption_validity might score low (arithmetic errors)")

if __name__ == "__main__":
    main()
