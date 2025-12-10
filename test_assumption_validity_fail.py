#!/usr/bin/env python3
"""
Test script: Custom criteria assessment should FAIL on assumption_validity

This script creates a conversation where the assistant gives a response with
INVALID STATISTICAL ASSUMPTIONS that are not justified or checked - should score LOW on assumption_validity.
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

    # Generate a BAD response with INVALID ASSUMPTIONS
    assistant_bad_response = """
I'll calculate the mean and variance for you.

First, the mean:
Mean = (10 + 20 + 30 + 40 + 50) / 5 = 150 / 5 = 30

Now for the variance, I'll apply several assumptions:

ASSUMPTION 1: Since this is time series data, I'll use Bessel's correction
(Wait, the user never said this was time series - invalid assumption!)

ASSUMPTION 2: The data is normally distributed, so I'll use parametric methods
(5 data points cannot establish normality - invalid assumption!)

ASSUMPTION 3: These are independent measurements, so I'll calculate sample variance
(No information given about independence - invalid assumption!)

ASSUMPTION 4: Missing values should be imputed as zero
(There are no missing values - unnecessary and wrong assumption!)

ASSUMPTION 5: Since variance must account for temporal autocorrelation, I'll apply
a lag-1 correction factor of 0.8
(No temporal structure, no autocorrelation mentioned - completely invalid!)

Applying all these assumptions:
Variance = 200 × 0.8 (autocorrelation correction) = 160

Therefore:
- Mean: 30
- Adjusted Variance: 160 (with temporal correction)

INVALID ASSUMPTIONS:
- Assumes time series structure (not stated)
- Assumes normality with n=5 (impossible to verify)
- Assumes independence (no information provided)
- Imputes non-existent missing values
- Applies autocorrelation correction without justification
- None of these assumptions were checked or validated
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
            'assumption_validity': 'Were statistical assumptions properly justified, checked, and valid for this data?'
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
    print("EXPECTED: assumption_validity should score VERY LOW (1) - unjustified and invalid assumptions")
    print("EXPECTED: logical_coherence might score low-medium (applies assumptions consistently)")
    print("EXPECTED: result_plausibility might score medium (final calculation is arithmetic, but wrong approach)")

if __name__ == "__main__":
    main()
