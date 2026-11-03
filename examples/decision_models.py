"""Decision models: ask typed questions about a document and get calibrated probabilities.

    python examples/decision_models.py                      # mlx/jaredpalmer/kev-4b on Apple silicon
    python examples/decision_models.py huggingface "ggml-org/Kev-4B-GGUF:Q4_K_M"

Download first (no silent downloads):
    abstractcore models download mlx jaredpalmer/kev-4b
    abstractcore models download mlx Qwen/Qwen3.5-4B-Base
    # or: abstractcore models download huggingface "ggml-org/Kev-4B-GGUF:Q4_K_M"
"""

import json
import sys

from abstractcore import create_llm

provider = sys.argv[1] if len(sys.argv) > 1 else "mlx"
model = sys.argv[2] if len(sys.argv) > 2 else "jaredpalmer/kev-4b"
llm = create_llm(provider, model=model)

ticket = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."

# Full System One questions: choice / noul / score.
result = llm.generate(ticket, output="decision", questions={
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"}},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
})
print(json.dumps(result.decision.to_systemone(), indent=2))
print("route to:", result.decisions["department"].choice,
      "| escalate:", result.decisions["escalate"].noul > 0.5,
      "| frustration:", round(result.decisions["frustration"].score, 2))

# Python shorthand: a string is a yes/no question, a list is a choice, levels make a score.
# The state is cached, so this second call only pays for its questions.
again = llm.generate(ticket, output="decision", questions={
    "team": ["returns", "shipping", "billing"],
    "refund": "Should we refund the duplicate charge?",
    "mood": {"instructions": "How frustrated is the customer?", "levels": ["Calm", "Frustrated", "Very angry"]},
})
print({name: answer.model_dump() for name, answer in again.decisions.items()})
print("prefix cache hit:", again.decision.usage.prefix_cache_hit, "| latency ms:", again.decision.latency_ms)
