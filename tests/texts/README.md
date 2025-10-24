# Intent Analysis Test Cases

This directory contains test cases for evaluating the intent analysis and deception detection capabilities.

## Test Files

### Core Intent Analysis Examples

**`intent1.json`** - Authentic Communication
- Workplace conversation about a late report submission
- Demonstrates authentic problem-solving with technical explanations
- Expected: Low deception likelihood (~0.05-0.10)
- Focus: Ben's genuine technical difficulties vs potential excuse-making

**`intent2.json`** - Blame Deflection
- Workplace conversation about missed client call
- Demonstrates blame deflection and responsibility avoidance
- Expected: Moderate deception likelihood (~0.30-0.40)
- Focus: Maria's deflection patterns and excuse-making

**`intent3_multi_participant.json`** - Multi-Participant Deception
- Three-way scenario: Emma deceives Ryan (manager) but confesses to Jordan (friend)
- Demonstrates audience-specific deception and relational context
- Expected: Low deception when lying to Ryan, moderate when confessing to Jordan
- Focus: Contextual deception and the "confession paradox"

### Truthful Baselines

**`truthful_baseline1.txt`** - Honest Accountability
- Direct admission of failure with full responsibility
- Expected: Very low deception likelihood (~0.05)
- Focus: Genuine honesty and accountability patterns

**`truthful_baseline2.txt`** - Honest Reporting
- Balanced report with admission of knowledge gaps
- Expected: Very low deception likelihood (~0.05)
- Focus: Transparent communication with limitations acknowledged

## Usage Examples

```bash
# Analyze authentic communication
python -m abstractcore.apps intent tests/texts/intent1.json --focus-participant user --format plain

# Analyze blame deflection
python -m abstractcore.apps intent tests/texts/intent2.json --focus-participant assistant --format plain

# Analyze multi-participant deception
python -m abstractcore.apps intent tests/texts/intent3_multi_participant.json --focus-participant emma --format plain

# Verify truthful baselines
python -m abstractcore.apps intent tests/texts/truthful_baseline1.txt --format plain
python -m abstractcore.apps intent tests/texts/truthful_baseline2.txt --format plain
```

## Expected Performance

The intent analyzer should:
- Correctly identify authentic communication (low deception scores)
- Detect blame deflection and face-saving patterns
- Understand audience-specific deception contexts
- Distinguish between honest admissions and deceptive content
- Apply appropriate healthy skepticism without over-flagging honest communication
