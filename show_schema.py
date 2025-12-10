#!/usr/bin/env python3
"""Show the JSON schema being sent to Ollama for structured output"""

from abstractcore.processing.basic_judge import Assessment
import json

# Get the JSON schema from the Pydantic model
schema = Assessment.model_json_schema()

print("="*80)
print("JSON SCHEMA SENT TO OLLAMA (format parameter):")
print("="*80)
print(json.dumps(schema, indent=2))
print("="*80)
print()
print("KEY FIELD TO CHECK:")
print("="*80)

# Check custom_scores field
if 'custom_scores' in schema.get('properties', {}):
    custom_scores_schema = schema['properties']['custom_scores']
    print(f"✅ custom_scores field found in schema")
    print(f"\nField definition:")
    print(json.dumps(custom_scores_schema, indent=2))

    # Check if it's required
    required_fields = schema.get('required', [])
    if 'custom_scores' in required_fields:
        print("\n✅ custom_scores is REQUIRED")
    else:
        print("\n❌ custom_scores is NOT required (optional)")
        print("   This allows Ollama to return an empty dict!")
else:
    print("❌ custom_scores NOT in schema!")

print()
print("="*80)
print("REQUIRED FIELDS:")
print("="*80)
print(schema.get('required', []))
