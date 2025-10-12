#!/bin/bash
cd /Users/albou/projects/abstractllm_core
source .venv/bin/activate 2>/dev/null || true
python -m pytest tests/test_openai_format_bug.py -v -s
