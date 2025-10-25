#!/usr/bin/env python3
"""Diagnose what's wrong with fast and enhanced modes"""

from abstractcore.processing import BasicDeepSearch
from abstractcore.core.factory import create_llm
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

llm = create_llm('ollama', model='qwen3:4b-instruct-2507-q4_K_M', max_tokens=32000)

# Patch to track sub-tasks and queries
original_develop_fast = BasicDeepSearch._develop_search_questions_fast
original_develop_standard = BasicDeepSearch._develop_search_questions

def traced_develop_fast(self, research_plan, max_sources):
    print(f'\n📋 _develop_search_questions_fast called:')
    print(f'   Sub-tasks: {len(research_plan.sub_tasks)}')
    print(f'   Max sources: {max_sources}')

    original_develop_fast(self, research_plan, max_sources)

    total_queries = sum(len(st.search_queries) for st in research_plan.sub_tasks)
    print(f'   Generated queries: {total_queries}')
    for st in research_plan.sub_tasks:
        print(f'   - {st.id}: {len(st.search_queries)} queries')

def traced_develop_standard(self, research_plan, max_sources):
    queries_per_task = max(2, max_sources // len(research_plan.sub_tasks))
    print(f'\n📋 _develop_search_questions called:')
    print(f'   Sub-tasks: {len(research_plan.sub_tasks)}')
    print(f'   Max sources: {max_sources}')
    print(f'   Queries per task: {queries_per_task}')

    original_develop_standard(self, research_plan, max_sources)

    total_queries = sum(len(st.search_queries) for st in research_plan.sub_tasks)
    print(f'   Generated queries: {total_queries}')

BasicDeepSearch._develop_search_questions_fast = traced_develop_fast
BasicDeepSearch._develop_search_questions = traced_develop_standard

# Test fast mode
print('\n' + '='*80)
print('TESTING FAST MODE')
print('='*80)

searcher_fast = BasicDeepSearch(llm=llm, research_mode='fast', debug_mode=False)

try:
    report = searcher_fast.research(
        query='Laurent-Philippe Albou',
        max_sources=25,
        search_depth='brief'  # Use brief to speed up test
    )

    print(f'\n✅ FAST MODE RESULTS:')
    print(f'   Sources in report: {len(report.sources)}')
    print(f'   Key findings: {len(report.key_findings)}')

except Exception as e:
    print(f'❌ Fast mode failed: {e}')
    import traceback
    traceback.print_exc()

# Restore
BasicDeepSearch._develop_search_questions_fast = original_develop_fast
BasicDeepSearch._develop_search_questions = original_develop_standard
