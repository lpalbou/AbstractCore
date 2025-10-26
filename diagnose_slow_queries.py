#!/usr/bin/env python3
"""
Diagnostic: Why are Strategy B queries taking 5+ minutes?

This script profiles Strategy B's execution to identify the bottleneck.
"""

import time
from abstractcore import create_llm
from abstractcore.processing.basic_deepresearcherB import BasicDeepResearcherB
from pydantic import BaseModel, Field
from typing import List

# Monkey-patch to track time spent
original_generate = None
call_times = []

def tracked_generate(*args, **kwargs):
    """Track every LLM call"""
    start = time.time()
    response_model = kwargs.get('response_model')

    print(f"\n{'='*80}")
    print(f"LLM CALL: {response_model.__name__ if response_model else 'No model'}")
    print(f"Prompt length: {len(args[0]) if args else 0} chars")

    try:
        result = original_generate(*args, **kwargs)
        duration = time.time() - start
        call_times.append({
            'model': response_model.__name__ if response_model else 'None',
            'duration': duration,
            'success': True
        })
        print(f"✅ SUCCESS in {duration:.2f}s")
        return result
    except Exception as e:
        duration = time.time() - start
        call_times.append({
            'model': response_model.__name__ if response_model else 'None',
            'duration': duration,
            'success': False,
            'error': type(e).__name__
        })
        print(f"❌ FAILED in {duration:.2f}s - {type(e).__name__}: {str(e)[:100]}")
        raise

def main():
    global original_generate

    print("="*80)
    print("DIAGNOSTIC: Why are Strategy B queries taking 5+ minutes?")
    print("="*80)

    # Create LLM
    print("\n1. Creating LLM instance...")
    llm = create_llm('lmstudio', model='qwen/qwen3-30b-a3b-2507')

    # Patch the generate method
    original_generate = llm.generate
    llm.generate = tracked_generate

    # Create researcher
    print("2. Creating BasicDeepResearcherB...")
    researcher = BasicDeepResearcherB(llm, max_sources=25)

    # Run a simple research query
    query = "What is deep learning?"

    print(f"\n3. Starting research for: '{query}'")
    print("   (This will show every LLM call with timing)")
    print("="*80)

    overall_start = time.time()

    try:
        result = researcher.research(query, depth="shallow")
        overall_duration = time.time() - overall_start

        print(f"\n{'='*80}")
        print(f"✅ RESEARCH COMPLETED in {overall_duration:.2f}s")
        print(f"{'='*80}")

    except Exception as e:
        overall_duration = time.time() - overall_start
        print(f"\n{'='*80}")
        print(f"❌ RESEARCH FAILED after {overall_duration:.2f}s")
        print(f"Error: {type(e).__name__}: {str(e)[:200]}")
        print(f"{'='*80}")

    # Analysis
    print(f"\n{'='*80}")
    print("CALL TIME ANALYSIS")
    print(f"{'='*80}")

    total_llm_time = sum(c['duration'] for c in call_times)
    successful_calls = [c for c in call_times if c['success']]
    failed_calls = [c for c in call_times if not c['success']]

    print(f"\nTotal LLM calls: {len(call_times)}")
    print(f"Successful: {len(successful_calls)}")
    print(f"Failed: {len(failed_calls)}")
    print(f"\nTotal LLM time: {total_llm_time:.2f}s")
    print(f"Overall duration: {overall_duration:.2f}s")
    print(f"Overhead (non-LLM): {overall_duration - total_llm_time:.2f}s")

    # Per-model breakdown
    print(f"\n{'─'*80}")
    print("Time breakdown by model:")
    print(f"{'─'*80}")

    by_model = {}
    for call in call_times:
        model = call['model']
        if model not in by_model:
            by_model[model] = {'total': 0, 'count': 0, 'failures': 0}
        by_model[model]['total'] += call['duration']
        by_model[model]['count'] += 1
        if not call['success']:
            by_model[model]['failures'] += 1

    for model, stats in sorted(by_model.items(), key=lambda x: x[1]['total'], reverse=True):
        avg = stats['total'] / stats['count']
        print(f"{model:30s}: {stats['total']:7.2f}s total ({stats['count']} calls, avg {avg:.2f}s/call, {stats['failures']} failures)")

    # Retry analysis
    print(f"\n{'─'*80}")
    print("RETRY ANALYSIS:")
    print(f"{'─'*80}")

    if failed_calls:
        print(f"\n⚠️  Found {len(failed_calls)} failed calls (likely validation errors)")
        print(f"   These trigger RETRIES, each retry = full LLM regeneration")
        print(f"\n   Failed call breakdown:")
        for i, call in enumerate(failed_calls, 1):
            print(f"   {i}. {call['model']:30s} - {call['duration']:.2f}s - {call.get('error', 'Unknown')}")

        # Calculate retry overhead
        retry_time = sum(c['duration'] for c in failed_calls)
        print(f"\n   Estimated retry overhead: {retry_time:.2f}s ({retry_time/total_llm_time*100:.1f}% of total LLM time)")
    else:
        print("✅ No failed calls detected")

    # Bottleneck identification
    print(f"\n{'='*80}")
    print("BOTTLENECK IDENTIFICATION")
    print(f"{'='*80}")

    if failed_calls:
        worst_model = max(by_model.items(), key=lambda x: x[1]['failures'])
        print(f"\n🔴 PRIMARY BOTTLENECK: {worst_model[0]}")
        print(f"   - {worst_model[1]['failures']} failures out of {worst_model[1]['count']} attempts")
        print(f"   - {worst_model[1]['total']:.2f}s total time spent")
        print(f"\n💡 SOLUTION: Simplify {worst_model[0]} or add fallback text parsing")

    slowest_model = max(by_model.items(), key=lambda x: x[1]['total'])
    print(f"\n⏱️  SLOWEST MODEL: {slowest_model[0]}")
    print(f"   - {slowest_model[1]['total']:.2f}s total time ({slowest_model[1]['total']/total_llm_time*100:.1f}% of LLM time)")
    print(f"   - {slowest_model[1]['count']} calls, avg {slowest_model[1]['total']/slowest_model[1]['count']:.2f}s/call")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    print(f"\n1. If validation failures are the issue:")
    print(f"   • Simplify Pydantic models (remove min_items, max_items constraints)")
    print(f"   • Add fallback text parsing (see Strategy A)")
    print(f"   • Reduce retry attempts from 3 to 1")

    print(f"\n2. If a specific model is slow:")
    print(f"   • Simplify the prompt for that model")
    print(f"   • Reduce the number of items requested")
    print(f"   • Cache results if possible")

    print(f"\n3. General optimization:")
    print(f"   • Use Strategy A instead (50-60s, 100% success)")
    print(f"   • Enable parallel execution where possible")
    print(f"   • Disable full content extraction (extract_full_content=False)")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
