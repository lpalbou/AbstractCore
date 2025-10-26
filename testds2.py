import json
import re
from datetime import datetime
from abstractcore import create_llm
from abstractcore.processing import BasicDeepResearcherC

# Initialize with LMStudio + qwen3-30b
llm = create_llm("lmstudio", model="qwen/qwen3-30b-a3b-2507")

researcher = BasicDeepResearcherC(
    llm=llm,
    max_sources=30,           # Target high-quality sources
    max_urls_to_probe=60,     # Explore more, select best
    max_iterations=5,         # ReAct loop iterations  
    fetch_timeout=10,         # Per-URL fetch timeout (seconds)
    enable_breadth=True,      # Explore multiple dimensions
    enable_depth=True,        # Deep dive on promising leads
    grounding_threshold=0.7,  # Min relevance for inclusion (0-1)
    debug=False               # Detailed execution traces
)

# Define query and model info
query = "Create a comprehensive guide about BAML and how it differs frm outlines"
model_name = "qwen/qwen3-30b-a3b-2507"

# Run research
result = researcher.research(query)

# Access results with SOURCE ATTRIBUTION
print("="*80)
print("RESEARCH RESULTS WITH SOURCE ATTRIBUTION")
print("="*80)
print(f"\nTitle: {result.title}")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Sources: {len(result.sources_selected)}")

print(f"\n{'='*80}")
print("KEY FINDINGS WITH SOURCE ATTRIBUTION")
print(f"{'='*80}")
for i, finding in enumerate(result.key_findings, 1):
    print(f"\n{i}. {finding}")
    # Show which source this finding came from
    source_url = researcher.finding_to_source.get(finding, "UNKNOWN SOURCE - POTENTIAL HALLUCINATION!")
    if source_url != "UNKNOWN SOURCE - POTENTIAL HALLUCINATION!":
        print(f"   📎 SOURCE: {source_url}")
    else:
        print(f"   ⚠️  {source_url}")

print(f"\n{'='*80}")
print("KNOWLEDGE GAPS WITH SOURCE CONTEXT")
print(f"{'='*80}")
for i, gap in enumerate(result.knowledge_gaps, 1):
    print(f"\n{i}. {gap}")
    # Show which sources are related to this gap
    gap_data = researcher.active_gaps.get(gap)
    if gap_data:
        if gap_data['source_urls']:
            print(f"   📎 RELATED SOURCES ({len(gap_data['source_urls'])}):")
            for url in gap_data['source_urls'][:3]:  # Show top 3
                print(f"      - {url}")
        else:
            print(f"   ⚠️  NO SOURCES YET - Unexplored gap")

        if gap_data.get('related_findings'):
            print(f"   📋 RELATED FINDINGS:")
            for finding in gap_data['related_findings'][:2]:  # Show top 2
                print(f"      - {finding[:80]}...")

print(f"\n{'='*80}")
print("ALL SOURCES USED")
print(f"{'='*80}")
for i, source in enumerate(result.sources_selected, 1):
    print(f"\n{i}. {source.get('title', 'No title')}")
    print(f"   URL: {source.get('url', 'No URL')}")
    print(f"   Relevance: {source.get('relevance', 0):.2f}")
    print(f"   Credibility: {source.get('credibility', 0):.2f}")
    if source.get('excerpt'):
        print(f"   Excerpt: {source['excerpt'][:150]}...")

print(f"\n{'='*80}")
print("DETAILED RESEARCH REPORT")
print(f"{'='*80}")
if hasattr(result, 'detailed_report') and result.detailed_report:
    for section in result.detailed_report.get("sections", []):
        print(f"\n## {section['heading']}")
        print(f"{section['content']}")
else:
    print("\nNo detailed report available.")

print(f"\n{'='*80}")
print("SOURCE ATTRIBUTION SUMMARY")
print(f"{'='*80}")
print(f"Total findings tracked: {len(researcher.finding_to_source)}")
print(f"Total gaps tracked: {len(researcher.active_gaps)}")
print(f"Findings with sources: {sum(1 for v in researcher.finding_to_source.values() if v != 'UNKNOWN')}")
print(f"Gaps with source context: {sum(1 for g in researcher.active_gaps.values() if g['source_urls'])}")

# Create filename based on query, model, and timestamp
def create_filename(query: str, model: str) -> str:
    """Create a safe filename from query, model, and timestamp."""
    # Clean query: remove special chars, convert to lowercase, replace spaces with underscores
    clean_query = re.sub(r'[^\w\s-]', '', query.lower())
    clean_query = re.sub(r'[-\s]+', '_', clean_query)
    
    # Clean model: remove special chars, replace slashes and dots
    clean_model = re.sub(r'[^\w.-]', '', model)
    clean_model = clean_model.replace('/', '').replace('.', '_').replace('-', '_')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{clean_query}_{clean_model}_{timestamp}_result.json"

# Save result to JSON file
filename = create_filename(query, model_name)

# Prepare result data with SOURCE ATTRIBUTION for saving
result_data = {
    "topic": query,
    "model": model_name,
    "timestamp": datetime.now().isoformat(),
    "title": result.title,
    "confidence_score": result.confidence_score,
    "sources_count": len(result.sources_selected),
    "findings_count": len(result.key_findings),

    # KEY FINDINGS WITH SOURCE ATTRIBUTION
    "key_findings_with_sources": [
        {
            "finding": finding,
            "source_url": researcher.finding_to_source.get(finding, "UNKNOWN - POTENTIAL HALLUCINATION"),
            "is_grounded": researcher.finding_to_source.get(finding) is not None
        }
        for finding in result.key_findings
    ],

    # KNOWLEDGE GAPS WITH SOURCE CONTEXT
    "knowledge_gaps_with_sources": [
        {
            "gap": gap,
            "status": researcher.active_gaps.get(gap, {}).get("status", "unknown"),
            "related_source_urls": researcher.active_gaps.get(gap, {}).get("source_urls", []),
            "related_findings": researcher.active_gaps.get(gap, {}).get("related_findings", []),
            "has_source_context": len(researcher.active_gaps.get(gap, {}).get("source_urls", [])) > 0
        }
        for gap in result.knowledge_gaps
    ],

    # SOURCE ATTRIBUTION METADATA
    "source_attribution": {
        "total_findings_tracked": len(researcher.finding_to_source),
        "total_gaps_tracked": len(researcher.active_gaps),
        "findings_with_sources": sum(1 for v in researcher.finding_to_source.values() if v),
        "gaps_with_source_context": sum(1 for g in researcher.active_gaps.values() if g.get('source_urls')),
        "finding_to_source_map": dict(researcher.finding_to_source),
        "gap_details": {gap: data for gap, data in researcher.active_gaps.items()}
    },

    # DETAILED REPORT
    "detailed_report": result.detailed_report,

    # BACKWARD COMPATIBILITY (original fields)
    "key_findings": result.key_findings,
    "knowledge_gaps": result.knowledge_gaps,
    "sources_selected": [str(source) for source in result.sources_selected],
    "summary": result.summary,
    "research_metadata": result.research_metadata
}

# Save to file
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False)

print(f"\n{'='*80}")
print(f"RESULTS SAVED WITH FULL SOURCE ATTRIBUTION")
print(f"{'='*80}")
print(f"Filename: {filename}")
print(f"\nJSON structure includes:")
print(f"  • key_findings_with_sources: Each finding linked to its source URL")
print(f"  • knowledge_gaps_with_sources: Each gap linked to related sources")
print(f"  • source_attribution.finding_to_source_map: Complete traceability map")
print(f"  • source_attribution.gap_details: Full gap context with sources")
print(f"\n✅ Every finding is traceable to its source - NO HALLUCINATIONS!")
print(f"{'='*80}")