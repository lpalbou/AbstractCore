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
query = "Laurent-Philippe Albou"
model_name = "qwen/qwen3-30b-a3b-2507"

# Run research
result = researcher.research(query)

# Access results
print(f"Title: {result.title}")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Sources: {len(result.sources_selected)}")
print(f"Findings: {result.key_findings}")
print(f"Gaps: {result.knowledge_gaps}")

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

# Prepare result data for saving
result_data = {
    "topic": query,
    "model": model_name,
    "timestamp": datetime.now().isoformat(),
    "title": result.title,
    "confidence_score": result.confidence_score,
    "sources_count": len(result.sources_selected),
    "findings_count": len(result.key_findings),
    "key_findings": result.key_findings,
    "knowledge_gaps": result.knowledge_gaps,
    "sources_selected": [str(source) for source in result.sources_selected],
    "summary": result.summary,
    "full_result": str(result)
}

# Save to file
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False)

print(f"\nResult saved to: {filename}")