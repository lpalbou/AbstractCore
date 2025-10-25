from abstractcore.processing import BasicDeepResearcherA
from abstractcore import create_llm
import json
import re
from datetime import datetime

# Initialize (any model/provider works)
#llm = create_llm("openai", model="gpt-4o-mini")
model = "openai/gpt-oss-20b"
llm = create_llm("lmstudio", model=model)
researcher = BasicDeepResearcherA(llm, max_sources=25)

# Research topic
topic = "Laurent-Philippe Albou"

# Research (fast, reliable, high-quality)
result = researcher.research(topic)

# Create safe filename from topic
safe_modelname = re.sub(r'[^\w\s-]', '', model).strip()
safe_modelname = re.sub(r'[-\s]+', '_', safe_modelname).lower()
safe_filename = re.sub(r'[^\w\s-]', '', topic).strip()
safe_filename = re.sub(r'[-\s]+', '_', safe_filename).lower()
json_filename = f"{safe_filename}_{safe_modelname}_result.json"

# Convert result to JSON-serializable format
result_data = {
    "topic": topic,
    "timestamp": datetime.now().isoformat(),
    "confidence_score": result.confidence_score,
    "sources_count": len(result.sources_selected),
    "findings_count": len(result.key_findings),
    "key_findings": result.key_findings,
    "sources_selected": [
        {
            "title": source.title,
            "url": source.url,
            "content": source.content,
            "metadata": source.metadata if hasattr(source, 'metadata') else {}
        } for source in result.sources_selected
    ],
    "summary": result.summary if hasattr(result, 'summary') else "",
    "full_result": str(result)
}

# Save to JSON file
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False)

# Results
print(result)
print(f"✅ Confidence: {result.confidence_score:.2f}")
print(f"📚 Sources: {len(result.sources_selected)}")
print(f"🔑 Findings: {len(result.key_findings)}")
print(f"💾 Results saved to: {json_filename}")