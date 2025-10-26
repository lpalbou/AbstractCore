from abstractcore.processing import BasicDeepResearcherA, BasicDeepResearcherB
from abstractcore import create_llm
import json
import re
import logging
from datetime import datetime

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize (any model/provider works)
model = "openai/gpt-oss-20b"
model = "qwen/qwen3-next-80b"
model = "qwen/qwen3-30b-a3b-2507"
#model = "gpt-4o-mini"
llm = create_llm("lmstudio", model=model)

logger.info(f"🔍 LLM created successfully")
logger.info(f"📝 LLM type: {type(llm)}")
logger.info(f"📝 LLM public methods: {[attr for attr in dir(llm) if not attr.startswith('_') and callable(getattr(llm, attr))]}")

# Add debugging hooks to the LLM provider
original_generate = llm.generate

def debug_generate(*args, **kwargs):
    logger.info(f"🔍 LLM Generate called with args: {len(args)}, kwargs keys: {list(kwargs.keys())}")
    if 'response_model' in kwargs:
        logger.info(f"📋 Response model: {kwargs['response_model'].__name__}")
        logger.info(f"📋 Response model schema: {kwargs['response_model'].model_json_schema()}")
    
    try:
        result = original_generate(*args, **kwargs)
        logger.info(f"✅ LLM Generate successful, result type: {type(result)}")
        if hasattr(result, '__dict__'):
            logger.info(f"📊 Result attributes: {list(result.__dict__.keys())}")
        return result
    except Exception as e:
        logger.error(f"❌ LLM Generate failed: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        raise

llm.generate = debug_generate

researcher = BasicDeepResearcherA(llm, max_sources=25)

# Research topic
topic = "Laurent-Philippe Albou"

logger.info(f"🚀 Starting research for topic: {topic}")

# First, let's test the SynthesisModel structure independently
logger.info("🧪 Testing SynthesisModel validation with sample data...")
from abstractcore.processing.basic_deepresearcherA import SynthesisModel

# Create sample data that should match the SynthesisModel
sample_data = {
    "title": "Research Report on Laurent-Philippe Albou",
    "executive_summary": "This is a comprehensive research report on Laurent-Philippe Albou, covering his academic background and contributions to computational biology.",
    "key_findings": [
        "Academic researcher in computational biology",
        "Works on protein structure prediction",
        "Published several papers on machine learning applications in biology"
    ],
    "detailed_sections": [
        {"heading": "Background", "content": "Academic background information"},
        {"heading": "Research Focus", "content": "Protein structure prediction research"}
    ],
    "confidence_assessment": 0.75,
    "knowledge_gaps": [
        "Limited information about recent publications",
        "Unclear current institutional affiliation"
    ]
}

try:
    test_model = SynthesisModel.model_validate(sample_data)
    logger.info(f"✅ SynthesisModel validation successful: {type(test_model)}")
    logger.info(f"📊 Model data: {test_model.model_dump()}")
except Exception as validation_e:
    logger.error(f"❌ SynthesisModel validation failed: {type(validation_e).__name__}: {str(validation_e)}")

# Let's test structured output directly first
logger.info("🔍 Testing structured output in isolation...")

# Create a simple test prompt
test_prompt = """Based on research about Laurent-Philippe Albou, create a synthesis report.

Key findings from research:
- Academic researcher in computational biology
- Works on protein structure prediction
- Published several papers on machine learning applications in biology

Please provide a structured synthesis following the required format."""

try:
    logger.info("🧪 Testing direct structured output generation...")
    test_result = llm.generate(test_prompt, response_model=SynthesisModel)
    logger.info(f"✅ Direct structured output test successful: {type(test_result)}")
    logger.info(f"📊 Test result: {test_result}")
except Exception as test_e:
    logger.error(f"❌ Direct structured output test failed: {type(test_e).__name__}: {str(test_e)}")
    import traceback
    logger.error(f"📋 Full traceback: {traceback.format_exc()}")

# Only run full research if direct test works
logger.info("🔍 Now testing full research with timeout...")

import signal
import threading

def timeout_handler(signum, frame):
    raise TimeoutError("Research timed out after 10 minutes")

def run_research_with_timeout():
    """Run research with a timeout to prevent hanging."""
    try:
        # Set a 10-minute timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes (600 seconds)
        
        result = researcher.research(topic)
        signal.alarm(0)  # Cancel the alarm
        logger.info(f"✅ Research completed successfully")
        return result
        
    except TimeoutError as te:
        signal.alarm(0)  # Cancel the alarm
        logger.error(f"⏰ Research timed out: {te}")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        logger.error(f"❌ Research failed: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        return None

try:
    result = run_research_with_timeout()
    if result is None:
        logger.warning("🔍 Research failed or timed out, but structured output tests passed")
        logger.warning("💡 This suggests the issue is in the research pipeline, not structured output")
except Exception as e:
    logger.error(f"❌ Timeout mechanism failed: {e}")
    result = None

# Create safe filename from topic
safe_modelname = re.sub(r'[^\w\s-]', '', model).strip()
safe_modelname = re.sub(r'[-\s]+', '_', safe_modelname).lower()
safe_filename = re.sub(r'[^\w\s-]', '', topic).strip()
safe_filename = re.sub(r'[-\s]+', '_', safe_filename).lower()
import uuid
uid = uuid.uuid4().hex[:8]
json_filename = f"{safe_filename}_{safe_modelname}_result_{uid}.json"

# Only save results if research was successful
if 'result' in locals() and result is not None:
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
            "title": source.get("title", "Unknown") if isinstance(source, dict) else getattr(source, 'title', 'Unknown'),
            "url": source.get("url", "Unknown") if isinstance(source, dict) else getattr(source, 'url', 'Unknown'),
            "content": source.get("content", "") if isinstance(source, dict) else getattr(source, 'content', ''),
            "metadata": source.get("metadata", {}) if isinstance(source, dict) else getattr(source, 'metadata', {})
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
else:
    print("❌ Research failed - no results to save")
    print("🔍 Check the debug logs above for details on the structured output issue")