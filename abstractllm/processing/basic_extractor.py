"""
Basic Extractor - High-quality semantic knowledge extraction with JSON-LD output

Features:
- Clean JSON-LD output with standard schema.org vocabulary
- Generic relationship IDs (r:1, r:2, etc.) with s:name property for type
- Single-pass LLM generation
- Robust post-processing validation
"""

from typing import Optional, List
import json
import logging
from pydantic import BaseModel, Field

from ..core.interface import AbstractLLMInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry

logger = logging.getLogger(__name__)


class BasicExtractor:
    """
    Basic Extractor for semantic knowledge extraction with multiple output formats

    Key features:
    - Multiple output formats: JSON-LD, RDF triples, minified JSON-LD
    - Generic relationship IDs (r:1, r:2, etc.) with s:name for relationship type
    - No orphaned entity references
    - Schema.org vocabulary
    - Production-ready output

    Examples:
        >>> extractor = BasicExtractor()

        # Default JSON-LD format
        >>> result = extractor.extract("Google created TensorFlow")
        >>> # {"@context": {...}, "@graph": [...]}

        # RDF triples format (SUBJECT PREDICATE OBJECT)
        >>> result = extractor.extract("Google created TensorFlow", output_format="triples")
        >>> # {"triples": [...], "simple_triples": ["Google creates TensorFlow"]}

        # Minified JSON-LD
        >>> result = extractor.extract("Google created TensorFlow", output_format="jsonld_minified")
        >>> # {"format": "jsonld_minified", "data": "{\"@context\":{...}}"}
    """

    def __init__(
        self,
        llm: Optional[AbstractLLMInterface] = None,
        max_chunk_size: int = 8000
    ):
        """Initialize the extractor"""
        if llm is None:
            try:
                self.llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", max_tokens=32000, max_output_tokens=8000)
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default Ollama model 'qwen3:4b-instruct-2507-q4_K_M': {e}\n\n"
                    "💡 To use the default model, please:\n"
                    "   1. Install Ollama from: https://ollama.com/\n"
                    "   2. Download the model: ollama pull qwen3:4b-instruct-2507-q4_K_M\n"
                    "   3. Start Ollama service\n\n"
                    "⚡ For best performance, consider these models:\n"
                    "   - qwen3-coder:30b (excellent for structured output, requires 32GB RAM)\n"
                    "   - gpt-oss:120b (highest quality, requires 120GB RAM)\n\n"
                    "🔧 Alternatively, provide a custom LLM instance:\n"
                    "   from abstractllm import create_llm\n"
                    "   from abstractllm.processing import BasicExtractor\n"
                    "   \n"
                    "   llm = create_llm('openai', model='gpt-4o-mini', max_tokens=32000, max_output_tokens=8000)\n"
                    "   extractor = BasicExtractor(llm)"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm

        self.max_chunk_size = max_chunk_size
        self.retry_strategy = FeedbackRetry(max_attempts=3)

    def extract(
        self,
        text: str,
        domain_focus: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        style: Optional[str] = None,
        length: Optional[str] = None,
        output_format: str = "jsonld"
    ) -> dict:
        """
        Extract entities and relationships from text

        Args:
            text: Text to extract knowledge from
            domain_focus: Optional domain to focus on
            entity_types: Optional specific entity types to extract (reserved for future use)
            style: Optional style parameter (reserved for future use)
            length: Extract length ("brief", "standard", "detailed", "comprehensive")
            output_format: Output format ("jsonld", "triples", "jsonld_minified")

        Returns:
            dict: Extracted knowledge in requested format
        """
        # Note: entity_types and style parameters are reserved for future enhancements
        _ = entity_types  # Reserved for future entity type filtering
        _ = style  # Reserved for future style customization

        if len(text) > self.max_chunk_size:
            result = self._extract_long_document(text, domain_focus, length)
        else:
            result = self._extract_single_chunk(text, domain_focus, length)

        return self._format_output(result, output_format)

    def _extract_single_chunk(
        self,
        text: str,
        domain_focus: Optional[str],
        length: Optional[str]
    ) -> dict:
        """Extract from a single chunk using single-pass LLM generation"""

        logger.info("Starting JSON-LD extraction with descriptive IDs")

        entity_limit = self._get_entity_limit(length)
        domain_note = f" Focus on {domain_focus} domain." if domain_focus else ""

        # Knowledge extraction prompt with JSON-LD output
        prompt = f"""You are an expert in Semantic extraction and your task it to create consistent high-quality information-rich knowledge graphs. Your output is a JSON-LD knowledge graph with entities and relationships.{domain_note}.

STEP 1 : always start by identifying the entities in the text and to classify them in 2 groups based on their importance: the primary entities are the main subjects, the main topics or main ideas; the secondary entities relates either to details of the primary entities or to additional information.

ENTITY TYPES must be one of:
- sk:Concept - Abstract concepts, technologies
- s:Person - People by name
- s:Organization - Companies, institutions
- s:Event - Events, meetings, conferences, etc
- s:Goal - Abstract goals, objectives
- s:Task - Abstract tasks, actions
- s:Place - Locations
- s:Product - Products, services
- s:SoftwareApplication - Software, libraries, frameworks, tools

LIMITS: try to limit the number of entities to {entity_limit}.

STEP 2 : ONCE all the entities have been created and annotated, then identify and characterize all the relationships between the selected entities.

RELATIONSHIP TYPES must be one of:
- is, is_not, part_of, transforms, provides, describes, mentions, integrates, supports, discourages, requires, uses, creates, compatible_with, works_with, enables, disables, constitutes, models, occurs_in, occurs_when

STEP 3 : create the JSON-LD knowledge graph with the entities and relationships identified and characterized in step 1 and 2. Be extra mindful to use the correct JSON-LD syntax. An example is provided just below.

----------------------------------
EXAMPLE: HERE IS AN INPUT TEXT: "OpenAI created GPT-4. Microsoft Copilot uses GPT-4 for code generation."

AND HERE IS THE EXPECTED JSON-LD KNOWLEDGE GRAPH OUTPUT FOR THAT INPUT TEXT:
{{
  "@context": {{
    "s": "https://schema.org/",
    "e": "http://example.org/entity/",
    "r": "http://example.org/relation/",
    "confidence": "http://example.org/confidence"
  }},
  "@graph": [
    {{
      "@id": "e:openai",
      "@type": "s:Organization",
      "s:name": "OpenAI",
      "s:description": "AI company that created GPT-4",
      "confidence": 0.95
    }},
    {{
      "@id": "e:gpt4",
      "@type": "s:SoftwareApplication",
      "s:name": "GPT-4",
      "s:description": "Language model",
      "confidence": 0.95
    }},
    {{
      "@id": "e:copilot",
      "@type": "s:SoftwareApplication",
      "s:name": "Microsoft Copilot",
      "s:description": "Code generation tool",
      "confidence": 0.95
    }},
    {{
      "@id": "r:1",
      "@type": "s:Relationship",
      "s:name": "creates",
      "s:about": {{"@id": "e:openai"}},
      "s:object": {{"@id": "e:gpt4"}},
      "s:description": "OpenAI created GPT-4",
      "confidence": 0.95,
      "strength": 0.9
    }},
    {{
      "@id": "r:2",
      "@type": "s:Relationship",
      "s:name": "uses",
      "s:about": {{"@id": "e:copilot"}},
      "s:object": {{"@id": "e:gpt4"}},
      "s:description": "Microsoft Copilot uses GPT-4",
      "confidence": 0.95,
      "strength": 0.9
    }}
  ]
}}
----------------------------------

FOLLOW STEPS 1, 2 AND 3 TO CREATE THE JSON-LD KNOWLEDGE GRAPH FOR THAT INPUT TEXT:
{text}

CRITICAL : ONLY OUTPUT THE FULL JSON-LD WITHOUT ANY OTHER TEXT OR COMMENTS.
"""

        # Generate
        response = self.llm.generate(prompt, retry_strategy=self.retry_strategy)

        # Extract text
        if hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)

        response_text = response_text.strip()

        # Parse JSON
        import json
        try:
            result = json.loads(response_text)

            # Validate structure
            if "@context" not in result or "@graph" not in result:
                logger.error("Invalid JSON-LD structure")
                return self._create_empty_graph()

            # Normalize JSON-LD references (convert strings to objects)
            result = self._normalize_jsonld_references(result)

            # Remove dangling references
            result = self._remove_dangling_references(result)

            # Log results
            entities = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('e:')]
            relationships = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('r:')]
            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")

            # Attempt self-correction
            from ..utils.self_fixes import fix_json
            corrected_json = fix_json(response_text)

            if corrected_json:
                try:
                    result = json.loads(corrected_json)
                    if "@context" in result and "@graph" in result:
                        result = self._normalize_jsonld_references(result)
                        result = self._remove_dangling_references(result)
                        entities = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('e:')]
                        relationships = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('r:')]
                        logger.info(f"✅ JSON self-correction successful! Extracted {len(entities)} entities and {len(relationships)} relationships")
                        return result
                except json.JSONDecodeError:
                    pass

            logger.error("JSON self-correction failed, returning empty graph")
            return self._create_empty_graph()

    def _remove_dangling_references(self, result: dict) -> dict:
        """Remove relationships that reference non-existent entities"""
        defined_entities = {
            item['@id']
            for item in result.get('@graph', [])
            if item.get('@id', '').startswith('e:')
        }

        cleaned_graph = []
        removed_count = 0

        for item in result.get('@graph', []):
            item_id = item.get('@id', '')

            # Keep all entities
            if item_id.startswith('e:'):
                cleaned_graph.append(item)

            # Keep only valid relationships
            elif item_id.startswith('r:'):
                # Safely extract source and target IDs
                source_ref = item.get('s:about', {})
                target_ref = item.get('s:object', {})

                if isinstance(source_ref, dict):
                    source_id = source_ref.get('@id', '')
                else:
                    source_id = str(source_ref) if source_ref else ''
                    logger.debug(f"Relationship {item_id} has non-dict s:about: {type(source_ref)} (normalized)")

                if isinstance(target_ref, dict):
                    target_id = target_ref.get('@id', '')
                else:
                    target_id = str(target_ref) if target_ref else ''
                    logger.debug(f"Relationship {item_id} has non-dict s:object: {type(target_ref)} (normalized)")

                if source_id in defined_entities and target_id in defined_entities:
                    cleaned_graph.append(item)
                else:
                    removed_count += 1
                    logger.debug(f"Removed dangling relationship: {item_id}")

        if removed_count > 0:
            logger.warning(f"Removed {removed_count} relationships with dangling references")

        result['@graph'] = cleaned_graph
        return result

    def _get_entity_limit(self, length: Optional[str]) -> int:
        """Get entity extraction limit based on length parameter"""
        if length == "brief":
            return 10
        elif length == "detailed":
            return 25
        elif length == "comprehensive":
            return 50
        else:  # standard
            return 15

    def _create_empty_graph(self) -> dict:
        """Create empty JSON-LD graph for error cases"""
        return {
            "@context": {
                "s": "https://schema.org/",
                "d": "http://purl.org/dc/terms/",
                "sk": "http://www.w3.org/2004/02/skos/core#",
                "e": "http://example.org/entity/",
                "r": "http://example.org/relation/",
                "confidence": "http://example.org/confidence"
            },
            "@graph": []
        }

    def _normalize_jsonld_references(self, result: dict) -> dict:
        """
        Normalize JSON-LD references to ensure proper object format.

        Some LLMs generate string references instead of object references:
        - Wrong: "s:about": "e:entity_id"
        - Correct: "s:about": {"@id": "e:entity_id"}

        This method fixes such inconsistencies.
        """
        if not isinstance(result, dict) or "@graph" not in result:
            return result

        normalized_graph = []

        for item in result.get("@graph", []):
            if not isinstance(item, dict):
                normalized_graph.append(item)
                continue

            # Only process relationships that have reference fields
            item_id = item.get("@id", "")
            if item_id.startswith("r:"):
                # Create a copy to avoid modifying the original
                normalized_item = item.copy()

                # Normalize s:about reference
                if "s:about" in normalized_item:
                    about_ref = normalized_item["s:about"]
                    if isinstance(about_ref, str) and about_ref.startswith("e:"):
                        normalized_item["s:about"] = {"@id": about_ref}

                # Normalize s:object reference
                if "s:object" in normalized_item:
                    object_ref = normalized_item["s:object"]
                    if isinstance(object_ref, str) and object_ref.startswith("e:"):
                        normalized_item["s:object"] = {"@id": object_ref}

                normalized_graph.append(normalized_item)
            else:
                # Keep entities and other items as-is
                normalized_graph.append(item)

        # Return result with normalized graph
        result_copy = result.copy()
        result_copy["@graph"] = normalized_graph
        return result_copy

    def _extract_long_document(
        self,
        text: str,
        domain_focus: Optional[str],
        length: Optional[str]
    ) -> dict:
        """Handle long documents using chunking"""
        chunks = self._split_text_into_chunks(text)

        if len(chunks) == 1:
            return self._extract_single_chunk(chunks[0], domain_focus, length)

        # Extract from each chunk
        all_entities = []
        all_relationships = []
        seen_entity_ids = set()
        seen_relationship_ids = set()

        chunk_result = None  # Initialize to handle case where no chunks succeed
        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting from chunk {i+1}/{len(chunks)}")
            chunk_result = self._extract_single_chunk(chunk, domain_focus, length)

            # Validate chunk result
            if not isinstance(chunk_result, dict):
                logger.error(f"Chunk {i+1} returned {type(chunk_result)} instead of dict")
                chunk_result = self._create_empty_graph()
            elif "@graph" not in chunk_result:
                logger.error(f"Chunk {i+1} missing @graph")
                chunk_result = self._create_empty_graph()

            # Safely merge entities with additional error handling
            try:
                graph_items = chunk_result.get("@graph", [])
                if not isinstance(graph_items, list):
                    logger.error(f"Chunk {i+1} @graph is not a list: {type(graph_items)}")
                    graph_items = []

                for entity in graph_items:
                    if not isinstance(entity, dict):
                        logger.warning(f"Chunk {i+1} contains non-dict entity: {type(entity)}")
                        continue
                    entity_id = entity.get("@id", "")
                    if entity_id.startswith("e:") and entity_id not in seen_entity_ids:
                        all_entities.append(entity)
                        seen_entity_ids.add(entity_id)

                # Merge relationships (deduplicate by source-target-relation)
                for item in graph_items:
                    if not isinstance(item, dict):
                        logger.warning(f"Chunk {i+1} contains non-dict item: {type(item)}")
                        continue
                    item_id = item.get("@id", "")
                    if item_id.startswith("r:"):
                        source_ref = item.get("s:about", {})
                        target_ref = item.get("s:object", {})

                        # Safely extract IDs
                        if isinstance(source_ref, dict):
                            source = source_ref.get("@id", "")
                        else:
                            source = str(source_ref) if source_ref else ""

                        if isinstance(target_ref, dict):
                            target = target_ref.get("@id", "")
                        else:
                            target = str(target_ref) if target_ref else ""

                        relation = item.get("s:name", "")
                        triple = (source, relation, target)

                        if triple not in seen_relationship_ids:
                            all_relationships.append(item)
                            seen_relationship_ids.add(triple)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue

        # Build final result with safe context extraction
        default_context = self._create_empty_graph()["@context"]
        final_context = default_context

        # Try to get context from the last valid chunk, fallback to default
        if isinstance(chunk_result, dict) and "@context" in chunk_result:
            final_context = chunk_result["@context"]

        final_result = {
            "@context": final_context,
            "@graph": all_entities + all_relationships
        }
        return final_result

    def _split_text_into_chunks(self, text: str, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            if end < len(text):
                # Try to break at sentence boundary
                sentence_end = text.rfind('. ', start + self.max_chunk_size - 500, end)
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 2

            chunks.append(text[start:end].strip())

            if end >= len(text):
                break
            start = max(start + self.max_chunk_size - overlap, end - overlap)

        return chunks

    def refine_extraction(
        self,
        text: str,
        previous_extraction: dict,
        domain_focus: Optional[str] = None,
        length: Optional[str] = None
    ) -> dict:
        """
        Refine a previous extraction by finding missing entities/relationships
        and creating a complete refined knowledge graph.

        Args:
            text: The original source text
            previous_extraction: The previous JSON-LD extraction result
            domain_focus: Optional domain focus
            length: Extract length for refinement scope

        Returns:
            Complete refined JSON-LD extraction
        """
        logger.info("Starting extraction refinement")

        if len(text) > self.max_chunk_size:
            return self._refine_long_document(text, previous_extraction, domain_focus, length)
        else:
            return self._refine_single_chunk(text, previous_extraction, domain_focus, length)

    def _refine_single_chunk(
        self,
        text: str,
        previous_extraction: dict,
        domain_focus: Optional[str],
        length: Optional[str]
    ) -> dict:
        """Refine extraction from a single chunk using the complete graph approach"""

        logger.info("Starting JSON-LD refinement with complete graph output")

        entity_limit = self._get_entity_limit(length) if length else 20  # More generous for refinement
        domain_note = f" Focus on {domain_focus} domain." if domain_focus else ""

        # Create formatted summary of existing extraction
        prev_entities = [item for item in previous_extraction.get('@graph', [])
                        if item.get('@id', '').startswith('e:')]
        prev_relationships = [item for item in previous_extraction.get('@graph', [])
                             if item.get('@id', '').startswith('r:')]

        # Build concise existing graph summary
        def safe_extract_id(ref):
            """Safely extract @id from a reference that might be dict or string"""
            if isinstance(ref, dict):
                return ref.get('@id', '?')
            elif ref:
                return str(ref)
            else:
                return '?'

        existing_entities = "\n".join([
            f"  - {e.get('s:name', e.get('@id'))} ({e.get('@type', 'Unknown')})"
            for e in prev_entities[:15]  # Limit to avoid token overflow
        ])

        existing_relationships = "\n".join([
            f"  - {safe_extract_id(r.get('s:about', {}))} --[{r.get('s:name', '?')}]--> {safe_extract_id(r.get('s:object', {}))}"
            for r in prev_relationships[:15]  # Limit to avoid token overflow
        ])

        # Knowledge graph refinement prompt with JSON-LD output (matches initial extraction format)
        prompt = f"""You are an expert in Semantic extraction and your task is to refine and improve an existing knowledge graph. Your output is a COMPLETE refined JSON-LD knowledge graph with entities and relationships.{domain_note}

EXISTING GRAPH TO REFINE:
Current Entities:
{existing_entities if existing_entities else "  (none)"}

Current Relationships:
{existing_relationships if existing_relationships else "  (none)"}

REFINEMENT TASK: Review the text below and the existing graph above. Create a COMPLETE refined knowledge graph that:
1. INCLUDES all valuable entities and relationships from the existing graph
2. ADDS any missing entities that should have been extracted
3. ADDS any missing relationships that should have been captured
4. CORRECTS any relationship directions that are wrong
5. REMOVES any entities or relationships that are not supported by the text

STEP 1: Identify the complete SET of entities in the text, starting with the existing entities and adding any missing ones. Classify them in 2 groups based on their importance: the primary entities are the main subjects, the main topics or main ideas; the secondary entities relate either to details of the primary entities or to additional information.

ENTITY TYPES must be one of:
- s:Person - People by name
- s:Organization - Companies, institutions
- s:Event - Events, meetings, conferences, etc
- s:Goal - Abstract goals, objectives
- s:Task - Abstract tasks, actions
- s:SoftwareApplication - Software, libraries, frameworks, tools
- s:Place - Locations
- s:Product - Products, services
- sk:Concept - Abstract concepts, technologies

LIMITS: try to limit the total number of entities to {entity_limit}.

STEP 2: ONCE all the entities have been created and annotated, then identify and characterize all the relationships between the selected entities.

RELATIONSHIP TYPES must be one of:
- is, is_not, part_of, transforms, provides, describes, mentions, integrates, supports, discourages, requires, uses, creates, compatible_with, works_with, enables, disables, constitutes, models, occurs_in, occurs_when

STEP 3: Create the COMPLETE refined JSON-LD knowledge graph with ALL entities and relationships (existing + new + corrected). Be extra mindful to use the correct JSON-LD syntax. An example is provided just below.

----------------------------------
EXAMPLE: HERE IS AN INPUT TEXT: "OpenAI created GPT-4. Microsoft Copilot uses GPT-4 for code generation."

AND HERE IS THE EXPECTED JSON-LD KNOWLEDGE GRAPH OUTPUT FOR THAT INPUT TEXT:
{{
  "@context": {{
    "s": "https://schema.org/",
    "e": "http://example.org/entity/",
    "r": "http://example.org/relation/",
    "confidence": "http://example.org/confidence"
  }},
  "@graph": [
    {{
      "@id": "e:openai",
      "@type": "s:Organization",
      "s:name": "OpenAI",
      "s:description": "AI company that created GPT-4",
      "confidence": 0.95
    }},
    {{
      "@id": "e:gpt4",
      "@type": "s:SoftwareApplication",
      "s:name": "GPT-4",
      "s:description": "Language model",
      "confidence": 0.95
    }},
    {{
      "@id": "e:copilot",
      "@type": "s:SoftwareApplication",
      "s:name": "Microsoft Copilot",
      "s:description": "Code generation tool",
      "confidence": 0.95
    }},
    {{
      "@id": "r:1",
      "@type": "s:Relationship",
      "s:name": "creates",
      "s:about": {{"@id": "e:openai"}},
      "s:object": {{"@id": "e:gpt4"}},
      "s:description": "OpenAI created GPT-4",
      "confidence": 0.95,
      "strength": 0.9
    }},
    {{
      "@id": "r:2",
      "@type": "s:Relationship",
      "s:name": "uses",
      "s:about": {{"@id": "e:copilot"}},
      "s:object": {{"@id": "e:gpt4"}},
      "s:description": "Microsoft Copilot uses GPT-4",
      "confidence": 0.95,
      "strength": 0.9
    }}
  ]
}}
----------------------------------

FOLLOW STEPS 1, 2 AND 3 TO CREATE THE COMPLETE REFINED JSON-LD KNOWLEDGE GRAPH FOR THIS TEXT:
{text}

CRITICAL: ONLY OUTPUT THE FULL JSON-LD WITHOUT ANY OTHER TEXT OR COMMENTS."""

        # Generate
        response = self.llm.generate(prompt, retry_strategy=self.retry_strategy)

        # Extract text
        if hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)

        response_text = response_text.strip()

        # Parse JSON
        import json
        try:
            result = json.loads(response_text)

            # Validate structure
            if "@context" not in result or "@graph" not in result:
                logger.error("Invalid refined JSON-LD structure")
                return previous_extraction

            # Normalize JSON-LD references (convert strings to objects)
            result = self._normalize_jsonld_references(result)

            # Remove dangling references
            result = self._remove_dangling_references(result)

            # Log results
            entities = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('e:')]
            relationships = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('r:')]
            prev_entities = [item for item in previous_extraction.get('@graph', []) if item.get('@id', '').startswith('e:')]
            prev_relationships = [item for item in previous_extraction.get('@graph', []) if item.get('@id', '').startswith('r:')]

            logger.info(f"Refinement: {len(prev_entities)} → {len(entities)} entities, {len(prev_relationships)} → {len(relationships)} relationships")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Refinement JSON parsing failed: {e}")

            # Attempt self-correction
            from ..utils.self_fixes import fix_json
            corrected_json = fix_json(response_text)

            if corrected_json:
                try:
                    result = json.loads(corrected_json)
                    if "@context" in result and "@graph" in result:
                        result = self._normalize_jsonld_references(result)
                        result = self._remove_dangling_references(result)
                        entities = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('e:')]
                        relationships = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('r:')]
                        logger.info(f"✅ Refinement JSON self-correction successful! Refined to {len(entities)} entities and {len(relationships)} relationships")
                        return result
                except json.JSONDecodeError:
                    pass

            logger.error("Refinement JSON self-correction failed, returning previous extraction")
            return previous_extraction

    def _refine_long_document(
        self,
        text: str,
        previous_extraction: dict,
        domain_focus: Optional[str],
        length: Optional[str]
    ) -> dict:
        """Handle long documents using chunking for refinement"""
        chunks = self._split_text_into_chunks(text)

        if len(chunks) == 1:
            return self._refine_single_chunk(chunks[0], previous_extraction, domain_focus, length)

        logger.info(f"Refining long document with {len(chunks)} chunks")

        # Start with the previous extraction as the base
        current_extraction = previous_extraction

        # Refine each chunk against the evolving extraction
        for i, chunk in enumerate(chunks):
            logger.info(f"Refining chunk {i+1}/{len(chunks)}")

            # Refine this chunk against the current extraction
            chunk_refinement = self._refine_single_chunk(chunk, current_extraction, domain_focus, length)

            # Validate chunk result
            if not isinstance(chunk_refinement, dict) or "@graph" not in chunk_refinement:
                logger.warning(f"Chunk {i+1} refinement failed, skipping")
                continue

            # Update current extraction with refinements from this chunk
            current_extraction = chunk_refinement

        logger.info("Long document refinement completed")
        return current_extraction


    def _format_output(self, jsonld_result: dict, output_format: str) -> dict:
        """
        Format the output according to the requested format

        Args:
            jsonld_result: The JSON-LD extraction result
            output_format: Desired output format ("jsonld", "triples", "jsonld_minified")

        Returns:
            dict: Formatted result
        """
        # Validate input - ensure it's a dict
        if not isinstance(jsonld_result, dict):
            logger.error(f"_format_output received {type(jsonld_result)} instead of dict: {repr(jsonld_result)}")
            jsonld_result = self._create_empty_graph()

        # Ensure it has the required structure
        if "@graph" not in jsonld_result:
            logger.error("_format_output received dict without @graph")
            jsonld_result = self._create_empty_graph()

        if output_format == "jsonld":
            return jsonld_result
        elif output_format == "jsonld_minified":
            return self._minify_jsonld(jsonld_result)
        elif output_format == "triples":
            return self._convert_to_triples(jsonld_result)
        else:
            logger.warning(f"Unknown output format '{output_format}', defaulting to jsonld")
            return jsonld_result

    def _minify_jsonld(self, jsonld_result: dict) -> dict:
        """
        Convert JSON-LD result to minified JSON string format

        Args:
            jsonld_result: The JSON-LD extraction result

        Returns:
            dict: Result with minified JSON string
        """
        import json
        minified_json = json.dumps(jsonld_result, ensure_ascii=False, separators=(',', ':'))
        return {
            "format": "jsonld_minified",
            "data": minified_json,
            "entities_count": len([item for item in jsonld_result.get('@graph', [])
                                  if item.get('@id', '').startswith('e:')]),
            "relationships_count": len([item for item in jsonld_result.get('@graph', [])
                                       if item.get('@id', '').startswith('r:')])
        }

    def _convert_to_triples(self, jsonld_result: dict) -> dict:
        """
        Convert JSON-LD result to RDF-style triples (SUBJECT PREDICATE OBJECT)

        Based on semantic web/RDF/OWL standards:
        - Subject: The entity being described (URI or identifier)
        - Predicate: The property/relationship (URI describing the relationship)
        - Object: The value or target entity (URI, identifier, or literal)

        Args:
            jsonld_result: The JSON-LD extraction result

        Returns:
            dict: Result with triples array and entity definitions
        """
        triples = []
        entities = {}

        graph = jsonld_result.get('@graph', [])
        context = jsonld_result.get('@context', {})

        # Extract entities for reference
        for item in graph:
            item_id = item.get('@id', '')
            if item_id.startswith('e:'):
                entities[item_id] = {
                    'id': item_id,
                    'type': item.get('@type', ''),
                    'name': item.get('s:name', ''),
                    'description': item.get('s:description', ''),
                    'confidence': item.get('confidence', 0.0)
                }

        # Extract relationship triples
        for item in graph:
            item_id = item.get('@id', '')
            if item_id.startswith('r:'):
                subject_ref = item.get('s:about', {})
                object_ref = item.get('s:object', {})
                predicate = item.get('s:name', '')

                subject_id = subject_ref.get('@id', '') if isinstance(subject_ref, dict) else str(subject_ref)
                object_id = object_ref.get('@id', '') if isinstance(object_ref, dict) else str(object_ref)

                if subject_id and predicate and object_id:
                    # Get human-readable names if available
                    subject_name = entities.get(subject_id, {}).get('name', subject_id)
                    object_name = entities.get(object_id, {}).get('name', object_id)

                    triple = {
                        'subject': subject_id,
                        'subject_name': subject_name,
                        'predicate': predicate,
                        'object': object_id,
                        'object_name': object_name,
                        'triple_text': f"{subject_name} {predicate} {object_name}",
                        'confidence': item.get('confidence', 0.0),
                        'strength': item.get('strength', 0.0),
                        'description': item.get('s:description', '')
                    }
                    triples.append(triple)

        # Simple text format for easy consumption
        simple_triples = []
        for triple in triples:
            simple_triples.append(f"{triple['subject_name']} {triple['predicate']} {triple['object_name']}")

        return {
            "format": "triples",
            "triples": triples,
            "simple_triples": simple_triples,
            "entities": entities,
            "context": context,
            "statistics": {
                "entities_count": len(entities),
                "relationships_count": len(triples),
                "total_triples": len(triples)
            }
        }

