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
                self.llm = create_llm("ollama", model="gemma3:1b-it-qat", max_tokens=16000)
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default Ollama model 'gemma3:1b-it-qat': {e}\n\n"
                    "💡 To use the default model, please:\n"
                    "   1. Install Ollama from: https://ollama.com/\n"
                    "   2. Download the model: ollama pull gemma3:1b-it-qat\n"
                    "   3. Start Ollama service\n\n"
                    "🔧 Alternatively, provide a custom LLM instance:\n"
                    "   from abstractllm import create_llm\n"
                    "   from abstractllm.processing import BasicExtractor\n"
                    "   \n"
                    "   llm = create_llm('openai', model='gpt-4o-mini')\n"
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
- s:Person - People by name
- s:Organization - Companies, institutions
- s:Event - Events, meetings, conferences, etc
- s:Goal - Abstract goals, objectives
- s:Task - Abstract tasks, actions
- s:SoftwareApplication - Software, libraries, frameworks, tools
- s:Place - Locations
- s:Product - Products, services
- sk:Concept - Abstract concepts, technologies

LIMITS: try to limit the number of entities to {entity_limit}.

STEP 2 : ONCE all the entities have been created and annotated, then identify and characterize all the relationships between the selected entities.

RELATIONSHIP TYPES must be one of:
- is_a, part_of, transforms, provides, describes, mentions, integrates, supports, discourages, requires, uses, creates, compatible_with, works_with, enables, disables, occurs_in, occurs_when

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
                source_id = item.get('s:about', {}).get('@id', '')
                target_id = item.get('s:object', {}).get('@id', '')

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

        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting from chunk {i+1}/{len(chunks)}")
            chunk_result = self._extract_single_chunk(chunk, domain_focus, length)

            # Merge entities
            for entity in chunk_result.get("@graph", []):
                entity_id = entity.get("@id", "")
                if entity_id.startswith("e:") and entity_id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity_id)

            # Merge relationships (deduplicate by source-target-relation)
            for item in chunk_result.get("@graph", []):
                item_id = item.get("@id", "")
                if item_id.startswith("r:"):
                    source = item.get("s:about", {}).get("@id", "")
                    target = item.get("s:object", {}).get("@id", "")
                    relation = item.get("s:name", "")
                    triple = (source, relation, target)

                    if triple not in seen_relationship_ids:
                        all_relationships.append(item)
                        seen_relationship_ids.add(triple)

        return {
            "@context": chunk_result.get("@context", self._create_empty_graph()["@context"]),
            "@graph": all_entities + all_relationships
        }

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
        domain_focus: Optional[str] = None
    ) -> dict:
        """
        Refine a previous extraction by finding missing entities/relationships
        and verifying relationship directionality.

        Args:
            text: The original source text
            previous_extraction: The previous JSON-LD extraction result
            domain_focus: Optional domain focus

        Returns:
            Refined JSON-LD extraction with merged results
        """
        logger.info("Starting extraction refinement")

        domain_note = f" Focus on {domain_focus} domain." if domain_focus else ""

        # Create summary of previous extraction for prompt
        prev_entities = [item for item in previous_extraction.get('@graph', [])
                        if item.get('@id', '').startswith('e:')]
        prev_relationships = [item for item in previous_extraction.get('@graph', [])
                             if item.get('@id', '').startswith('r:')]

        # Build entity summary for the prompt
        entity_summary = "\n".join([
            f"  - {e.get('s:name', e.get('@id'))} ({e.get('@type', 'Unknown')})"
            for e in prev_entities[:20]  # Limit to avoid token overflow
        ])

        # Build relationship summary for the prompt
        relationship_summary = "\n".join([
            f"  - {r.get('s:about', {}).get('@id', '?')} --[{r.get('s:name', '?')}]--> {r.get('s:object', {}).get('@id', '?')}"
            for r in prev_relationships[:20]  # Limit to avoid token overflow
        ])

        # Refinement prompt
        prompt = f"""You are refining a knowledge graph extraction. Review the text and the existing extraction to find MISSING entities and relationships, and to verify relationship directionality.

TEXT TO ANALYZE:
{text}

EXISTING EXTRACTION:
Entities found:
{entity_summary if entity_summary else "  (none)"}

Relationships found:
{relationship_summary if relationship_summary else "  (none)"}

TASK: Find missing entities and relationships, and verify directionality.{domain_note}

CRITICAL CHECKS:
1. Missing entities: Are there important entities in the text that were not extracted?
2. Missing relationships: Are there relationships mentioned in the text that were not captured?
3. Relationship directionality: Check CAREFULLY that relationships point in the correct direction.
   - Example: If text says "AbstractAgent uses AbstractCore", then relationship should be:
     s:about: e:abstractagent (source/subject)
     s:object: e:abstractcore (target/object)
   - NOT the reverse!

OUTPUT FORMAT: JSON-LD with ONLY the NEW/CORRECTED items (entities and relationships).

Example of corrected relationship:
{{
  "@context": {{
    "s": "https://schema.org/",
    "e": "http://example.org/entity/",
    "r": "http://example.org/relation/",
    "confidence": "http://example.org/confidence"
  }},
  "@graph": [
    {{
      "@id": "e:new_entity",
      "@type": "s:Organization",
      "s:name": "New Entity",
      "s:description": "A missing entity",
      "confidence": 0.9
    }},
    {{
      "@id": "r:corrected",
      "@type": "s:Relationship",
      "s:name": "uses",
      "s:about": {{"@id": "e:source"}},
      "s:object": {{"@id": "e:target"}},
      "s:description": "Corrected relationship direction",
      "confidence": 0.95,
      "strength": 0.9
    }}
  ]
}}

RULES:
1. Include ONLY new entities not in the existing extraction
2. Include ONLY new relationships not in the existing extraction
3. Include corrected versions of relationships with wrong directionality
4. Entity @id: "e:" + normalized_name
5. Relationship @id: "r:" + number
6. Relationship s:name: The relationship type (uses, creates, etc.)
7. s:about: Source/subject entity
8. s:object: Target/object entity
9. Verify directionality matches the text exactly!

Output ONLY valid JSON following the format above. If no changes needed, output empty @graph array."""

        # Generate refinements
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
            refinements = json.loads(response_text)

            if "@graph" not in refinements:
                logger.warning("Refinement response missing @graph, returning previous extraction")
                return previous_extraction

            # Merge refinements with previous extraction
            return self._merge_extractions(previous_extraction, refinements)

        except json.JSONDecodeError as e:
            logger.warning(f"Refinement JSON parsing failed: {e}")

            # Attempt self-correction
            from ..utils.self_fixes import fix_json
            corrected_json = fix_json(response_text)

            if corrected_json:
                try:
                    refinements = json.loads(corrected_json)
                    if "@graph" in refinements:
                        logger.info("✅ Refinement JSON self-correction successful!")
                        return self._merge_extractions(previous_extraction, refinements)
                except json.JSONDecodeError:
                    pass

            logger.warning("Refinement JSON self-correction failed, returning previous extraction")
            return previous_extraction

    def _merge_extractions(self, base: dict, refinements: dict) -> dict:
        """
        Merge refinement results with base extraction.

        Strategy:
        - Add new entities
        - Replace relationships that have been corrected (same source/target but corrected direction)
        - Add new relationships
        """
        # Get existing items
        base_graph = base.get('@graph', [])
        refinement_graph = refinements.get('@graph', [])

        # Build sets for tracking
        base_entity_ids = {item['@id'] for item in base_graph if item.get('@id', '').startswith('e:')}
        base_rel_triples = {}  # (source, relation, target) -> relationship object

        for item in base_graph:
            if item.get('@id', '').startswith('r:'):
                source = item.get('s:about', {}).get('@id', '')
                target = item.get('s:object', {}).get('@id', '')
                relation = item.get('s:name', '')
                if source and target:
                    base_rel_triples[(source, relation, target)] = item

        # Merge entities
        merged_graph = list(base_graph)  # Start with base
        next_rel_id = len([item for item in base_graph if item.get('@id', '').startswith('r:')]) + 1

        for item in refinement_graph:
            item_id = item.get('@id', '')

            # Add new entities
            if item_id.startswith('e:') and item_id not in base_entity_ids:
                merged_graph.append(item)
                logger.info(f"Added new entity: {item.get('s:name', item_id)}")

            # Handle relationships
            elif item_id.startswith('r:'):
                source = item.get('s:about', {}).get('@id', '')
                target = item.get('s:object', {}).get('@id', '')
                relation = item.get('s:name', '')

                if source and target:
                    triple = (source, relation, target)
                    reverse_triple = (target, relation, source)

                    # Check if this corrects a reversed relationship
                    if reverse_triple in base_rel_triples:
                        # Remove the incorrectly directed relationship
                        old_rel = base_rel_triples[reverse_triple]
                        merged_graph = [r for r in merged_graph if r.get('@id') != old_rel.get('@id')]
                        logger.info(f"Corrected relationship direction: {relation} from {source} to {target}")

                    # Add if new
                    if triple not in base_rel_triples:
                        # Assign new ID
                        item['@id'] = f"r:{next_rel_id}"
                        next_rel_id += 1
                        merged_graph.append(item)
                        logger.info(f"Added new relationship: {relation} from {source} to {target}")

        # Return merged result
        return {
            "@context": base.get("@context", refinements.get("@context", {})),
            "@graph": merged_graph
        }

    def _format_output(self, jsonld_result: dict, output_format: str) -> dict:
        """
        Format the output according to the requested format

        Args:
            jsonld_result: The JSON-LD extraction result
            output_format: Desired output format ("jsonld", "triples", "jsonld_minified")

        Returns:
            dict: Formatted result
        """
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

