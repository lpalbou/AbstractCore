"""
Fast Extractor - Optimized version of BasicExtractor for speed

Provides faster entity and relationship extraction by:
1. Single-pass extraction (no Chain of Verification by default)
2. Larger chunk sizes to reduce LLM calls
3. Optional embeddings for speed vs accuracy trade-offs
4. Simplified prompting for faster processing
"""

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from .basic_extractor import (
    BasicExtractor, EntityType, RelationType, Entity, Relationship,
    ExtractionOutput, EntityRegistry, LLMExtractionOutput
)
from ..core.interface import AbstractLLMInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry


class FastExtractor(BasicExtractor):
    """
    Fast Entity and Relationship Extractor optimized for speed

    Performance optimizations:
    - Single-pass extraction (no verification by default)
    - Larger default chunk size (15000 chars)
    - Simplified prompting for faster processing
    - Optional embeddings (disabled by default for speed)

    Use when speed > accuracy for initial KG construction.
    """

    def __init__(
        self,
        llm: Optional[AbstractLLMInterface] = None,
        use_embeddings: bool = False,  # Disabled by default for speed
        similarity_threshold: float = 0.85,
        max_chunk_size: int = 15000,   # Larger chunks = fewer LLM calls
        use_verification: bool = False  # Skip Chain of Verification for speed
    ):
        """
        Initialize the fast extractor

        Args:
            llm: AbstractLLM instance (any provider). If None, uses ollama gemma3:1b-it-qat
            use_embeddings: Whether to use semantic deduplication (default False for speed)
            similarity_threshold: Minimum similarity for entity merging (0-1)
            max_chunk_size: Maximum characters per chunk (default 15000 for speed)
            use_verification: Whether to use Chain of Verification (default False for speed)
        """
        # Initialize parent class
        super().__init__(llm, use_embeddings, similarity_threshold, max_chunk_size)

        self.use_verification = use_verification

    def _extract_single_chunk(
        self,
        text: str,
        domain_focus: Optional[str],
        entity_types: Optional[List[EntityType]],
        style: Optional[str] = None,
        length: Optional[str] = None
    ) -> ExtractionOutput:
        """Extract from a single chunk with optional verification for speed"""

        # Step 1: Initial extraction
        initial_prompt = self._build_extraction_prompt(text, domain_focus, entity_types, style, length)

        # Use AbstractCore's structured output with retry
        response = self.llm.generate(
            initial_prompt,
            response_model=LLMExtractionOutput,
            retry_strategy=self.retry_strategy
        )

        # Extract the structured output
        if isinstance(response, LLMExtractionOutput):
            extraction_result = response
        elif hasattr(response, 'structured_output') and response.structured_output:
            extraction_result = response.structured_output
        else:
            raise ValueError(f"Failed to generate structured extraction output. Response type: {type(response)}")

        # Step 2: Optional Chain of Verification (skip for speed by default)
        if self.use_verification:
            verification_prompt = self._build_verification_prompt(text, extraction_result)

            verified_response = self.llm.generate(
                verification_prompt,
                response_model=LLMExtractionOutput,
                retry_strategy=self.retry_strategy
            )

            # Extract verified results
            if isinstance(verified_response, LLMExtractionOutput):
                final_result = verified_response
            elif hasattr(verified_response, 'structured_output') and verified_response.structured_output:
                final_result = verified_response.structured_output
            else:
                # Fall back to initial results if verification fails
                final_result = extraction_result
        else:
            # Skip verification for speed - use initial results
            final_result = extraction_result

        # Step 3: Register entities and deduplicate
        canonical_entities = {}
        entity_name_mapping = {}  # original_name -> canonical_id

        for entity in final_result.entities:
            canonical_id = self.entity_registry.register_entity(entity)
            canonical_entities[canonical_id] = self.entity_registry.entities[canonical_id]
            entity_name_mapping[entity.name] = canonical_id

        # Step 4: Resolve relationships using canonical entity names
        resolved_relationships = []
        for rel in final_result.relationships:
            # Map relationship entity names to canonical IDs
            source_id = entity_name_mapping.get(rel.source)
            target_id = entity_name_mapping.get(rel.target)

            if source_id and target_id and source_id != target_id:
                # Update relationship to use canonical names
                source_name = canonical_entities[source_id].name
                target_name = canonical_entities[target_id].name

                resolved_rel = Relationship(
                    source=source_name,
                    target=target_name,
                    relation=rel.relation,
                    context=rel.context,
                    confidence=rel.confidence
                )
                resolved_relationships.append(resolved_rel)

        return ExtractionOutput(
            entities=canonical_entities,
            relationships=resolved_relationships,
            verification_confidence=final_result.verification_confidence,
            deduplication_summary=self.entity_registry.get_stats()
        )

    def _build_extraction_prompt(
        self,
        text: str,
        domain_focus: Optional[str],
        entity_types: Optional[List[EntityType]],
        style: Optional[str] = None,
        length: Optional[str] = None
    ) -> str:
        """Build a simplified extraction prompt optimized for speed"""

        # Domain-specific instructions
        domain_instruction = ""
        if domain_focus:
            domain_instruction = f"\nFocus on entities and relationships relevant to: {domain_focus}"

        # Entity type filtering
        type_instruction = ""
        if entity_types:
            type_names = [t.value for t in entity_types]
            type_instruction = f"\nFocus on these entity types: {', '.join(type_names)}"

        # Style instructions (simplified for fast extraction)
        style_instruction = ""
        if style == "structured":
            style_instruction = "\nExtraction approach: Organize entities and relationships systematically."
        elif style == "focused":
            style_instruction = "\nExtraction approach: Focus on key entities and relationships."
        elif style == "minimal":
            style_instruction = "\nExtraction approach: Extract only essential entities and relationships."

        # Length instructions (simplified for fast extraction)
        length_instruction = ""
        if length == "brief":
            length_instruction = "\nLimit to 5-8 entities and 3-5 relationships."
        elif length == "detailed":
            length_instruction = "\nExtract 15-20 entities and 8-12 relationships."
        elif length == "comprehensive":
            length_instruction = "\nExtract all relevant entities and relationships."

        # Fast semantic extraction with streamlined vocabulary
        prompt = f"""You are a specialized semantic knowledge extractor. Extract entities and relationships with semantic precision.

🎯 **FAST SEMANTIC EXTRACTION:**

**Entity Types (choose most specific):**
- schema:Person, schema:Organization, schema:Place, schema:Event
- skos:Concept, schema:SoftwareApplication, schema:Product
- dcterms:Text, schema:Method, schema:Process

**Semantic Relationships (be precise):**
- dcterms:creator, schema:about, schema:isPartOf, schema:hasPart
- schema:describes, schema:implements, schema:utilizes
- skos:broader, skos:narrower, skos:related
- schema:during, schema:enables, schema:produces

**Only use schema:relatedTo for generic connections**

{domain_instruction}{type_instruction}{style_instruction}{length_instruction}

📄 TEXT:
{text}

🚀 FAST EXTRACTION - Focus on core entities and meaningful relationships only."""

        return prompt