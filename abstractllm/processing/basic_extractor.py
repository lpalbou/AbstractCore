"""
Basic Extractor - Clean, powerful entity and relationship extraction built on AbstractCore

Demonstrates semantic extraction with entity deduplication using Chain of Verification
and semantic similarity clustering for Knowledge Graph construction.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from pydantic import BaseModel, Field
import hashlib

from ..core.interface import AbstractLLMInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry


class EntityType(Enum):
    """Common entity types for knowledge extraction"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    DATE = "date"
    OTHER = "other"


class RelationType(Enum):
    """Common relationship types for knowledge graphs"""
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    CREATED_BY = "created_by"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    USES = "uses"
    PARTICIPATES_IN = "participates_in"
    OCCURRED_ON = "occurred_on"
    SIMILAR_TO = "similar_to"
    OTHER = "other"


class Entity(BaseModel):
    """Extracted entity with metadata"""
    name: str = Field(description="Entity name or mention")
    type: EntityType = Field(description="Category of the entity")
    aliases: List[str] = Field(default=[], description="Alternative names or mentions", max_length=5)
    context: str = Field(description="Surrounding context where entity appears")
    confidence: float = Field(description="Confidence in extraction (0-1)", ge=0, le=1)


class Relationship(BaseModel):
    """Extracted relationship between entities"""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: RelationType = Field(description="Type of relationship")
    context: str = Field(description="Context where relationship appears")
    confidence: float = Field(description="Confidence in extraction (0-1)", ge=0, le=1)


class LLMExtractionOutput(BaseModel):
    """LLM-generated extraction output"""
    entities: List[Entity] = Field(description="Extracted entities", max_length=20)
    relationships: List[Relationship] = Field(description="Extracted relationships", max_length=15)
    verification_confidence: float = Field(description="Overall confidence after verification (0-1)", ge=0, le=1)


class ExtractionOutput(BaseModel):
    """Complete extraction output with deduplication"""
    entities: Dict[str, Entity] = Field(description="Deduplicated entities (canonical_id -> Entity)")
    relationships: List[Relationship] = Field(description="Extracted relationships with resolved entity names")
    verification_confidence: float = Field(description="Overall confidence after verification (0-1)", ge=0, le=1)
    deduplication_summary: Dict[str, int] = Field(description="Deduplication statistics")


class EntityRegistry:
    """
    Manages entity deduplication using semantic similarity

    Uses AbstractCore's EmbeddingManager for semantic comparison while maintaining
    a lightweight, efficient registry for real-time deduplication.
    """

    def __init__(self, embedder=None, similarity_threshold: float = 0.85):
        """
        Initialize entity registry

        Args:
            embedder: EmbeddingManager instance (optional, created if None)
            similarity_threshold: Minimum similarity for entity merging (0-1)
        """
        self.entities: Dict[str, Entity] = {}  # canonical_id -> Entity
        self.entity_embeddings: Dict[str, List[float]] = {}  # canonical_id -> embedding
        self.name_to_canonical: Dict[str, str] = {}  # entity_name -> canonical_id
        self.similarity_threshold = similarity_threshold
        self.stats = {"merged": 0, "created": 0, "total_processed": 0}

        # Optional embedding support for semantic deduplication
        self.embedder = embedder
        if embedder is None:
            try:
                from ..embeddings import EmbeddingManager
                self.embedder = EmbeddingManager()
            except ImportError:
                self.embedder = None

    def register_entity(self, entity: Entity) -> str:
        """
        Register an entity, returning its canonical ID

        Performs deduplication by checking:
        1. Exact name matches
        2. Alias matches
        3. Semantic similarity (if embeddings available)

        Args:
            entity: Entity to register

        Returns:
            str: Canonical ID for the entity
        """
        self.stats["total_processed"] += 1

        # Check exact name match first
        if entity.name in self.name_to_canonical:
            canonical_id = self.name_to_canonical[entity.name]
            self._merge_entity_info(canonical_id, entity)
            self.stats["merged"] += 1
            return canonical_id

        # Check alias matches
        for canonical_id, existing_entity in self.entities.items():
            if (entity.name.lower() in [alias.lower() for alias in existing_entity.aliases] or
                any(alias.lower() in [existing_entity.name.lower()] + [a.lower() for a in existing_entity.aliases]
                    for alias in entity.aliases)):
                self._merge_entity_info(canonical_id, entity)
                self.name_to_canonical[entity.name] = canonical_id
                self.stats["merged"] += 1
                return canonical_id

        # Semantic similarity check (if embeddings available)
        if self.embedder and len(self.entities) > 0:
            similar_id = self._find_semantically_similar(entity)
            if similar_id:
                self._merge_entity_info(similar_id, entity)
                self.name_to_canonical[entity.name] = similar_id
                self.stats["merged"] += 1
                return similar_id

        # No match found - create new canonical entity
        canonical_id = self._create_canonical_id(entity.name)
        self.entities[canonical_id] = entity
        self.name_to_canonical[entity.name] = canonical_id

        # Store embedding for future similarity checks
        if self.embedder:
            try:
                embedding = self.embedder.embed(f"{entity.name} {entity.context[:100]}")
                self.entity_embeddings[canonical_id] = embedding
            except Exception:
                pass  # Continue without embedding if it fails

        self.stats["created"] += 1
        return canonical_id

    def _find_semantically_similar(self, entity: Entity) -> Optional[str]:
        """Find semantically similar entity using embeddings"""
        if not self.embedder or not self.entity_embeddings:
            return None

        try:
            entity_text = f"{entity.name} {entity.context[:100]}"
            entity_embedding = self.embedder.embed(entity_text)

            best_similarity = 0
            best_canonical_id = None

            for canonical_id, existing_embedding in self.entity_embeddings.items():
                similarity = self.embedder.compute_similarity_direct(entity_embedding, existing_embedding)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    # Additional type check - only merge if types are compatible
                    existing_entity = self.entities[canonical_id]
                    if self._types_compatible(entity.type, existing_entity.type):
                        best_similarity = similarity
                        best_canonical_id = canonical_id

            return best_canonical_id

        except Exception:
            return None  # Fall back to no similarity matching if embeddings fail

    def _types_compatible(self, type1: EntityType, type2: EntityType) -> bool:
        """Check if two entity types are compatible for merging"""
        if type1 == type2:
            return True
        # Allow OTHER to merge with any type
        if type1 == EntityType.OTHER or type2 == EntityType.OTHER:
            return True
        return False

    def _merge_entity_info(self, canonical_id: str, new_entity: Entity):
        """Merge information from new entity into existing canonical entity"""
        existing = self.entities[canonical_id]

        # Add new aliases
        new_aliases = set(existing.aliases + [new_entity.name] + new_entity.aliases)
        new_aliases.discard(existing.name)  # Don't include canonical name as alias
        existing.aliases = list(new_aliases)[:5]  # Limit aliases

        # Update context if new one is more detailed
        if len(new_entity.context) > len(existing.context):
            existing.context = new_entity.context

        # Update confidence to average
        existing.confidence = (existing.confidence + new_entity.confidence) / 2

    def _create_canonical_id(self, name: str) -> str:
        """Create a canonical ID for an entity"""
        # Create a clean, consistent ID
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        base_id = f"entity_{clean_name}"

        # Handle duplicates by adding suffix
        counter = 1
        canonical_id = base_id
        while canonical_id in self.entities:
            canonical_id = f"{base_id}_{counter}"
            counter += 1

        return canonical_id

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        return self.stats.copy()


class BasicExtractor:
    """
    Basic Entity and Relationship Extractor using Chain of Verification

    Demonstrates AbstractCore best practices:
    - Structured output with Pydantic validation
    - Semantic entity deduplication using embeddings
    - Chain of Verification for reduced hallucinations
    - Provider-agnostic implementation
    - Built-in retry and error handling (inherited from AbstractCore)

    Optimized for Knowledge Graph construction:
    - Produces clean entities with canonical IDs
    - Resolves entity mentions across text chunks
    - Creates relationships using canonical entity names
    - Suitable for feeding into AbstractMemory package

    Optimized defaults (no setup required):
        extractor = BasicExtractor()  # Uses gemma3:1b-it-qat, semantic deduplication

    Custom setup for different needs:
        llm = create_llm("openai", model="gpt-4o-mini")
        extractor = BasicExtractor(llm)
    """

    def __init__(
        self,
        llm: Optional[AbstractLLMInterface] = None,
        use_embeddings: bool = True,
        similarity_threshold: float = 0.85,
        max_chunk_size: int = 6000
    ):
        """
        Initialize the extractor

        Args:
            llm: AbstractLLM instance (any provider). If None, uses ollama gemma3:1b-it-qat
            use_embeddings: Whether to use semantic deduplication (default True)
            similarity_threshold: Minimum similarity for entity merging (0-1)
            max_chunk_size: Maximum characters per chunk for long documents
        """
        if llm is None:
            try:
                # Default to gemma3:1b-it-qat with 16k context window
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

        # Initialize entity registry with optional embeddings
        embedder = None
        if use_embeddings:
            try:
                from ..embeddings import EmbeddingManager
                embedder = EmbeddingManager()
            except ImportError:
                pass  # Continue without embeddings

        self.entity_registry = EntityRegistry(embedder, similarity_threshold)

    def extract(
        self,
        text: str,
        domain_focus: Optional[str] = None,
        entity_types: Optional[List[EntityType]] = None
    ) -> ExtractionOutput:
        """
        Extract entities and relationships from text using Chain of Verification

        Args:
            text: Text to analyze
            domain_focus: Optional domain focus (e.g., "business", "technology", "medical")
            entity_types: Optional list of entity types to focus on

        Returns:
            ExtractionOutput: Extracted and deduplicated entities with relationships

        Example:
            >>> extractor = BasicExtractor()
            >>> result = extractor.extract(
            ...     "Apple Inc. was founded by Steve Jobs in Cupertino. "
            ...     "The iPhone was launched in 2007 by Apple."
            ... )
            >>> print(f"Found {len(result.entities)} unique entities")
            >>> print(f"Found {len(result.relationships)} relationships")
        """
        # Handle long documents through chunking
        if len(text) > self.max_chunk_size:
            return self._extract_long_document(text, domain_focus, entity_types)
        else:
            return self._extract_single_chunk(text, domain_focus, entity_types)

    def _extract_single_chunk(
        self,
        text: str,
        domain_focus: Optional[str],
        entity_types: Optional[List[EntityType]]
    ) -> ExtractionOutput:
        """Extract from a single chunk using Chain of Verification"""

        # Step 1: Initial extraction
        initial_prompt = self._build_extraction_prompt(text, domain_focus, entity_types)

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

        # Step 2: Chain of Verification - verify extracted facts
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

    def _extract_long_document(
        self,
        text: str,
        domain_focus: Optional[str],
        entity_types: Optional[List[EntityType]]
    ) -> ExtractionOutput:
        """Handle long documents using incremental extraction with entity registry"""

        chunks = self._split_text_into_chunks(text)
        all_relationships = []

        for chunk in chunks:
            chunk_result = self._extract_single_chunk(chunk, domain_focus, entity_types)
            all_relationships.extend(chunk_result.relationships)

        # Final output uses accumulated entities and relationships
        return ExtractionOutput(
            entities=self.entity_registry.entities.copy(),
            relationships=all_relationships,
            verification_confidence=0.8,  # Conservative estimate for long docs
            deduplication_summary=self.entity_registry.get_stats()
        )

    def _split_text_into_chunks(self, text: str, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks (same logic as BasicSummarizer)"""
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

    def _build_extraction_prompt(
        self,
        text: str,
        domain_focus: Optional[str],
        entity_types: Optional[List[EntityType]]
    ) -> str:
        """Build the initial extraction prompt"""

        # Domain-specific instructions
        domain_instruction = ""
        if domain_focus:
            domain_instruction = f"\nFocus on entities and relationships relevant to: {domain_focus}"

        # Entity type filtering
        type_instruction = ""
        if entity_types:
            type_names = [t.value for t in entity_types]
            type_instruction = f"\nFocus on these entity types: {', '.join(type_names)}"

        prompt = f"""Extract entities and relationships from the following text.

For each entity:
- Identify the name/mention as it appears in text
- Classify the type (person, organization, location, concept, event, technology, product, date, other)
- Note any alternative names or aliases mentioned
- Include relevant context where the entity appears
- Provide confidence score (0-1) for the extraction

For each relationship:
- Identify source and target entities by their exact names
- Classify the relationship type (works_for, located_in, created_by, related_to, causes, uses, participates_in, occurred_on, similar_to, other)
- Include context where the relationship is mentioned
- Provide confidence score (0-1) for the relationship

{domain_instruction}{type_instruction}

Text to analyze:
{text}

Requirements:
- Be precise and factual - avoid hallucinating entities or relationships not clearly stated
- Focus on the most important and clearly mentioned entities
- Relationships should connect entities that actually appear in the text
- Provide high confidence scores only for clearly stated facts
- Give an overall verification confidence for the entire extraction

Extract the entities and relationships following these guidelines."""

        return prompt

    def _build_verification_prompt(
        self,
        original_text: str,
        extraction: LLMExtractionOutput
    ) -> str:
        """Build Chain of Verification prompt to validate extractions"""

        # Format the extracted entities and relationships for verification
        entity_list = []
        for entity in extraction.entities:
            entity_list.append(f"- {entity.name} ({entity.type.value}): {entity.context[:100]}")

        relationship_list = []
        for rel in extraction.relationships:
            relationship_list.append(f"- {rel.source} --{rel.relation.value}--> {rel.target}: {rel.context[:100]}")

        entities_text = "\n".join(entity_list) if entity_list else "None extracted"
        relationships_text = "\n".join(relationship_list) if relationship_list else "None extracted"

        prompt = f"""Verify the following extracted entities and relationships against the original text.

ORIGINAL TEXT:
{original_text}

EXTRACTED ENTITIES:
{entities_text}

EXTRACTED RELATIONSHIPS:
{relationships_text}

Verification Task:
1. Check each entity - is it actually mentioned in the original text?
2. Check each relationship - is it actually stated or clearly implied?
3. Remove any hallucinated entities or relationships
4. Add any clearly stated entities or relationships that were missed
5. Adjust confidence scores based on how clearly each item is supported by the text
6. Provide an overall verification confidence score

Return the corrected and verified entities and relationships. Only include items that are clearly supported by the original text."""

        return prompt