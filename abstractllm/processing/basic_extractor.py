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
    """Comprehensive entity types for semantic knowledge extraction"""

    # Core Entities (First-class entities that can have relationships)
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"

    # Knowledge & Concepts
    CONCEPT = "concept"
    THEORY = "theory"
    METHOD = "method"
    HYPOTHESIS = "hypothesis"
    FIELD = "field"
    DOMAIN = "domain"
    PARADIGM = "paradigm"

    # Technology & Software
    SOFTWARE_APPLICATION = "software_application"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    ALGORITHM = "algorithm"
    FRAMEWORK = "framework"
    PLATFORM = "platform"

    # Information Artifacts
    DOCUMENT = "document"
    PUBLICATION = "publication"
    DATASET = "dataset"
    MODEL = "model"
    SPECIFICATION = "specification"

    # Processes & Activities
    PROCESS = "process"
    INVESTIGATION = "investigation"
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"
    INTERACTION = "interaction"

    # Temporal & Contextual
    DATE = "date"
    TIME_FRAME = "time_frame"
    CONTEXT = "context"
    STATE = "state"
    PHASE = "phase"

    # Systems & Structures
    SYSTEM = "system"
    STRUCTURE = "structure"
    INSTITUTION = "institution"
    GROUP = "group"

    # Generic
    OTHER = "other"


class RelationType(Enum):
    """Comprehensive relationship types for semantic knowledge graphs"""

    # Structural Relationships
    HAS_PART = "has_part"
    IS_PART_OF = "is_part_of"
    CONTAINS = "contains"
    BELONGS_TO = "belongs_to"
    MEMBER_OF = "member_of"

    # Causal Relationships
    CAUSES = "causes"
    ENABLES = "enables"
    ENABLED_BY = "enabled_by"
    PREVENTS = "prevents"
    INFLUENCES = "influences"
    TRIGGERS = "triggers"

    # Temporal Relationships
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONCURRENT = "concurrent"

    # Functional Relationships
    IMPLEMENTS = "implements"
    IMPLEMENTED_BY = "implemented_by"
    UTILIZES = "utilizes"
    USES = "uses"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    TRANSFORMS = "transforms"
    CONFIGURES = "configures"

    # Knowledge Relationships
    EXPLAINS = "explains"
    DESCRIBES = "describes"
    DEFINES = "defines"
    EXEMPLIFIES = "exemplifies"
    CATEGORIZES = "categorizes"

    # Evidential Relationships
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    VALIDATES = "validates"
    QUESTIONS = "questions"

    # Social/Organizational
    WORKS_FOR = "works_for"
    WORKS_WITH = "works_with"
    COLLABORATES_WITH = "collaborates_with"
    REPORTS_TO = "reports_to"
    MANAGES = "manages"

    # Spatial Relationships
    LOCATED_IN = "located_in"
    LOCATED_AT = "located_at"
    ADJACENT_TO = "adjacent_to"

    # Comparative Relationships
    SIMILAR_TO = "similar_to"
    DIFFERENT_FROM = "different_from"
    COMPARED_TO = "compared_to"
    ALTERNATIVE_TO = "alternative_to"
    COMPLEMENT_TO = "complement_to"
    COMPLEMENTS = "complements"

    # Creation/Attribution
    CREATED_BY = "created_by"
    AUTHORED_BY = "authored_by"
    DEVELOPED_BY = "developed_by"
    INVENTED_BY = "invented_by"

    # Participation
    PARTICIPATES_IN = "participates_in"
    INVOLVED_IN = "involved_in"
    CONTRIBUTES_TO = "contributes_to"

    # Generic (use only when no specific relationship applies)
    RELATED_TO = "related_to"
    ASSOCIATED_WITH = "associated_with"
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
    entities: List[Entity] = Field(description="Extracted entities")
    relationships: List[Relationship] = Field(description="Extracted relationships")
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
        # Extraction mode (presets other parameters)
        extraction_mode: str = "balanced",  # "fast", "balanced", "thorough"
        # Performance settings
        use_embeddings: bool = None,
        use_verification: bool = None,
        use_refinement: bool = None,
        use_consolidation: bool = None,
        # Chunking settings
        max_chunk_size: int = None,
        chunk_overlap: int = 500,
        # Quality settings
        similarity_threshold: float = 0.85,
        min_confidence: float = 0.7
    ):
        """
        Initialize the unified semantic extractor

        Args:
            llm: AbstractLLM instance (any provider). If None, uses ollama gemma3:1b-it-qat
            extraction_mode: Preset configuration ("fast", "balanced", "thorough")
            use_embeddings: Whether to use semantic deduplication for entity merging
            use_verification: Whether to use Chain of Verification (2nd LLM call for validation)
            use_refinement: Whether to use semantic refinement (3rd LLM call for enhancement)
            use_consolidation: Whether to consolidate isolated entities
            max_chunk_size: Maximum characters per chunk for long documents
            chunk_overlap: Character overlap between chunks
            similarity_threshold: Minimum similarity for entity merging (0-1)
            min_confidence: Minimum confidence for extracted entities/relationships

        Extraction Modes:
            - "fast": 2-3x faster, skip verification/refinement, disable embeddings, large chunks
            - "balanced": Default speed/quality tradeoff, verification enabled, embeddings enabled
            - "thorough": Highest quality, all features enabled, smaller chunks for precision
        """
        # Apply extraction mode presets
        mode_presets = {
            "fast": {
                "use_embeddings": False,
                "use_verification": False,
                "use_refinement": False,
                "use_consolidation": True,  # Keep consolidation as it's fast and useful
                "max_chunk_size": 15000
            },
            "balanced": {
                "use_embeddings": True,
                "use_verification": True,
                "use_refinement": True,
                "use_consolidation": True,
                "max_chunk_size": 6000
            },
            "thorough": {
                "use_embeddings": True,
                "use_verification": True,
                "use_refinement": True,
                "use_consolidation": True,
                "max_chunk_size": 3000
            }
        }

        if extraction_mode not in mode_presets:
            raise ValueError(f"Invalid extraction_mode '{extraction_mode}'. Use: fast, balanced, thorough")

        presets = mode_presets[extraction_mode]

        # Apply presets if parameters not explicitly set
        self.use_embeddings = use_embeddings if use_embeddings is not None else presets["use_embeddings"]
        self.use_verification = use_verification if use_verification is not None else presets["use_verification"]
        self.use_refinement = use_refinement if use_refinement is not None else presets["use_refinement"]
        self.use_consolidation = use_consolidation if use_consolidation is not None else presets["use_consolidation"]
        self.max_chunk_size = max_chunk_size if max_chunk_size is not None else presets["max_chunk_size"]

        # Store other settings
        self.extraction_mode = extraction_mode
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        # Initialize LLM
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

        # Initialize retry strategy
        self.retry_strategy = FeedbackRetry(max_attempts=3)

        # Initialize entity registry with optional embeddings
        embedder = None
        if self.use_embeddings:
            try:
                from ..embeddings import EmbeddingManager
                embedder = EmbeddingManager()
            except ImportError:
                pass  # Continue without embeddings

        self.entity_registry = EntityRegistry(embedder, self.similarity_threshold)

    def extract(
        self,
        text: str,
        domain_focus: Optional[str] = None,
        entity_types: Optional[List[EntityType]] = None,
        style: Optional[str] = None,
        length: Optional[str] = None
    ) -> ExtractionOutput:
        """
        Extract entities and relationships from text using Chain of Verification

        Args:
            text: Text to analyze
            domain_focus: Optional domain focus (e.g., "business", "technology", "medical")
            entity_types: Optional list of entity types to focus on
            style: Extraction style ("comprehensive", "focused", "minimal")
            length: Extraction depth ("brief", "standard", "detailed", "comprehensive")

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
            return self._extract_long_document(text, domain_focus, entity_types, style, length)
        else:
            return self._extract_single_chunk(text, domain_focus, entity_types, style, length)

    def _extract_single_chunk(
        self,
        text: str,
        domain_focus: Optional[str],
        entity_types: Optional[List[EntityType]],
        style: Optional[str] = None,
        length: Optional[str] = None
    ) -> ExtractionOutput:
        """Extract from a single chunk with configurable verification/refinement"""

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

        current_result = extraction_result

        # Step 2: Optional Verification - validate extraction accuracy
        if self.use_verification:
            verification_prompt = self._build_verification_prompt(text, current_result)

            verified_response = self.llm.generate(
                verification_prompt,
                response_model=LLMExtractionOutput,
                retry_strategy=self.retry_strategy
            )

            # Extract verified results
            if isinstance(verified_response, LLMExtractionOutput):
                current_result = verified_response
            elif hasattr(verified_response, 'structured_output') and verified_response.structured_output:
                current_result = verified_response.structured_output
            # If verification fails, continue with current results

        # Step 3: Optional Semantic Refinement - enhance semantic accuracy and completeness
        if self.use_refinement:
            refinement_prompt = self._build_semantic_refinement_prompt(text, current_result)

            refined_response = self.llm.generate(
                refinement_prompt,
                response_model=LLMExtractionOutput,
                retry_strategy=self.retry_strategy
            )

            # Extract refined results
            if isinstance(refined_response, LLMExtractionOutput):
                current_result = refined_response
            elif hasattr(refined_response, 'structured_output') and refined_response.structured_output:
                current_result = refined_response.structured_output
            # If refinement fails, continue with current results

        # Step 4: Optional Graph Consolidation - clean up isolated entities
        if self.use_consolidation:
            # Phase 1: Remove attribute-like isolated entities
            isolated_entity_names = self._identify_isolated_entities(current_result)
            entities_to_keep = []

            for entity in current_result.entities:
                if entity.name in isolated_entity_names and self._is_likely_attribute(entity):
                    # Skip this entity - it's likely an attribute, not a core entity
                    continue
                entities_to_keep.append(entity)

            # Update the entity list
            current_result.entities = entities_to_keep

            # Phase 2: Find relationships for remaining isolated entities
            remaining_isolated_names = self._identify_isolated_entities(current_result)
            if remaining_isolated_names:
                isolated_entities = [e for e in current_result.entities if e.name in remaining_isolated_names]
                new_relationships = self._find_missing_relationships(text, isolated_entities, current_result.entities)
                current_result.relationships.extend(new_relationships)

        final_result = current_result

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
        entity_types: Optional[List[EntityType]],
        style: Optional[str] = None,
        length: Optional[str] = None
    ) -> ExtractionOutput:
        """Handle long documents using incremental extraction with entity registry"""

        chunks = self._split_text_into_chunks(text)
        all_relationships = []

        for chunk in chunks:
            chunk_result = self._extract_single_chunk(chunk, domain_focus, entity_types, style, length)
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

    def _is_likely_attribute(self, entity: Entity) -> bool:
        """
        Simple heuristic to detect attribute-like entities that should be properties

        Args:
            entity: Entity to check

        Returns:
            True if entity is likely an attribute rather than a core entity
        """
        # Generic terms that are usually attributes, not entities
        ATTRIBUTE_PATTERNS = {
            'api', 'input', 'output', 'process', 'system', 'data', 'information',
            'type', 'status', 'method', 'interface', 'protocol', 'format',
            'mode', 'version', 'level', 'state', 'config', 'setting',
            'chat', 'interaction', 'session', 'completion', 'response'
        }

        name_lower = entity.name.lower().strip()

        # Check if it's a generic single word that's an attribute pattern
        if ' ' not in entity.name and name_lower in ATTRIBUTE_PATTERNS:
            return True

        # Check if it's a very short generic concept
        if len(entity.name) <= 4 and entity.type == EntityType.CONCEPT:
            return True

        # Check if it's a generic plural noun (often attributes)
        if name_lower.endswith('s') and name_lower[:-1] in ATTRIBUTE_PATTERNS:
            return True

        return False

    def _identify_isolated_entities(self, extraction: LLMExtractionOutput) -> Set[str]:
        """
        Identify entities that have no relationships (isolated nodes)

        Args:
            extraction: The extraction result to analyze

        Returns:
            Set of entity names that are isolated
        """
        # Get all entities mentioned in relationships
        connected_entities = set()
        for rel in extraction.relationships:
            connected_entities.add(rel.source)
            connected_entities.add(rel.target)

        # Find entities with no relationships
        all_entities = {entity.name for entity in extraction.entities}
        isolated_entities = all_entities - connected_entities

        return isolated_entities

    def _find_missing_relationships(
        self,
        text: str,
        isolated_entities: List[Entity],
        all_entities: List[Entity]
    ) -> List[Relationship]:
        """
        Find relationships for isolated entities using targeted LLM prompts

        Args:
            text: Original source text
            isolated_entities: List of entities that need connections
            all_entities: All entities available for connections

        Returns:
            List of new relationships connecting isolated entities
        """
        if not isolated_entities:
            return []

        # Build entity context for the LLM
        isolated_names = [e.name for e in isolated_entities]
        available_targets = [e.name for e in all_entities if e.name not in isolated_names]

        if not available_targets:
            return []  # No entities to connect to

        # Create a concise, focused prompt
        prompt = f"""Find relationships for specific isolated entities based on the source text.

ORIGINAL TEXT:
{text}

ISOLATED ENTITIES TO CONNECT:
{', '.join(isolated_names)}

AVAILABLE ENTITIES TO CONNECT TO:
{', '.join(available_targets)}

TASK: Find ONLY relationships that connect the isolated entities to available entities.
Each relationship must be clearly stated or strongly implied in the source text.

RULES:
- Only return relationships for the isolated entities listed above
- Only connect to entities in the available list
- Use precise semantic relationship types from the vocabulary: dcterms:creator, schema:isPartOf, schema:utilizes, schema:enables, schema:describes, etc.
- Base relationships on actual content from the text, not assumptions
- Return empty if no clear relationships exist

Return a JSON list of relationships in this exact format:
[
  {{
    "source": "EntityName",
    "target": "EntityName",
    "relation": "relationship_type",
    "context": "Brief context from text",
    "confidence": 0.8
  }}
]

Only include relationships that are grounded in the source text."""

        try:
            # Use a simpler response model for just relationships
            from pydantic import BaseModel, Field
            from typing import List

            class RelationshipList(BaseModel):
                relationships: List[Relationship] = Field(description="List of relationships")

            response = self.llm.generate(
                prompt,
                response_model=RelationshipList,
                retry_strategy=self.retry_strategy
            )

            if isinstance(response, RelationshipList):
                return response.relationships
            elif hasattr(response, 'structured_output') and response.structured_output:
                return response.structured_output.relationships
            else:
                return []

        except Exception:
            # If relationship discovery fails, return empty list
            # This ensures the system remains robust
            return []

    def _build_extraction_prompt(
        self,
        text: str,
        domain_focus: Optional[str],
        entity_types: Optional[List[EntityType]],
        style: Optional[str] = None,
        length: Optional[str] = None
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

        # Style instructions based on knowledge graph best practices
        style_instruction = ""
        if style == "structured":
            style_instruction = "\n🎯 STRUCTURED EXTRACTION:\n- Differentiate core entities (subjects) from attributes (properties)\n- Create clear hierarchies using part-of relationships\n- Group related information under parent entities\n- Use precise, directional relationship types"
        elif style == "focused":
            style_instruction = "\n🎯 FOCUSED EXTRACTION:\n- Extract only the most important core entities (avoid extracting attributes as entities)\n- Prioritize entities central to the domain focus\n- Use specific relationship types instead of generic ones\n- Focus on entities that are primary subjects, not descriptive details"
        elif style == "minimal":
            style_instruction = "\n🎯 MINIMAL EXTRACTION:\n- Extract only essential entities that cannot be attributes of other entities\n- Consolidate related information under fewer core entities\n- Use the most important, specific relationships only\n- Avoid redundancy and over-extraction"
        elif style == "comprehensive":
            style_instruction = "\n🎯 COMPREHENSIVE EXTRACTION:\n- Extract all significant entities while maintaining proper hierarchy\n- Distinguish between core entities and their attributes\n- Include both obvious and implied relationships with precise types\n- Build a complete but well-structured knowledge graph"

        # Length instructions
        length_instruction = ""
        if length == "brief":
            length_instruction = "\nExtraction depth: Limit to 5-10 key entities and 3-5 main relationships."
        elif length == "detailed":
            length_instruction = "\nExtraction depth: Extract comprehensive details including 15-25 entities and 10-15 relationships."
        elif length == "comprehensive":
            length_instruction = "\nExtraction depth: Perform exhaustive extraction including all entities and relationships, aliases, and detailed context."
        else:  # standard
            length_instruction = "\nExtraction depth: Extract 10-15 key entities and 5-10 important relationships."

        prompt = f"""You are a specialized semantic knowledge extractor. Your goal is to identify first-class entities and their relationships while avoiding the extraction of mere properties as entities.

🎯 **ENTITY vs PROPERTY DISTINCTION** (Critical Rule):

**EXTRACT as ENTITIES** (First-class objects):
✓ Proper nouns: Google, TensorFlow, Python, ChatGPT
✓ People: individuals with names (John Smith, Dr. Sarah Johnson)
✓ Organizations: companies, institutions (Microsoft, MIT, NATO)
✓ Technologies: specific systems/tools (Docker, React, MongoDB)
✓ Concepts: distinct ideas (machine learning, blockchain, democracy)
✓ Locations: places (San Francisco, Europe, Main Street)
✓ Products: specific items (iPhone, Tesla Model 3, Windows 11)

**DO NOT EXTRACT as ENTITIES** (These are properties):
✗ Generic terms: API, system, process, data, input, output
✗ Attributes: version, status, type, mode, format, level
✗ Descriptors: interface, protocol, method, function, service
✗ Common nouns without specificity: user, developer, company

🔬 **ENTITY TYPES** - Use most specific:

**Core Entities:** person, organization, location, event
**Knowledge:** concept, theory, method, field, domain, hypothesis
**Technology:** software_application, technology, product, algorithm, framework
**Information:** document, publication, dataset, model, specification
**Processes:** process, investigation, analysis, transformation

🔗 **RELATIONSHIP TYPES** - Be precise:

**Structural:** has_part, is_part_of, contains, belongs_to
**Causal:** causes, enables, prevents, influences, triggers
**Temporal:** before, after, during, precedes, follows
**Functional:** implements, utilizes, produces, consumes, transforms
**Knowledge:** explains, describes, defines, exemplifies
**Social:** works_for, collaborates_with, manages, reports_to
**Creation:** created_by, authored_by, developed_by, invented_by

📚 **EXAMPLES:**

**Good Example:**
Text: "Google's TensorFlow framework helps developers build AI models"
✓ Entities: Google (organization), TensorFlow (software_application), AI (concept)
✓ Relationships: Google created_by TensorFlow, TensorFlow enables AI
✗ Avoid: "framework", "developers", "models" as entities

**Bad Example:**
✗ Extracting: framework (generic), developers (generic), models (generic)
✗ These should be properties of the main entities

**Good Example:**
Text: "The API processes user input through machine learning algorithms"
✓ Entities: machine learning (concept)
✓ Relationships: machine learning enables processing
✗ Avoid: "API", "input", "algorithms" as separate entities

{domain_instruction}{type_instruction}{style_instruction}{length_instruction}

📄 TEXT TO ANALYZE:
{text}

🎯 EXTRACTION RULES:
1. **First-class entities only**: Extract entities that have independent identity
2. **Specific relationships**: Use precise relationship types, avoid "related_to"
3. **High confidence**: Only extract what's clearly stated (confidence ≥ 0.7)
4. **Semantic accuracy**: Choose most specific entity/relationship types
5. **Factual grounding**: Base everything on the actual text content

Extract entities and relationships following these semantic principles."""

        return prompt

    def _build_verification_prompt(
        self,
        original_text: str,
        extraction: LLMExtractionOutput
    ) -> str:
        """Build verification prompt to validate extraction accuracy"""

        # Format the extracted entities and relationships for verification
        entity_list = []
        for entity in extraction.entities:
            entity_list.append(f"- {entity.name} ({entity.type.value}): {entity.context[:100]}")

        relationship_list = []
        for rel in extraction.relationships:
            relationship_list.append(f"- {rel.source} --{rel.relation.value}--> {rel.target}: {rel.context[:100]}")

        entities_text = "\n".join(entity_list) if entity_list else "None extracted"
        relationships_text = "\n".join(relationship_list) if relationship_list else "None extracted"

        prompt = f"""Verify and validate this knowledge extraction for accuracy and completeness.

ORIGINAL TEXT:
{original_text}

EXTRACTED ENTITIES:
{entities_text}

EXTRACTED RELATIONSHIPS:
{relationships_text}

🔍 **VERIFICATION TASKS:**

1. **Accuracy Check**: Are all entities and relationships actually present in the text?
   - Remove any hallucinated or incorrectly extracted items
   - Ensure entity types are accurate
   - Verify relationship types are correct

2. **Completeness Check**: Are there obvious entities or relationships missing?
   - Add clearly mentioned entities that were missed
   - Add obvious relationships that were missed
   - Don't over-extract - only add what's clearly stated

3. **Quality Check**: Are entities properly categorized?
   - Ensure first-class entities vs properties distinction
   - Use most specific entity types possible
   - Use most specific relationship types possible

4. **Confidence Assessment**: Set realistic confidence scores
   - High confidence (0.9+): Explicitly stated in text
   - Medium confidence (0.7-0.9): Clearly implied
   - Low confidence (0.5-0.7): Somewhat inferred

Return the verified extraction with corrected entities, relationships, and confidence scores."""

        return prompt

    def _build_semantic_refinement_prompt(
        self,
        original_text: str,
        extraction: LLMExtractionOutput
    ) -> str:
        """Build semantic refinement prompt to enhance extraction quality"""

        # Format the extracted entities and relationships for refinement
        entity_list = []
        for entity in extraction.entities:
            entity_list.append(f"- {entity.name} ({entity.type.value}): {entity.context[:100]}")

        relationship_list = []
        for rel in extraction.relationships:
            relationship_list.append(f"- {rel.source} --{rel.relation.value}--> {rel.target}: {rel.context[:100]}")

        entities_text = "\n".join(entity_list) if entity_list else "None extracted"
        relationships_text = "\n".join(relationship_list) if relationship_list else "None extracted"

        prompt = f"""Refine this knowledge graph extraction to improve semantic accuracy and completeness.

ORIGINAL TEXT:
{original_text}

CURRENT EXTRACTION:

ENTITIES:
{entities_text}

RELATIONSHIPS:
{relationships_text}

🔬 **SEMANTIC REFINEMENT AREAS:**

1. **Completeness**: Look for ADDITIONAL entities referenced indirectly or implicitly
2. **Semantic Precision**: Improve entity types - use most specific semantic type possible
3. **Relationship Quality**: Find NEW semantic relationships not captured initially
4. **Hidden Connections**: Search for implicit semantic patterns or hierarchies

🎯 **REFINEMENT TASKS:**
- Identify any missed entities that are semantically important
- Upgrade generic entity types to more specific semantic types
- Replace vague relationships with precise semantic connections
- Add implicit relationships that provide semantic context
- Improve confidence scores based on semantic clarity
- Remove any semantically inconsistent extractions

**Focus on finding NEW semantic elements while maintaining accuracy. Each addition should provide unique semantic value to the knowledge graph.**

Return the enhanced entities and relationships with improved semantic precision."""

        return prompt