"""PydanticAI agent for entity extraction."""

import asyncio
from typing import Type, List, Optional, Dict, Any
import structlog

from morag_core.ai import MoRAGBaseAgent, EntityExtractionResult, Entity, ConfidenceLevel
from ..models import Entity as GraphEntity

logger = structlog.get_logger(__name__)


class EntityExtractionAgent(MoRAGBaseAgent[EntityExtractionResult]):
    """PydanticAI agent for extracting entities from text."""

    def __init__(self, min_confidence: float = 0.6, dynamic_types: bool = True, entity_types: Optional[Dict[str, str]] = None, language: Optional[str] = None, **kwargs):
        """Initialize the entity extraction agent.

        Args:
            min_confidence: Minimum confidence threshold for entities
            dynamic_types: Whether to use dynamic entity types (LLM-determined)
            entity_types: Custom entity types dict (type_name -> description). If None and dynamic_types=True, uses pure dynamic mode
            language: Language code for processing (e.g., 'en', 'de', 'fr')
            **kwargs: Additional arguments passed to base agent
        """
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.dynamic_types = dynamic_types
        self.entity_types = entity_types or {}
        self.language = language
        self.logger = logger.bind(agent="entity_extraction")

    def get_result_type(self) -> Type[EntityExtractionResult]:
        return EntityExtractionResult

    def get_system_prompt(self) -> str:
        # Build language instruction
        language_instruction = ""
        if self.language:
            language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST provide ALL entity descriptions and entity types in {self.language} language. This is mandatory. If the source text is in a different language, you MUST translate all descriptions to {self.language}. Do NOT provide descriptions in any other language.\n"

        if self.dynamic_types and not self.entity_types:
            # Pure dynamic mode - let LLM determine appropriate entity types
            return f"""You are an expert entity extraction agent. Your task is to identify and extract named entities from text with high accuracy.{language_instruction}

CRITICAL INSTRUCTION: Create BROAD, REUSABLE entity types. Avoid overly specific types that would create duplicate entities. Think of entity types as categories that many similar entities could share.

For each entity, provide:
1. name: The SINGULAR, UNCONJUGATED base form of the entity (NOT the exact text as it appears)
2. type: A BROAD, REUSABLE entity type that YOU determine
3. confidence: Your confidence in the extraction (0.0 to 1.0)
4. description: Generic, context-independent description of what this entity is (not its role in this specific document)

CRITICAL ENTITY NAME NORMALIZATION RULES:
- ALWAYS use SINGULAR form: "Schwermetall" not "Schwermetalle" or "Schwermetallen"
- ALWAYS use UNCONJUGATED base form: "Belastung" not "Belastungen"
- RESOLVE abbreviations where possible: "WHO" not "Weltgesundheitsorganisation" or "World Health Organization"
- If abbreviation is more commonly known, use the abbreviation: "DNA" not "Desoxyribonukleinsäure"
- Use the most CANONICAL form of the entity name
- Normalize case appropriately: proper nouns capitalized, common nouns lowercase

ENTITY TYPE CREATION RULES - FOLLOW STRICTLY:
- Use BROAD categories that can apply to many similar entities
- Prefer GENERAL types over specific ones
- Use SIMPLE, SINGLE-WORD types when possible (avoid complex compound types)
- If you must use compound types, keep them SHORT and GENERAL
- Think: "What is the SIMPLEST, most GENERAL category this entity belongs to?"
- Avoid creating types like "BIOLOGICAL_SOMETHING" - just use "BIOLOGICAL"

EXAMPLES OF GOOD BROAD TYPING:
- "Einstein", "Newton", "Darwin" → ALL should be SCIENTIST
- "Zirbeldrüse", "Herz", "Leber" → ALL should be BODY_PART
- "Harvard", "MIT", "Stanford" → ALL should be UNIVERSITY
- "Python", "Java", "C++" → ALL should be TECHNOLOGY
- "Photosynthesis", "Respiration", "Digestion" → ALL should be PROCESS
- "Berlin", "Paris", "London" → ALL should be CITY
- "Deutschland", "Frankreich", "England" → ALL should be COUNTRY

EXAMPLES OF BAD (TOO SPECIFIC) TYPING:
- "Einstein" → THEORETICAL_PHYSICIST (too specific, use SCIENTIST)
- "Zirbeldrüse" → PINEAL_GLAND (too specific, use BODY_PART)
- "Harvard" → IVY_LEAGUE_UNIVERSITY (too specific, use UNIVERSITY)
- "Python" → PROGRAMMING_LANGUAGE_INTERPRETED (too specific, use TECHNOLOGY)
- "Photosynthesis" → BIOLOGICAL_CELLULAR_PROCESS (too specific, use PROCESS)

PREFERRED SIMPLE CATEGORIES:
PERSON, SCIENTIST, AUTHOR, POLITICIAN, ARTIST, ATHLETE
ORGANIZATION, COMPANY, UNIVERSITY, HOSPITAL, GOVERNMENT
LOCATION, CITY, COUNTRY, REGION, BUILDING
CONCEPT, THEORY, PRINCIPLE, METHOD, TECHNIQUE
BIOLOGICAL, DISEASE, MEDICATION, TREATMENT
TECHNOLOGY, SOFTWARE, DEVICE, SYSTEM
DOCUMENT, BOOK, ARTICLE, REPORT
EVENT, CONFERENCE, WAR, DISCOVERY
SUBSTANCE, CHEMICAL, MATERIAL
PROCESS, PROCEDURE, ACTIVITY

REMEMBER: Keep types SIMPLE and GENERAL. If in doubt, choose the broader category!

Focus on entities that are:
- Clearly identifiable and significant
- Relevant to the document's main topics and content
- Mentioned with sufficient context to determine their type
- Represent meaningful concepts, people, places, or things discussed in the content

Avoid extracting:
- Technical metadata (file formats, timestamps, dimensions, codecs, etc.)
- Common words or generic terms
- Pronouns or vague references
- Numbers without semantic meaning (unless they represent important quantities)
- File properties, technical specifications, or system information
- Entities with very low confidence (<0.5)

REMEMBER: The goal is REUSABILITY. Multiple entities should share the same type when they belong to the same general category."""

        elif self.entity_types:
            # Custom types mode - use provided entity types
            types_section = "\n".join([f"- {type_name}: {description}" for type_name, description in self.entity_types.items()])
            return f"""You are an expert entity extraction agent. Your task is to identify and extract named entities from text with high accuracy.{language_instruction}

Extract entities that represent:
{types_section}

For each entity, provide:
1. name: The SINGULAR, UNCONJUGATED base form of the entity (NOT the exact text as it appears)
2. type: The most appropriate entity type from the list above
3. confidence: Your confidence in the extraction (0.0 to 1.0)
4. description: Generic, context-independent description of what this entity is (not its role in this specific document)

CRITICAL ENTITY NAME NORMALIZATION RULES:
- ALWAYS use SINGULAR form: "Schwermetall" not "Schwermetalle" or "Schwermetallen"
- ALWAYS use UNCONJUGATED base form: "Belastung" not "Belastungen"
- RESOLVE abbreviations where possible: "WHO" not "Weltgesundheitsorganisation" or "World Health Organization"
- If abbreviation is more commonly known, use the abbreviation: "DNA" not "Desoxyribonukleinsäure"
- Use the most CANONICAL form of the entity name
- Normalize case appropriately: proper nouns capitalized, common nouns lowercase

Focus on entities that are:
- Clearly identifiable and significant
- Relevant to the document's main topics
- Mentioned with sufficient context to determine their type

Avoid extracting:
- Common words or generic terms
- Pronouns or vague references
- Entities with very low confidence (<0.5)"""

        else:
            # No static types - always use dynamic mode
            return f"""You are an expert entity extraction agent. Your task is to identify and extract named entities from text with high accuracy.{language_instruction}

CRITICAL INSTRUCTION: Create BROAD, REUSABLE entity types. Avoid overly specific types that would create duplicate entities. Think of entity types as categories that many similar entities could share.

For each entity, provide:
1. name: The SINGULAR, UNCONJUGATED base form of the entity (NOT the exact text as it appears)
2. type: A BROAD, REUSABLE entity type that YOU determine
3. confidence: Your confidence in the extraction (0.0 to 1.0)
4. description: Generic, context-independent description of what this entity is (not its role in this specific document)

CRITICAL ENTITY NAME NORMALIZATION RULES:
- ALWAYS use SINGULAR form: "Schwermetall" not "Schwermetalle" or "Schwermetallen"
- ALWAYS use UNCONJUGATED base form: "Belastung" not "Belastungen"
- RESOLVE abbreviations where possible: "WHO" not "Weltgesundheitsorganisation" or "World Health Organization"
- If abbreviation is more commonly known, use the abbreviation: "DNA" not "Desoxyribonukleinsäure"
- Use the most CANONICAL form of the entity name
- Normalize case appropriately: proper nouns capitalized, common nouns lowercase

ENTITY TYPE CREATION RULES - FOLLOW STRICTLY:
- Use BROAD categories that can apply to many similar entities
- Prefer GENERAL types over specific ones
- Use SIMPLE, SINGLE-WORD types when possible (avoid complex compound types)
- If you must use compound types, keep them SHORT and GENERAL
- Think: "What is the SIMPLEST, most GENERAL category this entity belongs to?"
- Avoid creating types like "BIOLOGICAL_SOMETHING" - just use "BIOLOGICAL"

EXAMPLES OF GOOD BROAD TYPING:
- "Einstein", "Newton", "Darwin" → ALL should be SCIENTIST
- "Zirbeldrüse", "Herz", "Leber" → ALL should be BODY_PART
- "Harvard", "MIT", "Stanford" → ALL should be UNIVERSITY
- "Python", "Java", "C++" → ALL should be TECHNOLOGY
- "Photosynthesis", "Respiration", "Digestion" → ALL should be PROCESS
- "Berlin", "Paris", "London" → ALL should be CITY
- "Deutschland", "Frankreich", "England" → ALL should be COUNTRY

EXAMPLES OF BAD (TOO SPECIFIC) TYPING:
- "Einstein" → THEORETICAL_PHYSICIST (too specific, use SCIENTIST)
- "Zirbeldrüse" → PINEAL_GLAND (too specific, use BODY_PART)
- "Harvard" → IVY_LEAGUE_UNIVERSITY (too specific, use UNIVERSITY)
- "Python" → PROGRAMMING_LANGUAGE_INTERPRETED (too specific, use TECHNOLOGY)
- "Photosynthesis" → BIOLOGICAL_CELLULAR_PROCESS (too specific, use PROCESS)

PREFERRED SIMPLE CATEGORIES:
PERSON, SCIENTIST, AUTHOR, POLITICIAN, ARTIST, ATHLETE
ORGANIZATION, COMPANY, UNIVERSITY, HOSPITAL, GOVERNMENT
LOCATION, CITY, COUNTRY, REGION, BUILDING
CONCEPT, THEORY, PRINCIPLE, METHOD, TECHNIQUE
BIOLOGICAL, DISEASE, MEDICATION, TREATMENT
TECHNOLOGY, SOFTWARE, DEVICE, SYSTEM
DOCUMENT, BOOK, ARTICLE, REPORT
EVENT, CONFERENCE, WAR, DISCOVERY
SUBSTANCE, CHEMICAL, MATERIAL
PROCESS, PROCEDURE, ACTIVITY

Focus on entities that are:
- Clearly identifiable and significant
- Relevant to the document's main topics and content
- Mentioned with sufficient context to determine their type
- Represent meaningful concepts, people, places, or things discussed in the content

Avoid extracting:
- Technical metadata (file formats, timestamps, dimensions, codecs, etc.)
- Common words or generic terms
- Pronouns or vague references
- Numbers without semantic meaning (unless they represent important quantities)
- File properties, technical specifications, or system information
- Entities with very low confidence (<0.5)

REMEMBER: The goal is REUSABILITY. Multiple entities should share the same type when they belong to the same general category."""
    
    async def extract_entities(
        self,
        text: str,
        chunk_size: int = 4000,
        source_doc_id: Optional[str] = None
    ) -> List[GraphEntity]:
        """Extract entities from text with automatic chunking.
        
        Args:
            text: Text to extract entities from
            chunk_size: Maximum characters per chunk for large texts
            source_doc_id: Optional source document ID
            
        Returns:
            List of GraphEntity objects
        """
        if not text or not text.strip():
            return []
        
        self.logger.info(
            "Starting entity extraction",
            text_length=len(text),
            chunk_size=chunk_size,
            source_doc_id=source_doc_id
        )
        
        try:
            # Check if text needs chunking
            if len(text) <= chunk_size:
                entities = await self._extract_single_chunk(text)
            else:
                entities = await self._extract_chunked(text, chunk_size)
            
            # Convert to GraphEntity objects and filter by confidence
            graph_entities = []
            for entity in entities:
                try:
                    # Validate confidence is a float
                    if not isinstance(entity.confidence, (int, float)):
                        self.logger.warning(
                            "Invalid confidence type, skipping entity",
                            entity_name=entity.name,
                            confidence_type=type(entity.confidence).__name__,
                            confidence_value=str(entity.confidence)
                        )
                        continue

                    confidence = float(entity.confidence)
                    if confidence >= self.min_confidence:
                        graph_entity = self._convert_to_graph_entity(entity, source_doc_id)
                        graph_entities.append(graph_entity)
                except Exception as e:
                    self.logger.warning(
                        "Error processing entity, skipping",
                        entity_name=getattr(entity, 'name', 'unknown'),
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    continue
            
            # Deduplicate entities
            graph_entities = self._deduplicate_entities(graph_entities)
            
            self.logger.info(
                "Entity extraction completed",
                total_entities=len(graph_entities),
                min_confidence=self.min_confidence
            )
            
            return graph_entities
            
        except Exception as e:
            self.logger.error("Entity extraction failed", error=str(e), error_type=type(e).__name__)
            raise
    
    async def _extract_single_chunk(self, text: str) -> List[Entity]:
        """Extract entities from a single chunk of text."""
        prompt = f"Extract named entities from the following text:\n\n{text}"
        
        result = await self.run(prompt)
        return result.entities
    
    async def _extract_chunked(self, text: str, chunk_size: int) -> List[Entity]:
        """Extract entities from text using chunking for large documents."""
        chunks = self._split_text_into_chunks(text, chunk_size)
        all_entities = []
        
        self.logger.info(f"Processing {len(chunks)} chunks for entity extraction")
        
        # Process chunks concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
        
        async def process_chunk(chunk_idx: int, chunk: str) -> List[Entity]:
            async with semaphore:
                try:
                    self.logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
                    return await self._extract_single_chunk(chunk)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to process chunk {chunk_idx + 1}",
                        error=str(e)
                    )
                    return []
        
        # Process all chunks
        chunk_results = await asyncio.gather(
            *[process_chunk(i, chunk) for i, chunk in enumerate(chunks)],
            return_exceptions=True
        )
        
        # Collect results
        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                self.logger.warning(f"Chunk {i + 1} failed with exception: {result}")
            else:
                all_entities.extend(result)
        
        return all_entities
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks with word boundaries."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Find word boundary
            while end > start and text[end] not in ' \n\t':
                end -= 1
            
            if end == start:
                # No word boundary found, force split
                end = start + chunk_size
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def _simplify_entity_type(self, entity_type: str) -> str:
        """Simplify entity type by taking only the first part before underscore.

        Examples:
        - BIOLOGICAL_SOMETHING => BIOLOGICAL
        - TECHNOLOGY_FRAMEWORK => TECHNOLOGY
        - PERSON_SCIENTIST => PERSON
        """
        if '_' in entity_type:
            return entity_type.split('_')[0]
        return entity_type

    def _convert_to_graph_entity(self, entity: Entity, source_doc_id: Optional[str]) -> GraphEntity:
        """Convert AI entity to graph entity."""
        # Always use dynamic entity types - LLM determines the type
        if isinstance(entity.type, str):
            graph_type = entity.type
        else:
            # Handle enum types by extracting the value (fallback for compatibility)
            graph_type = entity.type.value if hasattr(entity.type, 'value') else str(entity.type)

        # Simplify the entity type to use only the first part
        graph_type = self._simplify_entity_type(graph_type)
        
        # Create attributes from metadata and context
        attributes = entity.metadata.copy() if entity.metadata else {}
        if entity.context:
            attributes['context'] = entity.context
        if entity.start_pos is not None:
            attributes['start_pos'] = entity.start_pos
        if entity.end_pos is not None:
            attributes['end_pos'] = entity.end_pos
        
        # Ensure confidence is a float
        confidence = float(entity.confidence) if isinstance(entity.confidence, (int, float)) else 0.5

        return GraphEntity(
            name=entity.name,
            type=graph_type,
            confidence=confidence,
            source_doc_id=source_doc_id,
            attributes=attributes
        )
    
    def _deduplicate_entities(self, entities: List[GraphEntity]) -> List[GraphEntity]:
        """Remove duplicate entities based on name only, merging different types."""
        # Group entities by normalized name (case-insensitive)
        entity_groups = {}

        for entity in entities:
            # Use only the normalized name as the key, ignoring type
            normalized_name = entity.name.lower().strip()

            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)

        # Merge entities in each group
        deduplicated = []
        for name_group in entity_groups.values():
            if len(name_group) == 1:
                # Single entity, no merging needed
                deduplicated.append(name_group[0])
            else:
                # Multiple entities with same name, merge them
                merged_entity = self._merge_entities_with_same_name(name_group)
                deduplicated.append(merged_entity)

        return deduplicated

    def _merge_entities_with_same_name(self, entities: List[GraphEntity]) -> GraphEntity:
        """Merge multiple entities with the same name but potentially different types."""
        if not entities:
            raise ValueError("Cannot merge empty entity list")

        if len(entities) == 1:
            return entities[0]

        # Find the entity with the highest confidence
        best_entity = max(entities, key=lambda e: e.confidence)

        # Use the type from the highest confidence entity
        # The LLM determines all types, so we trust the highest confidence extraction
        final_type = best_entity.type

        # Merge attributes from all entities
        merged_attributes = {}
        for entity in entities:
            if entity.attributes:
                merged_attributes.update(entity.attributes)

        # Create merged entity using the best entity as base
        merged_entity = GraphEntity(
            name=best_entity.name,  # Use the exact name from the best entity
            type=final_type,
            confidence=best_entity.confidence,
            source_doc_id=best_entity.source_doc_id,
            attributes=merged_attributes
        )

        self.logger.debug(
            "Merged entities with same name",
            entity_name=best_entity.name,
            original_types=[e.type for e in entities],
            final_type=final_type,
            merged_confidence=best_entity.confidence
        )

        return merged_entity
