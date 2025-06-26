"""PydanticAI agent for entity extraction."""

import asyncio
from typing import Type, List, Optional, Dict, Any
import structlog

from morag_core.ai import MoRAGBaseAgent, EntityExtractionResult, Entity, EntityType, ConfidenceLevel
from ..models import Entity as GraphEntity, EntityType as GraphEntityType

logger = structlog.get_logger(__name__)


class EntityExtractionAgent(MoRAGBaseAgent[EntityExtractionResult]):
    """PydanticAI agent for extracting entities from text."""

    def __init__(self, min_confidence: float = 0.6, dynamic_types: bool = True, entity_types: Optional[Dict[str, str]] = None, **kwargs):
        """Initialize the entity extraction agent.

        Args:
            min_confidence: Minimum confidence threshold for entities
            dynamic_types: Whether to use dynamic entity types (LLM-determined)
            entity_types: Custom entity types dict (type_name -> description). If None and dynamic_types=True, uses pure dynamic mode
            **kwargs: Additional arguments passed to base agent
        """
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.dynamic_types = dynamic_types
        self.entity_types = entity_types or {}
        self.logger = logger.bind(agent="entity_extraction")

    def get_result_type(self) -> Type[EntityExtractionResult]:
        return EntityExtractionResult

    def get_system_prompt(self) -> str:
        if self.dynamic_types and not self.entity_types:
            # Pure dynamic mode - let LLM determine appropriate entity types
            return """You are an expert entity extraction agent. Your task is to identify and extract named entities from text with high accuracy.

Determine the most appropriate entity type based on the semantic meaning and context of each entity. Do not limit yourself to predefined categories - use descriptive, domain-specific entity types that capture the essence of what each entity represents.

For each entity, provide:
1. name: The exact text as it appears in the source
2. type: A descriptive entity type that captures the semantic meaning (e.g., SOFTWARE_FRAMEWORK, MEDICAL_CONDITION, RESEARCH_METHODOLOGY, FINANCIAL_INSTRUMENT, etc.)
3. confidence: Your confidence in the extraction (0.0 to 1.0)
4. context: Brief description of the entity's role or significance

Guidelines for entity types:
- Use clear, descriptive names that capture the specific nature of the entity
- Prefer domain-specific types over generic ones when appropriate
- Use UPPER_CASE with underscores (e.g., PROGRAMMING_LANGUAGE, GOVERNMENT_AGENCY, SCIENTIFIC_CONCEPT)
- Be consistent within the same document/domain
- Consider the entity's role and function in the context

Focus on entities that are:
- Clearly identifiable and significant
- Relevant to the document's main topics
- Mentioned with sufficient context to determine their type

Avoid extracting:
- Common words or generic terms
- Pronouns or vague references
- Entities with very low confidence (<0.5)"""

        elif self.entity_types:
            # Custom types mode - use provided entity types
            types_section = "\n".join([f"- {type_name}: {description}" for type_name, description in self.entity_types.items()])
            return f"""You are an expert entity extraction agent. Your task is to identify and extract named entities from text with high accuracy.

Extract entities that represent:
{types_section}

For each entity, provide:
1. name: The exact text as it appears in the source
2. type: The most appropriate entity type from the list above
3. confidence: Your confidence in the extraction (0.0 to 1.0)
4. context: Brief description of the entity's role or significance

Focus on entities that are:
- Clearly identifiable and significant
- Relevant to the document's main topics
- Mentioned with sufficient context to determine their type

Avoid extracting:
- Common words or generic terms
- Pronouns or vague references
- Entities with very low confidence (<0.5)"""

        else:
            # Fallback to basic static types for backward compatibility
            return """You are an expert entity extraction agent. Your task is to identify and extract named entities from text with high accuracy.

Extract entities that represent:
- PERSON: Individual people, including names, titles, roles
- ORGANIZATION: Companies, institutions, government bodies, groups
- LOCATION: Places, addresses, geographical locations, buildings
- EVENT: Meetings, conferences, incidents, historical events
- CONCEPT: Ideas, theories, methodologies, abstract concepts
- PRODUCT: Software, hardware, services, brands, models
- TECHNOLOGY: Programming languages, frameworks, tools, systems
- DATE: Specific dates, time periods, deadlines
- MONEY: Financial amounts, budgets, costs, revenues
- OTHER: Any other significant named entities

For each entity, provide:
1. name: The exact text as it appears in the source
2. type: The most appropriate entity type from the list above
3. confidence: Your confidence in the extraction (0.0 to 1.0)
4. context: Brief description of the entity's role or significance

Focus on entities that are:
- Clearly identifiable and significant
- Relevant to the document's main topics
- Mentioned with sufficient context to determine their type

Avoid extracting:
- Common words or generic terms
- Pronouns or vague references
- Entities with very low confidence (<0.5)"""
    
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
    
    def _convert_to_graph_entity(self, entity: Entity, source_doc_id: Optional[str]) -> GraphEntity:
        """Convert AI entity to graph entity."""
        # Handle dynamic entity types
        if self.dynamic_types:
            # Use the entity type directly as a string for dynamic types
            if isinstance(entity.type, str):
                graph_type = entity.type
            else:
                # Handle enum types by extracting the value
                graph_type = entity.type.value if hasattr(entity.type, 'value') else str(entity.type)
        else:
            # Map AI entity types to graph entity types for static mode
            type_mapping = {
                EntityType.PERSON: GraphEntityType.PERSON,
                EntityType.ORGANIZATION: GraphEntityType.ORGANIZATION,
                EntityType.LOCATION: GraphEntityType.LOCATION,
                EntityType.EVENT: GraphEntityType.EVENT,
                EntityType.CONCEPT: GraphEntityType.CONCEPT,
                EntityType.PRODUCT: GraphEntityType.PRODUCT,
                EntityType.TECHNOLOGY: GraphEntityType.TECHNOLOGY,
                EntityType.DATE: GraphEntityType.DATE,
                EntityType.MONEY: GraphEntityType.MONEY,
                EntityType.OTHER: GraphEntityType.CUSTOM,
            }

            graph_type = type_mapping.get(entity.type, GraphEntityType.CUSTOM)
        
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
        """Remove duplicate entities based on name and type."""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            # Create a key based on normalized name and type
            key = (entity.name.lower().strip(), entity.type)
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
            else:
                # If we've seen this entity before, keep the one with higher confidence
                for i, existing in enumerate(deduplicated):
                    if (existing.name.lower().strip(), existing.type) == key:
                        if entity.confidence > existing.confidence:
                            deduplicated[i] = entity
                        break
        
        return deduplicated
