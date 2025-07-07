"""PydanticAI agent for relation extraction."""

import asyncio
from typing import Type, List, Optional, Dict, Any, Tuple
import structlog

from morag_core.ai import MoRAGBaseAgent, RelationExtractionResult, Relation, ConfidenceLevel
from ..models import Entity as GraphEntity, Relation as GraphRelation

logger = structlog.get_logger(__name__)


class RelationExtractionAgent(MoRAGBaseAgent[RelationExtractionResult]):
    """PydanticAI agent for extracting relations between entities."""

    def __init__(self, min_confidence: float = 0.6, dynamic_types: bool = True, relation_types: Optional[Dict[str, str]] = None, language: Optional[str] = None, **kwargs):
        """Initialize the relation extraction agent.

        Args:
            min_confidence: Minimum confidence threshold for relations
            dynamic_types: Whether to use dynamic relation types (LLM-determined)
            relation_types: Custom relation types dict (type_name -> description). If None and dynamic_types=True, uses pure dynamic mode
            language: Language code for processing (e.g., 'en', 'de', 'fr')
            **kwargs: Additional arguments passed to base agent
        """
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.dynamic_types = dynamic_types
        self.relation_types = relation_types or {}
        self.language = language
        self.logger = logger.bind(agent="relation_extraction")

    def get_result_type(self) -> Type[RelationExtractionResult]:
        return RelationExtractionResult

    def get_system_prompt(self) -> str:
        # Build language instruction
        language_instruction = ""
        if self.language:
            language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST provide ALL relation descriptions, context, and relation types in {self.language} language. This is mandatory. If the source text is in a different language, you MUST translate all descriptions and context to {self.language}. Do NOT provide descriptions in any other language.\n"

        if self.dynamic_types and not self.relation_types:
            # Pure dynamic mode - let LLM determine appropriate relation types
            return f"""You are an expert relation extraction agent. Your task is to identify meaningful relationships between entities mentioned in text.{language_instruction}

Extract relations that represent clear, factual connections between entities. Determine the most appropriate relation type based on the semantic meaning and context of the relationship. Do not limit yourself to predefined categories.

For each relation, provide:
1. source_entity: NORMALIZED name of the source entity (use SINGULAR, UNCONJUGATED form that matches the known entities)
2. target_entity: NORMALIZED name of the target entity (use SINGULAR, UNCONJUGATED form that matches the known entities)
3. relation_type: A SIMPLE, descriptive relation type (e.g., EMPLOYS, CREATES, INFLUENCES, CONTAINS, USES, etc.)
4. confidence: Your confidence in the relation (0.0 to 1.0)
5. context: Brief explanation of the relationship

CRITICAL ENTITY NAME MATCHING:
- Use the NORMALIZED form of entity names (singular, unconjugated) that match the known entities list
- If you see "Schwermetallen" in text, use "Schwermetall" (the normalized form from known entities)
- If you see "Belastungen" in text, use "Belastung" (the normalized form from known entities)
- Only extract relations between entities that are explicitly listed in the "Known entities" section

Guidelines for relation types:
- Use SIMPLE, clear names that capture the core relationship
- Prefer GENERAL types over overly specific ones
- Use SINGLE WORDS when possible (EMPLOYS, CREATES, INFLUENCES, CONTAINS, USES)
- If compound types are needed, keep them SHORT and GENERAL
- Avoid complex types like "COLLABORATES_WITH_ON_PROJECT" - use "COLLABORATES"
- Be consistent within the same document/domain
- Consider the direction of the relationship (source -> target)

Focus on relations that are:
- Explicitly stated or clearly implied in the text
- Factual and verifiable
- Significant to understanding the content
- Between entities that actually exist in the known entities list

Avoid extracting:
- Relations with entities not in the known entities list
- Vague or uncertain relationships
- Relations based on speculation
- Relations with very low confidence (<0.5)
- Duplicate or redundant relations
- Relations involving technical metadata or file properties"""

        elif self.relation_types:
            # Custom types mode - use provided relation types
            types_section = "\n".join([f"- {type_name}: {description}" for type_name, description in self.relation_types.items()])
            return f"""You are an expert relation extraction agent. Your task is to identify meaningful relationships between entities mentioned in text.

Extract relations that represent clear, factual connections between entities:

RELATION TYPES:
{types_section}

For each relation, provide:
1. source_entity: NORMALIZED name of the source entity (use SINGULAR, UNCONJUGATED form that matches the known entities)
2. target_entity: NORMALIZED name of the target entity (use SINGULAR, UNCONJUGATED form that matches the known entities)
3. relation_type: Most appropriate relation type from the list above
4. confidence: Your confidence in the relation (0.0 to 1.0)
5. context: Brief explanation of the relationship

CRITICAL ENTITY NAME MATCHING:
- Use the NORMALIZED form of entity names (singular, unconjugated) that match the known entities list
- If you see "Schwermetallen" in text, use "Schwermetall" (the normalized form from known entities)
- If you see "Belastungen" in text, use "Belastung" (the normalized form from known entities)
- Only extract relations between entities that are explicitly listed in the "Known entities" section

Focus on relations that are:
- Explicitly stated or clearly implied in the text
- Factual and verifiable
- Significant to understanding the content
- Between entities that actually exist in the known entities list

Avoid extracting:
- Relations with entities not in the known entities list
- Vague or uncertain relationships
- Relations based on speculation
- Relations with very low confidence (<0.5)
- Duplicate or redundant relations
- Relations involving technical metadata or file properties"""

        else:
            # No static types - always use dynamic mode
            return f"""You are an expert relation extraction agent. Your task is to identify meaningful relationships between entities mentioned in text.{language_instruction}

Extract relations that represent clear, factual connections between entities. Determine the most appropriate relation type based on the semantic meaning and context of the relationship. Do not limit yourself to predefined categories.

For each relation, provide:
1. source_entity: NORMALIZED name of the source entity (use SINGULAR, UNCONJUGATED form that matches the known entities)
2. target_entity: NORMALIZED name of the target entity (use SINGULAR, UNCONJUGATED form that matches the known entities)
3. relation_type: A SIMPLE, descriptive relation type (e.g., EMPLOYS, CREATES, INFLUENCES, CONTAINS, USES, etc.)
4. confidence: Your confidence in the relation (0.0 to 1.0)
5. context: Brief explanation of the relationship

CRITICAL ENTITY NAME MATCHING:
- Use the NORMALIZED form of entity names (singular, unconjugated) that match the known entities list
- If you see "Schwermetallen" in text, use "Schwermetall" (the normalized form from known entities)
- If you see "Belastungen" in text, use "Belastung" (the normalized form from known entities)
- Only extract relations between entities that are explicitly listed in the "Known entities" section

Guidelines for relation types:
- Use SIMPLE, clear names that capture the core relationship
- Prefer GENERAL types over overly specific ones
- Use SINGLE WORDS when possible (EMPLOYS, CREATES, INFLUENCES, CONTAINS, USES)
- If compound types are needed, keep them SHORT and GENERAL
- Avoid complex types like "COLLABORATES_WITH_ON_PROJECT" - use "COLLABORATES"
- Be consistent within the same document/domain
- Consider the direction of the relationship (source -> target)

Focus on relations that are:
- Explicitly stated or clearly implied in the text
- Factual and verifiable
- Significant to understanding the content

Avoid extracting:
- Vague or uncertain relationships
- Relations based on speculation
- Relations with very low confidence (<0.5)
- Duplicate or redundant relations"""
    
    async def extract_relations(
        self,
        text: str,
        entities: Optional[List[GraphEntity]] = None,
        chunk_size: int = 3000,
        source_doc_id: Optional[str] = None
    ) -> List[GraphRelation]:
        """Extract relations from text with known entities.
        
        Args:
            text: Text to extract relations from
            entities: List of known entities to consider
            chunk_size: Maximum characters per chunk for large texts
            source_doc_id: Optional source document ID
            
        Returns:
            List of GraphRelation objects
        """
        if not text or not text.strip():
            return []
        
        entity_names = [entity.name for entity in entities] if entities else []
        
        self.logger.info(
            "Starting relation extraction",
            text_length=len(text),
            num_entities=len(entity_names),
            chunk_size=chunk_size,
            source_doc_id=source_doc_id
        )
        
        try:
            # Check if text needs chunking
            if len(text) <= chunk_size:
                relations = await self._extract_single_chunk(text, entity_names)
            else:
                relations = await self._extract_chunked(text, entity_names, chunk_size)
            
            # Convert to GraphRelation objects and filter by confidence
            graph_relations = []
            for relation in relations:
                if relation.confidence >= self.min_confidence:
                    graph_relation = self._convert_to_graph_relation(
                        relation, entities, source_doc_id
                    )
                    if graph_relation:  # Only add if entity resolution succeeded
                        graph_relations.append(graph_relation)
            
            # Deduplicate relations
            graph_relations = self._deduplicate_relations(graph_relations)
            
            self.logger.info(
                "Relation extraction completed",
                total_relations=len(graph_relations),
                min_confidence=self.min_confidence
            )
            
            return graph_relations
            
        except Exception as e:
            self.logger.error("Relation extraction failed", error=str(e), error_type=type(e).__name__)
            raise
    
    async def _extract_single_chunk(self, text: str, entity_names: List[str]) -> List[Relation]:
        """Extract relations from a single chunk of text."""
        entity_context = ""
        if entity_names:
            entity_context = f"\n\nKnown entities in this text: {', '.join(entity_names)}"
        
        prompt = f"Extract relations between entities in the following text:{entity_context}\n\nText:\n{text}"
        
        result = await self.run(prompt)
        return result.relations
    
    async def _extract_chunked(self, text: str, entity_names: List[str], chunk_size: int) -> List[Relation]:
        """Extract relations from text using chunking for large documents."""
        chunks = self._split_text_into_chunks(text, chunk_size)
        all_relations = []
        
        self.logger.info(f"Processing {len(chunks)} chunks for relation extraction")
        
        # Process chunks concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
        
        async def process_chunk(chunk_idx: int, chunk: str) -> List[Relation]:
            async with semaphore:
                try:
                    self.logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
                    return await self._extract_single_chunk(chunk, entity_names)
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
                all_relations.extend(result)
        
        return all_relations
    
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

    def _simplify_relation_type(self, relation_type: str) -> str:
        """Simplify relation type by taking only the first part before underscore.

        Examples:
        - WORKS_FOR_COMPANY => WORKS
        - LOCATED_IN_CITY => LOCATED
        - COLLABORATES_WITH_PERSON => COLLABORATES
        """
        if '_' in relation_type:
            return relation_type.split('_')[0]
        return relation_type

    def _convert_to_graph_relation(
        self,
        relation: Relation,
        entities: Optional[List[GraphEntity]],
        source_doc_id: Optional[str]
    ) -> Optional[GraphRelation]:
        """Convert AI relation to graph relation with entity ID resolution."""
        if not entities:
            self.logger.warning("No entities provided for relation resolution")
            return None
        
        # Create entity name to ID mapping
        entity_name_to_id = {entity.name.lower().strip(): entity.id for entity in entities}
        
        # Resolve source entity ID
        source_key = relation.source_entity.lower().strip()
        source_entity_id = entity_name_to_id.get(source_key)
        
        # Resolve target entity ID
        target_key = relation.target_entity.lower().strip()
        target_entity_id = entity_name_to_id.get(target_key)
        
        if not source_entity_id or not target_entity_id:
            self.logger.debug(
                "Could not resolve entity IDs for relation",
                source_entity=relation.source_entity,
                target_entity=relation.target_entity,
                source_resolved=source_entity_id is not None,
                target_resolved=target_entity_id is not None
            )
            return None
        
        # Always use dynamic relation types - LLM determines the type
        if isinstance(relation.relation_type, str):
            graph_type = relation.relation_type
        else:
            # Handle enum types by extracting the value (fallback for compatibility)
            graph_type = relation.relation_type.value if hasattr(relation.relation_type, 'value') else str(relation.relation_type)

        # Simplify the relation type to use only the first part
        graph_type = self._simplify_relation_type(graph_type)
        
        # Create attributes from metadata and context
        attributes = relation.metadata.copy() if relation.metadata else {}
        if relation.context:
            attributes['context'] = relation.context
        
        return GraphRelation(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            type=graph_type,
            confidence=relation.confidence,
            source_doc_id=source_doc_id,
            attributes=attributes
        )
    
    def _deduplicate_relations(self, relations: List[GraphRelation]) -> List[GraphRelation]:
        """Remove duplicate relations based on source, target, and type."""
        seen = set()
        deduplicated = []
        
        for relation in relations:
            # Create a key based on source, target, and type
            key = (relation.source_entity_id, relation.target_entity_id, relation.type)
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(relation)
            else:
                # If we've seen this relation before, keep the one with higher confidence
                for i, existing in enumerate(deduplicated):
                    if (existing.source_entity_id, existing.target_entity_id, existing.type) == key:
                        if relation.confidence > existing.confidence:
                            deduplicated[i] = relation
                        break
        
        return deduplicated
