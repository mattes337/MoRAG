"""PydanticAI-based relation extraction."""

import structlog
from typing import List, Optional, Dict, Any

from ..models import Entity, Relation
from ..ai import RelationExtractionAgent

logger = structlog.get_logger(__name__)


class RelationExtractor:
    """PydanticAI-based relation extractor - completely new implementation."""
    
    def __init__(self, min_confidence: float = 0.6, chunk_size: int = 3000, **kwargs):
        """Initialize the relation extractor.
        
        Args:
            min_confidence: Minimum confidence threshold for relations
            chunk_size: Maximum characters per chunk for large texts
            **kwargs: Additional arguments passed to the agent
        """
        self.min_confidence = min_confidence
        self.chunk_size = chunk_size
        self.agent = RelationExtractionAgent(min_confidence=min_confidence, **kwargs)
        self.logger = logger.bind(component="relation_extractor")
    
    async def extract(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        doc_id: Optional[str] = None,
        source_doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Relation]:
        """Extract relations from text using PydanticAI agent.
        
        Args:
            text: Text to extract relations from
            entities: Optional list of known entities
            doc_id: Optional document ID (deprecated, use source_doc_id)
            source_doc_id: Optional document ID to associate with relations
            **kwargs: Additional arguments
            
        Returns:
            List of Relation objects
        """
        # Handle backward compatibility
        if source_doc_id is None and doc_id is not None:
            source_doc_id = doc_id
        
        try:
            # Use PydanticAI agent for extraction
            relations = await self.agent.extract_relations(
                text=text,
                entities=entities,
                chunk_size=self.chunk_size,
                source_doc_id=source_doc_id
            )
            
            self.logger.info(
                "Relation extraction completed",
                text_length=len(text),
                num_entities=len(entities) if entities else 0,
                relations_found=len(relations),
                source_doc_id=source_doc_id
            )
            
            return relations
            
        except Exception as e:
            self.logger.error(
                "Relation extraction failed",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text)
            )
            raise
    
    async def extract_relations(
        self,
        text: str,
        entities: List[Entity],
        source_doc_id: Optional[str] = None,
        **kwargs
    ) -> List[Relation]:
        """Extract relations from text (alias for extract method).
        
        Args:
            text: Text to extract relations from
            entities: List of entities to consider for relations
            source_doc_id: Optional document ID to associate with relations
            **kwargs: Additional arguments
            
        Returns:
            List of extracted relations
        """
        return await self.extract(text, entities=entities, source_doc_id=source_doc_id, **kwargs)
    
    async def extract_with_entities(
        self,
        text: str,
        entities: List[Entity],
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations with known entities.
        
        Args:
            text: Text to extract relations from
            entities: List of known entities
            source_doc_id: ID of the source document
            
        Returns:
            List of Relation objects with resolved entity IDs
        """
        return await self.extract(text, entities=entities, source_doc_id=source_doc_id)
    
    async def extract_from_entity_pairs(
        self,
        text: str,
        entity_pairs: List[tuple[Entity, Entity]],
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations for specific entity pairs.
        
        Args:
            text: Text to extract relations from
            entity_pairs: List of entity pairs to check for relations
            source_doc_id: ID of the source document
            
        Returns:
            List of Relation objects
        """
        # Convert entity pairs to a flat list of entities
        entities = []
        for source_entity, target_entity in entity_pairs:
            if source_entity not in entities:
                entities.append(source_entity)
            if target_entity not in entities:
                entities.append(target_entity)
        
        # Extract relations and filter for the specific pairs
        all_relations = await self.extract(text, entities=entities, source_doc_id=source_doc_id)
        
        # Filter relations to only include the specified pairs
        pair_ids = set()
        for source_entity, target_entity in entity_pairs:
            pair_ids.add((source_entity.id, target_entity.id))
            pair_ids.add((target_entity.id, source_entity.id))  # Include reverse direction
        
        filtered_relations = []
        for relation in all_relations:
            if (relation.source_entity_id, relation.target_entity_id) in pair_ids:
                filtered_relations.append(relation)
        
        return filtered_relations
