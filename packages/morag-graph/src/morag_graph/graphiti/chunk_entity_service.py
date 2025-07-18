"""Service for managing chunk-entity relationships in Graphiti."""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

from .entity_storage import GraphitiEntityStorage
from .search_service import GraphitiSearchService
from morag_graph.models import Entity, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkEntityMapping:
    """Mapping between chunks and entities."""
    chunk_id: str
    entity_ids: List[str]
    confidence_scores: Dict[str, float]
    mention_contexts: Dict[str, str]  # entity_id -> context text


@dataclass
class EntityMentionResult:
    """Result of entity mention analysis."""
    entity_id: str
    chunk_id: str
    mention_text: str
    context: str
    confidence: float
    start_position: int
    end_position: int


class ChunkEntityRelationshipService:
    """Service for managing chunk-entity relationships in Graphiti."""
    
    def __init__(
        self, 
        entity_storage: GraphitiEntityStorage,
        search_service: GraphitiSearchService
    ):
        self.entity_storage = entity_storage
        self.search_service = search_service
        self.graphiti = entity_storage.graphiti
    
    async def create_chunk_entity_mappings(
        self,
        chunks: List[DocumentChunk],
        entities: List[Entity],
        document_id: str
    ) -> List[ChunkEntityMapping]:
        """Create mappings between chunks and entities.
        
        Args:
            chunks: List of document chunks
            entities: List of extracted entities
            document_id: Document ID
            
        Returns:
            List of chunk-entity mappings
        """
        mappings = []
        
        for chunk in chunks:
            # Find entities mentioned in this chunk
            chunk_entities = await self._find_entities_in_chunk(chunk, entities)
            
            if chunk_entities:
                # Calculate confidence scores and extract contexts
                confidence_scores = {}
                mention_contexts = {}
                
                for entity in chunk_entities:
                    confidence, context = self._calculate_entity_confidence_and_context(
                        chunk, entity
                    )
                    confidence_scores[entity.id] = confidence
                    mention_contexts[entity.id] = context
                
                mapping = ChunkEntityMapping(
                    chunk_id=chunk.id,
                    entity_ids=[e.id for e in chunk_entities],
                    confidence_scores=confidence_scores,
                    mention_contexts=mention_contexts
                )
                mappings.append(mapping)
        
        return mappings
    
    async def store_chunk_entity_relationships(
        self,
        mappings: List[ChunkEntityMapping],
        document_id: str
    ) -> Dict[str, Any]:
        """Store chunk-entity relationships in Graphiti.
        
        Args:
            mappings: List of chunk-entity mappings
            document_id: Document ID
            
        Returns:
            Storage result information
        """
        if not self.graphiti:
            return {
                "success": False,
                "error": "Graphiti instance not available",
                "stored_relationships": 0
            }
        
        try:
            stored_count = 0
            
            for mapping in mappings:
                # Create relationship episodes linking chunks to entities
                for entity_id in mapping.entity_ids:
                    relationship_data = {
                        "chunk_id": mapping.chunk_id,
                        "entity_id": entity_id,
                        "document_id": document_id,
                        "confidence": mapping.confidence_scores.get(entity_id, 0.0),
                        "context": mapping.mention_contexts.get(entity_id, ""),
                        "relationship_type": "chunk_mentions_entity"
                    }
                    
                    # Store as episode metadata or separate relationship episode
                    await self._store_chunk_entity_relationship(relationship_data)
                    stored_count += 1
            
            return {
                "success": True,
                "stored_relationships": stored_count,
                "document_id": document_id,
                "chunk_count": len(mappings)
            }
            
        except Exception as e:
            logger.error(f"Failed to store chunk-entity relationships: {e}")
            return {
                "success": False,
                "error": str(e),
                "stored_relationships": 0
            }
    
    async def find_chunks_by_entity(
        self,
        entity_id: str,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find chunks that mention a specific entity.
        
        Args:
            entity_id: Entity ID to search for
            document_id: Optional document ID to limit search
            
        Returns:
            List of chunk information with mention details
        """
        if not self.graphiti:
            return []
        
        try:
            # Search for chunk-entity relationships
            query = f"entity_id:{entity_id}"
            if document_id:
                query += f" AND document_id:{document_id}"
            
            results, _ = await self.search_service.search(query, limit=100)
            
            chunk_info = []
            for result in results:
                if hasattr(result, 'metadata') and result.metadata.get('relationship_type') == 'chunk_mentions_entity':
                    chunk_info.append({
                        "chunk_id": result.metadata.get('chunk_id'),
                        "document_id": result.metadata.get('document_id'),
                        "confidence": result.metadata.get('confidence', 0.0),
                        "context": result.metadata.get('context', ''),
                        "score": result.score
                    })
            
            return chunk_info
            
        except Exception as e:
            logger.error(f"Failed to find chunks by entity: {e}")
            return []
    
    async def find_entities_by_chunk(
        self,
        chunk_id: str
    ) -> List[Dict[str, Any]]:
        """Find entities mentioned in a specific chunk.
        
        Args:
            chunk_id: Chunk ID to search for
            
        Returns:
            List of entity information with mention details
        """
        if not self.graphiti:
            return []
        
        try:
            # Search for chunk-entity relationships
            query = f"chunk_id:{chunk_id}"
            
            results, _ = await self.search_service.search(query, limit=100)
            
            entity_info = []
            for result in results:
                if hasattr(result, 'metadata') and result.metadata.get('relationship_type') == 'chunk_mentions_entity':
                    entity_info.append({
                        "entity_id": result.metadata.get('entity_id'),
                        "confidence": result.metadata.get('confidence', 0.0),
                        "context": result.metadata.get('context', ''),
                        "score": result.score
                    })
            
            return entity_info
            
        except Exception as e:
            logger.error(f"Failed to find entities by chunk: {e}")
            return []
    
    async def _find_entities_in_chunk(
        self,
        chunk: DocumentChunk,
        entities: List[Entity]
    ) -> List[Entity]:
        """Find entities that are mentioned in a chunk.
        
        Args:
            chunk: Document chunk
            entities: List of entities to check
            
        Returns:
            List of entities found in the chunk
        """
        chunk_entities = []
        chunk_text = chunk.content.lower()
        
        for entity in entities:
            # Simple text matching - in practice, you'd use more sophisticated NER
            entity_name = entity.name.lower()
            
            if entity_name in chunk_text:
                chunk_entities.append(entity)
            
            # Check aliases if available
            if hasattr(entity, 'aliases') and entity.aliases:
                for alias in entity.aliases:
                    if alias.lower() in chunk_text:
                        chunk_entities.append(entity)
                        break
        
        return chunk_entities
    
    def _calculate_entity_confidence_and_context(
        self,
        chunk: DocumentChunk,
        entity: Entity
    ) -> Tuple[float, str]:
        """Calculate confidence score and extract context for entity mention.
        
        Args:
            chunk: Document chunk
            entity: Entity
            
        Returns:
            Tuple of (confidence_score, context_text)
        """
        chunk_text = chunk.content
        entity_name = entity.name
        
        # Find entity mention in text
        pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
        match = pattern.search(chunk_text)
        
        if match:
            # Extract context around the mention
            start = max(0, match.start() - 50)
            end = min(len(chunk_text), match.end() + 50)
            context = chunk_text[start:end].strip()
            
            # Simple confidence calculation based on exact match
            confidence = 0.9 if match.group() == entity_name else 0.7
            
            return confidence, context
        
        # Fallback for aliases or partial matches
        return 0.5, chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
    
    async def _store_chunk_entity_relationship(
        self,
        relationship_data: Dict[str, Any]
    ):
        """Store a single chunk-entity relationship.
        
        Args:
            relationship_data: Relationship information
        """
        if not self.graphiti:
            return
        
        try:
            # Create episode for the relationship
            episode_name = f"chunk_entity_{relationship_data['chunk_id']}_{relationship_data['entity_id']}"
            
            content = (
                f"Chunk {relationship_data['chunk_id']} mentions entity {relationship_data['entity_id']} "
                f"with confidence {relationship_data['confidence']:.2f}. "
                f"Context: {relationship_data['context'][:200]}..."
            )
            
            await self.graphiti.add_episode(
                name=episode_name,
                content=content,
                source_description="Chunk-Entity Relationship",
                metadata=relationship_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store chunk-entity relationship: {e}")
    
    def get_relationship_stats(self) -> Dict[str, Any]:
        """Get statistics about chunk-entity relationships.
        
        Returns:
            Dictionary with relationship statistics
        """
        return {
            "service_available": self.graphiti is not None,
            "entity_storage_stats": self.entity_storage.get_storage_stats() if self.entity_storage else {},
            "search_service_available": self.search_service is not None
        }


def create_chunk_entity_service(
    entity_storage: GraphitiEntityStorage,
    search_service: GraphitiSearchService
) -> ChunkEntityRelationshipService:
    """Create a ChunkEntityRelationshipService instance.
    
    Args:
        entity_storage: Entity storage service
        search_service: Search service
        
    Returns:
        ChunkEntityRelationshipService instance
    """
    return ChunkEntityRelationshipService(entity_storage, search_service)
