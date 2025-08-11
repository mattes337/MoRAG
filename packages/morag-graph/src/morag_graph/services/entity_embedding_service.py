"""Entity embedding service for generating and managing entity embeddings."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models import Entity
from ..storage.neo4j_storage import Neo4jStorage
from morag_services.embedding import GeminiEmbeddingService

logger = logging.getLogger(__name__)


class EntityEmbeddingService:
    """Service for generating and managing entity embeddings."""
    
    def __init__(
        self,
        neo4j_storage: Neo4jStorage,
        embedding_service: GeminiEmbeddingService
    ):
        """Initialize entity embedding service.
        
        Args:
            neo4j_storage: Neo4j storage instance
            embedding_service: Gemini embedding service instance
        """
        self.neo4j_storage = neo4j_storage
        self.embedding_service = embedding_service
        self.logger = logger
    
    def _create_entity_text(self, entity: Dict[str, Any]) -> str:
        """Create text representation of entity for embedding.
        
        Args:
            entity: Entity data from Neo4j
            
        Returns:
            Text representation for embedding
        """
        name = entity.get('name', '')
        entity_type = entity.get('type', '')
        
        # Create base text
        text = f"{name} ({entity_type})"
        
        # Add metadata context if available
        metadata = entity.get('metadata', {})
        if isinstance(metadata, dict):
            domain = metadata.get('domain', '')
            if domain and domain != 'general':
                text += f" in {domain} domain"
            
            # Add fact component context if derived from fact
            fact_component = metadata.get('fact_component', '')
            if fact_component:
                text += f" as {fact_component}"
        
        return text
    
    async def generate_entity_embedding(self, entity_id: str) -> Optional[List[float]]:
        """Generate embedding for a single entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Get entity data
            query = """
            MATCH (e {id: $entity_id})
            RETURN e.id as id, e.name as name, e.type as type, e.metadata as metadata
            """
            
            results = await self.neo4j_storage._connection_ops._execute_query(
                query, {"entity_id": entity_id}
            )
            
            if not results:
                self.logger.warning(f"Entity not found: {entity_id}")
                return None
            
            entity = results[0]
            
            # Create text for embedding
            entity_text = self._create_entity_text(entity)
            
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(
                entity_text, task_type="retrieval_document"
            )
            
            self.logger.debug(f"Generated embedding for entity {entity_id}: {entity_text}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for entity {entity_id}: {e}")
            return None
    
    async def store_entity_embedding(
        self, 
        entity_id: str, 
        embedding: List[float]
    ) -> bool:
        """Store embedding for an entity in Neo4j.
        
        Args:
            entity_id: Entity ID
            embedding: Embedding vector
            
        Returns:
            True if successful
        """
        try:
            query = """
            MATCH (e {id: $entity_id})
            SET e.embedding_vector = $embedding,
                e.embedding_model = $model,
                e.embedding_dimensions = $dimensions,
                e.embedding_created_at = $created_at
            RETURN e.id as id
            """
            
            # Handle different embedding result types
            if hasattr(embedding, 'embedding'):
                # EmbeddingResult object
                embedding_vector = embedding.embedding
            elif isinstance(embedding, list):
                # Direct list of floats
                embedding_vector = embedding
            else:
                self.logger.error(f"Unexpected embedding type: {type(embedding)}")
                return False

            results = await self.neo4j_storage._connection_ops._execute_query(query, {
                "entity_id": entity_id,
                "embedding": embedding_vector,
                "model": "text-embedding-004",
                "dimensions": len(embedding_vector),
                "created_at": datetime.utcnow().isoformat()
            })
            
            if results:
                self.logger.debug(f"Stored embedding for entity {entity_id}")
                return True
            else:
                self.logger.warning(f"Entity not found when storing embedding: {entity_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to store embedding for entity {entity_id}: {e}")
            return False
    
    async def generate_and_store_entity_embedding(self, entity_id: str) -> bool:
        """Generate and store embedding for an entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            True if successful
        """
        embedding = await self.generate_entity_embedding(entity_id)
        if embedding:
            return await self.store_entity_embedding(entity_id, embedding)
        return False
    
    async def process_entities_batch(
        self, 
        entity_ids: List[str],
        batch_size: int = 10
    ) -> Dict[str, bool]:
        """Process multiple entities in batches.
        
        Args:
            entity_ids: List of entity IDs
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping entity_id to success status
        """
        results = {}
        
        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i:i + batch_size]
            
            self.logger.info(f"Processing entity batch {i//batch_size + 1}: {len(batch)} entities")
            
            # Process batch
            batch_tasks = [
                self.generate_and_store_entity_embedding(entity_id)
                for entity_id in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Store results
            for entity_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Exception processing entity {entity_id}: {result}")
                    results[entity_id] = False
                else:
                    results[entity_id] = result
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(entity_ids):
                await asyncio.sleep(0.1)
        
        return results
    
    async def get_entities_without_embeddings(self) -> List[str]:
        """Get list of entity IDs that don't have embeddings.
        
        Returns:
            List of entity IDs without embeddings
        """
        query = """
        MATCH (e)
        WHERE (e:SUBJECT OR e:OBJECT OR e:Entity) 
        AND e.embedding_vector IS NULL
        RETURN e.id as id
        """
        
        results = await self.neo4j_storage._connection_ops._execute_query(query)
        return [result['id'] for result in results]
    
    async def search_similar_entities(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for entities similar to query embedding.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar entities with similarity scores
        """
        try:
            # Use manual similarity calculation since GDS cosine function is not available
            # as a direct function in this Neo4j/GDS version
            self.logger.debug("Using manual similarity calculation for entity search")
            return await self._manual_similarity_search(
                query_embedding, limit, similarity_threshold
            )
                
        except Exception as e:
            self.logger.error(f"Entity similarity search failed: {e}")
            return []
    
    async def _manual_similarity_search(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Manual similarity search using Python calculations.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar entities with similarity scores
        """
        # Get all entities with embeddings
        query = """
        MATCH (e)
        WHERE (e:SUBJECT OR e:OBJECT OR e:Entity) 
        AND e.embedding_vector IS NOT NULL
        RETURN e.id as id, e.name as name, e.type as type, e.embedding_vector as embedding
        """
        
        results = await self.neo4j_storage._connection_ops._execute_query(query)
        
        # Calculate similarities
        similarities = []
        for entity in results:
            similarity = self._cosine_similarity(query_embedding, entity['embedding'])
            if similarity >= similarity_threshold:
                similarities.append({
                    'id': entity['id'],
                    'name': entity['name'],
                    'type': entity['type'],
                    'similarity': similarity
                })
        
        # Sort by similarity and limit
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about entity embeddings.
        
        Returns:
            Statistics dictionary
        """
        query = """
        MATCH (e)
        WHERE e:SUBJECT OR e:OBJECT OR e:Entity
        RETURN 
            labels(e)[0] as entity_type,
            count(*) as total_count,
            count(e.embedding_vector) as with_embeddings,
            avg(size(e.embedding_vector)) as avg_dimensions
        """
        
        results = await self.neo4j_storage._connection_ops._execute_query(query)
        
        stats = {
            'total_entities': 0,
            'entities_with_embeddings': 0,
            'entities_without_embeddings': 0,
            'by_type': {}
        }
        
        for result in results:
            entity_type = result['entity_type']
            total = result['total_count']
            with_embeddings = result['with_embeddings'] or 0
            without_embeddings = total - with_embeddings
            
            stats['total_entities'] += total
            stats['entities_with_embeddings'] += with_embeddings
            stats['entities_without_embeddings'] += without_embeddings
            
            stats['by_type'][entity_type] = {
                'total': total,
                'with_embeddings': with_embeddings,
                'without_embeddings': without_embeddings,
                'avg_dimensions': result['avg_dimensions']
            }
        
        return stats
