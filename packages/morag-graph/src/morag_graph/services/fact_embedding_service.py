"""Fact embedding service for generating and managing fact embeddings."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..storage.neo4j_storage import Neo4jStorage
from morag_services.embedding import GeminiEmbeddingService

logger = logging.getLogger(__name__)


class FactEmbeddingService:
    """Service for generating and managing fact embeddings."""
    
    def __init__(
        self,
        neo4j_storage: Neo4jStorage,
        embedding_service: GeminiEmbeddingService
    ):
        """Initialize fact embedding service.
        
        Args:
            neo4j_storage: Neo4j storage instance
            embedding_service: Gemini embedding service instance
        """
        self.neo4j_storage = neo4j_storage
        self.embedding_service = embedding_service
        self.logger = logger
    
    def _create_fact_text(self, fact: Dict[str, Any]) -> str:
        """Create text representation of fact for embedding.
        
        Args:
            fact: Fact data from Neo4j
            
        Returns:
            Text representation for embedding
        """
        # Get core fact components
        subject = fact.get('subject', '')
        approach = fact.get('approach', '')
        object_text = fact.get('object', '')
        solution = fact.get('solution', '')
        
        # Create structured fact text
        text_parts = []
        
        if subject:
            text_parts.append(f"Subject: {subject}")
        
        if approach:
            text_parts.append(f"Approach: {approach}")
        
        if object_text:
            text_parts.append(f"Object: {object_text}")
        
        if solution:
            text_parts.append(f"Solution: {solution}")
        
        # Add keywords if available
        keywords = fact.get('keywords', '')
        if keywords:
            text_parts.append(f"Keywords: {keywords}")
        
        # Add domain context
        domain = fact.get('domain', '')
        if domain and domain != 'general':
            text_parts.append(f"Domain: {domain}")
        
        # Join all parts
        fact_text = ". ".join(text_parts)
        
        # Fallback to any available text field if structured approach fails
        if not fact_text.strip():
            for field in ['fact_text', 'text', 'description']:
                if fact.get(field):
                    fact_text = fact[field]
                    break
        
        return fact_text
    
    async def generate_fact_embedding(self, fact_id: str) -> Optional[List[float]]:
        """Generate embedding for a single fact.
        
        Args:
            fact_id: Fact ID
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Get fact data
            query = """
            MATCH (f:Fact {id: $fact_id})
            RETURN f.id as id, f.subject as subject, f.approach as approach, 
                   f.object as object, f.solution as solution, f.keywords as keywords,
                   f.domain as domain, f.fact_text as fact_text, f.text as text,
                   f.description as description
            """
            
            results = await self.neo4j_storage._connection_ops._execute_query(
                query, {"fact_id": fact_id}
            )
            
            if not results:
                self.logger.warning(f"Fact not found: {fact_id}")
                return None
            
            fact = results[0]
            
            # Create text for embedding
            fact_text = self._create_fact_text(fact)
            
            if not fact_text.strip():
                self.logger.warning(f"No text content found for fact {fact_id}")
                return None
            
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(
                fact_text, task_type="retrieval_document"
            )
            
            self.logger.debug(f"Generated embedding for fact {fact_id}: {fact_text[:100]}...")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for fact {fact_id}: {e}")
            return None
    
    async def store_fact_embedding(
        self, 
        fact_id: str, 
        embedding: List[float]
    ) -> bool:
        """Store embedding for a fact in Neo4j.
        
        Args:
            fact_id: Fact ID
            embedding: Embedding vector
            
        Returns:
            True if successful
        """
        try:
            query = """
            MATCH (f:Fact {id: $fact_id})
            SET f.embedding_vector = $embedding,
                f.embedding_model = $model,
                f.embedding_dimensions = $dimensions,
                f.embedding_created_at = $created_at
            RETURN f.id as id
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
                "fact_id": fact_id,
                "embedding": embedding_vector,
                "model": "text-embedding-004",
                "dimensions": len(embedding_vector),
                "created_at": datetime.utcnow().isoformat()
            })
            
            if results:
                self.logger.debug(f"Stored embedding for fact {fact_id}")
                return True
            else:
                self.logger.warning(f"Fact not found when storing embedding: {fact_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to store embedding for fact {fact_id}: {e}")
            return False
    
    async def generate_and_store_fact_embedding(self, fact_id: str) -> bool:
        """Generate and store embedding for a fact.
        
        Args:
            fact_id: Fact ID
            
        Returns:
            True if successful
        """
        embedding = await self.generate_fact_embedding(fact_id)
        if embedding:
            return await self.store_fact_embedding(fact_id, embedding)
        return False
    
    async def process_facts_batch(
        self, 
        fact_ids: List[str],
        batch_size: int = 10
    ) -> Dict[str, bool]:
        """Process multiple facts in batches.
        
        Args:
            fact_ids: List of fact IDs
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping fact_id to success status
        """
        results = {}
        
        for i in range(0, len(fact_ids), batch_size):
            batch = fact_ids[i:i + batch_size]
            
            self.logger.info(f"Processing fact batch {i//batch_size + 1}: {len(batch)} facts")
            
            # Process batch
            batch_tasks = [
                self.generate_and_store_fact_embedding(fact_id)
                for fact_id in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Store results
            for fact_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Exception processing fact {fact_id}: {result}")
                    results[fact_id] = False
                else:
                    results[fact_id] = result
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(fact_ids):
                await asyncio.sleep(0.1)
        
        return results
    
    async def get_facts_without_embeddings(self) -> List[str]:
        """Get list of fact IDs that don't have embeddings.
        
        Returns:
            List of fact IDs without embeddings
        """
        query = """
        MATCH (f:Fact)
        WHERE f.embedding_vector IS NULL
        RETURN f.id as id
        """
        
        results = await self.neo4j_storage._connection_ops._execute_query(query)
        return [result['id'] for result in results]
    
    async def search_similar_facts(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for facts similar to query embedding.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar facts with similarity scores
        """
        try:
            # Try using Neo4j's vector similarity (if available)
            query = """
            MATCH (f:Fact)
            WHERE f.embedding_vector IS NOT NULL
            WITH f, gds.similarity.cosine(f.embedding_vector, $query_embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN f.id as id, f.subject as subject, f.approach as approach,
                   f.object as object, f.solution as solution, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            try:
                results = await self.neo4j_storage._connection_ops._execute_query(query, {
                    "query_embedding": query_embedding,
                    "threshold": similarity_threshold,
                    "limit": limit
                })
                
                return results
                
            except Exception as e:
                # Fallback to manual similarity calculation
                self.logger.warning(f"GDS similarity failed, using manual calculation: {e}")
                return await self._manual_similarity_search(
                    query_embedding, limit, similarity_threshold
                )
                
        except Exception as e:
            self.logger.error(f"Fact similarity search failed: {e}")
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
            List of similar facts with similarity scores
        """
        # Get all facts with embeddings
        query = """
        MATCH (f:Fact)
        WHERE f.embedding_vector IS NOT NULL
        RETURN f.id as id, f.subject as subject, f.approach as approach,
               f.object as object, f.solution as solution, f.embedding_vector as embedding
        """
        
        results = await self.neo4j_storage._connection_ops._execute_query(query)
        
        # Calculate similarities
        similarities = []
        for fact in results:
            similarity = self._cosine_similarity(query_embedding, fact['embedding'])
            if similarity >= similarity_threshold:
                similarities.append({
                    'id': fact['id'],
                    'subject': fact['subject'],
                    'approach': fact['approach'],
                    'object': fact['object'],
                    'solution': fact['solution'],
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
        """Get statistics about fact embeddings.
        
        Returns:
            Statistics dictionary
        """
        query = """
        MATCH (f:Fact)
        RETURN 
            count(*) as total_facts,
            count(f.embedding_vector) as facts_with_embeddings,
            avg(size(f.embedding_vector)) as avg_dimensions
        """
        
        results = await self.neo4j_storage._connection_ops._execute_query(query)
        
        if results:
            result = results[0]
            total = result['total_facts']
            with_embeddings = result['facts_with_embeddings'] or 0
            
            return {
                'total_facts': total,
                'facts_with_embeddings': with_embeddings,
                'facts_without_embeddings': total - with_embeddings,
                'avg_dimensions': result['avg_dimensions']
            }
        
        return {
            'total_facts': 0,
            'facts_with_embeddings': 0,
            'facts_without_embeddings': 0,
            'avg_dimensions': None
        }
