"""Intelligent entity-based retrieval service with recursive path following."""

import structlog
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from qdrant_client.models import Filter, FieldCondition, MatchValue

from morag_reasoning.llm import LLMClient
from morag_reasoning.intelligent_retrieval_models import (
    IntelligentRetrievalRequest, IntelligentRetrievalResponse,
    KeyFact, RetrievalIteration
)
from morag_reasoning.entity_identification import EntityIdentificationService
from morag_reasoning.recursive_path_follower import RecursivePathFollower
from morag_reasoning.fact_extraction import FactExtractionService
from morag_graph.storage.neo4j_storage import Neo4jStorage
from morag_graph.storage.qdrant_storage import QdrantStorage


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector storage."""
    id: str
    content: str
    document_id: str
    document_name: str
    score: float
    metadata: Dict[str, Any]


class IntelligentRetrievalService:
    """Service for intelligent entity-based retrieval with recursive path following."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        neo4j_storage: Neo4jStorage,
        qdrant_storage: QdrantStorage
    ):
        """Initialize the intelligent retrieval service.

        Args:
            llm_client: LLM client for various AI tasks
            neo4j_storage: Neo4j storage for graph operations
            qdrant_storage: Qdrant storage for vector operations
        """
        self.llm_client = llm_client
        self.neo4j_storage = neo4j_storage
        self.qdrant_storage = qdrant_storage
        self.logger = structlog.get_logger(__name__)

        # Initialize sub-services (will be updated with language in retrieve_intelligently)
        self.entity_service = EntityIdentificationService(
            llm_client=llm_client,
            graph_storage=neo4j_storage,
            min_confidence=0.2,
            max_entities=50
        )
        self.base_llm_client = llm_client
        self.base_neo4j_storage = neo4j_storage

        self.path_follower = RecursivePathFollower(
            llm_client=llm_client,
            graph_storage=neo4j_storage
        )

        self.fact_extractor = FactExtractionService(
            llm_client=llm_client
        )

    def _chunk_id_to_point_id(self, chunk_id: str) -> int:
        """Convert chunk ID to Qdrant point ID using the same logic as ingestion.

        Args:
            chunk_id: Neo4j chunk ID

        Returns:
            Qdrant point ID (integer)
        """
        # Use the same conversion logic as the ingestion coordinator
        return abs(hash(chunk_id)) % (2**31)
    
    async def retrieve_intelligently(
        self,
        request: IntelligentRetrievalRequest
    ) -> IntelligentRetrievalResponse:
        """Perform intelligent entity-based retrieval with recursive path following.
        
        Args:
            request: Intelligent retrieval request
            
        Returns:
            Intelligent retrieval response with key facts and sources
        """
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        llm_calls = 0
        
        self.logger.info(
            "Starting intelligent retrieval",
            query_id=query_id,
            query=request.query
        )
        
        try:
            # Initialize entity service with language parameter and increased limits
            if not self.entity_service or (hasattr(self.entity_service, 'language') and self.entity_service.language != request.language):
                self.entity_service = EntityIdentificationService(
                    llm_client=self.base_llm_client,
                    graph_storage=self.base_neo4j_storage,
                    min_confidence=0.2,  # Lower threshold for more entities
                    max_entities=request.max_entities_per_iteration * 3,  # Allow more initial entities
                    language=request.language
                )

            # Step 1: Identify entities from the query
            self.logger.info("Step 1: Identifying entities from query")
            identified_entities = await self.entity_service.identify_entities(request.query)
            llm_calls += 1
            
            initial_entity_names = [entity.name for entity in identified_entities]
            
            if not initial_entity_names:
                self.logger.warning("No entities identified from query")
                return self._create_empty_response(query_id, request, start_time, llm_calls)
            
            self.logger.info(
                "Entities identified",
                entities=initial_entity_names,
                count=len(initial_entity_names)
            )
            
            # Step 2: Recursive path following
            self.logger.info("Step 2: Following paths recursively")
            iterations = await self.path_follower.follow_paths_recursively(
                query=request.query,
                initial_entities=initial_entity_names,
                max_iterations=request.max_iterations,
                max_entities_per_iteration=request.max_entities_per_iteration
            )
            llm_calls += len(iterations)  # Each iteration involves an LLM call
            
            # Step 3: Collect all entities from all iterations
            all_entities: Set[str] = set(initial_entity_names)
            for iteration in iterations:
                all_entities.update(iteration.entities_explored)
                for path in iteration.paths_followed:
                    all_entities.update(path.path_entities)
            
            self.logger.info(
                "Path following completed",
                total_entities=len(all_entities),
                iterations=len(iterations)
            )
            
            # Step 4: Retrieve chunks for all identified entities
            self.logger.info("Step 3: Retrieving chunks for entities")
            all_chunks = await self._retrieve_chunks_for_entities(
                list(all_entities), request.query
            )
            
            # Update iteration chunk counts
            total_chunks = len(all_chunks)
            for iteration in iterations:
                iteration.chunks_retrieved = total_chunks // len(iterations) if iterations else 0
            
            self.logger.info(
                "Chunks retrieved",
                total_chunks=total_chunks,
                entities=len(all_entities)
            )
            
            # Step 5: Extract key facts from chunks
            self.logger.info("Step 4: Extracting key facts")
            chunk_dicts = [self._chunk_to_dict(chunk) for chunk in all_chunks]
            key_facts = await self.fact_extractor.extract_facts(
                query=request.query,
                chunks=chunk_dicts,
                max_facts=20,  # Could be made configurable
                min_confidence=request.min_relevance_threshold
            )
            llm_calls += 1
            
            self.logger.info(
                "Key facts extracted",
                total_facts=len(key_facts)
            )
            
            # Step 6: Calculate quality metrics
            confidence_score = self._calculate_confidence_score(key_facts)
            completeness_score = self._calculate_completeness_score(
                key_facts, len(all_entities), total_chunks
            )
            
            # Step 7: Build response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = IntelligentRetrievalResponse(
                query_id=query_id,
                query=request.query,
                key_facts=key_facts,
                total_iterations=len(iterations),
                iterations=iterations,
                initial_entities=initial_entity_names,
                total_entities_explored=len(all_entities),
                total_chunks_retrieved=total_chunks,
                confidence_score=confidence_score,
                completeness_score=completeness_score,
                processing_time_ms=processing_time,
                llm_calls_made=llm_calls,
                debug_info=self._create_debug_info(request, identified_entities) if request.include_debug_info else None
            )
            
            self.logger.info(
                "Intelligent retrieval completed",
                query_id=query_id,
                processing_time_ms=processing_time,
                key_facts=len(key_facts),
                llm_calls=llm_calls
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Intelligent retrieval failed",
                query_id=query_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _retrieve_chunks_for_entities(
        self,
        entities: List[str],
        query: str
    ) -> List[RetrievedChunk]:
        """Retrieve chunks for a list of entities.
        
        Args:
            entities: List of entity names
            query: Original query for context
            
        Returns:
            List of retrieved chunks
        """
        all_chunks = []
        
        # First, try to get chunks directly related to entities from Neo4j
        for entity_name in entities:
            try:
                # Find entity by name first
                entity_candidates = await self.neo4j_storage.search_entities(
                    entity_name, limit=1
                )
                
                if entity_candidates:
                    entity = entity_candidates[0]
                    # Get chunk IDs where this entity is mentioned
                    chunk_ids = await self.neo4j_storage.get_chunks_by_entity_id(entity.id)
                    
                    for chunk_id in chunk_ids:
                        try:
                            # First, get the actual chunk content from Neo4j
                            query = "MATCH (c:DocumentChunk {id: $chunk_id}) RETURN c.text as text, c.document_id as document_id"
                            neo4j_result = await self.neo4j_storage._execute_query(query, {"chunk_id": chunk_id})

                            if neo4j_result:
                                chunk_data = neo4j_result[0]
                                neo4j_content = chunk_data.get('text', '')
                                neo4j_document_id = chunk_data.get('document_id', '')

                                # Get metadata from Qdrant for additional information
                                qdrant_metadata = {}
                                document_name = ''
                                try:
                                    scroll_result = await self.qdrant_storage.client.scroll(
                                        collection_name=self.qdrant_storage.config.collection_name,
                                        scroll_filter=Filter(
                                            must=[
                                                FieldCondition(
                                                    key="chunk_id",
                                                    match=MatchValue(value=chunk_id)
                                                )
                                            ]
                                        ),
                                        limit=1,
                                        with_payload=True,
                                        with_vectors=False
                                    )
                                    points, _ = scroll_result
                                    if points:
                                        qdrant_metadata = points[0].payload
                                        document_name = qdrant_metadata.get('source_name', qdrant_metadata.get('file_name', ''))
                                except Exception as e:
                                    self.logger.warning(
                                        "Failed to retrieve metadata from Qdrant",
                                        chunk_id=chunk_id,
                                        error=str(e)
                                    )

                                # Use Neo4j content (which has the actual transcript) instead of Qdrant content
                                chunk = RetrievedChunk(
                                    id=chunk_id,
                                    content=neo4j_content,  # Use actual content from Neo4j
                                    document_id=neo4j_document_id or qdrant_metadata.get('document_id', ''),
                                    document_name=document_name,
                                    score=1.0,  # High score for entity-linked chunks
                                    metadata=qdrant_metadata
                                )
                                all_chunks.append(chunk)
                            else:
                                self.logger.warning(
                                    "Chunk not found in Neo4j",
                                    chunk_id=chunk_id
                                )
                        except Exception as e:
                            self.logger.warning(
                                "Failed to retrieve chunk content",
                                chunk_id=chunk_id,
                                error=str(e)
                            )
                            
            except Exception as e:
                self.logger.warning(
                    "Failed to retrieve chunks for entity",
                    entity=entity_name,
                    error=str(e)
                )
        
        # Also perform vector search with the original query
        try:
            # Note: This would require generating embeddings for the query
            # For now, we'll skip vector search and rely on entity-based retrieval
            # In a full implementation, you would:
            # 1. Generate embedding for the query using the same embedding service
            # 2. Use qdrant_storage.client.search() with the query vector
            # 3. Process the results similar to entity-based chunks
            
            self.logger.debug("Vector search skipped - relying on entity-based retrieval")
                
        except Exception as e:
            self.logger.warning(
                "Failed to perform vector search",
                query=query,
                error=str(e)
            )
        
        # Remove duplicates based on chunk ID
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _chunk_to_dict(self, chunk: RetrievedChunk) -> Dict[str, Any]:
        """Convert RetrievedChunk to dictionary format.
        
        Args:
            chunk: Retrieved chunk
            
        Returns:
            Dictionary representation
        """
        return {
            'id': chunk.id,
            'chunk_id': chunk.id,
            'content': chunk.content,
            'text': chunk.content,
            'document_id': chunk.document_id,
            'document_name': chunk.document_name,
            'score': chunk.score,
            'relevance_score': chunk.score,
            'metadata': chunk.metadata
        }
    
    def _calculate_confidence_score(self, key_facts: List[KeyFact]) -> float:
        """Calculate overall confidence score from key facts.
        
        Args:
            key_facts: List of extracted key facts
            
        Returns:
            Overall confidence score
        """
        if not key_facts:
            return 0.0
        
        return sum(fact.confidence for fact in key_facts) / len(key_facts)
    
    def _calculate_completeness_score(
        self,
        key_facts: List[KeyFact],
        entities_explored: int,
        chunks_retrieved: int
    ) -> float:
        """Calculate completeness score based on exploration coverage.
        
        Args:
            key_facts: List of extracted key facts
            entities_explored: Number of entities explored
            chunks_retrieved: Number of chunks retrieved
            
        Returns:
            Completeness score
        """
        # Simple heuristic based on facts extracted vs exploration effort
        if not key_facts or entities_explored == 0:
            return 0.0
        
        # More facts relative to exploration effort suggests better completeness
        fact_density = len(key_facts) / max(entities_explored, 1)
        chunk_utilization = len(key_facts) / max(chunks_retrieved, 1)
        
        # Combine metrics and normalize
        completeness = min((fact_density + chunk_utilization) / 2, 1.0)
        return completeness
    
    def _create_debug_info(
        self,
        request: IntelligentRetrievalRequest,
        identified_entities: List[Any]
    ) -> Dict[str, Any]:
        """Create debug information.
        
        Args:
            request: Original request
            identified_entities: Identified entities
            
        Returns:
            Debug information dictionary
        """
        return {
            'request_parameters': request.dict(),
            'identified_entities': [
                {
                    'name': entity.name,
                    'type': entity.entity_type,
                    'confidence': entity.confidence,
                    'graph_linked': entity.graph_entity_id is not None
                }
                for entity in identified_entities
            ]
        }
    
    def _create_empty_response(
        self,
        query_id: str,
        request: IntelligentRetrievalRequest,
        start_time: datetime,
        llm_calls: int
    ) -> IntelligentRetrievalResponse:
        """Create empty response when no entities are found.
        
        Args:
            query_id: Query identifier
            request: Original request
            start_time: Start time
            llm_calls: Number of LLM calls made
            
        Returns:
            Empty response
        """
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IntelligentRetrievalResponse(
            query_id=query_id,
            query=request.query,
            key_facts=[],
            total_iterations=0,
            iterations=[],
            initial_entities=[],
            total_entities_explored=0,
            total_chunks_retrieved=0,
            confidence_score=0.0,
            completeness_score=0.0,
            processing_time_ms=processing_time,
            llm_calls_made=llm_calls
        )
