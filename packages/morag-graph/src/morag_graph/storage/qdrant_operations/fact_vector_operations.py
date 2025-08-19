"""Qdrant vector operations for facts."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import structlog

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, CollectionInfo
from qdrant_client.http.exceptions import UnexpectedResponse

from ...models.fact import Fact
from morag_services.embedding import GeminiEmbeddingService


class FactVectorOperations:
    """Qdrant vector operations for fact storage and retrieval."""
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "morag_facts",
        embedding_service: Optional[GeminiEmbeddingService] = None
    ):
        """Initialize fact vector operations.
        
        Args:
            client: Qdrant client instance
            collection_name: Name of the collection for facts
            embedding_service: Service for generating embeddings
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.logger = structlog.get_logger(__name__)
        
        # Default embedding dimension (text-embedding-004)
        self.embedding_dimension = 768
    
    async def ensure_collection_exists(self) -> bool:
        """Ensure the facts collection exists in Qdrant.
        
        Returns:
            True if collection exists or was created successfully
        """
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            existing_names = [col.name for col in collections.collections]
            
            if self.collection_name in existing_names:
                self.logger.debug(
                    "Facts collection already exists",
                    collection=self.collection_name
                )
                return True
            
            # Create collection
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            
            self.logger.info(
                "Created facts collection",
                collection=self.collection_name,
                dimension=self.embedding_dimension
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to ensure facts collection exists",
                collection=self.collection_name,
                error=str(e)
            )
            return False
    
    async def store_fact_vector(self, fact: Fact, embedding: Optional[List[float]] = None) -> str:
        """Store a fact as a vector in Qdrant.
        
        Args:
            fact: Fact to store
            embedding: Pre-computed embedding (optional)
            
        Returns:
            Point ID of stored fact
        """
        # Generate embedding if not provided
        if embedding is None and self.embedding_service:
            fact_text = fact.get_search_text()
            embedding_result = await self.embedding_service.generate_embedding(
                fact_text,
                task_type="retrieval_document"
            )
            embedding = embedding_result.embedding
        
        if embedding is None:
            raise ValueError("No embedding provided and no embedding service available")
        
        # Ensure collection exists
        await self.ensure_collection_exists()
        
        # Create point ID
        point_id = str(uuid.uuid4())
        
        # Prepare payload with fact metadata
        payload = {
            "fact_id": fact.id,
            "fact_text": fact.fact_text,
            "primary_entities": fact.structured_metadata.primary_entities if fact.structured_metadata else [],
            "relationships": fact.structured_metadata.relationships if fact.structured_metadata else [],
            "domain_concepts": fact.structured_metadata.domain_concepts if fact.structured_metadata else [],
            "fact_type": fact.fact_type,
            "domain": fact.domain,
            "confidence": fact.extraction_confidence,
            "language": fact.language,
            "keywords": fact.keywords,
            "source_chunk_id": fact.source_chunk_id,
            "source_document_id": fact.source_document_id,
            "created_at": fact.created_at.isoformat(),
            # Source metadata for citation
            "source_file_path": fact.source_file_path,
            "source_file_name": fact.source_file_name,
            "page_number": fact.page_number,
            "chapter_title": fact.chapter_title,
            "chapter_index": fact.chapter_index,
            "timestamp_start": fact.timestamp_start,
            "timestamp_end": fact.timestamp_end,
            "topic_header": fact.topic_header,
            "speaker_label": fact.speaker_label,
            # Full text for search
            "search_text": fact.get_search_text(),
            "citation": fact.get_citation(),
            "machine_readable_source": fact.get_machine_readable_source()
        }
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        
        try:
            # Store point
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=[point]
            )
            
            self.logger.debug(
                "Fact vector stored successfully",
                fact_id=fact.id,
                point_id=point_id,
                fact_text=fact.fact_text[:50] + "..." if len(fact.fact_text) > 50 else fact.fact_text
            )
            
            return point_id
            
        except Exception as e:
            self.logger.error(
                "Failed to store fact vector",
                fact_id=fact.id,
                error=str(e)
            )
            raise
    
    async def store_facts_batch(self, facts: List[Fact], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """Store multiple facts as vectors in batch.
        
        Args:
            facts: List of facts to store
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List of point IDs
        """
        if not facts:
            return []
        
        # Generate embeddings if not provided
        if embeddings is None and self.embedding_service:
            fact_texts = [fact.get_search_text() for fact in facts]
            embedding_results = await self.embedding_service.generate_embeddings_batch(
                fact_texts,
                task_type="retrieval_document"
            )
            embeddings = [result.embedding for result in embedding_results]
        
        if embeddings is None:
            raise ValueError("No embeddings provided and no embedding service available")
        
        if len(facts) != len(embeddings):
            raise ValueError("Number of facts must match number of embeddings")
        
        # Ensure collection exists
        await self.ensure_collection_exists()
        
        # Prepare points
        points = []
        point_ids = []
        
        for fact, embedding in zip(facts, embeddings):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Prepare payload
            payload = {
                "fact_id": fact.id,
                "fact_text": fact.fact_text,
                "primary_entities": fact.structured_metadata.primary_entities if fact.structured_metadata else [],
                "relationships": fact.structured_metadata.relationships if fact.structured_metadata else [],
                "domain_concepts": fact.structured_metadata.domain_concepts if fact.structured_metadata else [],
                "fact_type": fact.fact_type,
                "domain": fact.domain,
                "confidence": fact.extraction_confidence,
                "language": fact.language,
                "keywords": fact.keywords,
                "source_chunk_id": fact.source_chunk_id,
                "source_document_id": fact.source_document_id,
                "created_at": fact.created_at.isoformat(),
                # Source metadata
                "source_file_path": fact.source_file_path,
                "source_file_name": fact.source_file_name,
                "page_number": fact.page_number,
                "chapter_title": fact.chapter_title,
                "chapter_index": fact.chapter_index,
                "timestamp_start": fact.timestamp_start,
                "timestamp_end": fact.timestamp_end,
                "topic_header": fact.topic_header,
                "speaker_label": fact.speaker_label,
                # Search and citation
                "search_text": fact.get_search_text(),
                "citation": fact.get_citation(),
                "machine_readable_source": fact.get_machine_readable_source()
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        try:
            # Store points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.collection_name,
                    points=batch
                )
            
            self.logger.info(
                "Fact vectors stored successfully",
                num_facts=len(facts),
                collection=self.collection_name
            )
            
            return point_ids
            
        except Exception as e:
            self.logger.error(
                "Failed to store fact vectors batch",
                num_facts=len(facts),
                error=str(e)
            )
            raise
    
    async def search_facts(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for facts using vector similarity.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional filters for metadata
            
        Returns:
            List of search results with fact data and scores
        """
        try:
            # Perform vector search
            search_result = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filters
            )
            
            # Convert results to fact data
            results = []
            for scored_point in search_result:
                result = {
                    "score": scored_point.score,
                    "fact_id": scored_point.payload.get("fact_id"),
                    "subject": scored_point.payload.get("subject"),
                    "object": scored_point.payload.get("object"),
                    "approach": scored_point.payload.get("approach"),
                    "solution": scored_point.payload.get("solution"),
                    "remarks": scored_point.payload.get("remarks"),
                    "fact_type": scored_point.payload.get("fact_type"),
                    "domain": scored_point.payload.get("domain"),
                    "confidence": scored_point.payload.get("confidence"),
                    "citation": scored_point.payload.get("citation"),
                    "machine_readable_source": scored_point.payload.get("machine_readable_source"),
                    "source_metadata": {
                        "source_file_name": scored_point.payload.get("source_file_name"),
                        "page_number": scored_point.payload.get("page_number"),
                        "chapter_title": scored_point.payload.get("chapter_title"),
                        "timestamp_start": scored_point.payload.get("timestamp_start"),
                        "timestamp_end": scored_point.payload.get("timestamp_end"),
                        "topic_header": scored_point.payload.get("topic_header"),
                        "speaker_label": scored_point.payload.get("speaker_label")
                    }
                }
                results.append(result)
            
            self.logger.debug(
                "Fact vector search completed",
                results_count=len(results),
                score_threshold=score_threshold
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                "Fact vector search failed",
                error=str(e)
            )
            raise
