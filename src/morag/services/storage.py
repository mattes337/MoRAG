from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
from typing import List, Dict, Any, Optional, Union
import structlog
import asyncio
from datetime import datetime
import uuid

from morag.core.config import settings
from morag.core.exceptions import StorageError

logger = structlog.get_logger()

class QdrantService:
    """Service for managing Qdrant vector database operations."""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collection_name = settings.qdrant_collection_name
        
    async def connect(self) -> None:
        """Initialize connection to Qdrant."""
        try:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                timeout=30
            )
            
            # Test connection
            collections = await asyncio.to_thread(self.client.get_collections)
            logger.info("Connected to Qdrant", collections_count=len(collections.collections))
            
        except Exception as e:
            logger.error("Failed to connect to Qdrant", error=str(e))
            raise StorageError(f"Failed to connect to Qdrant: {str(e)}")
    
    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        if self.client:
            await asyncio.to_thread(self.client.close)
            self.client = None
            logger.info("Disconnected from Qdrant")
    
    async def create_collection(self, vector_size: int = 768, force_recreate: bool = False) -> None:
        """Create or recreate the main collection."""
        if not self.client:
            await self.connect()
        
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if force_recreate:
                    logger.info("Deleting existing collection", collection=self.collection_name)
                    await asyncio.to_thread(
                        self.client.delete_collection,
                        collection_name=self.collection_name
                    )
                else:
                    logger.info("Collection already exists", collection=self.collection_name)
                    return
            
            # Create collection
            logger.info("Creating collection", collection=self.collection_name, vector_size=vector_size)
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info("Collection created successfully", collection=self.collection_name)
            
        except Exception as e:
            logger.error("Failed to create collection", error=str(e))
            raise StorageError(f"Failed to create collection: {str(e)}")
    
    async def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Store text chunks with their embeddings and metadata."""
        if not self.client:
            await self.connect()
        
        if len(chunks) != len(embeddings):
            raise StorageError("Number of chunks must match number of embeddings")
        
        try:
            points = []
            point_ids = []
            
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Prepare metadata
                payload = {
                    "text": chunk.get("text", ""),
                    "summary": chunk.get("summary", ""),
                    "source": chunk.get("source", ""),
                    "source_type": chunk.get("source_type", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": chunk.get("metadata", {})
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Store points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info("Stored chunks successfully", count=len(chunks))
            return point_ids
            
        except Exception as e:
            logger.error("Failed to store chunks", error=str(e))
            raise StorageError(f"Failed to store chunks: {str(e)}")
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        if not self.client:
            await self.connect()
        
        try:
            # Build filter if provided
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ))
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search
            results = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "summary": result.payload.get("summary", ""),
                    "source": result.payload.get("source", ""),
                    "source_type": result.payload.get("source_type", ""),
                    "metadata": result.payload.get("metadata", {})
                })
            
            logger.info("Search completed", results_count=len(formatted_results))
            return formatted_results
            
        except Exception as e:
            logger.error("Search failed", error=str(e))
            raise StorageError(f"Search failed: {str(e)}")
    
    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        if not self.client:
            await self.connect()
        
        try:
            # Search for points with the given source
            filter_condition = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )]
            )
            
            # Get points to delete
            search_result = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=10000,  # Adjust based on expected chunk count
                with_payload=False
            )
            
            point_ids = [point.id for point in search_result[0]]
            
            if point_ids:
                # Delete points
                await asyncio.to_thread(
                    self.client.delete,
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
            
            logger.info("Deleted chunks by source", source=source, count=len(point_ids))
            return len(point_ids)
            
        except Exception as e:
            logger.error("Failed to delete by source", error=str(e))
            raise StorageError(f"Failed to delete by source: {str(e)}")
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.client:
            await self.connect()
        
        try:
            info = await asyncio.to_thread(
                self.client.get_collection,
                collection_name=self.collection_name
            )
            
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "optimizer_status": info.optimizer_status.status.value,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value
                }
            }
            
        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            raise StorageError(f"Failed to get collection info: {str(e)}")

# Global instance
qdrant_service = QdrantService()
