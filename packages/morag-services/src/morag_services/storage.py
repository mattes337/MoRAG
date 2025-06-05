"""Vector storage services for MoRAG using Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
from typing import List, Dict, Any, Optional, Union
import structlog
import asyncio
from datetime import datetime, timezone
import uuid

from morag_core.interfaces.storage import VectorStorage as BaseVectorStorage
from morag_core.exceptions import StorageError

logger = structlog.get_logger(__name__)


class QdrantVectorStorage(BaseVectorStorage):
    """Qdrant vector storage implementation."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: str = "morag_vectors"
    ):
        """Initialize Qdrant storage.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            api_key: API key for authentication
            collection_name: Default collection name
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None
        
    async def connect(self) -> None:
        """Initialize connection to Qdrant."""
        try:
            # Check if host is a URL (starts with http:// or https://)
            if self.host.startswith(('http://', 'https://')):
                from urllib.parse import urlparse
                parsed = urlparse(self.host)
                hostname = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == 'https' else 6333)
                use_https = parsed.scheme == 'https'

                # Use host/port connection which works better with HTTPS
                self.client = QdrantClient(
                    host=hostname,
                    port=port,
                    https=use_https,
                    api_key=self.api_key,
                    timeout=30
                )

                logger.info("Connecting to Qdrant via URL",
                           url=self.host,
                           hostname=hostname,
                           port=port,
                           https=use_https)
            else:
                # Use host/port connection for local instances
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    timeout=30
                )

                logger.info("Connecting to Qdrant via host/port",
                           host=self.host,
                           port=self.port)

            # Test connection
            collections = await asyncio.to_thread(self.client.get_collections)
            logger.info("Connected to Qdrant successfully",
                       collections_count=len(collections.collections),
                       collections=[col.name for col in collections.collections])

        except Exception as e:
            logger.error("Failed to connect to Qdrant",
                        host=self.host,
                        port=self.port,
                        error=str(e))
            raise StorageError(f"Failed to connect to Qdrant: {str(e)}")
    
    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        if self.client:
            await asyncio.to_thread(self.client.close)
            self.client = None
            logger.info("Disconnected from Qdrant")
    
    async def create_collection(
        self, 
        collection_name: str, 
        vector_size: int = 768, 
        force_recreate: bool = False
    ) -> None:
        """Create or recreate a collection."""
        if not self.client:
            await self.connect()
        
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_exists = any(
                col.name == collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if force_recreate:
                    logger.info("Deleting existing collection", collection=collection_name)
                    await asyncio.to_thread(
                        self.client.delete_collection,
                        collection_name=collection_name
                    )
                else:
                    logger.info("Collection already exists", collection=collection_name)
                    return
            
            # Create collection
            logger.info("Creating collection", collection=collection_name, vector_size=vector_size)
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info("Collection created successfully", collection=collection_name)
            
        except Exception as e:
            logger.error("Failed to create collection", error=str(e))
            raise StorageError(f"Failed to create collection: {str(e)}")
    
    async def store_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Store vectors with metadata."""
        if not self.client:
            await self.connect()
        
        if len(vectors) != len(metadata):
            raise StorageError("Number of vectors must match number of metadata entries")
        
        target_collection = collection_name or self.collection_name
        
        try:
            # Ensure collection exists
            if vectors:
                await self.create_collection(target_collection, len(vectors[0]))
            
            points = []
            point_ids = []
            
            for vector, meta in zip(vectors, metadata):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Prepare payload
                payload = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **meta
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            # Store points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=target_collection,
                    points=batch
                )
            
            logger.info("Stored vectors successfully", 
                       count=len(vectors), 
                       collection=target_collection)
            return point_ids
            
        except Exception as e:
            logger.error("Failed to store vectors", error=str(e))
            raise StorageError(f"Failed to store vectors: {str(e)}")
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self.client:
            await self.connect()
        
        target_collection = collection_name or self.collection_name
        
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
                collection_name=target_collection,
                query_vector=query_vector,
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
                    "metadata": result.payload
                })
            
            logger.info("Search completed", 
                       results_count=len(formatted_results),
                       collection=target_collection)
            return formatted_results
            
        except Exception as e:
            logger.error("Search failed", error=str(e))
            raise StorageError(f"Search failed: {str(e)}")
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        collection_name: Optional[str] = None
    ) -> int:
        """Delete vectors by IDs."""
        if not self.client:
            await self.connect()
        
        target_collection = collection_name or self.collection_name
        
        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=target_collection,
                points_selector=vector_ids
            )
            
            logger.info("Deleted vectors", 
                       count=len(vector_ids),
                       collection=target_collection)
            return len(vector_ids)
            
        except Exception as e:
            logger.error("Failed to delete vectors", error=str(e))
            raise StorageError(f"Failed to delete vectors: {str(e)}")
    
    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get collection information."""
        if not self.client:
            await self.connect()
        
        target_collection = collection_name or self.collection_name
        
        try:
            info = await asyncio.to_thread(
                self.client.get_collection,
                collection_name=target_collection
            )
            
            return {
                "name": target_collection,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value
                }
            }
            
        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            raise StorageError(f"Failed to get collection info: {str(e)}")


# Convenience class for backward compatibility
class QdrantService(QdrantVectorStorage):
    """Legacy QdrantService for backward compatibility."""
    
    def __init__(self):
        # Initialize with default settings - these would come from config
        super().__init__(
            host="localhost",
            port=6333,
            api_key=None,
            collection_name="morag_vectors"
        )
