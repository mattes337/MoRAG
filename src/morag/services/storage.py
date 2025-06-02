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
                    "created_at": datetime.now(timezone.utc).isoformat(),
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
    
    async def create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_size: int = 768
    ) -> None:
        """Create collection if it doesn't exist."""
        if not self.client:
            await self.connect()

        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_exists = any(
                col.name == collection_name
                for col in collections.collections
            )

            if not collection_exists:
                logger.info("Creating new collection", collection=collection_name, vector_size=vector_size)
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully", collection=collection_name)
            else:
                logger.debug("Collection already exists", collection=collection_name)

        except Exception as e:
            logger.error("Failed to create collection", error=str(e))
            raise StorageError(f"Failed to create collection: {str(e)}")

    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with their info."""
        if not self.client:
            await self.connect()

        try:
            collections = await asyncio.to_thread(self.client.get_collections)

            result = []
            for collection in collections.collections:
                try:
                    info = await asyncio.to_thread(
                        self.client.get_collection,
                        collection_name=collection.name
                    )

                    result.append({
                        "name": collection.name,
                        "vectors_count": info.vectors_count,
                        "indexed_vectors_count": info.indexed_vectors_count,
                        "points_count": info.points_count,
                        "status": info.status.value,
                        "config": {
                            "vector_size": info.config.params.vectors.size,
                            "distance": info.config.params.vectors.distance.value
                        }
                    })
                except Exception as e:
                    logger.warning("Failed to get info for collection",
                                  collection=collection.name, error=str(e))
                    result.append({
                        "name": collection.name,
                        "error": str(e)
                    })

            return result

        except Exception as e:
            logger.error("Failed to list collections", error=str(e))
            raise StorageError(f"Failed to list collections: {str(e)}")

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

    async def store_embedding(
        self,
        embedding: Union[List[float], 'EmbeddingResult'],
        text: str,
        metadata: Dict[str, Any],
        collection_name: Optional[str] = None,
        point_id: Optional[str] = None
    ) -> str:
        """Store a single embedding with text and metadata."""
        if not self.client:
            await self.connect()

        # Handle both raw embedding list and EmbeddingResult object
        if hasattr(embedding, 'embedding'):
            embedding_vector = embedding.embedding
        else:
            embedding_vector = embedding

        try:
            # Use provided collection or default
            target_collection = collection_name or self.collection_name

            # Ensure collection exists
            await self.create_collection_if_not_exists(target_collection, len(embedding_vector))

            # Generate point ID if not provided
            if not point_id:
                point_id = str(uuid.uuid4())

            # Prepare payload
            payload = {
                "text": text,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload=payload
            )

            # Store point
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=target_collection,
                points=[point]
            )

            logger.info("Stored embedding successfully",
                       point_id=point_id,
                       collection=target_collection,
                       text_length=len(text))
            return point_id

        except Exception as e:
            logger.error("Failed to store embedding", error=str(e))
            raise StorageError(f"Failed to store embedding: {str(e)}")

    async def store_chunk(
        self,
        chunk_id: str,
        text: str,
        summary: str,
        embedding: Union[List[float], 'EmbeddingResult'],
        metadata: Dict[str, Any],
        collection_name: Optional[str] = None
    ) -> str:
        """Store a text chunk with embedding and metadata."""
        if not self.client:
            await self.connect()

        # Handle both raw embedding list and EmbeddingResult object
        if hasattr(embedding, 'embedding'):
            embedding_vector = embedding.embedding
        else:
            embedding_vector = embedding

        try:
            # Use provided collection or default
            target_collection = collection_name or self.collection_name

            # Ensure collection exists
            await self.create_collection_if_not_exists(target_collection, len(embedding_vector))

            # Prepare payload
            payload = {
                "text": text,
                "summary": summary,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata
            }

            # Create point
            point = PointStruct(
                id=chunk_id,
                vector=embedding_vector,
                payload=payload
            )

            # Store point
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=target_collection,
                points=[point]
            )

            logger.info("Stored chunk successfully",
                       chunk_id=chunk_id,
                       collection=target_collection,
                       text_length=len(text))
            return chunk_id

        except Exception as e:
            logger.error("Failed to store chunk", error=str(e))
            raise StorageError(f"Failed to store chunk: {str(e)}")

    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 10,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search by metadata filters only."""
        if not self.client:
            await self.connect()

        target_collection = collection_name or self.collection_name

        try:
            # Build filter conditions
            conditions = []
            for key, value in filters.items():
                # Handle nested metadata fields
                if key.startswith("metadata."):
                    field_key = key
                else:
                    field_key = f"metadata.{key}"

                conditions.append(FieldCondition(
                    key=field_key,
                    match=MatchValue(value=value)
                ))

            search_filter = Filter(must=conditions) if conditions else None

            # Use scroll to get points without vector search
            results, _ = await asyncio.to_thread(
                self.client.scroll,
                collection_name=target_collection,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "text": result.payload.get("text", ""),
                    "summary": result.payload.get("summary", ""),
                    "source": result.payload.get("source", ""),
                    "source_type": result.payload.get("source_type", ""),
                    "metadata": result.payload.get("metadata", {})
                })

            logger.info("Metadata search completed",
                       results_count=len(formatted_results),
                       collection=target_collection)
            return formatted_results

        except Exception as e:
            logger.error("Metadata search failed", error=str(e))
            raise StorageError(f"Metadata search failed: {str(e)}")

    async def batch_store_embeddings(
        self,
        embeddings_data: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Store multiple embeddings in batches."""
        if not self.client:
            await self.connect()

        target_collection = collection_name or self.collection_name

        try:
            # Ensure collection exists (use first embedding to determine vector size)
            if embeddings_data:
                first_embedding = embeddings_data[0]["embedding"]
                if hasattr(first_embedding, 'embedding'):
                    vector_size = len(first_embedding.embedding)
                else:
                    vector_size = len(first_embedding)

                await self.create_collection_if_not_exists(target_collection, vector_size)

            points = []
            point_ids = []

            for data in embeddings_data:
                point_id = data.get("point_id", str(uuid.uuid4()))
                point_ids.append(point_id)

                # Handle embedding format
                embedding = data["embedding"]
                if hasattr(embedding, 'embedding'):
                    embedding_vector = embedding.embedding
                else:
                    embedding_vector = embedding

                # Prepare payload
                payload = {
                    "text": data.get("text", ""),
                    "summary": data.get("summary", ""),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": data.get("metadata", {})
                }

                points.append(PointStruct(
                    id=point_id,
                    vector=embedding_vector,
                    payload=payload
                ))

            # Store points in batches
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=target_collection,
                    points=batch
                )

                logger.debug("Stored batch",
                           batch_num=i//batch_size + 1,
                           batch_size=len(batch),
                           collection=target_collection)

            logger.info("Batch store completed",
                       total_embeddings=len(embeddings_data),
                       collection=target_collection)
            return point_ids

        except Exception as e:
            logger.error("Batch store failed", error=str(e))
            raise StorageError(f"Batch store failed: {str(e)}")

# Global instance
qdrant_service = QdrantService()
