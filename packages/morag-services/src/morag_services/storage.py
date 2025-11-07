"""Vector storage services for MoRAG using Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
from typing import List, Dict, Any, Optional, Union, Callable, Awaitable
import structlog
import asyncio
from datetime import datetime, timezone
import uuid
import hashlib

from morag_core.interfaces.storage import VectorStorage as BaseVectorStorage
from morag_core.exceptions import StorageError

logger = structlog.get_logger(__name__)


class EmbeddingCache:
    """LRU cache for embeddings with content hashing."""

    def __init__(self, max_size: int = 10000):
        """Initialize the embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache (default: 10000)
        """
        self.cache: Dict[str, List[float]] = {}
        self.access_order: List[str] = []
        self.max_size = max_size
        logger.info("Initialized EmbeddingCache", max_size=max_size)

    def get_key(self, text: str) -> str:
        """Generate cache key from text content using SHA256.

        Args:
            text: Input text to hash

        Returns:
            16-character hash key
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def get_or_compute(
        self,
        text: str,
        compute_func: Callable[[str], Awaitable[List[float]]]
    ) -> List[float]:
        """Get embedding from cache or compute and cache it.

        Args:
            text: Text to get embedding for
            compute_func: Async function to compute embedding if not cached

        Returns:
            Embedding vector
        """
        key = self.get_key(text)

        # Check if embedding is in cache
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            logger.debug("Cache hit for embedding", key=key)
            return self.cache[key]

        # Compute embedding using provided function
        logger.debug("Cache miss, computing embedding", key=key)
        embedding = await compute_func(text)

        # Add to cache with LRU eviction
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
            logger.debug("LRU eviction", evicted_key=oldest)

        # Store in cache
        self.cache[key] = embedding
        self.access_order.append(key)

        logger.debug("Cached new embedding", key=key, cache_size=len(self.cache))
        return embedding

    def clear(self) -> None:
        """Clear all cached embeddings."""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cleared embedding cache", cleared_count=cleared_count)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
            "access_order_length": len(self.access_order)
        }

    def resize(self, new_max_size: int) -> None:
        """Resize the cache to a new maximum size.

        Args:
            new_max_size: New maximum cache size
        """
        if new_max_size < 0:
            raise ValueError("max_size must be non-negative")

        old_size = self.max_size
        self.max_size = new_max_size

        # Evict items if new size is smaller
        while len(self.cache) > new_max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        logger.info("Resized embedding cache",
                   old_size=old_size,
                   new_size=new_max_size,
                   current_entries=len(self.cache))


class QdrantVectorStorage(BaseVectorStorage):
    """Qdrant vector storage implementation."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        verify_ssl: Optional[bool] = None
    ):
        """Initialize Qdrant storage.

        Args:
            host: Qdrant host
            port: Qdrant port
            api_key: API key for authentication
            collection_name: Collection name (required, no default)
            verify_ssl: Whether to verify SSL certificates (default: from environment)
        """
        import os
        self.host = host
        self.port = port
        self.api_key = api_key
        # Use environment variable if verify_ssl not explicitly set
        if verify_ssl is None:
            self.verify_ssl = os.getenv('QDRANT_VERIFY_SSL', 'true').lower() == 'true'
        else:
            self.verify_ssl = verify_ssl
        if not collection_name:
            raise ValueError("collection_name is required - set QDRANT_COLLECTION_NAME environment variable")
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the storage.

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            await self.connect()
            # Ensure the collection exists
            await self.ensure_collection_exists()
            self._initialized = True
            logger.info("QdrantVectorStorage initialized successfully")
            return True
        except Exception as e:
            logger.error("Failed to initialize QdrantVectorStorage", error=str(e))
            self._initialized = False
            return False

    async def shutdown(self) -> None:
        """Shutdown the storage and release resources."""
        try:
            if self.client:
                # Qdrant client doesn't need explicit shutdown
                self.client = None
            self._initialized = False
            logger.info("QdrantVectorStorage shutdown completed")
        except Exception as e:
            logger.error("Error during QdrantVectorStorage shutdown", error=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Check storage health.

        Returns:
            Dictionary with health status information
        """
        try:
            if not self.client:
                await self.connect()

            # Test connection by getting collections
            collections = await asyncio.to_thread(self.client.get_collections)

            return {
                "status": "healthy",
                "host": self.host,
                "port": self.port,
                "collections_count": len(collections.collections),
                "collections": [col.name for col in collections.collections],
                "initialized": self._initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "host": self.host,
                "port": self.port,
                "initialized": self._initialized
            }

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
                    timeout=30,
                    verify=self.verify_ssl
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
                    timeout=30,
                    verify=self.verify_ssl
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

    async def ensure_collection_exists(self, vector_size: int = 768) -> None:
        """Ensure the collection exists, create it if it doesn't.

        Args:
            vector_size: Size of vectors for the collection (default: 768)
        """
        if not self.client:
            await self.connect()

        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            if not collection_exists:
                logger.info("Collection does not exist, creating it",
                           collection=self.collection_name,
                           vector_size=vector_size)
                await self.create_collection(self.collection_name, vector_size)
            else:
                logger.info("Collection already exists", collection=self.collection_name)

        except Exception as e:
            logger.error("Failed to ensure collection exists",
                        collection=self.collection_name,
                        error=str(e))
            raise StorageError(f"Failed to ensure collection exists: {str(e)}")

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

    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 10,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for vectors by metadata only (no vector similarity).

        Args:
            filters: Metadata filters
            limit: Maximum number of results
            collection_name: Optional collection name

        Returns:
            List of matching points with metadata
        """
        if not self.client:
            await self.connect()

        target_collection = collection_name or self.collection_name

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Build filter conditions
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                ))

            search_filter = Filter(must=conditions) if conditions else None

            # Use scroll to find matching points
            points, _ = await asyncio.to_thread(
                self.client.scroll,
                collection_name=target_collection,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            # Format results
            formatted_results = []
            for point in points:
                formatted_results.append({
                    "id": point.id,
                    "metadata": point.payload,
                    "text": point.payload.get("text", ""),
                    "score": 1.0  # No similarity score for metadata-only search
                })

            logger.info("Metadata search completed",
                       results_count=len(formatted_results),
                       collection=target_collection)
            return formatted_results

        except Exception as e:
            logger.error("Metadata search failed", error=str(e))
            raise StorageError(f"Metadata search failed: {str(e)}")

    async def find_document_points(
        self,
        document_id: str,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Find all vector points for a document by document ID.

        Args:
            document_id: Document identifier
            collection_name: Optional collection name

        Returns:
            List of point IDs for the document
        """
        if not self.client:
            await self.connect()

        target_collection = collection_name or self.collection_name

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Search for points with matching document_id
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

            # Scroll through all matching points
            points = []
            offset = None

            while True:
                result = await asyncio.to_thread(
                    self.client.scroll,
                    collection_name=target_collection,
                    scroll_filter=search_filter,
                    limit=100,
                    offset=offset,
                    with_payload=False  # We only need IDs
                )

                batch_points = result[0]
                if not batch_points:
                    break

                points.extend([point.id for point in batch_points])
                offset = result[1]  # Next offset

                if offset is None:
                    break

            logger.info("Found document points",
                       document_id=document_id,
                       points_count=len(points))
            return points

        except Exception as e:
            logger.error("Failed to find document points",
                        document_id=document_id,
                        error=str(e))
            return []

    async def delete_document_points(
        self,
        document_id: str,
        collection_name: Optional[str] = None
    ) -> int:
        """Delete all vector points for a document.

        Args:
            document_id: Document identifier
            collection_name: Optional collection name

        Returns:
            Number of points deleted
        """
        # Find all points for the document
        point_ids = await self.find_document_points(document_id, collection_name)

        if not point_ids:
            return 0

        # Delete the points
        deleted_count = await self.delete_vectors_legacy(point_ids, collection_name)

        logger.info("Deleted document points",
                   document_id=document_id,
                   deleted_count=deleted_count)
        return deleted_count

    async def delete_vectors_legacy(
        self,
        vector_ids: List[str],
        collection_name: Optional[str] = None
    ) -> int:
        """Delete vectors by IDs (legacy method)."""
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

    async def replace_document(
        self,
        document_id: str,
        new_vectors: List[List[float]],
        new_metadata: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Replace existing document with new vectors.

        Args:
            document_id: Document identifier
            new_vectors: New vector embeddings
            new_metadata: New metadata for each vector
            collection_name: Optional collection name

        Returns:
            List of new point IDs
        """
        target_collection = collection_name or self.collection_name

        try:
            # Find existing document chunks
            existing_points = await self.find_document_points(document_id, target_collection)

            # Delete existing points if any
            if existing_points:
                await self.delete_document_points(document_id, target_collection)
                logger.info("Deleted existing document points",
                           document_id=document_id,
                           points_deleted=len(existing_points))

            # Add document_id to all metadata entries
            for metadata in new_metadata:
                metadata['document_id'] = document_id
                metadata['replaced_at'] = datetime.now(timezone.utc).isoformat()

            # Store new vectors
            new_point_ids = await self.store_vectors(
                new_vectors,
                new_metadata,
                target_collection
            )

            logger.info("Document replaced successfully",
                       document_id=document_id,
                       old_points=len(existing_points) if existing_points else 0,
                       new_points=len(new_point_ids))

            return new_point_ids

        except Exception as e:
            logger.error("Failed to replace document",
                        document_id=document_id,
                        error=str(e))
            raise StorageError(f"Failed to replace document: {str(e)}")

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

    # Object storage methods (required by BaseStorage)
    async def put_object(
        self,
        key: str,
        data: Union[bytes, str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> "StorageObject":
        """Store an object in Qdrant as a point with metadata.

        Args:
            key: Object key/identifier
            data: Object data
            metadata: Optional metadata

        Returns:
            Storage object information
        """
        from morag_core.interfaces.storage import StorageObject, StorageType

        if not self.client:
            await self.connect()

        try:
            # Convert data to string for storage
            if isinstance(data, bytes):
                data_str = data.decode('utf-8')
            elif isinstance(data, dict):
                import json
                data_str = json.dumps(data)
            else:
                data_str = str(data)

            # Create a dummy vector (Qdrant requires vectors)
            dummy_vector = [0.0] * 384  # Standard embedding size

            payload = {
                "object_key": key,
                "object_data": data_str,
                "object_type": "generic",
                "created_at": datetime.now(timezone.utc).isoformat(),
                **(metadata or {})
            }

            point = PointStruct(
                id=key,
                vector=dummy_vector,
                payload=payload
            )

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=[point]
            )

            return StorageObject(
                key=key,
                storage_type=StorageType.VECTOR,
                metadata=metadata
            )

        except Exception as e:
            logger.error("Failed to put object", key=key, error=str(e))
            raise StorageError(f"Failed to put object: {str(e)}")

    async def get_object(self, key: str) -> Union[bytes, Dict[str, Any]]:
        """Retrieve an object by key.

        Args:
            key: Object key/identifier

        Returns:
            Object data
        """
        if not self.client:
            await self.connect()

        try:
            points = await asyncio.to_thread(
                self.client.retrieve,
                collection_name=self.collection_name,
                ids=[key]
            )

            if not points:
                raise StorageError(f"Object not found: {key}")

            point = points[0]
            data_str = point.payload.get("object_data")

            if not data_str:
                raise StorageError(f"No data found for object: {key}")

            # Try to parse as JSON, fallback to string
            try:
                import json
                return json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                return data_str

        except Exception as e:
            logger.error("Failed to get object", key=key, error=str(e))
            raise StorageError(f"Failed to get object: {str(e)}")

    async def delete_object(self, key: str) -> bool:
        """Delete an object by key.

        Args:
            key: Object key/identifier

        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.client:
            await self.connect()

        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=[key]
            )

            logger.info("Deleted object", key=key)
            return True

        except Exception as e:
            logger.error("Failed to delete object", key=key, error=str(e))
            return False

    async def list_objects(self, prefix: str = "") -> List["StorageObject"]:
        """List objects with prefix.

        Args:
            prefix: Key prefix

        Returns:
            List of storage objects
        """
        from morag_core.interfaces.storage import StorageObject, StorageType

        if not self.client:
            await self.connect()

        try:
            # Qdrant doesn't have native prefix search, so we'll scroll through all points
            # and filter by prefix
            points, _ = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                limit=1000  # Adjust as needed
            )

            objects = []
            for point in points:
                object_key = point.payload.get("object_key", str(point.id))
                if object_key.startswith(prefix):
                    objects.append(StorageObject(
                        key=object_key,
                        storage_type=StorageType.VECTOR,
                        metadata=point.payload
                    ))

            return objects

        except Exception as e:
            logger.error("Failed to list objects", prefix=prefix, error=str(e))
            raise StorageError(f"Failed to list objects: {str(e)}")

    async def get_object_metadata(self, key: str) -> "StorageMetadata":
        """Get object metadata.

        Args:
            key: Object key/identifier

        Returns:
            Object metadata
        """
        from morag_core.interfaces.storage import StorageMetadata

        if not self.client:
            await self.connect()

        try:
            points = await asyncio.to_thread(
                self.client.retrieve,
                collection_name=self.collection_name,
                ids=[key]
            )

            if not points:
                raise StorageError(f"Object not found: {key}")

            point = points[0]
            return StorageMetadata(
                size=len(str(point.payload.get("object_data", ""))),
                created_at=point.payload.get("created_at"),
                modified_at=point.payload.get("modified_at"),
                content_type=point.payload.get("content_type"),
                custom_metadata=point.payload
            )

        except Exception as e:
            logger.error("Failed to get object metadata", key=key, error=str(e))
            raise StorageError(f"Failed to get object metadata: {str(e)}")

    async def object_exists(self, key: str) -> bool:
        """Check if object exists.

        Args:
            key: Object key/identifier

        Returns:
            True if object exists, False otherwise
        """
        if not self.client:
            await self.connect()

        try:
            points = await asyncio.to_thread(
                self.client.retrieve,
                collection_name=self.collection_name,
                ids=[key]
            )

            return len(points) > 0

        except Exception as e:
            logger.error("Failed to check object existence", key=key, error=str(e))
            return False

    # Vector storage methods (required by VectorStorage)
    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors to storage.

        Args:
            vectors: List of vectors
            metadata: List of metadata dictionaries
            ids: Optional list of IDs

        Returns:
            List of assigned IDs
        """
        # Delegate to existing store_vectors method
        return await self.store_vectors(vectors, metadata, self.collection_name)

    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector
            limit: Maximum number of results
            filter_expr: Optional filter expression

        Returns:
            List of search results with scores and metadata
        """
        # Delegate to existing search_similar method
        return await self.search_similar(
            query_vector,
            limit=limit,
            score_threshold=0.0,  # Return all results up to limit
            filters=filter_expr,
            collection_name=self.collection_name
        )

    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            deleted_count = await self.delete_vectors_legacy(ids, self.collection_name)
            return deleted_count > 0
        except Exception as e:
            logger.error("Failed to delete vectors", ids=ids, error=str(e))
            return False

    async def update_vector_metadata(
        self,
        id: str,
        metadata: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        """Update vector metadata.

        Args:
            id: Vector ID
            metadata: New metadata
            upsert: Whether to insert if not exists

        Returns:
            True if update was successful, False otherwise
        """
        if not self.client:
            await self.connect()

        try:
            # Get existing point
            points = await asyncio.to_thread(
                self.client.retrieve,
                collection_name=self.collection_name,
                ids=[id]
            )

            if not points:
                if not upsert:
                    return False
                # For upsert, we'd need the vector, which we don't have
                logger.warning("Cannot upsert vector metadata without vector data", id=id)
                return False

            point = points[0]

            # Update metadata
            updated_payload = {**point.payload, **metadata}

            # Create updated point
            updated_point = PointStruct(
                id=id,
                vector=point.vector,
                payload=updated_payload
            )

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=[updated_point]
            )

            logger.info("Updated vector metadata", id=id)
            return True

        except Exception as e:
            logger.error("Failed to update vector metadata", id=id, error=str(e))
            return False



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
