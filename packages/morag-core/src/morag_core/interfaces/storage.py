"""Base interfaces for storage."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class StorageType(str, Enum):
    """Storage type enum."""

    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"
    DATABASE = "database"
    VECTOR = "vector"


@dataclass
class StorageMetadata:
    """Storage metadata."""

    content_type: Optional[str] = None
    content_length: Optional[int] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    etag: Optional[str] = None
    custom: Optional[Dict[str, Any]] = None


@dataclass
class StorageObject:
    """Storage object information."""

    key: str
    storage_type: StorageType
    metadata: Optional[StorageMetadata] = None
    url: Optional[str] = None


class BaseStorage(ABC):
    """Base class for storage implementations."""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the storage.

        Returns:
            True if initialization was successful, False otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the storage and release resources."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check storage health.

        Returns:
            Dictionary with health status information
        """
        pass

    @abstractmethod
    async def put_object(
        self,
        key: str,
        data: Union[bytes, str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StorageObject:
        """Store an object.

        Args:
            key: Object key/identifier
            data: Object data
            metadata: Optional metadata

        Returns:
            Storage object information
        """
        pass

    @abstractmethod
    async def get_object(self, key: str) -> Union[bytes, Dict[str, Any]]:
        """Retrieve an object.

        Args:
            key: Object key/identifier

        Returns:
            Object data
        """
        pass

    @abstractmethod
    async def delete_object(self, key: str) -> bool:
        """Delete an object.

        Args:
            key: Object key/identifier

        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    async def list_objects(self, prefix: str = "") -> List[StorageObject]:
        """List objects with prefix.

        Args:
            prefix: Key prefix

        Returns:
            List of storage objects
        """
        pass

    @abstractmethod
    async def get_object_metadata(self, key: str) -> StorageMetadata:
        """Get object metadata.

        Args:
            key: Object key/identifier

        Returns:
            Object metadata
        """
        pass

    @abstractmethod
    async def object_exists(self, key: str) -> bool:
        """Check if object exists.

        Args:
            key: Object key/identifier

        Returns:
            True if object exists, False otherwise
        """
        pass


class VectorStorage(BaseStorage):
    """Base class for vector storage implementations."""

    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add vectors to storage.

        Args:
            vectors: List of vectors
            metadata: List of metadata dictionaries
            ids: Optional list of IDs

        Returns:
            List of assigned IDs
        """
        pass

    @abstractmethod
    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector
            limit: Maximum number of results
            filter_expr: Optional filter expression

        Returns:
            List of search results with scores and metadata
        """
        pass

    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs

        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_vector_metadata(
        self, id: str, metadata: Dict[str, Any], upsert: bool = False
    ) -> bool:
        """Update vector metadata.

        Args:
            id: Vector ID
            metadata: New metadata
            upsert: Whether to insert if not exists

        Returns:
            True if update was successful, False otherwise
        """
        pass
