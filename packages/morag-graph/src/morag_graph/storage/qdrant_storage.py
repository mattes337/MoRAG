"""Qdrant storage backend for graph data."""

import logging
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, MatchValue, SearchRequest
    )

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, MatchValue, SearchRequest
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # Create placeholder classes for runtime
    class AsyncQdrantClient:  # type: ignore
        pass
    class Distance:  # type: ignore
        pass
    class VectorParams:  # type: ignore
        pass
    class CreateCollection:  # type: ignore
        pass
    class PointStruct:  # type: ignore
        pass
    class Filter:  # type: ignore
        pass
    class FieldCondition:  # type: ignore
        pass
    class MatchValue:  # type: ignore
        pass
    class SearchRequest:  # type: ignore
        pass

from pydantic import BaseModel

from ..models import Entity, Relation, Graph
from ..models.types import EntityId, RelationId
from ..utils.id_generation import UnifiedIDGenerator, IDValidator
from .base import BaseStorage

logger = logging.getLogger(__name__)


class QdrantConfig(BaseModel):
    """Configuration for Qdrant connection."""

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    https: bool = False
    api_key: Optional[str] = None
    prefix: Optional[str] = None
    timeout: Optional[float] = None
    collection_name: str = "morag_entities"
    vector_size: int = 384  # Default embedding size
    verify_ssl: Optional[bool] = None  # Whether to verify SSL certificates (None = use environment)

    def __init__(self, **data):
        """Initialize with environment variable defaults."""
        import os
        # Set verify_ssl from environment if not explicitly provided
        if 'verify_ssl' not in data or data['verify_ssl'] is None:
            data['verify_ssl'] = os.getenv('QDRANT_VERIFY_SSL', 'true').lower() == 'true'
        super().__init__(**data)


class QdrantStorage(BaseStorage):
    """Qdrant storage backend for graph data.
    
    This class implements the BaseStorage interface using Qdrant as the backend.
    It stores entities as vectors with metadata and relations as metadata connections.
    """
    
    def __init__(self, config: QdrantConfig):
        """Initialize Qdrant storage.
        
        Args:
            config: Qdrant configuration
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for QdrantStorage")
            
        self.config = config
        self.client: Optional["AsyncQdrantClient"] = None
    
    async def connect(self) -> None:
        """Connect to Qdrant database."""
        try:
            # Check if host is a URL (starts with http:// or https://)
            if self.config.host.startswith(('http://', 'https://')):
                from urllib.parse import urlparse
                parsed = urlparse(self.config.host)
                hostname = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == 'https' else self.config.port)
                use_https = parsed.scheme == 'https'

                logger.info(f"Connecting to Qdrant via URL: {self.config.host} "
                           f"(hostname={hostname}, port={port}, https={use_https}, "
                           f"verify_ssl={self.config.verify_ssl})")

                self.client = AsyncQdrantClient(
                    host=hostname,
                    port=port,
                    grpc_port=self.config.grpc_port,
                    prefer_grpc=self.config.prefer_grpc,
                    https=use_https,
                    api_key=self.config.api_key,
                    prefix=self.config.prefix,
                    timeout=self.config.timeout or 30,
                    verify=self.config.verify_ssl,
                )
            else:
                # Auto-detect HTTPS if port is 443
                use_https = self.config.https or (self.config.port == 443)

                logger.info(f"Connecting to Qdrant via host/port: {self.config.host}:{self.config.port} "
                           f"(https={use_https}, verify_ssl={self.config.verify_ssl})")

                self.client = AsyncQdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    grpc_port=self.config.grpc_port,
                    prefer_grpc=self.config.prefer_grpc,
                    https=use_https,
                    api_key=self.config.api_key,
                    prefix=self.config.prefix,
                    timeout=self.config.timeout or 30,
                    verify=self.config.verify_ssl,
                )
            
            # Create collection if it doesn't exist
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.config.collection_name not in collection_names:
                await self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.config.collection_name}")
            
            logger.info("Connected to Qdrant database")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Qdrant database."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Disconnected from Qdrant database")
    
    async def store_entity(self, entity: Entity) -> EntityId:
        """Store an entity in Qdrant.
        
        Args:
            entity: Entity to store
            
        Returns:
            ID of the stored entity
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        # Generate ID if not provided
        entity_id = entity.id or str(uuid.uuid4())
        
        # Prepare point for insertion
        point = PointStruct(
            id=entity_id,
            vector=entity.embedding or [0.0] * self.config.vector_size,
            payload={
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "attributes": entity.attributes,
                "source_doc_id": entity.source_doc_id,
                "confidence": entity.confidence,
                "created_at": entity.created_at.isoformat() if entity.created_at else None,
                "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
            }
        )
        
        # Upsert the point
        await self.client.upsert(
            collection_name=self.config.collection_name,
            points=[point]
        )
        
        logger.debug(f"Stored entity {entity_id} in Qdrant")
        return entity_id
    
    async def store_entities(self, entities: List[Entity]) -> List[EntityId]:
        """Store multiple entities in Qdrant.
        
        Args:
            entities: List of entities to store
            
        Returns:
            List of IDs of the stored entities
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        if not entities:
            return []
        
        points = []
        entity_ids = []
        
        for entity in entities:
            entity_id = entity.id or str(uuid.uuid4())
            entity_ids.append(entity_id)
            
            point = PointStruct(
                id=entity_id,
                vector=entity.embedding or [0.0] * self.config.vector_size,
                payload={
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "attributes": entity.attributes,
                    "source_doc_id": entity.source_doc_id,
                    "confidence": entity.confidence,
                    "created_at": entity.created_at.isoformat() if entity.created_at else None,
                    "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
                }
            )
            points.append(point)
        
        # Batch upsert
        await self.client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        logger.debug(f"Stored {len(entities)} entities in Qdrant")
        return entity_ids
    
    async def store_chunk_vector_with_unified_id(self, 
                                                chunk_id: str,
                                                vector: List[float],
                                                metadata: Dict[str, Any]) -> str:
        """Store chunk vector with unified ID format.
        
        Args:
            chunk_id: Unified chunk ID
            vector: Vector embedding
            metadata: Additional metadata
            
        Returns:
            Stored chunk ID
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        # Validate chunk ID format
        if not IDValidator.validate_chunk_id(chunk_id):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        # Extract document ID from chunk ID
        document_id = UnifiedIDGenerator.extract_document_id_from_chunk(chunk_id)
        chunk_index = UnifiedIDGenerator.extract_chunk_index_from_chunk(chunk_id)
        
        # Prepare enhanced metadata
        enhanced_metadata = {
            'document_id': document_id,
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'neo4j_chunk_id': chunk_id,  # Cross-reference
            'unified_id_format': True,
            **metadata
        }
        
        # Create point for insertion
        point = PointStruct(
            id=chunk_id,  # Use chunk_id as point ID
            vector=vector,
            payload=enhanced_metadata
        )
        
        # Upsert the point
        await self.client.upsert(
            collection_name=self.config.collection_name,
            points=[point]
        )
        
        logger.debug(f"Stored chunk vector {chunk_id} in Qdrant")
        return chunk_id
    
    async def get_chunk_vectors_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunk vectors for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk vector data
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        # Search for vectors with matching document_id
        search_result = await self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1000  # Adjust as needed
        )
        
        chunks = []
        for point in search_result[0]:  # search_result is (points, next_page_offset)
            chunks.append({
                'chunk_id': point.id,
                'vector': point.vector,
                'metadata': point.payload
            })
        
        logger.debug(f"Retrieved {len(chunks)} chunk vectors for document {document_id}")
        return chunks
    
    async def update_chunk_vector_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a chunk vector.
        
        Args:
            chunk_id: Chunk ID
            metadata: New metadata to merge
            
        Returns:
            True if successful
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            # Get existing point
            existing_points = await self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[chunk_id],
                with_payload=True
            )
            
            if not existing_points:
                logger.warning(f"Chunk {chunk_id} not found for metadata update")
                return False
            
            existing_point = existing_points[0]
            
            # Merge metadata
            updated_payload = {**existing_point.payload, **metadata}
            
            # Update the point
            await self.client.set_payload(
                collection_name=self.config.collection_name,
                payload=updated_payload,
                points=[chunk_id]
            )
            
            logger.debug(f"Updated metadata for chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata for chunk {chunk_id}: {e}")
            return False
     
    async def store_chunk_vector_with_entities(self, 
                                             chunk_id: str,
                                             vector: List[float],
                                             metadata: Dict[str, Any],
                                             entity_ids: List[str]) -> str:
        """Store chunk vector with entity references.
        
        Args:
            chunk_id: Unified chunk ID
            vector: Vector embedding
            metadata: Additional metadata
            entity_ids: List of entity IDs referenced in this chunk
            
        Returns:
            Stored chunk ID
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        # Validate chunk ID format
        if not IDValidator.validate_chunk_id(chunk_id):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        # Extract document ID from chunk ID
        document_id = UnifiedIDGenerator.extract_document_id_from_chunk(chunk_id)
        chunk_index = UnifiedIDGenerator.extract_chunk_index_from_chunk(chunk_id)
        
        # Prepare enhanced metadata with entity references
        enhanced_metadata = {
            'document_id': document_id,
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'neo4j_chunk_id': chunk_id,  # Cross-reference
            'unified_id_format': True,
            'entity_ids': entity_ids,  # Entity references
            'entity_count': len(entity_ids),
            **metadata
        }
        
        # Create point for insertion
        point = PointStruct(
            id=chunk_id,  # Use chunk_id as point ID
            vector=vector,
            payload=enhanced_metadata
        )
        
        # Upsert the point
        await self.client.upsert(
            collection_name=self.config.collection_name,
            points=[point]
        )
        
        logger.debug(f"Stored chunk vector {chunk_id} with {len(entity_ids)} entity references in Qdrant")
        return chunk_id
    
    async def get_chunks_by_entity_id(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all chunks that reference a specific entity.
        
        Args:
            entity_id: Entity ID to search for
            
        Returns:
            List of chunk data containing the entity
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        # Search for vectors with entity_id in entity_ids array
        search_result = await self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="entity_ids",
                        match=MatchValue(value=entity_id)
                    )
                ]
            ),
            limit=1000  # Adjust as needed
        )
        
        chunks = []
        for point in search_result[0]:  # search_result is (points, next_page_offset)
            chunks.append({
                'chunk_id': point.id,
                'vector': point.vector,
                'metadata': point.payload
            })
        
        logger.debug(f"Retrieved {len(chunks)} chunks referencing entity {entity_id}")
        return chunks
    
    async def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID from Qdrant.
        
        Args:
            entity_id: ID of the entity to get
            
        Returns:
            Entity if found, None otherwise
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            points = await self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[entity_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return None
            
            point = points[0]
            payload = point.payload
            
            return Entity(
                id=str(point.id),
                name=payload.get("name", ""),
                type=payload.get("type", ""),
                description=payload.get("description"),
                attributes=payload.get("attributes", {}),
                source_doc_id=payload.get("source_doc_id"),
                confidence=payload.get("confidence", 1.0),
                embedding=point.vector
            )
            
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    async def get_entities(self, entity_ids: List[EntityId]) -> List[Entity]:
        """Get multiple entities by IDs from Qdrant.
        
        Args:
            entity_ids: List of entity IDs to get
            
        Returns:
            List of entities (may be shorter than input if some not found)
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        if not entity_ids:
            return []
        
        try:
            points = await self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=entity_ids,
                with_payload=True,
                with_vectors=True
            )
            
            entities = []
            for point in points:
                payload = point.payload
                entity = Entity(
                    id=str(point.id),
                    name=payload.get("name", ""),
                    type=payload.get("type", ""),
                    description=payload.get("description"),
                    attributes=payload.get("attributes", {}),
                    source_doc_id=payload.get("source_doc_id"),
                    confidence=payload.get("confidence", 1.0),
                    embedding=point.vector
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to get entities: {e}")
            return []
    
    async def search_entities(
        self, 
        query: str, 
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search for entities by name or attributes in Qdrant.
        
        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        # For now, implement as a simple text search in payload
        # In a real implementation, you'd want to use vector search with query embeddings
        try:
            filter_conditions = []
            
            if entity_type:
                filter_conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=entity_type)
                    )
                )
            
            # Simple text search in name and description
            # Note: This is a basic implementation. For better search,
            # you'd want to embed the query and use vector search
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Use scroll to get all points and filter client-side
            # In production, you'd want to use proper vector search
            points, _ = await self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=search_filter,
                limit=limit * 2,  # Get more to filter client-side
                with_payload=True,
                with_vectors=True
            )
            
            # Filter by query text
            matching_entities = []
            query_lower = query.lower()
            
            for point in points:
                payload = point.payload
                name = payload.get("name", "").lower()
                description = payload.get("description", "").lower()
                
                if query_lower in name or query_lower in description:
                    entity = Entity(
                        id=str(point.id),
                        name=payload.get("name", ""),
                        type=payload.get("type", ""),
                        description=payload.get("description"),
                        attributes=payload.get("attributes", {}),
                        source_doc_id=payload.get("source_doc_id"),
                        confidence=payload.get("confidence", 1.0),
                        embedding=point.vector
                    )
                    matching_entities.append(entity)
                    
                    if len(matching_entities) >= limit:
                        break
            
            return matching_entities
            
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return []
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity in Qdrant.
        
        Args:
            entity: Entity to update
            
        Returns:
            True if successful, False otherwise
        """
        if not entity.id:
            return False
        
        try:
            # Qdrant upsert will update if exists
            await self.store_entity(entity)
            return True
        except Exception as e:
            logger.error(f"Failed to update entity {entity.id}: {e}")
            return False
    
    async def delete_entity(self, entity_id: EntityId) -> bool:
        """Delete an entity from Qdrant.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            await self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=[entity_id]
            )
            logger.debug(f"Deleted entity {entity_id} from Qdrant")
            return True
        except Exception as e:
            logger.error(f"Failed to delete entity {entity_id}: {e}")
            return False
    
    # Checksum management methods
    async def get_document_checksum(self, document_id: str) -> Optional[str]:
        """Get the stored checksum for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Stored checksum if found, None otherwise
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            # Use a special collection or namespace for checksums
            checksum_id = f"checksum_{document_id}"
            points = await self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[checksum_id],
                with_payload=True
            )
            
            if points and len(points) > 0:
                return points[0].payload.get("checksum")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document checksum for {document_id}: {e}")
            return None
    
    async def store_document_checksum(self, document_id: str, checksum: str) -> None:
        """Store the checksum for a document.
        
        Args:
            document_id: Document ID
            checksum: Document checksum
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            from datetime import datetime
            
            checksum_id = f"checksum_{document_id}"
            
            # Create a point for the checksum with a dummy vector
            point = PointStruct(
                id=checksum_id,
                vector=[0.0] * self.config.vector_size,  # Dummy vector
                payload={
                    "type": "document_checksum",
                    "document_id": document_id,
                    "checksum": checksum,
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
            await self.client.upsert(
                collection_name=self.config.collection_name,
                points=[point]
            )
            
            logger.debug(f"Stored checksum for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to store document checksum for {document_id}: {e}")
            raise
    
    async def delete_document_checksum(self, document_id: str) -> None:
        """Delete the checksum for a document.
        
        Args:
            document_id: Document ID
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            checksum_id = f"checksum_{document_id}"
            await self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=[checksum_id]
            )
            
            logger.debug(f"Deleted checksum for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete document checksum for {document_id}: {e}")
            # Don't raise here as this is cleanup
    
    async def get_entities_by_document(self, document_id: str) -> List[Entity]:
        """Get all entities associated with a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of entities associated with the document
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            
            # Create filter to get entities by source_doc_id, excluding checksum entries
            must_conditions = [
                FieldCondition(
                    key="source_doc_id",
                    match=MatchValue(value=document_id)
                )
            ]
            
            must_not_conditions = [
                FieldCondition(
                    key="type",
                    match=MatchValue(value="document_checksum")
                )
            ]
            
            filter_condition = Filter(
                must=must_conditions,
                must_not=must_not_conditions
            )
            
            # Search for entities with matching source_doc_id
            search_result = await self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=filter_condition,
                limit=1000,  # Adjust as needed
                with_payload=True,
                with_vectors=True
            )
            
            entities = []
            for point in search_result[0]:  # search_result is (points, next_page_offset)
                payload = point.payload
                # Skip checksum entries
                if payload.get("type") == "document_checksum":
                    continue
                    
                entity = Entity(
                    id=str(point.id),
                    name=payload.get("name", ""),
                    type=payload.get("type", ""),
                    description=payload.get("description"),
                    attributes=payload.get("attributes", {}),
                    source_doc_id=payload.get("source_doc_id"),
                    confidence=payload.get("confidence", 1.0),
                    embedding=point.vector
                )
                entities.append(entity)
            
            logger.debug(f"Retrieved {len(entities)} entities for document {document_id}")
            return entities
            
        except Exception as e:
            logger.error(f"Error getting entities by document: {e}")
            return []
    
    # Note: Qdrant doesn't have native graph relations like Neo4j
    # Relations would need to be stored as metadata or in a separate collection
    # For now, implementing minimal relation support
    
    async def store_relation(self, relation: Relation) -> RelationId:
        """Store a relation (as metadata in Qdrant).
        
        Note: Qdrant doesn't have native graph relations.
        This stores relation info in entity metadata.
        """
        # For Qdrant, we could store relations as separate points
        # or as metadata in entity points. This is a simplified implementation.
        relation_id = relation.id or str(uuid.uuid4())
        logger.debug(f"Relation storage in Qdrant is simplified: {relation_id}")
        return relation_id
    
    async def store_relations(self, relations: List[Relation]) -> List[RelationId]:
        """Store multiple relations."""
        return [await self.store_relation(rel) for rel in relations]
    
    async def get_relation(self, relation_id: RelationId) -> Optional[Relation]:
        """Get a relation by ID (simplified for Qdrant)."""
        logger.debug(f"Relation retrieval in Qdrant is simplified: {relation_id}")
        return None
    
    async def get_relations(self, relation_ids: List[RelationId]) -> List[Relation]:
        """Get multiple relations by IDs."""
        return []
    
    async def get_entity_relations(
        self, 
        entity_id: EntityId, 
        relation_type: Optional[str] = None
    ) -> List[Relation]:
        """Get relations for an entity."""
        return []
    
    async def delete_relation(self, relation_id: RelationId) -> bool:
        """Delete a relation."""
        return True
    
    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation (simplified for Qdrant)."""
        logger.debug(f"Relation update in Qdrant is simplified: {relation.id}")
        return True
    
    async def get_neighbors(
        self, 
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Entity]:
        """Get neighboring entities (simplified for Qdrant)."""
        # Qdrant doesn't have native graph traversal
        # This would require custom implementation based on stored relations
        logger.debug(f"Neighbor search in Qdrant is simplified for entity: {entity_id}")
        return []
    
    async def find_path(
        self, 
        source_entity_id: EntityId,
        target_entity_id: EntityId,
        max_depth: int = 3
    ) -> List[List[EntityId]]:
        """Find paths between two entities (simplified for Qdrant)."""
        # Qdrant doesn't have native graph pathfinding
        logger.debug(f"Path finding in Qdrant is simplified: {source_entity_id} -> {target_entity_id}")
        return []
    
    async def store_graph(self, graph: Graph) -> None:
        """Store an entire graph."""
        # Store all entities
        if graph.entities:
            await self.store_entities(list(graph.entities.values()))
        
        # Store all relations (simplified)
        if graph.relations:
            await self.store_relations(list(graph.relations.values()))
        
        logger.info(f"Stored graph with {len(graph.entities)} entities and {len(graph.relations)} relations")
    
    async def get_graph(
        self, 
        entity_ids: Optional[List[EntityId]] = None
    ) -> Graph:
        """Get a graph or subgraph."""
        from ..models import Graph
        
        if entity_ids:
            entities = await self.get_entities(entity_ids)
        else:
            # Get all entities (simplified - in practice you'd want pagination)
            entities = []
            try:
                search_result = await self.client.scroll(
                    collection_name=self.config.collection_name,
                    limit=10000,  # Large limit to get all entities
                    with_payload=True,
                    with_vectors=True
                )
                
                for point in search_result[0]:
                    payload = point.payload
                    # Skip checksum entries
                    if payload.get("type") == "document_checksum":
                        continue
                        
                    entity = Entity(
                        id=str(point.id),
                        name=payload.get("name", ""),
                        type=payload.get("type", ""),
                        description=payload.get("description"),
                        attributes=payload.get("attributes", {}),
                        source_doc_id=payload.get("source_doc_id"),
                        confidence=payload.get("confidence", 1.0),
                        embedding=point.vector
                    )
                    entities.append(entity)
            except Exception as e:
                logger.error(f"Failed to get all entities: {e}")
        
        # Create graph with entities (relations are simplified in Qdrant)
        graph = Graph()
        for entity in entities:
            graph.add_entity(entity)
        
        return graph
    
    async def clear(self) -> None:
        """Clear all data from the storage."""
        await self.clear_all()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            # Get collection info
            collection_info = await self.client.get_collection(self.config.collection_name)
            
            # Count entities (excluding checksums)
            search_result = await self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must_not=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="document_checksum")
                        )
                    ]
                ),
                limit=1,  # We just want the count
                with_payload=False,
                with_vectors=False
            )
            
            # Get total points count from collection info
            total_points = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
            
            return {
                "total_points": total_points,
                "collection_name": self.config.collection_name,
                "vector_size": self.config.vector_size,
                "host": self.config.host,
                "port": self.config.port
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def clear_all(self) -> bool:
        """Clear all data from Qdrant collection."""
        if not self.client:
            raise RuntimeError("Not connected to Qdrant database")
        
        try:
            # Delete and recreate collection
            await self.client.delete_collection(self.config.collection_name)
            await self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Cleared Qdrant collection: {self.config.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Qdrant collection: {e}")
            return False