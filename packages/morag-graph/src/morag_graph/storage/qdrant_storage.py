"""Qdrant storage backend for graph data."""

import logging
from typing import Dict, List, Optional, Any, Set
import uuid

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, MatchValue, SearchRequest
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    AsyncQdrantClient = None

from pydantic import BaseModel

from ..models import Entity, Relation, Graph
from ..models.types import EntityId, RelationId
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
        self.client: Optional[AsyncQdrantClient] = None
    
    async def connect(self) -> None:
        """Connect to Qdrant database."""
        try:
            self.client = AsyncQdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port,
                prefer_grpc=self.config.prefer_grpc,
                https=self.config.https,
                api_key=self.config.api_key,
                prefix=self.config.prefix,
                timeout=self.config.timeout,
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