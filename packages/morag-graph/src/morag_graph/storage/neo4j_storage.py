"""Neo4J storage backend for graph data - Refactored with modular operations."""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from pydantic import BaseModel

from ..models import Entity, Relation, Graph, Document, DocumentChunk, Fact
from ..models.types import EntityId, RelationId
from ..utils.id_generation import UnifiedIDGenerator, IDValidator
from .base import BaseStorage
from .neo4j_operations import (
    ConnectionOperations,
    DocumentOperations,
    EntityOperations,
    RelationOperations,
    FactOperations,
    GraphOperations,
    QueryOperations
)

# OpenIE operations removed - replaced by LangExtract

logger = logging.getLogger(__name__)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4J connection."""

    uri: str = "neo4j://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60
    verify_ssl: bool = True  # Whether to verify SSL certificates
    trust_all_certificates: bool = False  # Trust all certificates (for self-signed)


class Neo4jStorage(BaseStorage):
    """Neo4J storage backend for graph data.
    
    This class implements the BaseStorage interface using Neo4J as the backend.
    It provides efficient storage and retrieval of entities and relations.
    """
    
    def __init__(self, config: Neo4jConfig):
        """Initialize Neo4J storage.
        
        Args:
            config: Neo4J configuration
        """
        self.config = config
        self.driver: Optional[AsyncDriver] = None
        
        # Initialize operation handlers
        self._connection_ops: Optional[ConnectionOperations] = None
        self._document_ops: Optional[DocumentOperations] = None
        self._entity_ops: Optional[EntityOperations] = None
        self._relation_ops: Optional[RelationOperations] = None
        self._fact_ops: Optional[FactOperations] = None
        self._graph_ops: Optional[GraphOperations] = None
        self._query_ops: Optional[QueryOperations] = None

    
    async def connect(self) -> None:
        """Connect to Neo4J database."""
        # Initialize connection operations
        self._connection_ops = ConnectionOperations(self.config)
        await self._connection_ops.connect()
        
        # Store driver reference for other operations
        self.driver = self._connection_ops.driver
        
        # Initialize other operation handlers
        self._document_ops = DocumentOperations(self.driver, self.config.database)
        self._entity_ops = EntityOperations(self.driver, self.config.database)
        self._relation_ops = RelationOperations(self.driver, self.config.database)
        self._fact_ops = FactOperations(self.driver, self.config.database)
        self._graph_ops = GraphOperations(self.driver, self.config.database)
        self._query_ops = QueryOperations(self.driver, self.config.database)

        logger.info("Neo4J storage initialized with modular operations")
    
    async def disconnect(self) -> None:
        """Disconnect from Neo4J database."""
        if self._connection_ops:
            await self._connection_ops.disconnect()
            self.driver = None
            
            # Clear operation handlers
            self._connection_ops = None
            self._document_ops = None
            self._entity_ops = None
            self._relation_ops = None
            self._fact_ops = None
            self._graph_ops = None
            self._query_ops = None

    # Connection delegation methods
    async def create_database_if_not_exists(self, database_name: str) -> bool:
        """Create a database if it doesn't exist."""
        if not self._connection_ops:
            raise RuntimeError("Connection not initialized")
        return await self._connection_ops.create_database_if_not_exists(database_name)
    
    # Document operations delegation
    async def store_document_with_unified_id(self, document: Document) -> str:
        """Store document with unified ID format."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.store_document_with_unified_id(document)
    
    async def store_chunk_with_unified_id(self, chunk: DocumentChunk) -> str:
        """Store document chunk with unified ID format."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.store_chunk_with_unified_id(chunk)
    
    async def get_document_by_unified_id(self, document_id: str) -> Optional[Document]:
        """Retrieve document by unified ID."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.get_document_by_unified_id(document_id)
    
    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document by unified ID."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.get_chunks_by_document_id(document_id)
    
    async def validate_id_consistency(self) -> Dict[str, Any]:
        """Validate ID consistency across the database."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.validate_id_consistency()
    
    async def store_document(self, document: Document) -> str:
        """Store a document in Neo4J."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.store_document(document)
    
    async def store_document_chunk(self, chunk: DocumentChunk) -> str:
        """Store a document chunk in Neo4J."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.store_document_chunk(chunk)

    async def store_documents(self, documents: List[Document]) -> List[str]:
        """Store multiple documents in Neo4J."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.store_documents(documents)

    async def store_document_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Store multiple document chunks in Neo4J."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.store_document_chunks(chunks)

    async def create_document_contains_chunk_relation(self, document_id: str, chunk_id: str) -> None:
        """Create a CONTAINS relationship between a document and a chunk."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.create_document_contains_chunk_relation(document_id, chunk_id)
    
    async def get_document_checksum(self, document_id: str) -> Optional[str]:
        """Get stored checksum for a document."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.get_document_checksum(document_id)
    
    async def store_document_checksum(self, document_id: str, checksum: str) -> None:
        """Store document checksum."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.store_document_checksum(document_id, checksum)
    
    async def delete_document_checksum(self, document_id: str) -> None:
        """Remove stored checksum for a document."""
        if not self._document_ops:
            raise RuntimeError("Connection not initialized")
        return await self._document_ops.delete_document_checksum(document_id)
    
    # Entity operations delegation
    async def store_entity(self, entity: Entity) -> EntityId:
        """Store an entity in Neo4J."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.store_entity(entity)
    
    async def store_entities(self, entities: List[Entity]) -> List[EntityId]:
        """Store multiple entities in Neo4J."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.store_entities(entities)
    
    async def store_entity_with_chunk_references(self, entity: Entity, chunk_ids: List[str]) -> EntityId:
        """Store entity with references to chunks where it's mentioned."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.store_entity_with_chunk_references(entity, chunk_ids)
    
    async def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.get_entity(entity_id)
    
    async def get_entities(self, entity_ids: List[EntityId]) -> List[Entity]:
        """Get multiple entities by IDs."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.get_entities(entity_ids)
    
    async def get_all_entities(self) -> List[Entity]:
        """Get all entities from the storage."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.get_all_entities()
    
    async def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 10) -> List[Entity]:
        """Search for entities by name or content."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.search_entities(query, entity_type, limit)
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.update_entity(entity)
    
    async def delete_entity(self, entity_id: EntityId) -> bool:
        """Delete an entity and all its relations."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.delete_entity(entity_id)
    
    async def get_entities_by_chunk_id(self, chunk_id: str) -> List[Entity]:
        """Get all entities mentioned in a specific chunk."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.get_entities_by_chunk_id(chunk_id)
    
    async def get_chunks_by_entity_id(self, entity_id: EntityId) -> List[str]:
        """Get all chunk IDs where an entity is mentioned."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.get_chunks_by_entity_id(entity_id)
    
    async def get_document_chunks_by_entity_names(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """Get all DocumentChunk nodes related to specific entity names with full metadata."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.get_document_chunks_by_entity_names(entity_names)
    
    async def update_entity_chunk_references(self, entity_id: EntityId, chunk_ids: List[str]) -> None:
        """Update entity's chunk references by replacing all existing relationships."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.update_entity_chunk_references(entity_id, chunk_ids)
    
    async def get_entities_by_document(self, document_id: str) -> List[Entity]:
        """Get all entities associated with a document."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.get_entities_by_document(document_id)
    
    async def create_chunk_mentions_entity_relation(self, chunk_id: str, entity_id: str, context: str) -> None:
        """Create a MENTIONS relationship between a chunk and an entity."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.create_chunk_mentions_entity_relation(chunk_id, entity_id, context)

    async def create_chunk_contains_fact_relation(self, chunk_id: str, fact_id: str, context: str = "") -> None:
        """Create a CONTAINS relationship between a chunk and a fact."""
        if not self._fact_ops:
            raise RuntimeError("Connection not initialized")
        return await self._fact_ops.create_chunk_contains_fact_relation(chunk_id, fact_id, context)

    # Fact operations delegation
    async def store_fact(self, fact: Fact) -> str:
        """Store a fact in Neo4J."""
        if not self._fact_ops:
            raise RuntimeError("Connection not initialized")
        return await self._fact_ops.store_fact(fact)

    async def store_facts(self, facts: List[Fact]) -> List[str]:
        """Store multiple facts in Neo4J."""
        if not self._fact_ops:
            raise RuntimeError("Connection not initialized")
        return await self._fact_ops.store_facts(facts)

    async def fix_unconnected_entities(self) -> int:
        """DEPRECATED: Find and fix entities that are not connected to any chunks."""
        if not self._entity_ops:
            raise RuntimeError("Connection not initialized")
        return await self._entity_ops.fix_unconnected_entities()

    # Relation operations delegation
    async def store_relation(self, relation: Relation) -> RelationId:
        """Store a relation in Neo4J."""
        if not self._relation_ops:
            raise RuntimeError("Connection not initialized")
        return await self._relation_ops.store_relation(relation)

    async def store_relations(self, relations: List[Relation]) -> List[RelationId]:
        """Store multiple relations in Neo4J."""
        if not self._relation_ops:
            raise RuntimeError("Connection not initialized")
        return await self._relation_ops.store_relations(relations)

    async def get_relation(self, relation_id: RelationId) -> Optional[Relation]:
        """Get a relation by ID."""
        if not self._relation_ops:
            raise RuntimeError("Connection not initialized")
        return await self._relation_ops.get_relation(relation_id)

    async def get_relations(self, relation_ids: List[RelationId]) -> List[Relation]:
        """Get multiple relations by IDs."""
        if not self._relation_ops:
            raise RuntimeError("Connection not initialized")
        return await self._relation_ops.get_relations(relation_ids)

    async def get_entity_relations(self, entity_id: EntityId, relation_type: Optional[str] = None, direction: str = "both") -> List[Relation]:
        """Get all relations for an entity."""
        if not self._relation_ops:
            raise RuntimeError("Connection not initialized")
        return await self._relation_ops.get_entity_relations(entity_id, relation_type, direction)

    async def get_all_relations(self) -> List[Relation]:
        """Get all relations from the storage."""
        if not self._relation_ops:
            raise RuntimeError("Connection not initialized")
        return await self._relation_ops.get_all_relations()

    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation."""
        if not self._relation_ops:
            raise RuntimeError("Connection not initialized")
        return await self._relation_ops.update_relation(relation)

    async def delete_relation(self, relation_id: RelationId) -> bool:
        """Delete a relation."""
        if not self._relation_ops:
            raise RuntimeError("Connection not initialized")
        return await self._relation_ops.delete_relation(relation_id)

    # Query operations delegation
    async def get_neighbors(self, entity_id: EntityId, relation_type: Optional[str] = None, max_depth: int = 1) -> List[Entity]:
        """Get neighboring entities."""
        if not self._query_ops:
            raise RuntimeError("Connection not initialized")
        return await self._query_ops.get_neighbors(entity_id, relation_type, max_depth)

    async def find_path(self, source_entity_id: EntityId, target_entity_id: EntityId, max_depth: int = 3) -> List[List[EntityId]]:
        """Find paths between two entities."""
        if not self._query_ops:
            raise RuntimeError("Connection not initialized")
        return await self._query_ops.find_path(source_entity_id, target_entity_id, max_depth)

    async def search_entities_by_content(self, search_term: str, entity_type: Optional[str] = None, limit: int = 10) -> List[Entity]:
        """Search entities by content using full-text search."""
        if not self._query_ops:
            raise RuntimeError("Connection not initialized")
        return await self._query_ops.search_entities_by_content(search_term, entity_type, limit)

    async def find_related_entities(self, entity_id: EntityId, relation_types: Optional[List[str]] = None, max_depth: int = 2, limit: int = 20) -> List[Dict[str, Any]]:
        """Find entities related to a given entity with relationship context."""
        if not self._query_ops:
            raise RuntimeError("Connection not initialized")
        return await self._query_ops.find_related_entities(entity_id, relation_types, max_depth, limit)

    async def get_entity_clusters(self, min_cluster_size: int = 3, max_clusters: int = 10) -> List[Dict[str, Any]]:
        """Find clusters of highly connected entities."""
        if not self._query_ops:
            raise RuntimeError("Connection not initialized")
        return await self._query_ops.get_entity_clusters(min_cluster_size, max_clusters)

    async def find_shortest_paths(self, source_entity_id: EntityId, target_entity_id: EntityId, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Find shortest paths between two entities with relationship details."""
        if not self._query_ops:
            raise RuntimeError("Connection not initialized")
        return await self._query_ops.find_shortest_paths(source_entity_id, target_entity_id, max_depth)

    # Graph operations delegation
    async def store_graph(self, graph: Graph) -> None:
        """Store an entire graph."""
        if not self._graph_ops:
            raise RuntimeError("Connection not initialized")
        return await self._graph_ops.store_graph(graph)

    async def get_graph(self, entity_ids: Optional[List[EntityId]] = None) -> Graph:
        """Get a graph or subgraph."""
        if not self._graph_ops:
            raise RuntimeError("Connection not initialized")
        return await self._graph_ops.get_graph(entity_ids)

    async def clear(self) -> None:
        """Clear all data from the storage."""
        if not self._graph_ops:
            raise RuntimeError("Connection not initialized")
        return await self._graph_ops.clear()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self._graph_ops:
            raise RuntimeError("Connection not initialized")
        return await self._graph_ops.get_statistics()

    async def test_connection(self) -> bool:
        """Test the database connection.

        Returns:
            True if connection is working
        """
        if not self._connection_ops:
            raise RuntimeError("Connection not initialized")

        # Use the connection operations to test the connection
        result = await self._connection_ops._execute_query("RETURN 1 as test", {})
        return len(result) > 0 and result[0].get("test") == 1

    async def get_graph_metrics(self) -> Dict[str, Any]:
        """Get advanced graph metrics."""
        if not self._graph_ops:
            raise RuntimeError("Connection not initialized")
        return await self._graph_ops.get_graph_metrics()

    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance by creating indexes and constraints."""
        if not self._graph_ops:
            raise RuntimeError("Connection not initialized")
        return await self._graph_ops.optimize_database()

    # OpenIE operations removed - LangExtract handles all extraction
