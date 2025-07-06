"""Neo4J storage backend for graph data."""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from pydantic import BaseModel

from ..models import Entity, Relation, Graph, Document, DocumentChunk
from ..models.types import EntityId, RelationId
from ..utils.id_generation import UnifiedIDGenerator, IDValidator
from .base import BaseStorage

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
    
    async def connect(self) -> None:
        """Connect to Neo4J database."""
        try:
            # Configure basic driver settings
            driver_kwargs = {
                "auth": (self.config.username, self.config.password),
                "max_connection_lifetime": self.config.max_connection_lifetime,
                "max_connection_pool_size": self.config.max_connection_pool_size,
                "connection_acquisition_timeout": self.config.connection_acquisition_timeout,
            }

            # For encrypted URI schemes (bolt+s://, bolt+ssc://, neo4j+s://, neo4j+ssc://),
            # encryption and trust settings are handled by the URI scheme itself
            # Only add SSL configuration for non-encrypted URIs
            if self.config.uri.startswith(('bolt://', 'neo4j://')) and not self.config.verify_ssl:
                # For non-encrypted URIs, add SSL context if needed
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                driver_kwargs["ssl_context"] = ssl_context

            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                **driver_kwargs
            )

            # Ensure database exists before testing connection
            await self._ensure_database_exists()

            # Test connection to the specific database
            async with self.driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")

            logger.info("Connected to Neo4J database")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4J: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Neo4J database."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4J database")

    async def _ensure_database_exists(self) -> None:
        """Ensure the specified database exists, create it if it doesn't."""
        try:
            # First try to connect to the system database to check/create the target database
            # This works in Neo4j Enterprise Edition
            async with self.driver.session(database="system") as session:
                # Check if database exists
                result = await session.run(
                    "SHOW DATABASES YIELD name WHERE name = $db_name",
                    {"db_name": self.config.database}
                )
                databases = await result.data()

                if not databases:
                    # Database doesn't exist, create it
                    logger.info(f"Creating Neo4j database: {self.config.database}")
                    await session.run(f"CREATE DATABASE `{self.config.database}`")
                    logger.info(f"Successfully created Neo4j database: {self.config.database}")
                else:
                    logger.info(f"Neo4j database already exists: {self.config.database}")

        except Exception as e:
            # If we can't create the database (e.g., Neo4j Community Edition or auth issues),
            # try to connect directly to the target database
            logger.warning(f"Could not ensure database exists via system database (this is normal for Neo4j Community Edition): {e}")

            # For Community Edition or when we can't access system database,
            # try to connect directly to the target database
            try:
                async with self.driver.session(database=self.config.database) as session:
                    await session.run("RETURN 1")
                logger.info(f"Successfully connected to existing Neo4j database: {self.config.database}")
            except Exception as direct_error:
                # If the database doesn't exist and we can't create it, suggest using 'neo4j' database
                if "DatabaseNotFound" in str(direct_error):
                    logger.error(f"Database '{self.config.database}' does not exist and cannot be created automatically. "
                               f"Please either: 1) Create the database manually, 2) Use the default 'neo4j' database, "
                               f"or 3) Use Neo4j Enterprise Edition for automatic database creation.")
                raise direct_error

    async def create_database_if_not_exists(self, database_name: str) -> bool:
        """
        Manually create a database if it doesn't exist.

        Args:
            database_name: Name of the database to create

        Returns:
            True if database was created or already exists, False if creation failed
        """
        try:
            async with self.driver.session(database="system") as session:
                # Check if database exists
                result = await session.run(
                    "SHOW DATABASES YIELD name WHERE name = $db_name",
                    {"db_name": database_name}
                )
                databases = await result.data()

                if not databases:
                    # Database doesn't exist, create it
                    logger.info(f"Creating Neo4j database: {database_name}")
                    await session.run(f"CREATE DATABASE `{database_name}`")
                    logger.info(f"Successfully created Neo4j database: {database_name}")
                    return True
                else:
                    logger.info(f"Neo4j database already exists: {database_name}")
                    return True

        except Exception as e:
            logger.error(f"Failed to create database {database_name}: {e}")
            return False

    async def _execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4J database")
        
        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(query, parameters or {})
            records = []
            async for record in result:
                records.append(record.data())
            return records
    
    async def store_document_with_unified_id(self, document: Document) -> str:
        """Store document with unified ID format.
        
        Args:
            document: Document instance with unified ID
            
        Returns:
            Document ID
        """
        # Validate ID format
        if not IDValidator.validate_document_id(document.id):
            raise ValueError(f"Invalid document ID format: {document.id}")
        
        # Store document with unified ID
        query = """
        MERGE (d:Document {id: $id})
        SET d.source_file = $source_file,
            d.file_name = $file_name,
            d.checksum = $checksum,
            d.ingestion_timestamp = $ingestion_timestamp,
            d.metadata = $metadata,
            d.unified_id_format = true
        RETURN d.id as document_id
        """
        
        result = await self._execute_query(
            query,
            {
                "id": document.id,
                "source_file": document.source_file,
                "file_name": document.file_name,
                "checksum": document.checksum,
                "ingestion_timestamp": document.ingestion_timestamp.isoformat(),
                "metadata": document.metadata or {}
            }
        )
        
        return result[0]['document_id']
    
    async def store_chunk_with_unified_id(self, chunk: DocumentChunk) -> str:
        """Store document chunk with unified ID format.
        
        Args:
            chunk: DocumentChunk instance with unified ID
            
        Returns:
            Chunk ID
        """
        # Validate ID formats
        if not IDValidator.validate_chunk_id(chunk.id):
            raise ValueError(f"Invalid chunk ID format: {chunk.id}")
        
        if not IDValidator.validate_document_id(chunk.document_id):
            raise ValueError(f"Invalid document ID format: {chunk.document_id}")
        
        # Verify document exists
        doc_check = await self._execute_query(
            "MATCH (d:Document {id: $doc_id}) RETURN d.id",
            {"doc_id": chunk.document_id}
        )
        
        if not doc_check:
            raise ValueError(f"Document {chunk.document_id} not found")
        
        # Store chunk with unified ID
        query = """
        MATCH (d:Document {id: $document_id})
        MERGE (c:DocumentChunk {id: $id})
        SET c.document_id = $document_id,
            c.chunk_index = $chunk_index,
            c.text = $text,
            c.metadata = $metadata,
            c.unified_id_format = true
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c.id as chunk_id
        """
        
        result = await self._execute_query(
            query,
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "metadata": chunk.metadata or {}
            }
        )
        
        return result[0]['chunk_id']
    
    async def get_document_by_unified_id(self, document_id: str) -> Optional[Document]:
        """Retrieve document by unified ID.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            Document instance or None
        """
        if not IDValidator.validate_document_id(document_id):
            raise ValueError(f"Invalid document ID format: {document_id}")
        
        query = """
        MATCH (d:Document {id: $id})
        RETURN d.id as id,
               d.source_file as source_file,
               d.file_name as file_name,
               d.checksum as checksum,
               d.ingestion_timestamp as ingestion_timestamp,
               d.metadata as metadata
        """
        
        result = await self._execute_query(query, {"id": document_id})
        
        if not result:
            return None
        
        doc_data = result[0]
        return Document(
            id=doc_data['id'],
            source_file=doc_data['source_file'],
            file_name=doc_data['file_name'],
            checksum=doc_data['checksum'],
            ingestion_timestamp=datetime.fromisoformat(doc_data['ingestion_timestamp']),
            metadata=doc_data['metadata']
        )
    
    async def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document by unified ID.
        
        Args:
            document_id: Unified document ID
            
        Returns:
            List of DocumentChunk instances
        """
        if not IDValidator.validate_document_id(document_id):
            raise ValueError(f"Invalid document ID format: {document_id}")
        
        query = """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
        RETURN c.id as id,
               c.document_id as document_id,
               c.chunk_index as chunk_index,
               c.text as text,
               c.metadata as metadata
        ORDER BY c.chunk_index
        """
        
        result = await self._execute_query(query, {"document_id": document_id})
        
        chunks = []
        for chunk_data in result:
            chunks.append(DocumentChunk(
                id=chunk_data['id'],
                document_id=chunk_data['document_id'],
                chunk_index=chunk_data['chunk_index'],
                text=chunk_data['text'],
                metadata=chunk_data['metadata']
            ))
        
        return chunks
    
    async def validate_id_consistency(self) -> Dict[str, Any]:
        """Validate ID consistency across the database.
        
        Returns:
            Validation report
        """
        # Check for documents with invalid ID formats
        doc_query = """
        MATCH (d:Document)
        WHERE NOT d.id STARTS WITH 'doc_'
        RETURN count(d) as invalid_docs, collect(d.id)[0..10] as sample_ids
        """
        
        # Check for chunks with invalid ID formats
        chunk_query = """
        MATCH (c:DocumentChunk)
        WHERE NOT c.id CONTAINS ':chunk:'
        RETURN count(c) as invalid_chunks, collect(c.id)[0..10] as sample_ids
        """
        
        # Check for orphaned chunks
        orphan_query = """
        MATCH (c:DocumentChunk)
        WHERE NOT EXISTS((c)<-[:HAS_CHUNK]-(:Document))
        RETURN count(c) as orphaned_chunks, collect(c.id)[0..10] as sample_ids
        """
        
        doc_result = await self._execute_query(doc_query)
        chunk_result = await self._execute_query(chunk_query)
        orphan_result = await self._execute_query(orphan_query)
        
        return {
            "invalid_documents": {
                "count": doc_result[0]['invalid_docs'],
                "sample_ids": doc_result[0]['sample_ids']
            },
            "invalid_chunks": {
                "count": chunk_result[0]['invalid_chunks'],
                "sample_ids": chunk_result[0]['sample_ids']
            },
            "orphaned_chunks": {
                "count": orphan_result[0]['orphaned_chunks'],
                "sample_ids": orphan_result[0]['sample_ids']
            }
        }
    
    async def store_entity(self, entity: Entity) -> EntityId:
        """Store an entity in Neo4J.
        
        Uses MERGE based on name and type to prevent duplicate entities.
        If an entity with the same name and type exists, it will be updated
        with new attributes and the highest confidence score.
        
        Args:
            entity: Entity to store
            
        Returns:
            ID of the stored entity
        """
        properties = entity.to_neo4j_node()
        labels = properties.pop('_labels', ['Entity'])
        
        # Create labels string for Cypher
        labels_str = ':'.join(labels)
        
        # MERGE based on name and type to prevent duplicates
        query = f"""
        MERGE (e:{labels_str} {{name: $name, type: $type}})
        ON CREATE SET e += $properties
        ON MATCH SET 
            e.confidence = CASE WHEN e.confidence > $confidence THEN e.confidence ELSE $confidence END,
            e.attributes = $attributes,
            e.source_doc_id = CASE WHEN e.source_doc_id IS NULL THEN $source_doc_id ELSE e.source_doc_id END,
            e.id = $id
        RETURN e.id as id
        """
        
        parameters = {
            "name": entity.name,
            "type": entity.type.value if hasattr(entity.type, 'value') else str(entity.type),  # Handle both enum and string types
            "confidence": entity.confidence,
            "attributes": properties.get('attributes', '{}'),
            "source_doc_id": entity.source_doc_id,
            "id": entity.id,
            "properties": properties
        }
        
        result = await self._execute_query(query, parameters)
        return result[0]["id"] if result else entity.id

    async def _create_missing_entity(self, entity_id: str, entity_name: str) -> str:
        """Create a missing entity with minimal information.

        Args:
            entity_id: ID of the entity to create
            entity_name: Name of the entity

        Returns:
            ID of the created entity
        """
        from morag_graph.models.entity import Entity

        # Extract entity name from ID if not provided
        if not entity_name and entity_id.startswith('ent_'):
            # Try to extract name from entity ID
            parts = entity_id.split('_')
            if len(parts) >= 2:
                entity_name = ' '.join(parts[1:-1]).replace('_', ' ').title()

        if not entity_name:
            entity_name = entity_id

        # Create a minimal entity
        entity = Entity(
            id=entity_id,
            name=entity_name,
            type="CUSTOM",
            confidence=0.5,  # Lower confidence for auto-created entities
            attributes={
                "auto_created": True,
                "creation_reason": "missing_entity_for_relation"
            }
        )

        # Store the entity
        await self.store_entity(entity)
        logger.info(f"Created missing entity: {entity_id} (name: {entity_name})")
        return entity_id
    
    async def store_entities(self, entities: List[Entity]) -> List[EntityId]:
        """Store multiple entities in Neo4J.
        
        Uses MERGE based on name and type to prevent duplicate entities.
        Falls back to individual entity storage for proper deduplication.
        
        Args:
            entities: List of entities to store
            
        Returns:
            List of IDs of the stored entities
        """
        if not entities:
            return []
        
        # Use individual store_entity calls to ensure proper MERGE logic
        # This is more reliable than batch operations for deduplication
        return [await self.store_entity(entity) for entity in entities]
    
    async def store_entity_with_chunk_references(self, entity: Entity, chunk_ids: List[str]) -> EntityId:
        """Store entity with references to chunks where it's mentioned.
        
        Args:
            entity: Entity instance
            chunk_ids: List of chunk IDs where entity is mentioned
            
        Returns:
            Entity ID
        """
        # Validate chunk IDs
        for chunk_id in chunk_ids:
            if not IDValidator.validate_chunk_id(chunk_id):
                raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        # Add chunk references to entity
        for chunk_id in chunk_ids:
            entity.add_chunk_reference(chunk_id)
        
        # Store the entity
        entity_id = await self.store_entity(entity)
        
        # Create relationships with chunks
        for chunk_id in chunk_ids:
            await self._create_entity_chunk_relationship(entity_id, chunk_id)
        
        return entity_id
    
    async def _create_entity_chunk_relationship(self, entity_id: EntityId, chunk_id: str) -> None:
        """Create MENTIONED_IN relationship between entity and chunk.
        
        Args:
            entity_id: Entity ID
            chunk_id: Chunk ID
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (c:DocumentChunk {id: $chunk_id})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """
        
        await self._execute_query(
            query,
            {
                "entity_id": entity_id,
                "chunk_id": chunk_id
            }
        )
    
    async def get_entities_by_chunk_id(self, chunk_id: str) -> List[Entity]:
        """Get all entities mentioned in a specific chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            List of Entity instances
        """
        if not IDValidator.validate_chunk_id(chunk_id):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:DocumentChunk {id: $chunk_id})
        RETURN e
        """
        
        result = await self._execute_query(query, {"chunk_id": chunk_id})
        
        entities = []
        for record in result:
            entity_data = dict(record['e'])
            entities.append(Entity.from_neo4j_node(entity_data))
        
        return entities
    
    async def get_chunks_by_entity_id(self, entity_id: EntityId) -> List[str]:
        """Get all chunk IDs where an entity is mentioned.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            List of chunk IDs
        """
        query = """
        MATCH (e:Entity {id: $entity_id})-[:MENTIONED_IN]->(c:DocumentChunk)
        RETURN c.id as chunk_id
        ORDER BY c.chunk_index
        """
        
        result = await self._execute_query(query, {"entity_id": entity_id})
        
        return [record['chunk_id'] for record in result]
    
    async def update_entity_chunk_references(self, entity_id: EntityId, chunk_ids: List[str]) -> None:
        """Update entity's chunk references by replacing all existing relationships.
        
        Args:
            entity_id: Entity ID
            chunk_ids: New list of chunk IDs
        """
        # Validate chunk IDs
        for chunk_id in chunk_ids:
            if not IDValidator.validate_chunk_id(chunk_id):
                raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        # Remove all existing relationships
        delete_query = """
        MATCH (e:Entity {id: $entity_id})-[r:MENTIONED_IN]->()
        DELETE r
        """
        
        await self._execute_query(delete_query, {"entity_id": entity_id})
        
        # Create new relationships
        for chunk_id in chunk_ids:
            await self._create_entity_chunk_relationship(entity_id, chunk_id)
        
        # Update entity's mentioned_in_chunks field
        entity = await self.get_entity(entity_id)
        if entity:
            entity.mentioned_in_chunks = set(chunk_ids)
            await self.store_entity(entity)
    
    async def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID.
        
        Args:
            entity_id: ID of the entity to get
            
        Returns:
            Entity if found, None otherwise
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e
        """
        
        result = await self._execute_query(query, {"entity_id": entity_id})
        
        if result:
            node_data = result[0]["e"]
            return Entity.from_neo4j_node(node_data)
        
        return None
    
    async def get_entities(self, entity_ids: List[EntityId]) -> List[Entity]:
        """Get multiple entities by IDs.
        
        Args:
            entity_ids: List of entity IDs to get
            
        Returns:
            List of entities
        """
        if not entity_ids:
            return []
        
        query = """
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        RETURN e
        """
        
        result = await self._execute_query(query, {"entity_ids": entity_ids})
        
        entities = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["e"])
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity from Neo4J: {e}")
        
        return entities
    
    async def get_all_entities(self) -> List[Entity]:
        """Get all entities from the storage.
        
        Returns:
            List of all entities
        """
        query = "MATCH (e:Entity) RETURN e"
        result = await self._execute_query(query)
        
        entities = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["e"])
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity: {e}")
        
        return entities
    
    async def search_entities(
        self, 
        query: str, 
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search for entities by name or attributes.
        
        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        cypher_query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
        """
        
        parameters = {"query": query, "limit": limit}
        
        if entity_type:
            cypher_query += " AND e.type = $entity_type"
            parameters["entity_type"] = entity_type
        
        cypher_query += """
        RETURN e
        ORDER BY e.name
        LIMIT $limit
        """
        
        result = await self._execute_query(cypher_query, parameters)
        
        entities = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["e"])
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity from search: {e}")
        
        return entities
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity.
        
        Args:
            entity: Entity with updated data
            
        Returns:
            True if entity was updated, False if not found
        """
        properties = entity.to_neo4j_node()
        properties.pop('_labels', None)  # Don't update labels
        
        query = """
        MATCH (e:Entity {id: $id})
        SET e += $properties
        RETURN e.id as id
        """
        
        result = await self._execute_query(query, {
            "id": entity.id,
            "properties": properties
        })
        
        return len(result) > 0
    
    async def delete_entity(self, entity_id: EntityId) -> bool:
        """Delete an entity and all its relations.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        DETACH DELETE e
        RETURN count(e) as deleted_count
        """
        
        result = await self._execute_query(query, {"entity_id": entity_id})
        return result[0]["deleted_count"] > 0 if result else False
    
    async def store_relation(self, relation: Relation) -> RelationId:
        """Store a relation in Neo4J.

        If either entity doesn't exist, it will be created automatically.

        Args:
            relation: Relation to store

        Returns:
            ID of the stored relation
        """
        properties = relation.to_neo4j_relationship()
        relation_type = relation.get_neo4j_type()

        # First check if both entities exist
        check_query = """
        OPTIONAL MATCH (source:Entity {id: $source_id})
        OPTIONAL MATCH (target:Entity {id: $target_id})
        RETURN source.id as source_exists, target.id as target_exists
        """

        check_result = await self._execute_query(check_query, {
            "source_id": relation.source_entity_id,
            "target_id": relation.target_entity_id
        })

        if not check_result:
            logger.warning(f"Failed to check entity existence for relation {relation.id}")
            return relation.id

        source_exists = check_result[0]["source_exists"] is not None
        target_exists = check_result[0]["target_exists"] is not None

        # Create missing entities automatically
        if not source_exists:
            logger.info(f"Creating missing source entity {relation.source_entity_id} for relation {relation.id}")
            await self._create_missing_entity(relation.source_entity_id, relation.attributes.get('source_entity_name', ''))

        if not target_exists:
            logger.info(f"Creating missing target entity {relation.target_entity_id} for relation {relation.id}")
            await self._create_missing_entity(relation.target_entity_id, relation.attributes.get('target_entity_name', ''))

        # Both entities now exist (or have been created), create the relation
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{relation_type} {{id: $relation_id}}]->(target)
        SET r += $properties
        RETURN r.id as id
        """

        parameters = {
            "source_id": relation.source_entity_id,
            "target_id": relation.target_entity_id,
            "relation_id": relation.id,
            "properties": properties
        }

        result = await self._execute_query(query, parameters)
        if result:
            logger.debug(f"Successfully stored relation {relation.id}: {relation.source_entity_id} -> {relation.target_entity_id}")
            return result[0]["id"]
        else:
            logger.warning(f"Failed to store relation {relation.id}")
            return relation.id
    
    async def store_relations(self, relations: List[Relation]) -> List[RelationId]:
        """Store multiple relations in Neo4J.
        
        Args:
            relations: List of relations to store
            
        Returns:
            List of IDs of the stored relations
        """
        if not relations:
            return []
        
        # For now, use individual inserts (could be optimized with batch operations)
        return [await self.store_relation(relation) for relation in relations]
    
    async def get_relation(self, relation_id: RelationId) -> Optional[Relation]:
        """Get a relation by ID.
        
        Args:
            relation_id: ID of the relation to get
            
        Returns:
            Relation if found, None otherwise
        """
        query = """
        MATCH (source)-[r {id: $relation_id}]->(target)
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query, {"relation_id": relation_id})
        
        if result:
            record = result[0]
            return Relation.from_neo4j_relationship(
                record["r"], 
                record["source_id"], 
                record["target_id"]
            )
        
        return None
    
    async def get_relations(self, relation_ids: List[RelationId]) -> List[Relation]:
        """Get multiple relations by IDs.
        
        Args:
            relation_ids: List of relation IDs to get
            
        Returns:
            List of relations
        """
        if not relation_ids:
            return []
        
        query = """
        MATCH (source)-[r]->(target)
        WHERE r.id IN $relation_ids
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query, {"relation_ids": relation_ids})
        
        relations = []
        for record in result:
            try:
                relation = Relation.from_neo4j_relationship(
                    record["r"], 
                    record["source_id"], 
                    record["target_id"]
                )
                relations.append(relation)
            except Exception as e:
                logger.warning(f"Failed to parse relation from Neo4J: {e}")
        
        return relations
    
    async def get_entity_relations(
        self, 
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Relation]:
        """Get all relations for an entity.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional relation type filter
            direction: Direction of relations ("in", "out", "both")
            
        Returns:
            List of relations involving the entity
        """
        if direction == "out":
            pattern = "(e)-[r]->(target)"
            return_clause = "r, e.id as source_id, target.id as target_id"
        elif direction == "in":
            pattern = "(source)-[r]->(e)"
            return_clause = "r, source.id as source_id, e.id as target_id"
        else:  # both
            pattern = "(n1)-[r]-(e)-[r2]-(n2)"
            return_clause = "r, n1.id as source_id, e.id as target_id"
        
        query = f"""
        MATCH (e:Entity {{id: $entity_id}})
        MATCH {pattern}
        """
        
        parameters = {"entity_id": entity_id}
        
        if relation_type:
            query = query.replace("[r]", f"[r:{relation_type}]")
        
        query += f" RETURN {return_clause}"
        
        result = await self._execute_query(query, parameters)
        
        relations = []
        for record in result:
            try:
                relation = Relation.from_neo4j_relationship(
                    record["r"], 
                    record["source_id"], 
                    record["target_id"]
                )
                relations.append(relation)
            except Exception as e:
                logger.warning(f"Failed to parse relation: {e}")
        
        return relations
    
    async def get_all_relations(self) -> List[Relation]:
        """Get all relations from the storage.
        
        Returns:
            List of all relations
        """
        query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query)
        
        relations = []
        for record in result:
            try:
                relation = Relation.from_neo4j_relationship(
                    record["r"], 
                    record["source_id"], 
                    record["target_id"]
                )
                relations.append(relation)
            except Exception as e:
                logger.warning(f"Failed to parse relation: {e}")
        
        return relations
    
    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation.
        
        Args:
            relation: Relation with updated data
            
        Returns:
            True if relation was updated, False if not found
        """
        properties = relation.to_neo4j_relationship()
        
        query = """
        MATCH ()-[r {id: $id}]->()
        SET r += $properties
        RETURN r.id as id
        """
        
        result = await self._execute_query(query, {
            "id": relation.id,
            "properties": properties
        })
        
        return len(result) > 0
    
    async def delete_relation(self, relation_id: RelationId) -> bool:
        """Delete a relation.
        
        Args:
            relation_id: ID of the relation to delete
            
        Returns:
            True if relation was deleted, False if not found
        """
        query = """
        MATCH ()-[r {id: $relation_id}]->()
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await self._execute_query(query, {"relation_id": relation_id})
        return result[0]["deleted_count"] > 0 if result else False
    
    async def get_neighbors(
        self, 
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Entity]:
        """Get neighboring entities.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional relation type filter
            max_depth: Maximum traversal depth
            
        Returns:
            List of neighboring entities
        """
        relation_filter = f":{relation_type}" if relation_type else ""
        
        query = f"""
        MATCH (e:Entity {{id: $entity_id}})
        MATCH (e)-[{relation_filter}*1..{max_depth}]-(neighbor:Entity)
        WHERE neighbor.id <> e.id
        RETURN DISTINCT neighbor
        """
        
        result = await self._execute_query(query, {"entity_id": entity_id})
        
        neighbors = []
        for record in result:
            try:
                entity = Entity.from_neo4j_node(record["neighbor"])
                neighbors.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse neighbor entity: {e}")
        
        return neighbors
    
    async def find_path(
        self, 
        source_entity_id: EntityId,
        target_entity_id: EntityId,
        max_depth: int = 3
    ) -> List[List[EntityId]]:
        """Find paths between two entities.
        
        Args:
            source_entity_id: ID of the source entity
            target_entity_id: ID of the target entity
            max_depth: Maximum path length
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        query = f"""
        MATCH path = (source:Entity {{id: $source_id}})
        -[*1..{max_depth}]-
        (target:Entity {{id: $target_id}})
        RETURN [node in nodes(path) | node.id] as path_ids
        LIMIT 10
        """
        
        result = await self._execute_query(query, {
            "source_id": source_entity_id,
            "target_id": target_entity_id
        })
        
        return [record["path_ids"] for record in result]
    
    async def store_graph(self, graph: Graph) -> None:
        """Store an entire graph.
        
        Args:
            graph: Graph to store
        """
        # Store entities first
        if graph.entities:
            await self.store_entities(list(graph.entities.values()))
        
        # Then store relations
        if graph.relations:
            await self.store_relations(list(graph.relations.values()))
    
    async def get_graph(
        self, 
        entity_ids: Optional[List[EntityId]] = None
    ) -> Graph:
        """Get a graph or subgraph.
        
        Args:
            entity_ids: Optional list of entity IDs to include (None for all)
            
        Returns:
            Graph containing the requested entities and their relations
        """
        graph = Graph()
        
        # Get entities
        if entity_ids:
            entities = await self.get_entities(entity_ids)
        else:
            # Get all entities
            query = "MATCH (e:Entity) RETURN e"
            result = await self._execute_query(query)
            entities = []
            for record in result:
                try:
                    entity = Entity.from_neo4j_node(record["e"])
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to parse entity: {e}")
        
        # Add entities to graph
        for entity in entities:
            graph.add_entity(entity)
        
        # Get relations between these entities
        if graph.entities:
            entity_id_list = list(graph.entities.keys())
            query = """
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE source.id IN $entity_ids AND target.id IN $entity_ids
            RETURN r, source.id as source_id, target.id as target_id
            """
            
            result = await self._execute_query(query, {"entity_ids": entity_id_list})
            
            for record in result:
                try:
                    relation = Relation.from_neo4j_relationship(
                        record["r"], 
                        record["source_id"], 
                        record["target_id"]
                    )
                    graph.add_relation(relation)
                except Exception as e:
                    logger.warning(f"Failed to parse relation: {e}")
        
        return graph
    
    async def clear(self) -> None:
        """Clear all data from the storage."""
        query = "MATCH (n) DETACH DELETE n"
        await self._execute_query(query)
        logger.info("Cleared all data from Neo4J database")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary containing statistics
        """
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_types": """
                MATCH (n) 
                RETURN labels(n)[0] as type, count(n) as count 
                ORDER BY count DESC
            """,
            "relationship_types": """
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count 
                ORDER BY count DESC
            """,
            "entity_count": "MATCH (e:Entity) RETURN count(e) as count",
            "document_count": "MATCH (d:Document) RETURN count(d) as count",
            "chunk_count": "MATCH (c:DocumentChunk) RETURN count(c) as count",
            "entity_types": """
                MATCH (e:Entity) 
                RETURN e.type as type, count(e) as count 
                ORDER BY count DESC
            """
        }
        
        stats = {}
        
        for stat_name, query in queries.items():
            try:
                result = await self._execute_query(query)
                if stat_name in ["total_nodes", "total_relationships", "entity_count", "document_count", "chunk_count"]:
                    stats[stat_name] = result[0]["count"] if result else 0
                elif stat_name in ["node_types", "relationship_types", "entity_types"]:
                    # Convert list of dicts to dict for easier reading
                    stats[stat_name] = {item["type"]: item["count"] for item in result} if result else {}
                else:
                    stats[stat_name] = result
            except Exception as e:
                logger.warning(f"Failed to get {stat_name}: {e}")
                stats[stat_name] = 0 if "count" in stat_name else {}
        
        return stats
    
    async def store_document(self, document: Document) -> str:
        """Store a document in Neo4J.
        
        Args:
            document: Document to store
            
        Returns:
            ID of the stored document
        """
        properties = document.to_neo4j_node()
        labels = properties.pop('_labels', ['Document'])
        
        # Create labels string for Cypher
        labels_str = ':'.join(labels)
        
        query = f"""
        MERGE (d:{labels_str} {{id: $id}})
        SET d += $properties
        RETURN d.id as id
        """
        
        parameters = {
            "id": document.id,
            "properties": properties
        }
        
        result = await self._execute_query(query, parameters)
        return result[0]["id"] if result else document.id
    
    async def store_document_chunk(self, chunk: DocumentChunk) -> str:
        """Store a document chunk in Neo4J.
        
        Args:
            chunk: DocumentChunk to store
            
        Returns:
            ID of the stored chunk
        """
        properties = chunk.to_neo4j_node()
        labels = properties.pop('_labels', ['DocumentChunk'])
        
        # Create labels string for Cypher
        labels_str = ':'.join(labels)
        
        query = f"""
        MERGE (c:{labels_str} {{id: $id}})
        SET c += $properties
        RETURN c.id as id
        """
        
        parameters = {
            "id": chunk.id,
            "properties": properties
        }
        
        result = await self._execute_query(query, parameters)
        return result[0]["id"] if result else chunk.id
    
    async def create_document_contains_chunk_relation(self, document_id: str, chunk_id: str) -> None:
        """Create a CONTAINS relationship between a document and a chunk.
        
        Args:
            document_id: ID of the document
            chunk_id: ID of the chunk
        """
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (c:DocumentChunk {id: $chunk_id})
        MERGE (d)-[:CONTAINS]->(c)
        """
        
        await self._execute_query(query, {
            "document_id": document_id,
            "chunk_id": chunk_id
        })
    
    async def create_chunk_mentions_entity_relation(self, chunk_id: str, entity_id: str, context: str) -> None:
        """Create a MENTIONS relationship between a chunk and an entity.
        
        Args:
            chunk_id: ID of the chunk
            entity_id: ID of the entity
            context: Context where the entity is mentioned
        """
        query = """
        MATCH (c:DocumentChunk {id: $chunk_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (c)-[r:MENTIONS]->(e)
        SET r.context = $context
        """
        
        await self._execute_query(query, {
            "chunk_id": chunk_id,
            "entity_id": entity_id,
            "context": context
        })
    
    # Checksum management methods
    
    async def get_document_checksum(self, document_id: str) -> Optional[str]:
        """Get stored checksum for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Stored checksum if found, None otherwise
        """
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d.checksum as checksum
        """
        
        result = await self._execute_query(query, {"document_id": document_id})
        return result[0]["checksum"] if result and result[0]["checksum"] else None
    
    async def store_document_checksum(self, document_id: str, checksum: str) -> None:
        """Store document checksum.
        
        Args:
            document_id: Document identifier
            checksum: Document checksum to store
        """
        query = """
        MERGE (d:Document {id: $document_id})
        SET d.checksum = $checksum, d.checksum_updated = datetime()
        """
        
        await self._execute_query(query, {
            "document_id": document_id,
            "checksum": checksum
        })
    
    async def delete_document_checksum(self, document_id: str) -> None:
        """Remove stored checksum for a document.
        
        Args:
            document_id: Document identifier
        """
        query = """
        MATCH (d:Document {id: $document_id})
        REMOVE d.checksum, d.checksum_updated
        """
        
        await self._execute_query(query, {"document_id": document_id})
    
    async def get_entities_by_document(self, document_id: str) -> List[Entity]:
        """Get all entities associated with a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of entities from this document
        """
        query = """
        MATCH (d:Document {id: $document_id})-[:CONTAINS]->(c:DocumentChunk)-[:MENTIONS]->(e:Entity)
        RETURN DISTINCT e
        UNION
        MATCH (e:Entity {source_doc_id: $document_id})
        RETURN e
        """
        
        result = await self._execute_query(query, {"document_id": document_id})
        entities = []
        
        for record in result:
            entity_data = dict(record["e"])
            # Convert Neo4j node to Entity model
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                type=entity_data["type"],
                description=entity_data.get("description"),
                properties=entity_data.get("properties", {}),
                source_doc_id=entity_data.get("source_doc_id")
            )
            entities.append(entity)
        
        return entities