"""Entity operations for Neo4j storage."""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from ...models import Entity
from ...models.types import EntityId
from ...utils.id_generation import UnifiedIDGenerator, IDValidator
from .base_operations import BaseOperations

logger = logging.getLogger(__name__)


class EntityOperations(BaseOperations):
    """Handles entity storage, retrieval, and management operations."""
    
    async def store_entity(self, entity: Entity) -> EntityId:
        """Store an entity in Neo4J.

        Uses MERGE based on name only to prevent duplicate entities.
        If an entity with the same name exists, it will be updated with the
        highest confidence score and most recent type.

        Args:
            entity: Entity to store

        Returns:
            Entity ID
        """
        # Validate entity ID format
        if not IDValidator.is_valid_entity_id(entity.id):
            logger.warning(f"Invalid entity ID format: {entity.id}")
            # Generate a new valid ID
            entity.id = UnifiedIDGenerator.generate_entity_id(entity.name, entity.type)
            logger.info(f"Generated new entity ID: {entity.id}")

        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET 
            e.id = $id,
            e.type = $type,
            e.confidence = $confidence,
            e.metadata = $metadata,
            e.created_at = datetime(),
            e.updated_at = datetime()
        ON MATCH SET
            e.type = CASE 
                WHEN $confidence > coalesce(e.confidence, 0.0) THEN $type 
                ELSE e.type 
            END,
            e.confidence = CASE 
                WHEN $confidence > coalesce(e.confidence, 0.0) THEN $confidence 
                ELSE e.confidence 
            END,
            e.metadata = $metadata,
            e.updated_at = datetime()
        RETURN e.id as id, e.type as final_type
        """

        properties = entity.metadata.copy() if entity.metadata else {}
        
        result = await self._execute_query(query, {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "confidence": entity.confidence,
            "metadata": properties
        })

        if result:
            if result[0].get('final_type') != entity.type:
                logger.debug(f"Entity {entity.name} type updated from {entity.type} to {result[0].get('final_type')} "
                        f"due to higher confidence score. Entity ID: {result[0].get('entity_id')}")
            return result[0]["id"]
        return entity.id

    async def _create_missing_entity(self, entity_id: str, entity_name: str) -> str:
        """Create a missing entity with minimal information.

        Args:
            entity_id: ID of the entity to create
            entity_name: Name of the entity

        Returns:
            Entity ID
        """
        # Extract type from entity ID if possible
        entity_type = "unknown"
        if "_" in entity_id:
            parts = entity_id.split("_")
            if len(parts) >= 2:
                entity_type = parts[1]

        # Create minimal entity
        entity = Entity(
            id=entity_id,
            name=entity_name,
            type=entity_type,
            confidence=0.1,  # Low confidence for auto-created entities
            metadata={"auto_created": True, "created_at": datetime.now().isoformat()}
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
            List of entity IDs
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
        # First store the entity
        entity_id = await self.store_entity(entity)
        
        # Then create relationships to chunks
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
        MATCH (e:Entity {id: $entity_id}), (c:DocumentChunk {id: $chunk_id})
        MERGE (e)-[:MENTIONED_IN]->(c)
        RETURN e.id as entity_id, c.id as chunk_id
        """

        result = await self._execute_query(query, {
            "entity_id": entity_id,
            "chunk_id": chunk_id
        })

        if not result:
            logger.warning(f"Failed to create entity-chunk relationship: entity {entity_id} or chunk {chunk_id} not found")
        else:
            logger.debug(f"Created MENTIONED_IN relationship: entity {entity_id} -> chunk {chunk_id}")

    async def fix_unconnected_entities(self) -> int:
        """DEPRECATED: Find and fix entities that are not connected to any chunks.

        This method should no longer be needed with chunk-based extraction.
        Entities are now extracted directly from chunks and automatically connected.

        Returns:
            Number of entities that were fixed (always 0 now)
        """
        logger.info("fix_unconnected_entities called but is deprecated with chunk-based extraction")
        return 0
    
    async def get_entities_by_chunk_id(self, chunk_id: str) -> List[Entity]:
        """Get all entities mentioned in a specific chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            List of entities mentioned in the chunk
        """
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
        """

        result = await self._execute_query(query, {"entity_id": entity_id})

        return [record['chunk_id'] for record in result]

    async def get_document_chunks_by_entity_names(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """Get all DocumentChunk nodes related to specific entity names with full metadata.

        Since the current graph doesn't have direct entity-chunk relationships,
        we'll search for chunks that contain the entity names in their text.

        Args:
            entity_names: List of entity names to search for

        Returns:
            List of chunk data dictionaries with metadata
        """
        if not entity_names:
            return []

        # Create a case-insensitive regex pattern for each entity name
        # Use word boundaries to avoid partial matches
        patterns = [f"(?i)\\b{name.replace(' ', '\\s+')}\\b" for name in entity_names]
        pattern_string = "|".join(patterns)

        query = """
        MATCH (c:DocumentChunk)
        WHERE any(pattern IN $patterns WHERE c.content =~ pattern)
        RETURN c.id as chunk_id,
               c.document_id as document_id,
               c.chunk_index as chunk_index,
               c.content as content,
               c.start_char as start_char,
               c.end_char as end_char,
               c.metadata as metadata
        ORDER BY c.document_id, c.chunk_index
        """

        result = await self._execute_query(query, {"patterns": patterns})

        chunks = []
        for record in result:
            chunk_data = {
                "chunk_id": record["chunk_id"],
                "document_id": record["document_id"],
                "chunk_index": record["chunk_index"],
                "content": record["content"],
                "start_char": record["start_char"],
                "end_char": record["end_char"],
                "metadata": record["metadata"] or {}
            }
            chunks.append(chunk_data)

        return chunks
    
    async def update_entity_chunk_references(self, entity_id: EntityId, chunk_ids: List[str]) -> None:
        """Update entity's chunk references by replacing all existing relationships.

        Args:
            entity_id: Entity ID
            chunk_ids: New list of chunk IDs
        """
        # First remove all existing MENTIONED_IN relationships
        delete_query = """
        MATCH (e:Entity {id: $entity_id})-[r:MENTIONED_IN]->()
        DELETE r
        """
        await self._execute_query(delete_query, {"entity_id": entity_id})

        # Create new relationships
        for chunk_id in chunk_ids:
            await self._create_entity_chunk_relationship(entity_id, chunk_id)

    async def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID.

        Args:
            entity_id: ID of the entity to get

        Returns:
            Entity or None if not found
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
            List of entities (may be fewer than requested if some don't exist)
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
                node_data = record["e"]
                entities.append(Entity.from_neo4j_node(node_data))
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
                entities.append(Entity.from_neo4j_node(record["e"]))
            except Exception as e:
                logger.warning(f"Failed to parse entity: {e}")

        return entities

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search for entities by name or content.

        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        # Build the Cypher query
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
        ORDER BY
            CASE WHEN toLower(e.name) = toLower($query) THEN 0 ELSE 1 END,
            e.confidence DESC,
            e.name
        LIMIT $limit
        """

        result = await self._execute_query(cypher_query, parameters)

        entities = []
        for record in result:
            try:
                entities.append(Entity.from_neo4j_node(record["e"]))
            except Exception as e:
                logger.warning(
                    f"Failed to parse entity from search result: {e}",
                    extra={
                        "query": query,
                        "entity_type": entity_type,
                        "record": record,
                        "error": str(e)
                    }
                )

        return entities

    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity.

        Args:
            entity: Entity with updated data

        Returns:
            True if entity was updated, False otherwise
        """
        properties = entity.metadata.copy() if entity.metadata else {}

        query = """
        MATCH (e:Entity {id: $entity_id})
        SET e += $properties,
            e.updated_at = datetime()
        RETURN e
        """

        result = await self._execute_query(query, {
            "entity_id": entity.id,
            "properties": properties
        })

        return len(result) > 0

    async def delete_entity(self, entity_id: EntityId) -> bool:
        """Delete an entity and all its relations.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if entity was deleted, False otherwise
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        DETACH DELETE e
        RETURN count(e) as deleted_count
        """

        result = await self._execute_query(query, {"entity_id": entity_id})
        return result[0]["deleted_count"] > 0 if result else False

    async def get_entities_by_document(self, document_id: str) -> List[Entity]:
        """Get all entities associated with a document.

        Args:
            document_id: Document identifier

        Returns:
            List of entities found in the document
        """
        query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:DocumentChunk {document_id: $document_id})
        RETURN DISTINCT e
        ORDER BY e.name
        """

        result = await self._execute_query(query, {"document_id": document_id})

        entities = []
        for record in result:
            try:
                entities.append(Entity.from_neo4j_node(record["e"]))
            except Exception as e:
                logger.warning(f"Failed to parse entity: {e}")

        return entities

    async def create_chunk_mentions_entity_relation(self, chunk_id: str, entity_id: str, context: str) -> None:
        """Create a MENTIONS relationship between a chunk and an entity.

        Args:
            chunk_id: ID of the chunk
            entity_id: ID of the entity
            context: Context in which the entity is mentioned
        """
        query = """
        MATCH (c:DocumentChunk {id: $chunk_id}), (e:Entity {id: $entity_id})
        MERGE (c)-[r:MENTIONS]->(e)
        SET r.context = $context,
            r.created_at = datetime()
        """
        await self._execute_query(query, {
            "chunk_id": chunk_id,
            "entity_id": entity_id,
            "context": context
        })
