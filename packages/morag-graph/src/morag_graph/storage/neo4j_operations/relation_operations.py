"""Relation operations for Neo4j storage."""

import json
import logging
from typing import Dict, List, Optional, Any

from ...models import Relation
from ...models.types import EntityId, RelationId
from .base_operations import BaseOperations

logger = logging.getLogger(__name__)


class RelationOperations(BaseOperations):
    """Handles relation storage, retrieval, and management operations."""
    
    async def store_relation(self, relation: Relation) -> RelationId:
        """Store a relation in Neo4J.

        If either entity doesn't exist, it will be created automatically.

        Args:
            relation: Relation to store

        Returns:
            Relation ID
        """
        # First ensure both entities exist
        source_exists_query = "MATCH (e:Entity {id: $entity_id}) RETURN count(e) as count"
        target_exists_query = "MATCH (e:Entity {id: $entity_id}) RETURN count(e) as count"
        
        source_result = await self._execute_query(source_exists_query, {"entity_id": relation.source_entity_id})
        target_result = await self._execute_query(target_exists_query, {"entity_id": relation.target_entity_id})
        
        # Create missing entities if needed
        if source_result[0]["count"] == 0:
            logger.warning(f"Source entity {relation.source_entity_id} not found, creating placeholder")
            await self._create_missing_entity(relation.source_entity_id, f"Entity_{relation.source_entity_id}")
        
        if target_result[0]["count"] == 0:
            logger.warning(f"Target entity {relation.target_entity_id} not found, creating placeholder")
            await self._create_missing_entity(relation.target_entity_id, f"Entity_{relation.target_entity_id}")

        # Get the normalized Neo4j relationship type
        neo4j_rel_type = relation.get_neo4j_type()

        # Store the relation with dynamic relationship type
        # Note: We need to match entities by any label since they now have specific labels
        query = f"""
        MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
        MERGE (source)-[r:{neo4j_rel_type} {{
            id: $relation_id
        }}]->(target)
        SET r.type = $relation_type,
            r.confidence = $confidence,
            r.metadata = $metadata,
            r.created_at = coalesce(r.created_at, datetime()),
            r.updated_at = datetime()
        RETURN r.id as id
        """

        properties = relation.attributes.copy() if relation.attributes else {}

        result = await self._execute_query(query, {
            "source_id": relation.source_entity_id,
            "target_id": relation.target_entity_id,
            "relation_id": relation.id,
            "relation_type": relation.type,
            "confidence": relation.confidence,
            "metadata": json.dumps(properties) if properties else "{}"
        })

        if result:
            return result[0]["id"]
        else:
            logger.warning(f"Failed to store relation {relation.id}")
            return relation.id
    
    async def _create_missing_entity(self, entity_id: str, entity_name: str) -> str:
        """Create a missing entity with minimal information.

        Args:
            entity_id: ID of the entity to create (may be invalid format)
            entity_name: Name of the entity

        Returns:
            Entity ID (properly formatted)
        """
        from ...models import Entity
        from ...utils.id_generation import UnifiedIDGenerator, IDValidator
        from datetime import datetime

        # Check if the provided entity_id is in valid format
        try:
            IDValidator.validate_entity_id(entity_id)
            # If valid, use it as-is
            valid_entity_id = entity_id
        except:
            # If invalid, generate a proper unified ID
            logger.warning(f"Invalid entity ID format: {entity_id}, generating new unified ID")
            valid_entity_id = UnifiedIDGenerator.generate_entity_id(
                name=entity_name,
                entity_type="CUSTOM",  # Default type for auto-created entities
                source_doc_id=""  # No specific source document
            )
            logger.info(f"Generated new unified entity ID: {valid_entity_id}")

        # Extract type from entity ID if possible
        entity_type = "CUSTOM"
        if "_" in valid_entity_id:
            parts = valid_entity_id.split("_")
            if len(parts) >= 2:
                entity_type = parts[1].upper()

        # Create minimal entity with valid ID
        entity = Entity(
            name=entity_name,
            type=entity_type,
            confidence=0.1,  # Low confidence for auto-created entities
            attributes={"auto_created": True, "created_at": datetime.now().isoformat()}
        )

        # Store the entity using a simple query
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.confidence = $confidence,
            e.metadata = $metadata,
            e.created_at = coalesce(e.created_at, datetime()),
            e.updated_at = datetime()
        RETURN e.id as id
        """

        await self._execute_query(query, {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "confidence": entity.confidence,
            "metadata": json.dumps(entity.attributes) if entity.attributes else "{}"
        })

        logger.info(f"Created missing entity: {entity.id} (name: {entity_name})")
        return entity.id  # Return the valid entity ID, not the original invalid one
    
    async def store_relations(self, relations: List[Relation]) -> List[RelationId]:
        """Store multiple relations in Neo4J.
        
        Args:
            relations: List of relations to store
            
        Returns:
            List of relation IDs
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
            Relation or None if not found
        """
        query = """
        MATCH (source:Entity)-[r:RELATION {id: $relation_id}]->(target:Entity)
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query, {"relation_id": relation_id})
        
        if result:
            rel_data = result[0]["r"]
            # Parse metadata JSON string back to dict for attributes
            metadata_str = rel_data.get("metadata", "{}")
            try:
                attributes = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            except (json.JSONDecodeError, TypeError):
                attributes = {}

            return Relation(
                id=rel_data["id"],
                type=rel_data["type"],
                source_entity_id=result[0]["source_id"],
                target_entity_id=result[0]["target_id"],
                confidence=rel_data.get("confidence", 1.0),
                attributes=attributes
            )
        
        return None
    
    async def get_relations(self, relation_ids: List[RelationId]) -> List[Relation]:
        """Get multiple relations by IDs.
        
        Args:
            relation_ids: List of relation IDs to get
            
        Returns:
            List of relations (may be fewer than requested if some don't exist)
        """
        if not relation_ids:
            return []
        
        query = """
        MATCH (source:Entity)-[r:RELATION]->(target:Entity)
        WHERE r.id IN $relation_ids
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query, {"relation_ids": relation_ids})
        
        relations = []
        for record in result:
            try:
                rel_data = record["r"]
                # Parse metadata JSON string back to dict for attributes
                metadata_str = rel_data.get("metadata", "{}")
                try:
                    attributes = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except (json.JSONDecodeError, TypeError):
                    attributes = {}

                relations.append(Relation(
                    id=rel_data["id"],
                    type=rel_data["type"],
                    source_entity_id=record["source_id"],
                    target_entity_id=record["target_id"],
                    confidence=rel_data.get("confidence", 1.0),
                    attributes=attributes
                ))
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
            direction: Direction filter ("incoming", "outgoing", "both")
            
        Returns:
            List of relations
        """
        # Build query based on direction
        if direction == "outgoing":
            match_pattern = "(e:Entity {id: $entity_id})-[r:RELATION]->(target:Entity)"
            return_pattern = "r, e.id as source_id, target.id as target_id"
        elif direction == "incoming":
            match_pattern = "(source:Entity)-[r:RELATION]->(e:Entity {id: $entity_id})"
            return_pattern = "r, source.id as source_id, e.id as target_id"
        else:  # both
            match_pattern = "(e:Entity {id: $entity_id})-[r:RELATION]-(other:Entity)"
            return_pattern = """r, 
                CASE WHEN startNode(r).id = $entity_id 
                     THEN e.id 
                     ELSE other.id 
                END as source_id,
                CASE WHEN endNode(r).id = $entity_id 
                     THEN e.id 
                     ELSE other.id 
                END as target_id"""
        
        query = f"MATCH {match_pattern}"
        parameters = {"entity_id": entity_id}
        
        if relation_type:
            query += " AND r.type = $relation_type"
            parameters["relation_type"] = relation_type
        
        query += f" RETURN {return_pattern}"
        
        result = await self._execute_query(query, parameters)
        
        relations = []
        for record in result:
            try:
                rel_data = record["r"]
                # Parse metadata JSON string back to dict for attributes
                metadata_str = rel_data.get("metadata", "{}")
                try:
                    attributes = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except (json.JSONDecodeError, TypeError):
                    attributes = {}

                relations.append(Relation(
                    id=rel_data["id"],
                    type=rel_data["type"],
                    source_entity_id=record["source_id"],
                    target_entity_id=record["target_id"],
                    confidence=rel_data.get("confidence", 1.0),
                    attributes=attributes
                ))
            except Exception as e:
                logger.warning(f"Failed to parse relation: {e}")
        
        return relations
    
    async def get_all_relations(self) -> List[Relation]:
        """Get all relations from the storage.
        
        Returns:
            List of all relations
        """
        query = """
        MATCH (source:Entity)-[r:RELATION]->(target:Entity)
        RETURN r, source.id as source_id, target.id as target_id
        """
        
        result = await self._execute_query(query)
        
        relations = []
        for record in result:
            try:
                rel_data = record["r"]
                # Parse metadata JSON string back to dict for attributes
                metadata_str = rel_data.get("metadata", "{}")
                try:
                    attributes = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except (json.JSONDecodeError, TypeError):
                    attributes = {}

                relations.append(Relation(
                    id=rel_data["id"],
                    type=rel_data["type"],
                    source_entity_id=record["source_id"],
                    target_entity_id=record["target_id"],
                    confidence=rel_data.get("confidence", 1.0),
                    attributes=attributes
                ))
            except Exception as e:
                logger.warning(f"Failed to parse relation: {e}")
        
        return relations
    
    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation.
        
        Args:
            relation: Relation with updated data
            
        Returns:
            True if relation was updated, False otherwise
        """
        properties = relation.attributes.copy() if relation.attributes else {}

        query = """
        MATCH ()-[r:RELATION {id: $relation_id}]->()
        SET r.type = $type,
            r.confidence = $confidence,
            r.metadata = $metadata,
            r.updated_at = datetime()
        RETURN r
        """

        result = await self._execute_query(query, {
            "relation_id": relation.id,
            "type": relation.type,
            "confidence": relation.confidence,
            "metadata": json.dumps(properties) if properties else "{}"
        })
        
        return len(result) > 0
    
    async def delete_relation(self, relation_id: RelationId) -> bool:
        """Delete a relation.
        
        Args:
            relation_id: ID of the relation to delete
            
        Returns:
            True if relation was deleted, False otherwise
        """
        query = """
        MATCH ()-[r:RELATION {id: $relation_id}]->()
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await self._execute_query(query, {"relation_id": relation_id})
        return result[0]["deleted_count"] > 0 if result else False
