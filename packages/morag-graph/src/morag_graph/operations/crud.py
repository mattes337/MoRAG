"""CRUD operations for graph entities and relations.

This module provides high-level CRUD (Create, Read, Update, Delete)
operations for graph entities and relations.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from ..models import Entity, Relation, Graph
from ..storage.base import BaseStorage
from ..storage.neo4j_storage import Neo4jStorage

logger = logging.getLogger(__name__)


class GraphCRUD:
    """High-level CRUD operations for graph data.
    
    This class provides convenient methods for creating, reading,
    updating, and deleting entities and relations in the graph.
    """
    
    def __init__(self, storage: BaseStorage):
        """Initialize GraphCRUD with a storage backend.
        
        Args:
            storage: Storage backend (Neo4j, JSON, etc.)
        """
        self.storage = storage
        self.logger = logger.getChild(self.__class__.__name__)
    
    async def create_entity(self, entity: Entity) -> Entity:
        """Create or update an entity in the graph.

        Uses MERGE strategy to handle entity deduplication by name.
        If an entity with the same name exists, it will be updated with
        the highest confidence score and most recent type.

        Args:
            entity: Entity to create or update

        Returns:
            Created or updated entity with any generated fields
        """
        self.logger.info(f"Creating/updating entity: {entity.name} ({entity.type})")

        # Use storage layer's MERGE strategy for deduplication
        await self.storage.store_entity(entity)
        self.logger.debug(f"Entity created/updated successfully: {entity.id}")
        return entity
    
    async def create_relation(self, relation: Relation) -> Relation:
        """Create a new relation in the graph.
        
        Args:
            relation: Relation to create
            
        Returns:
            Created relation with any generated fields
            
        Raises:
            ValueError: If source or target entities don't exist
        """
        self.logger.info(f"Creating relation: {relation.source_id} -> {relation.target_id} ({relation.type})")
        
        # Verify source and target entities exist
        source_entity = await self.get_entity(relation.source_id)
        target_entity = await self.get_entity(relation.target_id)
        
        if not source_entity:
            raise ValueError(f"Source entity {relation.source_id} not found")
        if not target_entity:
            raise ValueError(f"Target entity {relation.target_id} not found")
        
        await self.storage.store_relation(relation)
        self.logger.debug(f"Relation created successfully: {relation.id}")
        return relation
    
    async def get_entity(self, entity_id: Union[str, UUID]) -> Optional[Entity]:
        """Get an entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity if found, None otherwise
        """
        entity_id_str = str(entity_id)
        entities = await self.storage.get_all_entities()
        
        for entity in entities:
            if str(entity.id) == entity_id_str:
                return entity
        return None
    
    async def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by name.
        
        Args:
            name: Entity name
            
        Returns:
            Entity if found, None otherwise
        """
        entities = await self.storage.get_entities()
        
        for entity in entities:
            if entity.name == name:
                return entity
        return None
    
    async def get_entities_by_type(self, entity_type: str, limit: Optional[int] = None) -> List[Entity]:
        """Get entities by type.
        
        Args:
            entity_type: Type of entities to retrieve
            limit: Maximum number of entities to return
            
        Returns:
            List of entities of the specified type
        """
        entities = await self.storage.get_all_entities()
        filtered = [e for e in entities if e.type == entity_type]
        
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    async def get_relation(self, relation_id: Union[str, UUID]) -> Optional[Relation]:
        """Get a relation by ID.
        
        Args:
            relation_id: Relation ID
            
        Returns:
            Relation if found, None otherwise
        """
        relation_id_str = str(relation_id)
        relations = await self.storage.get_all_relations()
        
        for relation in relations:
            if str(relation.id) == relation_id_str:
                return relation
        return None
    
    async def get_entity_relations(self, entity_id: Union[str, UUID], 
                                 direction: str = "both") -> List[Relation]:
        """Get all relations for an entity.
        
        Args:
            entity_id: Entity ID
            direction: Direction of relations ("incoming", "outgoing", "both")
            
        Returns:
            List of relations involving the entity
        """
        entity_id_str = str(entity_id)
        relations = await self.storage.get_all_relations()
        
        if direction == "incoming":
            return [r for r in relations if str(r.target_id) == entity_id_str]
        elif direction == "outgoing":
            return [r for r in relations if str(r.source_id) == entity_id_str]
        else:  # both
            return [r for r in relations if 
                   str(r.source_id) == entity_id_str or str(r.target_id) == entity_id_str]
    
    async def update_entity(self, entity_id: Union[str, UUID], 
                          updates: Dict[str, Any]) -> Optional[Entity]:
        """Update an entity.
        
        Args:
            entity_id: Entity ID
            updates: Dictionary of fields to update
            
        Returns:
            Updated entity if found, None otherwise
        """
        entity = await self.get_entity(entity_id)
        if not entity:
            return None
        
        # Update entity fields
        for field, value in updates.items():
            if hasattr(entity, field):
                setattr(entity, field, value)
        
        # Store updated entity
        await self.storage.store_entity(entity)
        self.logger.info(f"Entity updated: {entity.id}")
        return entity
    
    async def update_relation(self, relation_id: Union[str, UUID], 
                            updates: Dict[str, Any]) -> Optional[Relation]:
        """Update a relation.
        
        Args:
            relation_id: Relation ID
            updates: Dictionary of fields to update
            
        Returns:
            Updated relation if found, None otherwise
        """
        relation = await self.get_relation(relation_id)
        if not relation:
            return None
        
        # Update relation fields
        for field, value in updates.items():
            if hasattr(relation, field):
                setattr(relation, field, value)
        
        # Store updated relation
        await self.storage.store_relation(relation)
        self.logger.info(f"Relation updated: {relation.id}")
        return relation
    
    async def delete_entity(self, entity_id: Union[str, UUID], 
                          cascade: bool = True) -> bool:
        """Delete an entity.
        
        Args:
            entity_id: Entity ID
            cascade: If True, also delete all relations involving this entity
            
        Returns:
            True if entity was deleted, False if not found
        """
        entity = await self.get_entity(entity_id)
        if not entity:
            return False
        
        if cascade:
            # Delete all relations involving this entity
            relations = await self.get_entity_relations(entity_id)
            for relation in relations:
                await self.delete_relation(relation.id)
        
        # Delete the entity (implementation depends on storage backend)
        if isinstance(self.storage, Neo4jStorage):
            query = "MATCH (e:Entity {id: $entity_id}) DELETE e"
            await self.storage._execute_query(query, {"entity_id": str(entity_id)})
        
        self.logger.info(f"Entity deleted: {entity_id}")
        return True
    
    async def delete_relation(self, relation_id: Union[str, UUID]) -> bool:
        """Delete a relation.
        
        Args:
            relation_id: Relation ID
            
        Returns:
            True if relation was deleted, False if not found
        """
        relation = await self.get_relation(relation_id)
        if not relation:
            return False
        
        # Delete the relation (implementation depends on storage backend)
        if isinstance(self.storage, Neo4jStorage):
            query = "MATCH ()-[r {id: $relation_id}]-() DELETE r"
            await self.storage._execute_query(query, {"relation_id": str(relation_id)})
        
        self.logger.info(f"Relation deleted: {relation_id}")
        return True
    
    async def bulk_create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple entities in bulk.
        
        Args:
            entities: List of entities to create
            
        Returns:
            List of created entities
        """
        self.logger.info(f"Bulk creating {len(entities)} entities")
        
        created_entities = []
        for entity in entities:
            try:
                created = await self.create_entity(entity)
                created_entities.append(created)
            except ValueError as e:
                self.logger.warning(f"Failed to create entity {entity.name}: {e}")
                # Continue with other entities
        
        self.logger.info(f"Successfully created {len(created_entities)} entities")
        return created_entities
    
    async def bulk_create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple relations in bulk.
        
        Args:
            relations: List of relations to create
            
        Returns:
            List of created relations
        """
        self.logger.info(f"Bulk creating {len(relations)} relations")
        
        created_relations = []
        for relation in relations:
            try:
                created = await self.create_relation(relation)
                created_relations.append(created)
            except ValueError as e:
                self.logger.warning(f"Failed to create relation {relation.id}: {e}")
                # Continue with other relations
        
        self.logger.info(f"Successfully created {len(created_relations)} relations")
        return created_relations
    
    async def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        entities = await self.storage.get_all_entities()
        relations = await self.storage.get_all_relations()
        
        # Count entities by type
        entity_types = {}
        for entity in entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        # Count relations by type
        relation_types = {}
        for relation in relations:
            relation_types[relation.type] = relation_types.get(relation.type, 0) + 1
        
        return {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entity_types": entity_types,
            "relation_types": relation_types
        }