"""Graph storage interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import Entity, Relation


class GraphStorage(ABC):
    """Abstract base class for graph storage."""
    
    @abstractmethod
    async def store_entity(self, entity: Entity) -> bool:
        """Store an entity in the graph."""
        pass
    
    @abstractmethod
    async def store_relation(self, relation: Relation) -> bool:
        """Store a relation in the graph."""
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        pass
    
    @abstractmethod
    async def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Retrieve entities by type."""
        pass
    
    @abstractmethod
    async def get_relations(self, source_id: str, target_id: str = None) -> List[Relation]:
        """Retrieve relations for an entity."""
        pass
    
    @abstractmethod
    async def search_entities(self, query: str) -> List[Entity]:
        """Search entities by name or properties."""
        pass


class InMemoryGraphStorage(GraphStorage):
    """In-memory graph storage for testing."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
    
    async def store_entity(self, entity: Entity) -> bool:
        """Store an entity in memory."""
        self.entities[entity.id] = entity
        return True
    
    async def store_relation(self, relation: Relation) -> bool:
        """Store a relation in memory."""
        self.relations[relation.id] = relation
        return True
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self.entities.get(entity_id)
    
    async def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Retrieve entities by type."""
        return [
            entity for entity in self.entities.values()
            if entity.type == entity_type
        ]
    
    async def get_relations(self, source_id: str, target_id: str = None) -> List[Relation]:
        """Retrieve relations for an entity."""
        relations = [
            relation for relation in self.relations.values()
            if relation.source_id == source_id
        ]
        
        if target_id:
            relations = [
                relation for relation in relations
                if relation.target_id == target_id
            ]
        
        return relations
    
    async def search_entities(self, query: str) -> List[Entity]:
        """Search entities by name."""
        query_lower = query.lower()
        return [
            entity for entity in self.entities.values()
            if query_lower in entity.name.lower()
        ]


class DummyGraphStorage(GraphStorage):
    """Dummy graph storage that does nothing."""
    
    async def store_entity(self, entity: Entity) -> bool:
        return True
    
    async def store_relation(self, relation: Relation) -> bool:
        return True
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        return None
    
    async def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        return []
    
    async def get_relations(self, source_id: str, target_id: str = None) -> List[Relation]:
        return []
    
    async def search_entities(self, query: str) -> List[Entity]:
        return []
