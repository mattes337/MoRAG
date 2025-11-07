"""Graph model for graph-augmented RAG."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .entity import Entity
from .relation import Relation
from .types import EntityId, RelationId


@dataclass
class GraphNode:
    """Generic graph node."""

    id: str
    label: str
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    """Generic graph edge."""

    source: str
    target: str
    type: str
    properties: Dict[str, Any]


class Graph(BaseModel):
    """Graph model representing a knowledge graph.

    A graph consists of entities (nodes) and relations (edges).
    This model provides methods for adding, removing, and querying entities and relations.

    Attributes:
        entities: Dictionary of entities by ID
        relations: Dictionary of relations by ID
        name: Optional name of the graph
        description: Optional description of the graph
    """

    entities: Dict[EntityId, Entity] = Field(default_factory=dict)
    relations: Dict[RelationId, Relation] = Field(default_factory=dict)
    name: Optional[str] = None
    description: Optional[str] = None

    def add_entity(self, entity: Entity) -> EntityId:
        """Add an entity to the graph.

        Args:
            entity: Entity to add

        Returns:
            ID of the added entity
        """
        self.entities[entity.id] = entity
        return entity.id

    def add_relation(self, relation: Relation) -> RelationId:
        """Add a relation to the graph.

        Args:
            relation: Relation to add

        Returns:
            ID of the added relation

        Raises:
            ValueError: If source or target entity does not exist in the graph
        """
        if relation.source_entity_id not in self.entities:
            raise ValueError(
                f"Source entity {relation.source_entity_id} does not exist in the graph"
            )
        if relation.target_entity_id not in self.entities:
            raise ValueError(
                f"Target entity {relation.target_entity_id} does not exist in the graph"
            )

        self.relations[relation.id] = relation
        return relation.id

    def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID.

        Args:
            entity_id: ID of the entity to get

        Returns:
            Entity if found, None otherwise
        """
        return self.entities.get(entity_id)

    def get_relation(self, relation_id: RelationId) -> Optional[Relation]:
        """Get a relation by ID.

        Args:
            relation_id: ID of the relation to get

        Returns:
            Relation if found, None otherwise
        """
        return self.relations.get(relation_id)

    def remove_entity(self, entity_id: EntityId) -> bool:
        """Remove an entity and all its relations from the graph.

        Args:
            entity_id: ID of the entity to remove

        Returns:
            True if entity was removed, False if entity was not found
        """
        if entity_id not in self.entities:
            return False

        # Remove all relations involving this entity
        relation_ids_to_remove = [
            r_id for r_id, r in self.relations.items()
            if r.source_entity_id == entity_id or r.target_entity_id == entity_id
        ]

        for r_id in relation_ids_to_remove:
            self.relations.pop(r_id)

        # Remove the entity
        self.entities.pop(entity_id)

        return True

    def remove_relation(self, relation_id: RelationId) -> bool:
        """Remove a relation from the graph.

        Args:
            relation_id: ID of the relation to remove

        Returns:
            True if relation was removed, False if relation was not found
        """
        if relation_id not in self.relations:
            return False

        self.relations.pop(relation_id)
        return True

    def get_entity_relations(self, entity_id: EntityId) -> List[Relation]:
        """Get all relations involving an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            List of relations involving the entity
        """
        return [
            r for r in self.relations.values()
            if r.source_entity_id == entity_id or r.target_entity_id == entity_id
        ]

    def get_outgoing_relations(self, entity_id: EntityId) -> List[Relation]:
        """Get all outgoing relations from an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            List of outgoing relations from the entity
        """
        return [
            r for r in self.relations.values()
            if r.source_entity_id == entity_id
        ]

    def get_incoming_relations(self, entity_id: EntityId) -> List[Relation]:
        """Get all incoming relations to an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            List of incoming relations to the entity
        """
        return [
            r for r in self.relations.values()
            if r.target_entity_id == entity_id
        ]

    def get_neighbors(self, entity_id: EntityId) -> List[Entity]:
        """Get all neighboring entities of an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            List of neighboring entities
        """
        neighbor_ids = set()

        for r in self.relations.values():
            if r.source_entity_id == entity_id:
                neighbor_ids.add(r.target_entity_id)
            elif r.target_entity_id == entity_id:
                neighbor_ids.add(r.source_entity_id)

        return [self.entities[n_id] for n_id in neighbor_ids if n_id in self.entities]

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type.

        Args:
            entity_type: Type of entities to get

        Returns:
            List of entities of the specified type
        """
        return [
            e for e in self.entities.values()
            if str(e.type) == entity_type
        ]

    def get_relations_by_type(self, relation_type: str) -> List[Relation]:
        """Get all relations of a specific type.

        Args:
            relation_type: Type of relations to get

        Returns:
            List of relations of the specified type
        """
        return [
            r for r in self.relations.values()
            if str(r.type) == relation_type
        ]

    def merge(self, other: 'Graph') -> None:
        """Merge another graph into this graph.

        Args:
            other: Graph to merge into this graph
        """
        # Add entities from other graph
        for entity_id, entity in other.entities.items():
            if entity_id not in self.entities:
                self.entities[entity_id] = entity

        # Add relations from other graph
        for relation_id, relation in other.relations.items():
            if relation_id not in self.relations:
                # Only add relation if both entities exist in this graph
                if (relation.source_entity_id in self.entities and
                    relation.target_entity_id in self.entities):
                    self.relations[relation_id] = relation

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for JSON serialization."""
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
            "name": self.name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Graph':
        """Create graph from dictionary.

        Args:
            data: Dictionary representation of the graph

        Returns:
            Graph instance
        """
        entities = {}
        relations = {}

        # Create entities
        for entity_data in data.get("entities", []):
            entity = Entity(**entity_data)
            entities[entity.id] = entity

        # Create relations
        for relation_data in data.get("relations", []):
            relation = Relation(**relation_data)
            relations[relation.id] = relation

        return cls(
            entities=entities,
            relations=relations,
            name=data.get("name"),
            description=data.get("description"),
        )
