"""Base storage interface for graph data."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set

from ..models import Entity, Relation, Graph
from ..models.types import EntityId, RelationId


class BaseStorage(ABC):
    """Base class for graph storage backends.

    This abstract class defines the interface that all storage backends
    must implement for storing and retrieving graph data.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the storage backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        pass

    @abstractmethod
    async def store_entity(self, entity: Entity) -> EntityId:
        """Store an entity.

        Args:
            entity: Entity to store

        Returns:
            ID of the stored entity
        """
        pass

    @abstractmethod
    async def store_entities(self, entities: List[Entity]) -> List[EntityId]:
        """Store multiple entities.

        Args:
            entities: List of entities to store

        Returns:
            List of IDs of the stored entities
        """
        pass

    @abstractmethod
    async def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID.

        Args:
            entity_id: ID of the entity to get

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_entities(
        self,
        entity_ids: List[EntityId]
    ) -> List[Entity]:
        """Get multiple entities by IDs.

        Args:
            entity_ids: List of entity IDs to get

        Returns:
            List of entities (may be shorter than input if some not found)
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity.

        Args:
            entity: Entity with updated data

        Returns:
            True if entity was updated, False if not found
        """
        pass

    @abstractmethod
    async def delete_entity(self, entity_id: EntityId) -> bool:
        """Delete an entity and all its relations.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if entity was deleted, False if not found
        """
        pass

    @abstractmethod
    async def store_relation(self, relation: Relation) -> RelationId:
        """Store a relation.

        Args:
            relation: Relation to store

        Returns:
            ID of the stored relation
        """
        pass

    @abstractmethod
    async def store_relations(self, relations: List[Relation]) -> List[RelationId]:
        """Store multiple relations.

        Args:
            relations: List of relations to store

        Returns:
            List of IDs of the stored relations
        """
        pass

    @abstractmethod
    async def get_relation(self, relation_id: RelationId) -> Optional[Relation]:
        """Get a relation by ID.

        Args:
            relation_id: ID of the relation to get

        Returns:
            Relation if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_relations(
        self,
        relation_ids: List[RelationId]
    ) -> List[Relation]:
        """Get multiple relations by IDs.

        Args:
            relation_ids: List of relation IDs to get

        Returns:
            List of relations (may be shorter than input if some not found)
        """
        pass

    @abstractmethod
    async def get_entity_relations(
        self,
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        direction: str = "both"  # "in", "out", "both"
    ) -> List[Relation]:
        """Get all relations for an entity.

        Args:
            entity_id: ID of the entity
            relation_type: Optional relation type filter
            direction: Direction of relations ("in", "out", "both")

        Returns:
            List of relations involving the entity
        """
        pass

    @abstractmethod
    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation.

        Args:
            relation: Relation with updated data

        Returns:
            True if relation was updated, False if not found
        """
        pass

    @abstractmethod
    async def delete_relation(self, relation_id: RelationId) -> bool:
        """Delete a relation.

        Args:
            relation_id: ID of the relation to delete

        Returns:
            True if relation was deleted, False if not found
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def store_graph(self, graph: Graph) -> None:
        """Store an entire graph.

        Args:
            graph: Graph to store
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all data from the storage."""
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary containing statistics like entity count, relation count, etc.
        """
        pass

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
