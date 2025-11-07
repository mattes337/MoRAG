"""Unit tests for BaseStorage class."""

import pytest
from typing import List, Optional

from morag_graph.storage.base import BaseStorage
from morag_graph.models import Entity, Relation, Graph
from morag_graph.models.types import EntityId, RelationId


class MockStorage(BaseStorage):
    """Mock storage implementation for testing."""

    def __init__(self):
        super().__init__()
        self._connected = False
        self._entities = {}
        self._relations = {}
        self._entity_relations = {}

    async def connect(self) -> None:
        """Connect to the storage backend."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        self._connected = False

    async def store_entity(self, entity: Entity) -> EntityId:
        """Store an entity."""
        if not self._connected:
            raise ConnectionError("Not connected to storage")

        entity_id = f"entity_{len(self._entities)}"
        self._entities[entity_id] = entity
        self._entity_relations[entity_id] = []
        return entity_id

    async def store_entities(self, entities: List[Entity]) -> List[EntityId]:
        """Store multiple entities."""
        entity_ids = []
        for entity in entities:
            entity_id = await self.store_entity(entity)
            entity_ids.append(entity_id)
        return entity_ids

    async def get_entity(self, entity_id: EntityId) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    async def get_entities(self, entity_ids: List[EntityId]) -> List[Entity]:
        """Get multiple entities by IDs."""
        entities = []
        for entity_id in entity_ids:
            entity = await self.get_entity(entity_id)
            if entity:
                entities.append(entity)
        return entities

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search for entities by name or attributes."""
        results = []
        for entity in self._entities.values():
            if query.lower() in entity.name.lower():
                if entity_type is None or entity.type == entity_type:
                    results.append(entity)
                    if len(results) >= limit:
                        break
        return results

    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        if entity.id in self._entities:
            self._entities[entity.id] = entity
            return True
        return False

    async def delete_entity(self, entity_id: EntityId) -> bool:
        """Delete an entity and all its relations."""
        if entity_id in self._entities:
            del self._entities[entity_id]
            if entity_id in self._entity_relations:
                del self._entity_relations[entity_id]
            return True
        return False

    async def store_relation(self, relation: Relation) -> RelationId:
        """Store a relation."""
        relation_id = f"relation_{len(self._relations)}"
        self._relations[relation_id] = relation

        # Add to entity relations tracking
        if relation.source_id in self._entity_relations:
            self._entity_relations[relation.source_id].append(relation_id)
        if relation.target_id in self._entity_relations:
            self._entity_relations[relation.target_id].append(relation_id)

        return relation_id

    async def store_relations(self, relations: List[Relation]) -> List[RelationId]:
        """Store multiple relations."""
        relation_ids = []
        for relation in relations:
            relation_id = await self.store_relation(relation)
            relation_ids.append(relation_id)
        return relation_ids

    async def get_relation(self, relation_id: RelationId) -> Optional[Relation]:
        """Get a relation by ID."""
        return self._relations.get(relation_id)

    async def get_relations(self, relation_ids: List[RelationId]) -> List[Relation]:
        """Get multiple relations by IDs."""
        relations = []
        for relation_id in relation_ids:
            relation = await self.get_relation(relation_id)
            if relation:
                relations.append(relation)
        return relations

    async def get_entity_relations(
        self,
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Relation]:
        """Get all relations for an entity."""
        relation_ids = self._entity_relations.get(entity_id, [])
        relations = []

        for relation_id in relation_ids:
            relation = self._relations.get(relation_id)
            if relation:
                # Filter by type if specified
                if relation_type and relation.type != relation_type:
                    continue

                # Filter by direction
                if direction == "out" and relation.source_id != entity_id:
                    continue
                elif direction == "in" and relation.target_id != entity_id:
                    continue

                relations.append(relation)

        return relations

    async def update_relation(self, relation: Relation) -> bool:
        """Update an existing relation."""
        if relation.id in self._relations:
            self._relations[relation.id] = relation
            return True
        return False

    async def delete_relation(self, relation_id: RelationId) -> bool:
        """Delete a relation."""
        if relation_id in self._relations:
            del self._relations[relation_id]
            return True
        return False

    async def get_neighbors(
        self,
        entity_id: EntityId,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Entity]:
        """Get neighboring entities."""
        neighbors = []
        relations = await self.get_entity_relations(entity_id, relation_type)

        for relation in relations:
            neighbor_id = relation.target_id if relation.source_id == entity_id else relation.source_id
            neighbor = await self.get_entity(neighbor_id)
            if neighbor:
                neighbors.append(neighbor)

        return neighbors

    async def find_path(
        self,
        source_entity_id: EntityId,
        target_entity_id: EntityId,
        max_depth: int = 3
    ) -> List[List[EntityId]]:
        """Find paths between two entities."""
        # Simple implementation for testing
        if source_entity_id == target_entity_id:
            return [[source_entity_id]]

        # For testing, just return empty or simple path
        return []

    async def store_graph(self, graph: Graph) -> None:
        """Store an entire graph."""
        await self.store_entities(graph.entities)
        await self.store_relations(graph.relations)

    async def get_graph(self, entity_ids: Optional[List[EntityId]] = None) -> Graph:
        """Get a graph or subgraph."""
        if entity_ids is None:
            entities = list(self._entities.values())
            relations = list(self._relations.values())
        else:
            entities = await self.get_entities(entity_ids)
            # Get relations between these entities
            relations = []
            for relation in self._relations.values():
                if relation.source_id in entity_ids and relation.target_id in entity_ids:
                    relations.append(relation)

        return Graph(entities=entities, relations=relations)

    async def clear(self) -> None:
        """Clear all data from the storage."""
        self._entities.clear()
        self._relations.clear()
        self._entity_relations.clear()

    async def get_statistics(self):
        """Get storage statistics."""
        return {
            "entity_count": len(self._entities),
            "relation_count": len(self._relations),
            "connected": self._connected
        }


class TestBaseStorage:
    """Test BaseStorage abstract class functionality."""

    @pytest.fixture
    async def mock_storage(self):
        """Create a mock storage instance."""
        storage = MockStorage()
        await storage.connect()
        return storage

    @pytest.fixture
    def sample_entity(self):
        """Create a sample entity for testing."""
        return Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={"age": 30, "occupation": "Engineer"}
        )

    @pytest.fixture
    def sample_relation(self):
        """Create a sample relation for testing."""
        return Relation(
            id="test_relation",
            source_id="entity_1",
            target_id="entity_2",
            type="WORKS_WITH",
            attributes={"since": "2020"}
        )

    async def test_connection_lifecycle(self):
        """Test storage connection and disconnection."""
        storage = MockStorage()

        # Initially not connected
        assert storage._connected is False

        # Connect
        await storage.connect()
        assert storage._connected is True

        # Disconnect
        await storage.disconnect()
        assert storage._connected is False

    async def test_context_manager(self):
        """Test storage as async context manager."""
        storage = MockStorage()

        async with storage as connected_storage:
            assert connected_storage._connected is True
            assert connected_storage is storage

        # Should be disconnected after context
        assert storage._connected is False

    async def test_store_single_entity(self, mock_storage, sample_entity):
        """Test storing a single entity."""
        entity_id = await mock_storage.store_entity(sample_entity)

        assert entity_id is not None
        assert isinstance(entity_id, str)
        assert entity_id.startswith("entity_")

        # Verify entity was stored
        stored_entity = await mock_storage.get_entity(entity_id)
        assert stored_entity == sample_entity

    async def test_store_multiple_entities(self, mock_storage):
        """Test storing multiple entities."""
        entities = [
            Entity(id="e1", name="Entity 1", type="Person"),
            Entity(id="e2", name="Entity 2", type="Company"),
            Entity(id="e3", name="Entity 3", type="Location")
        ]

        entity_ids = await mock_storage.store_entities(entities)

        assert len(entity_ids) == 3
        assert all(isinstance(eid, str) for eid in entity_ids)

        # Verify all entities were stored
        stored_entities = await mock_storage.get_entities(entity_ids)
        assert len(stored_entities) == 3

    async def test_get_nonexistent_entity(self, mock_storage):
        """Test getting a non-existent entity."""
        entity = await mock_storage.get_entity("nonexistent")
        assert entity is None

    async def test_search_entities(self, mock_storage):
        """Test searching for entities."""
        # Store some entities
        entities = [
            Entity(id="e1", name="John Doe", type="Person"),
            Entity(id="e2", name="Jane Doe", type="Person"),
            Entity(id="e3", name="Acme Corp", type="Company")
        ]
        await mock_storage.store_entities(entities)

        # Search for entities containing "Doe"
        results = await mock_storage.search_entities("Doe")
        assert len(results) == 2
        assert all("Doe" in entity.name for entity in results)

        # Search with type filter
        person_results = await mock_storage.search_entities("Doe", entity_type="Person")
        assert len(person_results) == 2
        assert all(entity.type == "Person" for entity in person_results)

        # Search with limit
        limited_results = await mock_storage.search_entities("Doe", limit=1)
        assert len(limited_results) == 1

    async def test_update_entity(self, mock_storage, sample_entity):
        """Test updating an existing entity."""
        # Store entity first
        entity_id = await mock_storage.store_entity(sample_entity)

        # Update entity
        updated_entity = Entity(
            id=entity_id,
            name="Updated Entity",
            type="Person",
            attributes={"age": 31}
        )

        result = await mock_storage.update_entity(updated_entity)
        assert result is True

        # Verify update
        stored_entity = await mock_storage.get_entity(entity_id)
        assert stored_entity.name == "Updated Entity"
        assert stored_entity.attributes["age"] == 31

    async def test_update_nonexistent_entity(self, mock_storage):
        """Test updating a non-existent entity."""
        entity = Entity(id="nonexistent", name="Test", type="Person")
        result = await mock_storage.update_entity(entity)
        assert result is False

    async def test_delete_entity(self, mock_storage, sample_entity):
        """Test deleting an entity."""
        entity_id = await mock_storage.store_entity(sample_entity)

        # Verify entity exists
        entity = await mock_storage.get_entity(entity_id)
        assert entity is not None

        # Delete entity
        result = await mock_storage.delete_entity(entity_id)
        assert result is True

        # Verify entity was deleted
        entity = await mock_storage.get_entity(entity_id)
        assert entity is None

    async def test_delete_nonexistent_entity(self, mock_storage):
        """Test deleting a non-existent entity."""
        result = await mock_storage.delete_entity("nonexistent")
        assert result is False

    async def test_store_relation(self, mock_storage, sample_relation):
        """Test storing a relation."""
        # First store the entities referenced by the relation
        entity1 = Entity(id="entity_1", name="Entity 1", type="Person")
        entity2 = Entity(id="entity_2", name="Entity 2", type="Person")
        await mock_storage.store_entity(entity1)
        await mock_storage.store_entity(entity2)

        relation_id = await mock_storage.store_relation(sample_relation)

        assert relation_id is not None
        assert isinstance(relation_id, str)
        assert relation_id.startswith("relation_")

        # Verify relation was stored
        stored_relation = await mock_storage.get_relation(relation_id)
        assert stored_relation == sample_relation

    async def test_get_entity_relations(self, mock_storage):
        """Test getting relations for an entity."""
        # Store entities
        entity1 = Entity(id="e1", name="Entity 1", type="Person")
        entity2 = Entity(id="e2", name="Entity 2", type="Person")
        entity1_id = await mock_storage.store_entity(entity1)
        entity2_id = await mock_storage.store_entity(entity2)

        # Store relations
        relation1 = Relation(id="r1", source_id=entity1_id, target_id=entity2_id, type="KNOWS")
        relation2 = Relation(id="r2", source_id=entity2_id, target_id=entity1_id, type="WORKS_WITH")
        await mock_storage.store_relation(relation1)
        await mock_storage.store_relation(relation2)

        # Get all relations for entity1
        relations = await mock_storage.get_entity_relations(entity1_id)
        assert len(relations) == 2

        # Get outgoing relations only
        out_relations = await mock_storage.get_entity_relations(entity1_id, direction="out")
        assert len(out_relations) == 1
        assert out_relations[0].source_id == entity1_id

        # Get incoming relations only
        in_relations = await mock_storage.get_entity_relations(entity1_id, direction="in")
        assert len(in_relations) == 1
        assert in_relations[0].target_id == entity1_id

        # Filter by relation type
        knows_relations = await mock_storage.get_entity_relations(entity1_id, relation_type="KNOWS")
        assert len(knows_relations) == 1
        assert knows_relations[0].type == "KNOWS"

    async def test_get_neighbors(self, mock_storage):
        """Test getting neighboring entities."""
        # Store entities
        entity1 = Entity(id="e1", name="Entity 1", type="Person")
        entity2 = Entity(id="e2", name="Entity 2", type="Person")
        entity3 = Entity(id="e3", name="Entity 3", type="Company")
        entity1_id = await mock_storage.store_entity(entity1)
        entity2_id = await mock_storage.store_entity(entity2)
        entity3_id = await mock_storage.store_entity(entity3)

        # Store relations
        relation1 = Relation(id="r1", source_id=entity1_id, target_id=entity2_id, type="KNOWS")
        relation2 = Relation(id="r2", source_id=entity1_id, target_id=entity3_id, type="WORKS_FOR")
        await mock_storage.store_relation(relation1)
        await mock_storage.store_relation(relation2)

        # Get all neighbors
        neighbors = await mock_storage.get_neighbors(entity1_id)
        assert len(neighbors) == 2

        # Get neighbors by relation type
        work_neighbors = await mock_storage.get_neighbors(entity1_id, relation_type="WORKS_FOR")
        assert len(work_neighbors) == 1
        assert work_neighbors[0].type == "Company"

    async def test_clear_storage(self, mock_storage, sample_entity, sample_relation):
        """Test clearing all storage data."""
        # Store some data
        await mock_storage.store_entity(sample_entity)

        # Clear storage
        await mock_storage.clear()

        # Verify storage is empty
        stats = await mock_storage.get_statistics()
        assert stats["entity_count"] == 0
        assert stats["relation_count"] == 0

    async def test_get_statistics(self, mock_storage):
        """Test getting storage statistics."""
        # Initially empty
        stats = await mock_storage.get_statistics()
        assert stats["entity_count"] == 0
        assert stats["relation_count"] == 0
        assert stats["connected"] is True

        # Add some data
        entity = Entity(id="e1", name="Test Entity", type="Person")
        await mock_storage.store_entity(entity)

        # Check updated stats
        stats = await mock_storage.get_statistics()
        assert stats["entity_count"] == 1
        assert stats["relation_count"] == 0

    async def test_store_and_get_graph(self, mock_storage):
        """Test storing and retrieving a complete graph."""
        # Create a sample graph
        entities = [
            Entity(id="e1", name="Entity 1", type="Person"),
            Entity(id="e2", name="Entity 2", type="Person")
        ]
        relations = [
            Relation(id="r1", source_id="e1", target_id="e2", type="KNOWS")
        ]
        graph = Graph(entities=entities, relations=relations)

        # Store the graph
        await mock_storage.store_graph(graph)

        # Retrieve the graph
        stored_graph = await mock_storage.get_graph()
        assert len(stored_graph.entities) >= 2
        assert len(stored_graph.relations) >= 1

    async def test_connection_required_for_operations(self):
        """Test that operations require connection."""
        storage = MockStorage()  # Not connected

        entity = Entity(id="e1", name="Test", type="Person")

        with pytest.raises(ConnectionError, match="Not connected to storage"):
            await storage.store_entity(entity)
