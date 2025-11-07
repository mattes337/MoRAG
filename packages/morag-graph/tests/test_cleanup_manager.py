"""Tests for DocumentCleanupManager."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from morag_graph.models.entity import Entity
from morag_graph.models.relation import Relation
from morag_graph.storage.base import BaseStorage
from morag_graph.updates.cleanup_manager import CleanupResult, DocumentCleanupManager


class MockStorage(BaseStorage):
    """Mock storage for testing."""

    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.checksums = {}
        self.entity_relations = {}  # entity_id -> list of relations

    async def get_entities_by_document(self, document_id: str):
        """Return entities for a document."""
        entities = []
        for entity in self.entities.values():
            if hasattr(entity, "source_doc_id") and entity.source_doc_id == document_id:
                entities.append(entity)
        return entities

    async def get_entity_relations(
        self, entity_id, relation_type=None, direction="both"
    ):
        """Return relations for an entity."""
        return self.entity_relations.get(entity_id, [])

    async def delete_entity(self, entity_id):
        """Delete an entity."""
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False

    async def delete_relation(self, relation_id):
        """Delete a relation."""
        if relation_id in self.relations:
            del self.relations[relation_id]
            return True
        return False

    async def delete_document_checksum(self, document_id: str):
        """Delete document checksum."""
        if document_id in self.checksums:
            del self.checksums[document_id]

    # Required abstract methods (not used in tests)
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def store_entity(self, entity):
        pass

    async def store_entities(self, entities):
        return []

    async def get_entity(self, entity_id):
        pass

    async def get_entities(self, entity_ids):
        pass

    async def search_entities(self, query, entity_type=None, limit=10):
        return []

    async def update_entity(self, entity):
        pass

    async def store_relation(self, relation):
        pass

    async def store_relations(self, relations):
        return []

    async def get_relation(self, relation_id):
        pass

    async def get_relations(self, relation_ids):
        pass

    async def update_relation(self, relation):
        pass

    async def get_neighbors(self, entity_id, relation_type=None, max_depth=1):
        pass

    async def find_path(self, source_entity_id, target_entity_id, max_depth=3):
        pass

    async def store_graph(self, graph):
        pass

    async def get_graph(self, entity_ids=None):
        pass

    async def clear(self):
        pass

    async def get_statistics(self):
        pass


@pytest.fixture
def mock_storage():
    return MockStorage()


@pytest.fixture
def cleanup_manager(mock_storage):
    return DocumentCleanupManager(mock_storage)


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(name="Entity 1", type="PERSON", source_doc_id="doc_test1_abc123"),
        Entity(name="Entity 2", type="ORGANIZATION", source_doc_id="doc_test1_abc123"),
        Entity(
            name="Entity 3",
            type="PERSON",
            source_doc_id="doc_test2_abc123",  # Different document
        ),
    ]


@pytest.fixture
def sample_relations(sample_entities):
    """Create sample relations for testing."""
    return [
        Relation(
            source_entity_id=sample_entities[0].id,
            target_entity_id=sample_entities[1].id,
            type=RelationType.WORKS_FOR,
        ),
        Relation(
            source_entity_id=sample_entities[1].id,
            target_entity_id=sample_entities[2].id,
            type=RelationType.COMMUNICATES_WITH,
        ),
    ]


class TestDocumentCleanupManager:
    """Test cases for DocumentCleanupManager."""

    @pytest.mark.asyncio
    async def test_cleanup_document_data_no_entities(
        self, cleanup_manager, mock_storage
    ):
        """Test cleanup when document has no entities."""
        document_id = "empty_doc"

        result = await cleanup_manager.cleanup_document_data(document_id)

        assert isinstance(result, CleanupResult)
        assert result.document_id == document_id
        assert result.entities_deleted == 0
        assert result.relations_deleted == 0
        assert len(result.entity_ids_deleted) == 0
        assert len(result.relation_ids_deleted) == 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_cleanup_document_data_with_entities(
        self, cleanup_manager, mock_storage, sample_entities, sample_relations
    ):
        """Test cleanup when document has entities and relations."""
        document_id = "doc_test1_abc123"

        # Setup mock storage with entities and relations
        for entity in sample_entities:
            mock_storage.entities[entity.id] = entity

        for relation in sample_relations:
            mock_storage.relations[relation.id] = relation

        # Setup entity relations mapping
        mock_storage.entity_relations[sample_entities[0].id] = [
            sample_relations[0]
        ]  # rel1
        mock_storage.entity_relations[sample_entities[1].id] = [
            sample_relations[0],
            sample_relations[1],
        ]  # rel1, rel2

        # Add checksum
        mock_storage.checksums[document_id] = "test_checksum"

        result = await cleanup_manager.cleanup_document_data(document_id)

        assert isinstance(result, CleanupResult)
        assert result.document_id == document_id
        assert result.entities_deleted == 2  # entity1 and entity2 from doc1
        assert result.relations_deleted == 2  # rel1 and rel2
        assert sample_entities[0].id in result.entity_ids_deleted
        assert sample_entities[1].id in result.entity_ids_deleted
        assert sample_relations[0].id in result.relation_ids_deleted
        assert sample_relations[1].id in result.relation_ids_deleted
        assert len(result.errors) == 0

        # Verify entities were deleted from storage
        assert sample_entities[0].id not in mock_storage.entities
        assert sample_entities[1].id not in mock_storage.entities
        assert sample_entities[2].id in mock_storage.entities  # From different document

        # Verify relations were deleted from storage
        assert sample_relations[0].id not in mock_storage.relations
        assert sample_relations[1].id not in mock_storage.relations

        # Verify checksum was deleted
        assert document_id not in mock_storage.checksums

    @pytest.mark.asyncio
    async def test_cleanup_document_data_partial_failure(
        self, cleanup_manager, mock_storage, sample_entities
    ):
        """Test cleanup with partial failures."""
        document_id = "doc_test1_abc123"

        # Setup entities
        for entity in sample_entities[:2]:  # Only first two entities
            mock_storage.entities[entity.id] = entity

        # Mock delete_entity to fail for entity2
        original_delete_entity = mock_storage.delete_entity

        async def failing_delete_entity(entity_id):
            if entity_id == sample_entities[1].id:
                raise Exception("Simulated delete failure")
            return await original_delete_entity(entity_id)

        mock_storage.delete_entity = failing_delete_entity

        result = await cleanup_manager.cleanup_document_data(document_id)

        assert isinstance(result, CleanupResult)
        assert result.document_id == document_id
        assert result.entities_deleted == 1  # Only entity1 deleted successfully
        assert result.relations_deleted == 0
        assert sample_entities[0].id in result.entity_ids_deleted
        assert sample_entities[1].id not in result.entity_ids_deleted
        assert len(result.errors) == 1
        assert f"Failed to delete entity {sample_entities[1].id}" in result.errors[0]

    @pytest.mark.asyncio
    async def test_get_document_entities(
        self, cleanup_manager, mock_storage, sample_entities
    ):
        """Test getting entities for a document."""
        document_id = "doc_test1_abc123"

        # Setup entities in storage
        for entity in sample_entities:
            mock_storage.entities[entity.id] = entity

        entity_ids = await cleanup_manager._get_document_entities(document_id)

        # Should return entity1 and entity2 (both from doc_test1_abc123)
        assert len(entity_ids) == 2
        assert sample_entities[0].id in entity_ids
        assert sample_entities[1].id in entity_ids
        assert sample_entities[2].id not in entity_ids  # From different document

    @pytest.mark.asyncio
    async def test_get_document_relations(
        self, cleanup_manager, mock_storage, sample_entities, sample_relations
    ):
        """Test getting relations for entities."""
        entity_ids = [sample_entities[0].id, sample_entities[1].id]

        # Setup relations in storage
        for relation in sample_relations:
            mock_storage.relations[relation.id] = relation

        # Setup entity relations mapping
        mock_storage.entity_relations[sample_entities[0].id] = [
            sample_relations[0]
        ]  # rel1
        mock_storage.entity_relations[sample_entities[1].id] = [
            sample_relations[0],
            sample_relations[1],
        ]  # rel1, rel2

        relation_ids = await cleanup_manager._get_document_relations(entity_ids)

        # Should return unique relation IDs
        assert len(relation_ids) == 2
        assert sample_relations[0].id in relation_ids
        assert sample_relations[1].id in relation_ids

    @pytest.mark.asyncio
    async def test_remove_document_checksum(self, cleanup_manager, mock_storage):
        """Test removing document checksum."""
        document_id = "test_doc"
        mock_storage.checksums[document_id] = "test_checksum"

        await cleanup_manager._remove_document_checksum(document_id)

        assert document_id not in mock_storage.checksums

    def test_cleanup_result_initialization(self):
        """Test CleanupResult initialization."""
        result = CleanupResult(document_id="test_doc")

        assert result.document_id == "test_doc"
        assert result.entities_deleted == 0
        assert result.relations_deleted == 0
        assert result.entity_ids_deleted == []
        assert result.relation_ids_deleted == []
        assert result.errors == []

    def test_cleanup_result_with_data(self):
        """Test CleanupResult with data."""
        result = CleanupResult(
            document_id="test_doc",
            entities_deleted=2,
            relations_deleted=1,
            entity_ids_deleted=["e1", "e2"],
            relation_ids_deleted=["r1"],
            errors=["error1"],
        )

        assert result.document_id == "test_doc"
        assert result.entities_deleted == 2
        assert result.relations_deleted == 1
        assert result.entity_ids_deleted == ["e1", "e2"]
        assert result.relation_ids_deleted == ["r1"]
        assert result.errors == ["error1"]
