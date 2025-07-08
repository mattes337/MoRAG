"""Integration tests for GraphBuilder with checksum-based change detection."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from morag_graph.builders.graph_builder import GraphBuilder, GraphBuildResult
from morag_graph.models.entity import Entity
from morag_graph.models.relation import Relation
from morag_graph.updates.cleanup_manager import CleanupResult
from morag_graph.storage.base import BaseStorage


class MockStorage(BaseStorage):
    """Mock storage for testing."""
    
    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.checksums = {}
        self.entity_relations = {}
    
    async def get_document_checksum(self, document_id: str):
        return self.checksums.get(document_id)
    
    async def store_document_checksum(self, document_id: str, checksum: str):
        self.checksums[document_id] = checksum
    
    async def delete_document_checksum(self, document_id: str):
        if document_id in self.checksums:
            del self.checksums[document_id]
    
    async def get_entities_by_document(self, document_id: str):
        entities = []
        for entity in self.entities.values():
            if hasattr(entity, 'source_doc_id') and entity.source_doc_id == document_id:
                entities.append(entity)
        return entities
    
    async def get_entity_relations(self, entity_id, relation_type=None, direction="both"):
        return self.entity_relations.get(entity_id, [])
    
    async def delete_entity(self, entity_id):
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False
    
    async def delete_relation(self, relation_id):
        if relation_id in self.relations:
            del self.relations[relation_id]
            return True
        return False
    
    async def store_entity(self, entity):
        self.entities[entity.id] = entity
        return entity.id
    
    async def store_relation(self, relation):
        self.relations[relation.id] = relation
        return relation.id
    
    # Required abstract methods (not used in tests)
    async def connect(self): pass
    async def disconnect(self): pass
    async def get_entity(self, entity_id): pass
    async def get_entities(self, entity_ids): pass
    async def search_entities(self, query, entity_type=None, limit=10): return []
    async def store_entities(self, entities): return []
    async def update_entity(self, entity): pass
    async def get_relation(self, relation_id): pass
    async def get_relations(self, relation_ids): pass
    async def store_relations(self, relations): return []
    async def update_relation(self, relation): pass
    async def get_neighbors(self, entity_id, relation_type=None, max_depth=1): pass
    async def find_path(self, source_entity_id, target_entity_id, max_depth=3): pass
    async def store_graph(self, graph): pass
    async def get_graph(self, entity_ids=None): pass
    async def clear(self): pass
    async def get_statistics(self): pass


@pytest.fixture
def mock_storage():
    return MockStorage()


@pytest.fixture
def mock_llm_config():
    return {
        "model": "test-model",
        "api_key": "test-key"
    }


@pytest.fixture
def sample_entities():
    return [
        Entity(
            name="John Doe",
            type="PERSON",
            source_doc_id="doc_test_abc123"
        ),
        Entity(
            name="Acme Corp",
            type="ORGANIZATION",
            source_doc_id="doc_test_abc123"
        )
    ]


@pytest.fixture
def sample_relations(sample_entities):
    return [
        Relation(
            source_entity_id=sample_entities[0].id,
            target_entity_id=sample_entities[1].id,
            type="WORKS_FOR"
        )
    ]


class TestGraphBuilderChecksum:
    """Test cases for GraphBuilder with checksum-based change detection."""
    
    @pytest.mark.asyncio
    async def test_process_document_new_document(self, mock_storage, mock_llm_config, sample_entities, sample_relations):
        """Test processing a new document."""
        with patch('morag_graph.builders.graph_builder.EntityExtractor') as mock_entity_extractor, \
             patch('morag_graph.builders.graph_builder.RelationExtractor') as mock_relation_extractor:
            
            # Setup mock extractors
            mock_entity_extractor.return_value.extract = AsyncMock(return_value=sample_entities)
            mock_relation_extractor.return_value.extract = AsyncMock(return_value=sample_relations)
            
            builder = GraphBuilder(
                storage=mock_storage,
                llm_config=mock_llm_config
            )
            
            # Mock the _store_entities_and_relations method
            with patch.object(builder, '_store_entities_and_relations') as mock_store:
                mock_store.return_value = GraphBuildResult(
                    document_id="test_doc",
                    entities_created=2,
                    relations_created=1,
                    entity_ids=[sample_entities[0].id, sample_entities[1].id],
                    relation_ids=[sample_relations[0].id]
                )
                
                result = await builder.process_document(
                    content="Test content",
                    document_id="test_doc"
                )
                
                assert isinstance(result, GraphBuildResult)
                assert result.document_id == "test_doc"
                assert result.entities_created == 2
                assert result.relations_created == 1
                assert result.skipped is False
                assert result.cleanup_result is not None
                assert result.cleanup_result.entities_deleted == 0  # New document, nothing to clean
                
                # Verify checksum was stored
                assert "test_doc" in mock_storage.checksums
    
    @pytest.mark.asyncio
    async def test_process_document_unchanged_document(self, mock_storage, mock_llm_config):
        """Test processing an unchanged document (should be skipped)."""
        with patch('morag_graph.builders.graph_builder.EntityExtractor') as mock_entity_extractor, \
             patch('morag_graph.builders.graph_builder.RelationExtractor') as mock_relation_extractor:
            
            builder = GraphBuilder(
                storage=mock_storage,
                llm_config=mock_llm_config
            )
            
            content = "Test content"
            document_id = "test_doc"
            
            # Pre-store the checksum for this content
            checksum = builder.checksum_manager.calculate_document_checksum(content)
            mock_storage.checksums[document_id] = checksum
            
            result = await builder.process_document(
                content=content,
                document_id=document_id
            )
            
            assert isinstance(result, GraphBuildResult)
            assert result.document_id == document_id
            assert result.entities_created == 0
            assert result.relations_created == 0
            assert result.skipped is True
            assert result.cleanup_result is None
            
            # Verify extractors were not called
            mock_entity_extractor.return_value.extract.assert_not_called()
            mock_relation_extractor.return_value.extract.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_document_changed_document(self, mock_storage, mock_llm_config, sample_entities, sample_relations):
        """Test processing a changed document (should cleanup and reprocess)."""
        with patch('morag_graph.builders.graph_builder.EntityExtractor') as mock_entity_extractor, \
             patch('morag_graph.builders.graph_builder.RelationExtractor') as mock_relation_extractor:
            
            # Setup mock extractors
            mock_entity_extractor.return_value.extract = AsyncMock(return_value=sample_entities)
            mock_relation_extractor.return_value.extract = AsyncMock(return_value=sample_relations)
            
            builder = GraphBuilder(
                storage=mock_storage,
                llm_config=mock_llm_config
            )
            
            document_id = "test_doc"
            old_content = "Old content"
            new_content = "New content"
            
            # Pre-store checksum for old content
            old_checksum = builder.checksum_manager.calculate_document_checksum(old_content)
            mock_storage.checksums[document_id] = old_checksum
            
            # Add some existing entities and relations to storage
            old_entity = Entity(
                name="Old Entity",
                type="PERSON",
                source_doc_id=document_id
            )
            old_relation = Relation(
                source_entity_id=old_entity.id,
                target_entity_id=sample_entities[0].id,
                type="RELATED_TO"
            )
            
            mock_storage.entities[old_entity.id] = old_entity
            mock_storage.relations[old_relation.id] = old_relation
            mock_storage.entity_relations[old_entity.id] = [old_relation]
            
            # Mock the _store_entities_and_relations method
            with patch.object(builder, '_store_entities_and_relations') as mock_store:
                mock_store.return_value = GraphBuildResult(
                    document_id=document_id,
                    entities_created=2,
                    relations_created=1,
                    entity_ids=[sample_entities[0].id, sample_entities[1].id],
                    relation_ids=[sample_relations[0].id]
                )
                
                result = await builder.process_document(
                    content=new_content,
                    document_id=document_id
                )
                
                assert isinstance(result, GraphBuildResult)
                assert result.document_id == document_id
                assert result.entities_created == 2
                assert result.relations_created == 1
                assert result.skipped is False
                assert result.cleanup_result is not None
                assert result.cleanup_result.entities_deleted == 1  # old_entity deleted
                assert result.cleanup_result.relations_deleted == 1  # old_rel deleted
                
                # Verify old data was cleaned up
                assert old_entity.id not in mock_storage.entities
                assert old_relation.id not in mock_storage.relations
                
                # Verify new checksum was stored
                new_checksum = builder.checksum_manager.calculate_document_checksum(new_content)
                assert mock_storage.checksums[document_id] == new_checksum
    
    @pytest.mark.asyncio
    async def test_process_document_with_metadata_change(self, mock_storage, mock_llm_config, sample_entities, sample_relations):
        """Test processing document when metadata changes."""
        with patch('morag_graph.builders.graph_builder.EntityExtractor') as mock_entity_extractor, \
             patch('morag_graph.builders.graph_builder.RelationExtractor') as mock_relation_extractor:
            
            # Setup mock extractors
            mock_entity_extractor.return_value.extract = AsyncMock(return_value=sample_entities)
            mock_relation_extractor.return_value.extract = AsyncMock(return_value=sample_relations)
            
            builder = GraphBuilder(
                storage=mock_storage,
                llm_config=mock_llm_config
            )
            
            document_id = "test_doc"
            content = "Same content"
            old_metadata = {"version": 1}
            new_metadata = {"version": 2}
            
            # Pre-store checksum with old metadata
            old_checksum = builder.checksum_manager.calculate_document_checksum(content, old_metadata)
            mock_storage.checksums[document_id] = old_checksum
            
            # Mock the _store_entities_and_relations method
            with patch.object(builder, '_store_entities_and_relations') as mock_store:
                mock_store.return_value = GraphBuildResult(
                    document_id=document_id,
                    entities_created=2,
                    relations_created=1,
                    entity_ids=[sample_entities[0].id, sample_entities[1].id],
                    relation_ids=[sample_relations[0].id]
                )
                
                result = await builder.process_document(
                    content=content,
                    document_id=document_id,
                    metadata=new_metadata
                )
                
                assert isinstance(result, GraphBuildResult)
                assert result.skipped is False  # Should reprocess due to metadata change
                
                # Verify new checksum includes new metadata
                new_checksum = builder.checksum_manager.calculate_document_checksum(content, new_metadata)
                assert mock_storage.checksums[document_id] == new_checksum
                assert mock_storage.checksums[document_id] != old_checksum
    
    @pytest.mark.asyncio
    async def test_process_document_extraction_failure(self, mock_storage, mock_llm_config):
        """Test processing document when extraction fails."""
        with patch('morag_graph.builders.graph_builder.EntityExtractor') as mock_entity_extractor, \
             patch('morag_graph.builders.graph_builder.RelationExtractor') as mock_relation_extractor:
            
            # Setup mock extractors to fail
            mock_entity_extractor.return_value.extract = AsyncMock(side_effect=Exception("Extraction failed"))
            
            builder = GraphBuilder(
                storage=mock_storage,
                llm_config=mock_llm_config
            )
            
            result = await builder.process_document(
                content="Test content",
                document_id="test_doc"
            )
            
            assert isinstance(result, GraphBuildResult)
            assert result.document_id == "test_doc"
            assert result.entities_created == 0
            assert result.relations_created == 0
            assert result.skipped is False
            assert len(result.errors) == 1
            assert "Extraction failed" in result.errors[0]
    
    def test_graph_build_result_with_cleanup(self):
        """Test GraphBuildResult with cleanup result."""
        cleanup_result = CleanupResult(
            document_id="test_doc",
            entities_deleted=2,
            relations_deleted=1
        )
        
        result = GraphBuildResult(
            document_id="test_doc",
            entities_created=3,
            relations_created=2,
            entity_ids=["e1", "e2", "e3"],
            relation_ids=["r1", "r2"],
            cleanup_result=cleanup_result
        )
        
        assert result.document_id == "test_doc"
        assert result.entities_created == 3
        assert result.relations_created == 2
        assert result.cleanup_result == cleanup_result
        assert result.skipped is False
    
    def test_graph_build_result_skipped(self):
        """Test GraphBuildResult for skipped document."""
        result = GraphBuildResult(
            document_id="test_doc",
            entities_created=0,
            relations_created=0,
            entity_ids=[],
            relation_ids=[],
            skipped=True
        )
        
        assert result.document_id == "test_doc"
        assert result.entities_created == 0
        assert result.relations_created == 0
        assert result.skipped is True
        assert result.cleanup_result is None