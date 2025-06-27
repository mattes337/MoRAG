"""Tests for DocumentChecksumManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from morag_graph.updates.checksum_manager import DocumentChecksumManager
from morag_graph.storage.base import BaseStorage


class MockStorage(BaseStorage):
    """Mock storage for testing."""
    
    def __init__(self):
        self.checksums = {}
    
    async def get_document_checksum(self, document_id: str):
        return self.checksums.get(document_id)
    
    async def store_document_checksum(self, document_id: str, checksum: str):
        self.checksums[document_id] = checksum
    
    async def delete_document_checksum(self, document_id: str):
        if document_id in self.checksums:
            del self.checksums[document_id]
    
    # Required abstract methods (not used in tests)
    async def connect(self): pass
    async def disconnect(self): pass
    async def store_entity(self, entity): pass
    async def store_entities(self, entities): return []
    async def get_entity(self, entity_id): pass
    async def get_entities(self, entity_ids): pass
    async def search_entities(self, query, entity_type=None, limit=10): return []
    async def update_entity(self, entity): pass
    async def delete_entity(self, entity_id): pass
    async def store_relation(self, relation): pass
    async def store_relations(self, relations): return []
    async def get_relation(self, relation_id): pass
    async def get_relations(self, relation_ids): pass
    async def get_entity_relations(self, entity_id, relation_type=None, direction="both"): pass
    async def update_relation(self, relation): pass
    async def delete_relation(self, relation_id): pass
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
def checksum_manager(mock_storage):
    return DocumentChecksumManager(mock_storage)


class TestDocumentChecksumManager:
    """Test cases for DocumentChecksumManager."""
    
    def test_calculate_document_checksum_content_only(self, checksum_manager):
        """Test checksum calculation with content only."""
        content = "This is test content"
        checksum = checksum_manager.calculate_document_checksum(content)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest length
        
        # Same content should produce same checksum
        checksum2 = checksum_manager.calculate_document_checksum(content)
        assert checksum == checksum2
    
    def test_calculate_document_checksum_with_metadata(self, checksum_manager):
        """Test checksum calculation with content and metadata."""
        content = "This is test content"
        metadata = {"author": "test", "version": 1}
        
        checksum_with_meta = checksum_manager.calculate_document_checksum(content, metadata)
        checksum_without_meta = checksum_manager.calculate_document_checksum(content)
        
        # Checksums should be different
        assert checksum_with_meta != checksum_without_meta
        
        # Same content and metadata should produce same checksum
        checksum2 = checksum_manager.calculate_document_checksum(content, metadata)
        assert checksum_with_meta == checksum2
    
    def test_calculate_document_checksum_metadata_order(self, checksum_manager):
        """Test that metadata order doesn't affect checksum."""
        content = "This is test content"
        metadata1 = {"author": "test", "version": 1}
        metadata2 = {"version": 1, "author": "test"}  # Different order
        
        checksum1 = checksum_manager.calculate_document_checksum(content, metadata1)
        checksum2 = checksum_manager.calculate_document_checksum(content, metadata2)
        
        # Should be the same despite different order
        assert checksum1 == checksum2
    
    @pytest.mark.asyncio
    async def test_get_stored_checksum_exists(self, checksum_manager, mock_storage):
        """Test getting existing checksum."""
        document_id = "test_doc"
        expected_checksum = "abc123"
        mock_storage.checksums[document_id] = expected_checksum
        
        result = await checksum_manager.get_stored_checksum(document_id)
        assert result == expected_checksum
    
    @pytest.mark.asyncio
    async def test_get_stored_checksum_not_exists(self, checksum_manager):
        """Test getting non-existent checksum."""
        document_id = "nonexistent_doc"
        
        result = await checksum_manager.get_stored_checksum(document_id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_store_document_checksum(self, checksum_manager, mock_storage):
        """Test storing document checksum."""
        document_id = "test_doc"
        checksum = "abc123"
        
        await checksum_manager.store_document_checksum(document_id, checksum)
        
        # Verify it was stored
        assert mock_storage.checksums[document_id] == checksum
    
    @pytest.mark.asyncio
    async def test_needs_update_new_document(self, checksum_manager):
        """Test needs_update for new document."""
        document_id = "new_doc"
        content = "New content"
        
        result = await checksum_manager.needs_update(document_id, content)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_needs_update_unchanged_document(self, checksum_manager, mock_storage):
        """Test needs_update for unchanged document."""
        document_id = "existing_doc"
        content = "Existing content"
        
        # Calculate and store checksum
        checksum = checksum_manager.calculate_document_checksum(content)
        mock_storage.checksums[document_id] = checksum
        
        result = await checksum_manager.needs_update(document_id, content)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_needs_update_changed_document(self, checksum_manager, mock_storage):
        """Test needs_update for changed document."""
        document_id = "existing_doc"
        old_content = "Old content"
        new_content = "New content"
        
        # Store checksum for old content
        old_checksum = checksum_manager.calculate_document_checksum(old_content)
        mock_storage.checksums[document_id] = old_checksum
        
        # Check with new content
        result = await checksum_manager.needs_update(document_id, new_content)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_needs_update_with_metadata_change(self, checksum_manager, mock_storage):
        """Test needs_update when metadata changes."""
        document_id = "existing_doc"
        content = "Same content"
        old_metadata = {"version": 1}
        new_metadata = {"version": 2}
        
        # Store checksum with old metadata
        old_checksum = checksum_manager.calculate_document_checksum(content, old_metadata)
        mock_storage.checksums[document_id] = old_checksum
        
        # Check with new metadata
        result = await checksum_manager.needs_update(document_id, content, new_metadata)
        assert result is True