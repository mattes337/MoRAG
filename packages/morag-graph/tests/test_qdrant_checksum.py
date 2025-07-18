"""Tests for QdrantStorage checksum management functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional

from morag_graph.storage.qdrant_storage import QdrantStorage, QdrantConfig
from morag_graph.models.entity import Entity


class MockFilter:
    """Mock Filter for testing."""
    def __init__(self, must=None, must_not=None):
        self.must = must or []
        self.must_not = must_not or []


class MockFieldCondition:
    """Mock FieldCondition for testing."""
    def __init__(self, key, match, should_not=False):
        self.key = key
        self.match = match
        self.should_not = should_not


class MockMatchValue:
    """Mock MatchValue for testing."""
    def __init__(self, value):
        self.value = value


class MockQdrantClient:
    """Mock Qdrant client for testing."""
    
    def __init__(self):
        self.points = {}  # Store points by ID
        self.collections = []
    
    async def get_collections(self):
        """Mock get_collections."""
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="test_collection")]
        return mock_collections
    
    async def create_collection(self, collection_name, vectors_config):
        """Mock create_collection."""
        self.collections.append(collection_name)
    
    async def upsert(self, collection_name, points):
        """Mock upsert."""
        for point in points:
            self.points[point.id] = point
    
    async def retrieve(self, collection_name, ids, with_payload=False, with_vectors=False):
        """Mock retrieve."""
        result = []
        for point_id in ids:
            if point_id in self.points:
                result.append(self.points[point_id])
        return result
    
    async def delete(self, collection_name, points_selector):
        """Mock delete."""
        for point_id in points_selector:
            if point_id in self.points:
                del self.points[point_id]
    
    async def scroll(self, collection_name, scroll_filter=None, limit=None, with_payload=True, with_vectors=True, offset=None):
        """Mock scroll method that filters points based on the provided filter."""
        # Mock scroll method
        filtered_points = []
        
        # If no filter, return all points
        if not scroll_filter:
            filtered_points = list(self.points.values())
        else:
            # Apply filter logic
            for point in self.points.values():
                matches = True
                
                # Check must conditions
                if hasattr(scroll_filter, 'must') and scroll_filter.must:
                    for condition in scroll_filter.must:
                        if hasattr(condition, 'key') and hasattr(condition, 'match'):
                            key = condition.key
                            expected_value = condition.match.value if hasattr(condition.match, 'value') else condition.match
                            actual_value = point.payload.get(key)
                            if actual_value != expected_value:
                                matches = False
                                break
                
                # Check must_not conditions
                if matches and hasattr(scroll_filter, 'must_not') and scroll_filter.must_not:
                    for condition in scroll_filter.must_not:
                        if hasattr(condition, 'key') and hasattr(condition, 'match'):
                            key = condition.key
                            expected_value = condition.match.value if hasattr(condition.match, 'value') else condition.match
                            actual_value = point.payload.get(key)
                            if actual_value == expected_value:
                                matches = False
                                break
                
                if matches:
                    filtered_points.append(point)
        
        # Apply limit
        if limit and len(filtered_points) > limit:
            filtered_points = filtered_points[:limit]
        
        return (filtered_points, None)  # Return tuple (points, next_page_offset)
    
    async def close(self):
        """Mock close."""
        pass


class MockPointStruct:
    """Mock PointStruct for testing."""
    
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    return MockQdrantClient()


@pytest.fixture
def qdrant_config():
    """Create QdrantConfig for testing."""
    return QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        vector_size=384
    )


@pytest.fixture
def qdrant_storage(qdrant_config):
    """Create QdrantStorage instance for testing."""
    with patch('morag_graph.storage.qdrant_storage.QDRANT_AVAILABLE', True):
        storage = QdrantStorage(qdrant_config)
        return storage


class TestQdrantStorageChecksum:
    """Test cases for QdrantStorage checksum management."""
    
    @pytest.mark.asyncio
    async def test_get_document_checksum_not_found(self, qdrant_storage, mock_qdrant_client):
        """Test getting checksum for non-existent document."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            qdrant_storage.client = mock_qdrant_client
            
            result = await qdrant_storage.get_document_checksum("nonexistent_doc")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_store_and_get_document_checksum(self, qdrant_storage, mock_qdrant_client):
        """Test storing and retrieving document checksum."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client), \
             patch('morag_graph.storage.qdrant_storage.PointStruct', MockPointStruct):
            
            qdrant_storage.client = mock_qdrant_client
            
            document_id = "test_doc"
            checksum = "abc123def456"
            
            # Store checksum
            await qdrant_storage.store_document_checksum(document_id, checksum)
            
            # Verify it was stored
            checksum_id = f"checksum_{document_id}"
            assert checksum_id in mock_qdrant_client.points
            
            stored_point = mock_qdrant_client.points[checksum_id]
            assert stored_point.payload["checksum"] == checksum
            assert stored_point.payload["document_id"] == document_id
            assert stored_point.payload["type"] == "document_checksum"
            
            # Retrieve checksum
            retrieved_checksum = await qdrant_storage.get_document_checksum(document_id)
            assert retrieved_checksum == checksum
    
    @pytest.mark.asyncio
    async def test_delete_document_checksum(self, qdrant_storage, mock_qdrant_client):
        """Test deleting document checksum."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client), \
             patch('morag_graph.storage.qdrant_storage.PointStruct', MockPointStruct):
            
            qdrant_storage.client = mock_qdrant_client
            
            document_id = "test_doc"
            checksum = "abc123def456"
            
            # Store checksum first
            await qdrant_storage.store_document_checksum(document_id, checksum)
            
            # Verify it exists
            checksum_id = f"checksum_{document_id}"
            assert checksum_id in mock_qdrant_client.points
            
            # Delete checksum
            await qdrant_storage.delete_document_checksum(document_id)
            
            # Verify it was deleted
            assert checksum_id not in mock_qdrant_client.points
            
            # Verify get returns None
            retrieved_checksum = await qdrant_storage.get_document_checksum(document_id)
            assert retrieved_checksum is None
    
    @pytest.mark.asyncio
    async def test_get_entities_by_document(self, qdrant_storage, mock_qdrant_client):
        """Test getting entities by document ID."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client), \
             patch('morag_graph.storage.qdrant_storage.PointStruct', MockPointStruct), \
             patch('morag_graph.storage.qdrant_storage.Filter', MockFilter), \
             patch('morag_graph.storage.qdrant_storage.FieldCondition', MockFieldCondition), \
             patch('morag_graph.storage.qdrant_storage.MatchValue', MockMatchValue):
            
            qdrant_storage.client = mock_qdrant_client
            
            document_id = "test_doc_123"
            
            # Create test entities
            entity1 = MockPointStruct(
                id="ent_john_doe_person_test_doc_123",
                vector=[0.1] * 384,
                payload={
                    "name": "John Doe",
                    "type": "PERSON",
                    "source_doc_id": document_id,
                    "confidence": 0.9
                }
            )
            
            entity2 = MockPointStruct(
                id="ent_acme_corp_organization_test_doc_123",
                vector=[0.2] * 384,
                payload={
                    "name": "Acme Corp",
                    "type": "ORGANIZATION",
                    "source_doc_id": document_id,
                    "confidence": 0.8
                }
            )
            
            # Entity from different document (should not be returned)
            entity3 = MockPointStruct(
                id="ent_other_entity_person_other_doc",
                vector=[0.3] * 384,
                payload={
                    "name": "Other Entity",
                    "type": "PERSON",
                    "source_doc_id": "other_doc",
                    "confidence": 0.7
                }
            )
            
            # Checksum entry (should be excluded)
            checksum_point = MockPointStruct(
                id=f"checksum_{document_id}",
                vector=[0.0] * 384,
                payload={
                    "type": "document_checksum",
                    "document_id": document_id,
                    "checksum": "abc123"
                }
            )
            
            # Add to mock client
            mock_qdrant_client.points["ent_john_doe_person_test_doc_123"] = entity1
            mock_qdrant_client.points["ent_acme_corp_organization_test_doc_123"] = entity2
            mock_qdrant_client.points["ent_other_entity_person_other_doc"] = entity3
            mock_qdrant_client.points[f"checksum_{document_id}"] = checksum_point
            
            # Get entities by document
            entities = await qdrant_storage.get_entities_by_document(document_id)
            
            # Should return only entities from the specified document (excluding checksum)
            assert len(entities) == 2, f"Expected 2 entities, got {len(entities)}"
            
            entity_names = [e.name for e in entities]
            assert "John Doe" in entity_names
            assert "Acme Corp" in entity_names
            assert "Other Entity" not in entity_names
            
            # Verify entity properties
            john_entity = next(e for e in entities if e.name == "John Doe")
            assert john_entity.id == "ent_john_doe_person_test_doc_123"
            assert john_entity.type == "PERSON"
            assert john_entity.source_doc_id == document_id
            assert john_entity.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_get_entities_by_document_empty(self, qdrant_storage, mock_qdrant_client):
        """Test getting entities for document with no entities."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            qdrant_storage.client = mock_qdrant_client
            
            entities = await qdrant_storage.get_entities_by_document("nonexistent_doc")
            assert entities == []
    
    @pytest.mark.asyncio
    async def test_checksum_methods_without_client(self, qdrant_storage):
        """Test checksum methods raise error when not connected."""
        # Ensure client is None
        qdrant_storage.client = None
        
        with pytest.raises(RuntimeError, match="Not connected to Qdrant database"):
            await qdrant_storage.get_document_checksum("test_doc")
        
        with pytest.raises(RuntimeError, match="Not connected to Qdrant database"):
            await qdrant_storage.store_document_checksum("test_doc", "checksum")
        
        with pytest.raises(RuntimeError, match="Not connected to Qdrant database"):
            await qdrant_storage.delete_document_checksum("test_doc")
        
        with pytest.raises(RuntimeError, match="Not connected to Qdrant database"):
            await qdrant_storage.get_entities_by_document("test_doc")
    
    @pytest.mark.asyncio
    async def test_store_checksum_error_handling(self, qdrant_storage, mock_qdrant_client):
        """Test error handling in store_document_checksum."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            qdrant_storage.client = mock_qdrant_client
            
            # Mock upsert to raise an exception
            mock_qdrant_client.upsert = AsyncMock(side_effect=Exception("Upsert failed"))
            
            with pytest.raises(Exception, match="Upsert failed"):
                await qdrant_storage.store_document_checksum("test_doc", "checksum")
    
    @pytest.mark.asyncio
    async def test_get_checksum_error_handling(self, qdrant_storage, mock_qdrant_client):
        """Test error handling in get_document_checksum."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            qdrant_storage.client = mock_qdrant_client
            
            # Mock retrieve to raise an exception
            mock_qdrant_client.retrieve = AsyncMock(side_effect=Exception("Retrieve failed"))
            
            result = await qdrant_storage.get_document_checksum("test_doc")
            assert result is None  # Should return None on error
    
    @pytest.mark.asyncio
    async def test_delete_checksum_error_handling(self, qdrant_storage, mock_qdrant_client):
        """Test error handling in delete_document_checksum."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            qdrant_storage.client = mock_qdrant_client
            
            # Mock delete to raise an exception
            mock_qdrant_client.delete = AsyncMock(side_effect=Exception("Delete failed"))
            
            # Should not raise exception (error is logged but not raised)
            await qdrant_storage.delete_document_checksum("test_doc")
    
    @pytest.mark.asyncio
    async def test_get_entities_error_handling(self, qdrant_storage, mock_qdrant_client):
        """Test error handling in get_entities_by_document."""
        with patch('morag_graph.storage.qdrant_storage.AsyncQdrantClient', return_value=mock_qdrant_client):
            qdrant_storage.client = mock_qdrant_client
            
            # Mock scroll to raise an exception
            mock_qdrant_client.scroll = AsyncMock(side_effect=Exception("Scroll failed"))
            
            result = await qdrant_storage.get_entities_by_document("test_doc")
            assert result == []  # Should return empty list on error