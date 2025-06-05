"""Unit tests for vector storage implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from morag_services.storage import QdrantService
from morag_core.exceptions import StorageError


class MockEmbeddingResult:
    """Mock embedding result for testing."""
    def __init__(self, embedding: List[float]):
        self.embedding = embedding
        self.token_count = 100
        self.model = "test-model"


class TestVectorStorage:
    """Test vector storage functionality."""

    @pytest.fixture
    def qdrant_service(self):
        """Create QdrantService instance for testing."""
        return QdrantService()

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        client.create_collection = MagicMock()
        client.upsert = MagicMock()
        client.search = MagicMock()
        client.scroll = MagicMock()
        client.delete = MagicMock()
        client.close = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_store_embedding_with_embedding_result(self, qdrant_service, mock_client):
        """Test storing embedding with EmbeddingResult object."""
        qdrant_service.client = mock_client
        
        # Mock collection check
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None,  # create_collection
                None   # upsert
            ]
            
            embedding_result = MockEmbeddingResult([0.1, 0.2, 0.3] * 256)  # 768 dimensions
            text = "Test document content"
            metadata = {"source": "test", "type": "document"}
            
            point_id = await qdrant_service.store_embedding(
                embedding=embedding_result,
                text=text,
                metadata=metadata,
                collection_name="test_collection"
            )
            
            assert point_id is not None
            assert len(point_id) > 0
            
            # Verify upsert was called
            assert mock_to_thread.call_count == 3

    @pytest.mark.asyncio
    async def test_store_embedding_with_raw_list(self, qdrant_service, mock_client):
        """Test storing embedding with raw float list."""
        qdrant_service.client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None,  # create_collection
                None   # upsert
            ]
            
            embedding_vector = [0.1, 0.2, 0.3] * 256  # 768 dimensions
            text = "Test document content"
            metadata = {"source": "test", "type": "document"}
            
            point_id = await qdrant_service.store_embedding(
                embedding=embedding_vector,
                text=text,
                metadata=metadata
            )
            
            assert point_id is not None
            assert len(point_id) > 0

    @pytest.mark.asyncio
    async def test_store_chunk(self, qdrant_service, mock_client):
        """Test storing text chunk with embedding."""
        qdrant_service.client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None,  # create_collection
                None   # upsert
            ]
            
            chunk_id = "test_chunk_001"
            text = "This is a test chunk of text"
            summary = "Test chunk summary"
            embedding = [0.1, 0.2, 0.3] * 256
            metadata = {"source": "test_doc", "chunk_index": 0}
            
            result_id = await qdrant_service.store_chunk(
                chunk_id=chunk_id,
                text=text,
                summary=summary,
                embedding=embedding,
                metadata=metadata
            )
            
            assert result_id == chunk_id

    @pytest.mark.asyncio
    async def test_create_collection_if_not_exists_new(self, qdrant_service, mock_client):
        """Test creating new collection."""
        qdrant_service.client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock empty collections list
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None  # create_collection
            ]
            
            await qdrant_service.create_collection_if_not_exists("new_collection", 768)
            
            assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_create_collection_if_not_exists_existing(self, qdrant_service, mock_client):
        """Test with existing collection."""
        qdrant_service.client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock existing collection
            existing_collection = MagicMock()
            existing_collection.name = "existing_collection"
            mock_to_thread.return_value = MagicMock(collections=[existing_collection])
            
            await qdrant_service.create_collection_if_not_exists("existing_collection", 768)
            
            # Should only call get_collections, not create_collection
            assert mock_to_thread.call_count == 1

    @pytest.mark.asyncio
    async def test_search_by_metadata(self, qdrant_service, mock_client):
        """Test metadata-only search."""
        qdrant_service.client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock search results
            mock_point = MagicMock()
            mock_point.id = "test_id"
            mock_point.payload = {
                "text": "Test content",
                "summary": "Test summary",
                "metadata": {"source": "test"}
            }
            
            mock_to_thread.return_value = ([mock_point], None)
            
            filters = {"source": "test"}
            results = await qdrant_service.search_by_metadata(filters, limit=5)
            
            assert len(results) == 1
            assert results[0]["id"] == "test_id"
            assert results[0]["text"] == "Test content"

    @pytest.mark.asyncio
    async def test_batch_store_embeddings(self, qdrant_service, mock_client):
        """Test batch storage of embeddings."""
        qdrant_service.client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None,  # create_collection
                None,  # upsert batch 1
                None   # upsert batch 2 (if needed)
            ]
            
            embeddings_data = [
                {
                    "embedding": [0.1, 0.2, 0.3] * 256,
                    "text": "First document",
                    "metadata": {"source": "doc1"}
                },
                {
                    "embedding": [0.4, 0.5, 0.6] * 256,
                    "text": "Second document", 
                    "metadata": {"source": "doc2"}
                }
            ]
            
            point_ids = await qdrant_service.batch_store_embeddings(
                embeddings_data, 
                batch_size=2
            )
            
            assert len(point_ids) == 2
            assert all(isinstance(pid, str) for pid in point_ids)

    @pytest.mark.asyncio
    async def test_list_collections(self, qdrant_service, mock_client):
        """Test listing all collections."""
        qdrant_service.client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock collections
            mock_collection = MagicMock()
            mock_collection.name = "test_collection"
            
            mock_info = MagicMock()
            mock_info.vectors_count = 100
            mock_info.points_count = 100
            mock_info.status.value = "green"
            mock_info.config.params.vectors.size = 768
            mock_info.config.params.vectors.distance.value = "Cosine"
            
            mock_to_thread.side_effect = [
                MagicMock(collections=[mock_collection]),  # get_collections
                mock_info  # get_collection
            ]
            
            collections = await qdrant_service.list_collections()
            
            assert len(collections) == 1
            assert collections[0]["name"] == "test_collection"
            assert collections[0]["vectors_count"] == 100

    @pytest.mark.asyncio
    async def test_error_handling(self, qdrant_service, mock_client):
        """Test error handling in storage operations."""
        qdrant_service.client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = Exception("Connection failed")
            
            with pytest.raises(StorageError):
                await qdrant_service.store_embedding(
                    embedding=[0.1] * 768,
                    text="test",
                    metadata={}
                )

    @pytest.mark.asyncio
    async def test_connection_auto_initialization(self, qdrant_service, mock_client):
        """Test that connection is automatically initialized when needed."""
        # Ensure client is None initially
        qdrant_service.client = None

        async def mock_connect():
            qdrant_service.client = mock_client

        with patch.object(qdrant_service, 'connect', side_effect=mock_connect) as mock_connect:
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.side_effect = [
                    MagicMock(collections=[]),  # get_collections
                    None,  # create_collection
                    None   # upsert
                ]

                await qdrant_service.store_embedding(
                    embedding=[0.1] * 768,
                    text="test",
                    metadata={}
                )

                mock_connect.assert_called_once()
