import pytest
import asyncio
import uuid
from unittest.mock import patch, AsyncMock, MagicMock
from morag_services.storage import qdrant_service, QdrantService
from morag_core.exceptions import StorageError

@pytest.fixture
def mock_qdrant_service():
    """Create a mock Qdrant service for testing."""
    service = QdrantService()
    service.collection_name = "test_morag_documents"

    # Mock the client
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.create_collection = MagicMock()
    mock_client.get_collection.return_value = MagicMock(
        vectors_count=0,
        indexed_vectors_count=0,
        points_count=0,
        status=MagicMock(value="green"),
        optimizer_status=MagicMock(status=MagicMock(value="ok")),
        config=MagicMock(
            params=MagicMock(
                vectors=MagicMock(
                    size=768,
                    distance=MagicMock(value="Cosine")
                )
            )
        )
    )
    mock_client.upsert = MagicMock()
    mock_client.search.return_value = []
    mock_client.scroll.return_value = ([], None)
    mock_client.delete = MagicMock()
    mock_client.delete_collection = MagicMock()
    mock_client.close = MagicMock()

    service.client = mock_client
    return service

class TestQdrantService:
    """Test Qdrant service functionality."""

    @pytest.mark.asyncio
    async def test_connection_success(self):
        """Test successful Qdrant connection."""
        service = QdrantService()

        with patch('morag.services.storage.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_client_class.return_value = mock_client

            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = MagicMock(collections=[])

                await service.connect()

                assert service.client is not None
                mock_client_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test Qdrant connection failure."""
        service = QdrantService()

        with patch('morag.services.storage.QdrantClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            with pytest.raises(StorageError) as exc_info:
                await service.connect()

            assert "Failed to connect to Qdrant" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_qdrant_service):
        """Test Qdrant disconnection."""
        service = mock_qdrant_service

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            await service.disconnect()

            mock_to_thread.assert_called_once()
            assert service.client is None

    @pytest.mark.asyncio
    async def test_create_collection_new(self, mock_qdrant_service):
        """Test creating a new collection."""
        service = mock_qdrant_service

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock empty collections list
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None  # create_collection
            ]

            await service.create_collection(vector_size=768, force_recreate=False)

            assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_create_collection_exists_no_force(self, mock_qdrant_service):
        """Test creating collection when it already exists without force."""
        service = mock_qdrant_service

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock existing collection
            existing_collection = MagicMock()
            existing_collection.name = service.collection_name
            mock_to_thread.return_value = MagicMock(collections=[existing_collection])

            await service.create_collection(vector_size=768, force_recreate=False)

            # Should only call get_collections, not create_collection
            assert mock_to_thread.call_count == 1

    @pytest.mark.asyncio
    async def test_create_collection_force_recreate(self, mock_qdrant_service):
        """Test force recreating an existing collection."""
        service = mock_qdrant_service

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock existing collection
            existing_collection = MagicMock()
            existing_collection.name = service.collection_name
            mock_to_thread.side_effect = [
                MagicMock(collections=[existing_collection]),  # get_collections
                None,  # delete_collection
                None   # create_collection
            ]

            await service.create_collection(vector_size=768, force_recreate=True)

            assert mock_to_thread.call_count == 3

    @pytest.mark.asyncio
    async def test_store_chunks_success(self, mock_qdrant_service):
        """Test successful chunk storage."""
        service = mock_qdrant_service

        chunks = [
            {
                "text": "Test chunk 1",
                "summary": "Summary 1",
                "source": "test.pdf",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"test": True}
            }
        ]

        embeddings = [[0.1] * 768]

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            point_ids = await service.store_chunks(chunks, embeddings)

            assert len(point_ids) == 1
            assert all(isinstance(pid, str) for pid in point_ids)
            mock_to_thread.assert_called()

    @pytest.mark.asyncio
    async def test_store_chunks_mismatch(self, mock_qdrant_service):
        """Test chunk storage with mismatched chunks and embeddings."""
        service = mock_qdrant_service

        chunks = [{"text": "test"}]
        embeddings = [[0.1] * 768, [0.2] * 768]  # More embeddings than chunks

        with pytest.raises(StorageError) as exc_info:
            await service.store_chunks(chunks, embeddings)

        assert "Number of chunks must match number of embeddings" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_similar_success(self, mock_qdrant_service):
        """Test successful similarity search."""
        service = mock_qdrant_service

        # Mock search results
        mock_result = MagicMock()
        mock_result.id = "test-id"
        mock_result.score = 0.95
        mock_result.payload = {
            "text": "Test text",
            "summary": "Test summary",
            "source": "test.pdf",
            "source_type": "document",
            "metadata": {"test": True}
        }

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = [mock_result]

            results = await service.search_similar(
                query_embedding=[0.1] * 768,
                limit=10,
                score_threshold=0.7
            )

            assert len(results) == 1
            assert results[0]["id"] == "test-id"
            assert results[0]["score"] == 0.95
            assert results[0]["text"] == "Test text"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_qdrant_service):
        """Test search with filters."""
        service = mock_qdrant_service

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = []

            results = await service.search_similar(
                query_embedding=[0.1] * 768,
                limit=10,
                score_threshold=0.7,
                filters={"source": "test.pdf"}
            )

            assert len(results) == 0
            mock_to_thread.assert_called()

    @pytest.mark.asyncio
    async def test_delete_by_source(self, mock_qdrant_service):
        """Test deletion by source."""
        service = mock_qdrant_service

        # Mock scroll results
        mock_point = MagicMock()
        mock_point.id = "test-id"

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                ([mock_point], None),  # scroll result
                None  # delete result
            ]

            deleted_count = await service.delete_by_source("test.pdf")

            assert deleted_count == 1
            assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_get_collection_info(self, mock_qdrant_service):
        """Test getting collection information."""
        service = mock_qdrant_service

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_info = MagicMock()
            mock_info.vectors_count = 100
            mock_info.indexed_vectors_count = 100
            mock_info.points_count = 100
            mock_info.status.value = "green"
            mock_info.optimizer_status.status.value = "ok"
            mock_info.config.params.vectors.size = 768
            mock_info.config.params.vectors.distance.value = "Cosine"

            mock_to_thread.return_value = mock_info

            info = await service.get_collection_info()

            assert info["vectors_count"] == 100
            assert info["points_count"] == 100
            assert info["status"] == "green"
            assert info["config"]["vector_size"] == 768
            assert info["config"]["distance"] == "Cosine"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various operations."""
        service = QdrantService()

        # Test operations without connection
        with pytest.raises(StorageError):
            await service.get_collection_info()

        # Test with connection but operation failure
        service.client = MagicMock()

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = Exception("Operation failed")

            with pytest.raises(StorageError):
                await service.get_collection_info()

class TestQdrantServiceIntegration:
    """Test Qdrant service integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_workflow_mock(self, mock_qdrant_service):
        """Test a complete workflow with mocked Qdrant."""
        service = mock_qdrant_service

        # Test collection creation
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None  # create_collection
            ]

            await service.create_collection(vector_size=768)

        # Test storing chunks
        chunks = [
            {
                "text": "Integration test chunk",
                "summary": "Integration summary",
                "source": "integration.pdf",
                "source_type": "document",
                "chunk_index": 0,
                "metadata": {"integration": True}
            }
        ]
        embeddings = [[0.5] * 768]

        with patch('asyncio.to_thread', new_callable=AsyncMock):
            point_ids = await service.store_chunks(chunks, embeddings)
            assert len(point_ids) == 1

        # Test search
        mock_result = MagicMock()
        mock_result.id = point_ids[0]
        mock_result.score = 0.95
        mock_result.payload = chunks[0]

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = [mock_result]

            results = await service.search_similar([0.5] * 768)
            assert len(results) == 1
            assert results[0]["text"] == "Integration test chunk"
