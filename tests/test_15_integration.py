"""Integration tests for vector storage implementation."""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from morag_core.exceptions import StorageError
from morag_services.embedding import EmbeddingResult, GeminiService
from morag_services.storage import QdrantService


class TestVectorStorageIntegration:
    """Integration tests for vector storage."""

    @pytest.fixture
    def qdrant_service(self):
        """Create QdrantService instance for testing."""
        return QdrantService()

    @pytest.fixture
    def mock_gemini_service(self):
        """Create mock Gemini service."""
        service = MagicMock(spec=GeminiService)

        async def mock_generate_embedding(text, task_type="retrieval_document"):
            # Return consistent embedding based on text hash
            embedding = [hash(text) % 1000 / 1000.0] * 768
            return EmbeddingResult(
                embedding=embedding,
                token_count=len(text.split()),
                model="text-embedding-004",
            )

        service.generate_embedding = mock_generate_embedding
        return service

    @pytest.mark.asyncio
    async def test_end_to_end_storage_workflow(
        self, qdrant_service, mock_gemini_service
    ):
        """Test complete storage workflow from text to retrieval."""

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            mock_client.search.return_value = []
            mock_client_class.return_value = mock_client

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.side_effect = [
                    MagicMock(collections=[]),  # connect - get_collections
                    MagicMock(
                        collections=[]
                    ),  # create_collection_if_not_exists - get_collections
                    None,  # create_collection
                    None,  # store_embedding - upsert
                    [],  # search_similar - search
                ]

                # Step 1: Connect to Qdrant
                await qdrant_service.connect()

                # Step 2: Generate embedding
                text = "This is a test document about machine learning and AI."
                embedding_result = await mock_gemini_service.generate_embedding(text)

                # Step 3: Store embedding
                metadata = {
                    "source": "test_document.txt",
                    "source_type": "document",
                    "topic": "machine_learning",
                }

                point_id = await qdrant_service.store_embedding(
                    embedding=embedding_result,
                    text=text,
                    metadata=metadata,
                    collection_name="test_documents",
                )

                assert point_id is not None

                # Step 4: Search for similar content
                query_embedding = await mock_gemini_service.generate_embedding(
                    "AI and machine learning"
                )

                results = await qdrant_service.search_similar(
                    query_embedding=query_embedding.embedding,
                    limit=5,
                    score_threshold=0.5,
                )

                # Verify the workflow completed without errors
                assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_multiple_collections_workflow(self, qdrant_service):
        """Test working with multiple collections."""

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            mock_client_class.return_value = mock_client

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.side_effect = [
                    MagicMock(collections=[]),  # connect
                    MagicMock(collections=[]),  # create collection 1
                    None,  # create collection 1
                    None,  # store in collection 1
                    MagicMock(collections=[]),  # create collection 2
                    None,  # create collection 2
                    None,  # store in collection 2
                    MagicMock(collections=[]),  # list collections
                ]

                await qdrant_service.connect()

                # Store in documents collection
                await qdrant_service.store_embedding(
                    embedding=[0.1] * 768,
                    text="Document content",
                    metadata={"type": "document"},
                    collection_name="documents",
                )

                # Store in images collection
                await qdrant_service.store_embedding(
                    embedding=[0.2] * 768,
                    text="Image caption",
                    metadata={"type": "image"},
                    collection_name="images",
                )

                # List all collections
                collections = await qdrant_service.list_collections()

                # Verify operations completed
                assert mock_to_thread.call_count >= 6

    @pytest.mark.asyncio
    async def test_large_batch_operations(self, qdrant_service):
        """Test performance with large batches."""

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            mock_client_class.return_value = mock_client

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                # Prepare large batch
                batch_size = 250
                embeddings_data = []

                for i in range(batch_size):
                    embeddings_data.append(
                        {
                            "embedding": [i / batch_size] * 768,
                            "text": f"Document {i} content",
                            "metadata": {"doc_id": i, "batch": "large_test"},
                        }
                    )

                # Mock responses for batch operations
                mock_responses = [
                    MagicMock(collections=[]),  # connect
                    MagicMock(collections=[]),  # create_collection_if_not_exists
                    None,  # create_collection
                ]

                # Add upsert responses for each batch (250 items / 100 batch_size = 3 batches)
                for _ in range(3):
                    mock_responses.append(None)  # upsert

                mock_to_thread.side_effect = mock_responses

                await qdrant_service.connect()

                # Perform batch storage
                point_ids = await qdrant_service.batch_store_embeddings(
                    embeddings_data=embeddings_data, batch_size=100
                )

                assert len(point_ids) == batch_size
                assert all(isinstance(pid, str) for pid in point_ids)

    @pytest.mark.asyncio
    async def test_metadata_search_integration(self, qdrant_service):
        """Test metadata search functionality."""

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_client_class.return_value = mock_client

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                # Mock search results
                mock_point = MagicMock()
                mock_point.id = "doc_123"
                mock_point.payload = {
                    "text": "Machine learning document",
                    "summary": "About ML algorithms",
                    "metadata": {
                        "source": "ml_paper.pdf",
                        "topic": "machine_learning",
                        "author": "Dr. Smith",
                    },
                }

                mock_to_thread.side_effect = [
                    MagicMock(collections=[]),  # connect
                    ([mock_point], None),  # scroll (metadata search)
                ]

                await qdrant_service.connect()

                # Search by metadata
                results = await qdrant_service.search_by_metadata(
                    filters={"topic": "machine_learning"}, limit=10
                )

                assert len(results) == 1
                assert results[0]["id"] == "doc_123"
                assert results[0]["metadata"]["topic"] == "machine_learning"

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, qdrant_service):
        """Test error recovery and system resilience."""

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                # First call fails, second succeeds
                mock_to_thread.side_effect = [
                    Exception("Network timeout"),  # First attempt fails
                    MagicMock(collections=[]),  # Retry succeeds
                    None,  # Subsequent operation
                ]

                # First operation should fail
                with pytest.raises(StorageError):
                    await qdrant_service.connect()

                # Reset the client to simulate recovery
                qdrant_service.client = None

                # Second operation should succeed
                await qdrant_service.connect()

                # Verify recovery worked
                assert qdrant_service.client is not None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, qdrant_service):
        """Test concurrent storage operations."""

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_client.create_collection = MagicMock()
            mock_client.upsert = MagicMock()
            mock_client_class.return_value = mock_client

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                # Mock responses for concurrent operations
                mock_to_thread.side_effect = [
                    MagicMock(collections=[]),  # connect
                ] + [
                    MagicMock(collections=[]),
                    None,
                    None,
                ] * 5  # 5 concurrent operations

                await qdrant_service.connect()

                # Create concurrent storage tasks
                tasks = []
                for i in range(5):
                    task = qdrant_service.store_embedding(
                        embedding=[i / 10.0] * 768,
                        text=f"Concurrent document {i}",
                        metadata={"doc_id": i, "test": "concurrent"},
                    )
                    tasks.append(task)

                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Verify all operations completed successfully
                assert len(results) == 5
                assert all(
                    isinstance(result, str) for result in results
                )  # All should return point IDs

    @pytest.mark.asyncio
    async def test_collection_lifecycle_management(self, qdrant_service):
        """Test complete collection lifecycle."""

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_client.create_collection = MagicMock()
            mock_client.delete_collection = MagicMock()
            mock_client.get_collection.return_value = MagicMock(
                vectors_count=0,
                points_count=0,
                status=MagicMock(value="green"),
                config=MagicMock(
                    params=MagicMock(
                        vectors=MagicMock(size=768, distance=MagicMock(value="Cosine"))
                    )
                ),
            )
            mock_client_class.return_value = mock_client

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.side_effect = [
                    MagicMock(collections=[]),  # connect
                    MagicMock(collections=[]),  # create_collection - check
                    None,  # create_collection - create
                    mock_client.get_collection.return_value,  # get_collection_info
                    ([], None),  # scroll (delete_by_source)
                ]

                await qdrant_service.connect()

                # Create collection
                await qdrant_service.create_collection_if_not_exists(
                    "test_lifecycle", 768
                )

                # Get collection info
                info = await qdrant_service.get_collection_info()
                assert info["config"]["vector_size"] == 768

                # Delete collection (using existing method)
                await qdrant_service.delete_by_source("test_source")

                # Verify lifecycle completed
                assert mock_to_thread.call_count >= 4
