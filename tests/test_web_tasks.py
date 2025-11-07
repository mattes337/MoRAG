"""Tests for web tasks."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.morag.processors.document import DocumentChunk
from src.morag.processors.web import WebContent, WebScrapingResult
from src.morag.tasks.web_tasks import process_web_url, process_web_urls_batch


class TestWebTasks:
    """Test cases for web processing tasks."""

    @pytest.fixture
    def mock_web_content(self):
        """Create mock web content."""
        return WebContent(
            url="https://example.com",
            title="Test Page",
            content="This is test content for web scraping.",
            markdown_content="# Test Page\n\nThis is test content for web scraping.",
            metadata={
                "title": "Test Page",
                "description": "A test page",
                "domain": "example.com",
            },
            links=["https://example.com/link1", "https://example.com/link2"],
            images=["https://example.com/image1.jpg"],
            extraction_time=1.5,
            content_length=100,
            content_type="text/html",
        )

    @pytest.fixture
    def mock_chunks(self):
        """Create mock document chunks."""
        return [
            DocumentChunk(
                text="This is test content",
                chunk_type="text",
                page_number=1,
                element_id="chunk_1",
                metadata={"source": "web"},
            ),
            DocumentChunk(
                text="for web scraping.",
                chunk_type="text",
                page_number=1,
                element_id="chunk_2",
                metadata={"source": "web"},
            ),
        ]

    @pytest.fixture
    def mock_successful_result(self, mock_web_content, mock_chunks):
        """Create mock successful web scraping result."""
        return WebScrapingResult(
            url="https://example.com",
            content=mock_web_content,
            chunks=mock_chunks,
            processing_time=2.0,
            success=True,
        )

    @pytest.fixture
    def mock_failed_result(self):
        """Create mock failed web scraping result."""
        return WebScrapingResult(
            url="https://example.com",
            content=None,
            chunks=[],
            processing_time=1.0,
            success=False,
            error_message="Network error occurred",
        )

    @pytest.mark.asyncio
    async def test_process_web_url_success(self, mock_successful_result):
        """Test successful web URL processing."""
        # Mock the web processor
        with patch("src.morag.tasks.web_tasks.web_processor") as mock_processor:
            mock_processor.process_url = AsyncMock(return_value=mock_successful_result)

            # Mock the embedding and storage services
            with patch("src.morag.tasks.web_tasks.embedding_service") as mock_embedding:
                with patch("src.morag.tasks.web_tasks.storage_service") as mock_storage:
                    from src.morag.services.embedding import EmbeddingResult

                    mock_embedding.generate_embedding = AsyncMock(
                        return_value=EmbeddingResult(
                            embedding=[0.1, 0.2, 0.3], token_count=10, model="test"
                        )
                    )
                    mock_storage.store_chunk_with_embedding = AsyncMock()

                    # Create a mock task instance
                    mock_task = Mock()
                    mock_task.update_status = AsyncMock()

                    # Call the task function
                    result = await process_web_url(
                        mock_task,
                        "https://example.com",
                        {"timeout": 30},
                        "test_task_id",
                    )

                    # Verify the result
                    assert result["success"] is True
                    assert result["url"] == "https://example.com"
                    assert result["title"] == "Test Page"
                    assert result["content_length"] == 100
                    assert result["chunks_created"] == 2
                    assert result["links_found"] == 2
                    assert result["images_found"] == 1
                    assert result["embeddings_stored"] == 2
                    assert "processing_time" in result

                    # Verify status updates were called
                    mock_task.update_status.assert_any_call(
                        "PROCESSING", {"stage": "url_validation"}
                    )
                    mock_task.update_status.assert_any_call(
                        "PROCESSING", {"stage": "content_extraction"}
                    )
                    mock_task.update_status.assert_any_call(
                        "PROCESSING", {"stage": "content_chunking"}
                    )
                    mock_task.update_status.assert_any_call(
                        "PROCESSING", {"stage": "embedding_generation"}
                    )
                    mock_task.update_status.assert_any_call("COMPLETED", result)

    @pytest.mark.asyncio
    async def test_process_web_url_failure(self, mock_failed_result):
        """Test failed web URL processing."""
        # Mock the web processor to return failure
        with patch("src.morag.tasks.web_tasks.web_processor") as mock_processor:
            mock_processor.process_url = AsyncMock(return_value=mock_failed_result)

            # Create a mock task instance
            mock_task = Mock()
            mock_task.update_status = AsyncMock()

            # Call the task function
            result = await process_web_url(
                mock_task, "https://example.com", None, "test_task_id"
            )

            # Verify the result
            assert result["success"] is False
            assert result["error"] == "Network error occurred"
            assert result["url"] == "https://example.com"

            # Verify status updates
            mock_task.update_status.assert_any_call(
                "PROCESSING", {"stage": "url_validation"}
            )
            mock_task.update_status.assert_any_call(
                "FAILED", {"error": "Network error occurred"}
            )

    @pytest.mark.asyncio
    async def test_process_web_url_exception(self):
        """Test web URL processing with exception."""
        # Mock the web processor to raise an exception
        with patch("src.morag.tasks.web_tasks.web_processor") as mock_processor:
            mock_processor.process_url = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            # Create a mock task instance
            mock_task = Mock()
            mock_task.update_status = AsyncMock()

            # Call the task function
            result = await process_web_url(
                mock_task, "https://example.com", None, "test_task_id"
            )

            # Verify the result
            assert result["success"] is False
            assert "Unexpected error" in result["error"]
            assert result["url"] == "https://example.com"

            # Verify status updates
            mock_task.update_status.assert_any_call(
                "FAILED", {"error": result["error"]}
            )

    @pytest.mark.asyncio
    async def test_process_web_url_embedding_failure(self, mock_successful_result):
        """Test web URL processing with embedding failure."""
        # Mock the web processor
        with patch("src.morag.tasks.web_tasks.web_processor") as mock_processor:
            mock_processor.process_url = AsyncMock(return_value=mock_successful_result)

            # Mock embedding service to fail
            with patch("src.morag.tasks.web_tasks.embedding_service") as mock_embedding:
                mock_embedding.generate_embedding = AsyncMock(
                    side_effect=Exception("Embedding error")
                )

                # Create a mock task instance
                mock_task = Mock()
                mock_task.update_status = AsyncMock()

                # Call the task function
                result = await process_web_url(
                    mock_task, "https://example.com", None, "test_task_id"
                )

                # Verify the result - should still succeed but with embedding error
                assert result["success"] is True
                assert "embedding_error" in result
                assert "Embedding error" in result["embedding_error"]

    @pytest.mark.asyncio
    async def test_process_web_urls_batch_success(self, mock_successful_result):
        """Test successful batch web URL processing."""
        urls = ["https://example1.com", "https://example2.com"]

        # Mock the web processor
        with patch("src.morag.tasks.web_tasks.web_processor") as mock_processor:
            mock_processor.process_urls = AsyncMock(
                return_value=[mock_successful_result, mock_successful_result]
            )

            # Mock the embedding and storage services
            with patch("src.morag.tasks.web_tasks.embedding_service") as mock_embedding:
                with patch("src.morag.tasks.web_tasks.storage_service") as mock_storage:
                    from src.morag.services.embedding import EmbeddingResult

                    mock_embedding.generate_embedding = AsyncMock(
                        return_value=EmbeddingResult(
                            embedding=[0.1, 0.2, 0.3], token_count=10, model="test"
                        )
                    )
                    mock_storage.store_chunk_with_embedding = AsyncMock()

                    # Create a mock task instance
                    mock_task = Mock()
                    mock_task.update_status = AsyncMock()

                    # Call the task function
                    result = await process_web_urls_batch(
                        mock_task, urls, {"timeout": 30}, "test_task_id"
                    )

                    # Verify the result
                    assert result["success"] is True
                    assert result["total_urls"] == 2
                    assert result["successful"] == 2
                    assert result["failed"] == 0
                    assert result["total_chunks"] == 4  # 2 chunks per URL
                    assert result["embeddings_stored"] == 4
                    assert len(result["successful_urls"]) == 2
                    assert len(result["failed_urls"]) == 0

                    # Verify status updates
                    mock_task.update_status.assert_any_call(
                        "PROCESSING", {"stage": "batch_initialization"}
                    )
                    mock_task.update_status.assert_any_call(
                        "PROCESSING", {"stage": "batch_processing"}
                    )
                    mock_task.update_status.assert_any_call(
                        "PROCESSING", {"stage": "embedding_generation"}
                    )
                    mock_task.update_status.assert_any_call("COMPLETED", result)

    @pytest.mark.asyncio
    async def test_process_web_urls_batch_mixed_results(
        self, mock_successful_result, mock_failed_result
    ):
        """Test batch web URL processing with mixed results."""
        urls = ["https://example1.com", "https://example2.com"]

        # Mock the web processor with mixed results
        with patch("src.morag.tasks.web_tasks.web_processor") as mock_processor:
            mock_processor.process_urls = AsyncMock(
                return_value=[mock_successful_result, mock_failed_result]
            )

            # Mock the embedding and storage services
            with patch("src.morag.tasks.web_tasks.embedding_service") as mock_embedding:
                with patch("src.morag.tasks.web_tasks.storage_service") as mock_storage:
                    from src.morag.services.embedding import EmbeddingResult

                    mock_embedding.generate_embedding = AsyncMock(
                        return_value=EmbeddingResult(
                            embedding=[0.1, 0.2, 0.3], token_count=10, model="test"
                        )
                    )
                    mock_storage.store_chunk_with_embedding = AsyncMock()

                    # Create a mock task instance
                    mock_task = Mock()
                    mock_task.update_status = AsyncMock()

                    # Call the task function
                    result = await process_web_urls_batch(
                        mock_task, urls, None, "test_task_id"
                    )

                    # Verify the result
                    assert result["success"] is True
                    assert result["total_urls"] == 2
                    assert result["successful"] == 1
                    assert result["failed"] == 1
                    assert result["total_chunks"] == 2  # Only from successful result
                    assert result["embeddings_stored"] == 2
                    assert len(result["successful_urls"]) == 1
                    assert len(result["failed_urls"]) == 1
                    assert result["failed_urls"][0]["error"] == "Network error occurred"

    @pytest.mark.asyncio
    async def test_process_web_urls_batch_exception(self):
        """Test batch web URL processing with exception."""
        urls = ["https://example1.com", "https://example2.com"]

        # Mock the web processor to raise an exception
        with patch("src.morag.tasks.web_tasks.web_processor") as mock_processor:
            mock_processor.process_urls = AsyncMock(
                side_effect=Exception("Batch processing error")
            )

            # Create a mock task instance
            mock_task = Mock()
            mock_task.update_status = AsyncMock()

            # Call the task function
            result = await process_web_urls_batch(mock_task, urls, None, "test_task_id")

            # Verify the result
            assert result["success"] is False
            assert "Batch processing error" in result["error"]
            assert result["total_urls"] == 2

            # Verify status updates
            mock_task.update_status.assert_any_call(
                "FAILED", {"error": result["error"]}
            )
