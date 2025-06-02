import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from morag.tasks.document_tasks import process_document_task
from morag.services.embedding import EmbeddingResult, SummaryResult

@pytest.fixture
def mock_gemini_service():
    """Mock Gemini service for testing."""
    with patch('morag.tasks.document_tasks.gemini_service') as mock:
        # Mock embedding generation
        mock.generate_embedding.return_value = EmbeddingResult(
            embedding=[0.1] * 768,
            token_count=10,
            model="text-embedding-004"
        )
        
        # Mock batch embedding generation
        mock.generate_embeddings_batch.return_value = [
            EmbeddingResult(
                embedding=[0.1] * 768,
                token_count=10,
                model="text-embedding-004"
            )
        ]
        
        # Mock summary generation
        mock.generate_summary.return_value = SummaryResult(
            summary="Test summary",
            token_count=5,
            model="gemini-2.0-flash-001"
        )
        
        yield mock

@pytest.fixture
def mock_qdrant_service():
    """Mock Qdrant service for testing."""
    with patch('morag.tasks.document_tasks.qdrant_service') as mock:
        mock.store_chunks.return_value = ["point_1", "point_2"]
        yield mock

@pytest.mark.asyncio
async def test_document_processing_task_success(mock_gemini_service, mock_qdrant_service):
    """Test successful document processing task."""
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""
# Test Document

This is a test document for processing.

## Section 1

Some content in section 1 with enough text to make it meaningful.

## Section 2

Some content in section 2 with additional information.
        """)
        temp_path = f.name
    
    try:
        # Create a mock task instance
        mock_task = MagicMock()
        mock_task.log_step = MagicMock()
        mock_task.update_progress = MagicMock()
        
        # Process document
        result = await process_document_task(
            mock_task,
            file_path=temp_path,
            source_type="document",
            metadata={"test": True, "source": "test"}
        )
        
        # Verify result
        assert result["status"] == "success"
        assert result["file_path"] == temp_path
        assert result["chunks_processed"] > 0
        assert result["word_count"] > 0
        assert "point_ids" in result
        assert "metadata" in result
        
        # Verify service calls
        assert mock_gemini_service.generate_summary.called
        assert mock_gemini_service.generate_embeddings_batch.called
        assert mock_qdrant_service.store_chunks.called
        
        # Verify progress updates
        assert mock_task.update_progress.called
        assert mock_task.log_step.called
        
    finally:
        Path(temp_path).unlink()

@pytest.mark.asyncio
async def test_document_processing_with_docling(mock_gemini_service, mock_qdrant_service):
    """Test document processing with docling option."""
    # Create a test PDF-like file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        f.write("This is a fake PDF file for testing.")
        temp_path = f.name
    
    try:
        mock_task = MagicMock()
        mock_task.log_step = MagicMock()
        mock_task.update_progress = MagicMock()
        
        # Process document with docling option
        result = await process_document_task(
            mock_task,
            file_path=temp_path,
            source_type="document",
            metadata={"test": True},
            use_docling=True
        )
        
        assert result["status"] == "success"
        
    finally:
        Path(temp_path).unlink()

@pytest.mark.asyncio
async def test_document_processing_file_not_found():
    """Test document processing with non-existent file."""
    mock_task = MagicMock()
    mock_task.log_step = MagicMock()
    mock_task.update_progress = MagicMock()
    
    with pytest.raises(Exception):
        await process_document_task(
            mock_task,
            file_path="non_existent_file.pdf",
            source_type="document",
            metadata={"test": True}
        )

@pytest.mark.asyncio
async def test_document_processing_summary_failure(mock_qdrant_service):
    """Test document processing when summary generation fails."""
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for processing.")
        temp_path = f.name
    
    try:
        # Mock Gemini service with summary failure
        with patch('morag.tasks.document_tasks.gemini_service') as mock_gemini:
            # Summary generation fails
            mock_gemini.generate_summary.side_effect = Exception("Summary generation failed")
            
            # Embedding generation succeeds
            mock_gemini.generate_embeddings_batch.return_value = [
                EmbeddingResult(
                    embedding=[0.1] * 768,
                    token_count=10,
                    model="text-embedding-004"
                )
            ]
            
            mock_task = MagicMock()
            mock_task.log_step = MagicMock()
            mock_task.update_progress = MagicMock()
            
            # Should still succeed with fallback summary
            result = await process_document_task(
                mock_task,
                file_path=temp_path,
                source_type="document",
                metadata={"test": True}
            )
            
            assert result["status"] == "success"
            assert result["chunks_processed"] > 0
            
    finally:
        Path(temp_path).unlink()

@pytest.mark.asyncio
async def test_document_processing_empty_file(mock_gemini_service, mock_qdrant_service):
    """Test document processing with empty file."""
    # Create an empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name
    
    try:
        mock_task = MagicMock()
        mock_task.log_step = MagicMock()
        mock_task.update_progress = MagicMock()
        
        result = await process_document_task(
            mock_task,
            file_path=temp_path,
            source_type="document",
            metadata={"test": True}
        )
        
        # Should handle empty file gracefully
        assert result["status"] == "success"
        assert result["chunks_processed"] == 0
        assert result["word_count"] == 0
        
    finally:
        Path(temp_path).unlink()

@pytest.mark.asyncio
async def test_document_processing_metadata_preservation():
    """Test that metadata is properly preserved and enhanced."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test\nThis is a test document.")
        temp_path = f.name
    
    try:
        with patch('morag.tasks.document_tasks.gemini_service') as mock_gemini, \
             patch('morag.tasks.document_tasks.qdrant_service') as mock_qdrant:
            
            mock_gemini.generate_summary.return_value = SummaryResult(
                summary="Test summary",
                token_count=5,
                model="gemini-2.0-flash-001"
            )
            
            mock_gemini.generate_embeddings_batch.return_value = [
                EmbeddingResult(
                    embedding=[0.1] * 768,
                    token_count=10,
                    model="text-embedding-004"
                )
            ]
            
            mock_qdrant.store_chunks.return_value = ["point_1"]
            
            mock_task = MagicMock()
            mock_task.log_step = MagicMock()
            mock_task.update_progress = MagicMock()
            
            original_metadata = {
                "source_url": "https://example.com",
                "author": "Test Author",
                "created_at": "2024-01-01"
            }
            
            result = await process_document_task(
                mock_task,
                file_path=temp_path,
                source_type="document",
                metadata=original_metadata
            )
            
            # Check that chunks were stored with enhanced metadata
            store_call = mock_qdrant.store_chunks.call_args
            stored_chunks = store_call[0][0]  # First argument (chunks)
            
            # Verify metadata is preserved and enhanced
            chunk_metadata = stored_chunks[0]["metadata"]
            assert chunk_metadata["source_url"] == "https://example.com"
            assert chunk_metadata["author"] == "Test Author"
            assert chunk_metadata["created_at"] == "2024-01-01"
            assert "parser" in chunk_metadata
            assert "file_name" in chunk_metadata
            
    finally:
        Path(temp_path).unlink()

@pytest.mark.asyncio
async def test_document_processing_progress_tracking():
    """Test that progress is properly tracked during processing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test document content.")
        temp_path = f.name
    
    try:
        with patch('morag.tasks.document_tasks.gemini_service') as mock_gemini, \
             patch('morag.tasks.document_tasks.qdrant_service') as mock_qdrant:
            
            mock_gemini.generate_summary.return_value = SummaryResult(
                summary="Test summary",
                token_count=5,
                model="gemini-2.0-flash-001"
            )
            
            mock_gemini.generate_embeddings_batch.return_value = [
                EmbeddingResult(
                    embedding=[0.1] * 768,
                    token_count=10,
                    model="text-embedding-004"
                )
            ]
            
            mock_qdrant.store_chunks.return_value = ["point_1"]
            
            mock_task = MagicMock()
            mock_task.log_step = MagicMock()
            mock_task.update_progress = MagicMock()
            
            await process_document_task(
                mock_task,
                file_path=temp_path,
                source_type="document",
                metadata={"test": True}
            )
            
            # Verify progress updates were called with expected values
            progress_calls = mock_task.update_progress.call_args_list
            progress_values = [call[0][0] for call in progress_calls]
            
            # Should have progress updates from 0.1 to 1.0
            assert 0.1 in progress_values
            assert 1.0 in progress_values
            assert len(progress_values) >= 4  # At least validation, parsing, processing, completion
            
    finally:
        Path(temp_path).unlink()
