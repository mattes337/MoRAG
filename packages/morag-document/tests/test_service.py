"""Tests for document service."""

import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from morag_core.interfaces.service import ServiceConfig, ServiceStatus
from morag_core.interfaces.converter import ChunkingStrategy
from morag_core.models.document import Document
from morag_core.models.embedding import EmbeddingResult, BatchEmbeddingResult, SummaryResult
from morag_core.exceptions import ValidationError, ProcessingError

from morag_document.service import DocumentService


@pytest.fixture
def service():
    """Create document service fixture."""
    return DocumentService()


@pytest.fixture
def service_with_embedding():
    """Create document service with embedding fixture."""
    config = ServiceConfig(enable_embedding=True, embedding={"api_key": "test_key"})
    service = DocumentService(config)
    service.embedding_service = MagicMock()
    service.embedding_service.initialize = AsyncMock(return_value=True)
    service.embedding_service.shutdown = AsyncMock(return_value=True)
    service.embedding_service.health_check = AsyncMock(return_value={"status": "ready"})
    service.embedding_service.embed_batch = AsyncMock()
    service.embedding_service.summarize = AsyncMock()
    return service


@pytest.fixture
def sample_document():
    """Create sample document fixture."""
    doc = Document(
        raw_text="Sample document text for testing.",
        metadata={
            "title": "Test Document",
            "file_type": "text",
            "word_count": 5,
        }
    )
    doc.add_chunk(content="Sample document text for testing.")
    return doc


@pytest.mark.asyncio
async def test_service_initialization(service):
    """Test service initialization."""
    # Check initial state
    assert service.processor is not None
    assert service.embedding_service is None
    assert service._status == ServiceStatus.INITIALIZING
    
    # Initialize service
    result = await service.initialize()
    assert result is True
    assert service._status == ServiceStatus.READY


@pytest.mark.asyncio
async def test_service_with_embedding_initialization(service_with_embedding):
    """Test service with embedding initialization."""
    # Initialize service
    result = await service_with_embedding.initialize()
    assert result is True
    assert service_with_embedding._status == ServiceStatus.READY
    service_with_embedding.embedding_service.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_service_shutdown(service):
    """Test service shutdown."""
    # Initialize service
    await service.initialize()
    
    # Shutdown service
    result = await service.shutdown()
    assert result is True
    assert service._status == ServiceStatus.STOPPED


@pytest.mark.asyncio
async def test_service_with_embedding_shutdown(service_with_embedding):
    """Test service with embedding shutdown."""
    # Initialize service
    await service_with_embedding.initialize()
    
    # Shutdown service
    result = await service_with_embedding.shutdown()
    assert result is True
    assert service_with_embedding._status == ServiceStatus.STOPPED
    service_with_embedding.embedding_service.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_service_health_check(service):
    """Test service health check."""
    # Initialize service
    await service.initialize()
    
    # Check health
    health = await service.health_check()
    assert health["status"] == ServiceStatus.READY.value
    assert health["processor"] == "ok"
    assert "embedding" not in health


@pytest.mark.asyncio
async def test_service_with_embedding_health_check(service_with_embedding):
    """Test service with embedding health check."""
    # Initialize service
    await service_with_embedding.initialize()
    
    # Check health
    health = await service_with_embedding.health_check()
    assert health["status"] == ServiceStatus.READY.value
    assert health["processor"] == "ok"
    assert health["embedding"] == {"status": "ready"}
    service_with_embedding.embedding_service.health_check.assert_called_once()


@pytest.mark.asyncio
async def test_process_document(service, tmp_path):
    """Test document processing."""
    # Initialize service
    await service.initialize()
    
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    # Mock processor
    service.processor.process_file = AsyncMock()
    service.processor.process_file.return_value = MagicMock(
        document=MagicMock(),
        metadata={"quality_score": 0.9}
    )
    
    # Process document
    result = await service.process_document(test_file)
    assert result is not None
    assert result.metadata["quality_score"] == 0.9
    
    # Check processor was called with correct arguments
    service.processor.process_file.assert_called_once_with(
        test_file,
        generate_embeddings=False
    )


@pytest.mark.asyncio
async def test_process_document_with_embeddings(service_with_embedding, sample_document, tmp_path):
    """Test document processing with embeddings."""
    # Initialize service
    await service_with_embedding.initialize()
    
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    # Mock processor
    service_with_embedding.processor.process_file = AsyncMock()
    service_with_embedding.processor.process_file.return_value = MagicMock(
        document=sample_document,
        metadata={"quality_score": 0.9}
    )
    
    # Mock embedding service
    embedding_result = EmbeddingResult(
        id="test",
        text="Sample document text for testing.",
        embedding=[0.1, 0.2, 0.3],
        model="test-model"
    )
    batch_result = BatchEmbeddingResult(
        results=[embedding_result],
        model="test-model"
    )
    service_with_embedding.embedding_service.embed_batch.return_value = batch_result
    
    # Process document with embeddings
    result = await service_with_embedding.process_document(
        test_file,
        generate_embeddings=True
    )
    
    # Check result
    assert result is not None
    assert result.document == sample_document
    
    # Check embedding service was called
    service_with_embedding.embedding_service.embed_batch.assert_called_once()
    assert service_with_embedding.embedding_service.embed_batch.call_args[0][0] == [
        "Sample document text for testing."
    ]


@pytest.mark.asyncio
async def test_process_text(service):
    """Test text processing."""
    # Initialize service
    await service.initialize()
    
    # Process text
    result = await service.process_text("Sample text for processing.")
    
    # Check result
    assert result is not None
    assert result.document.raw_text == "Sample text for processing."
    assert result.document.metadata.title == "Text Document"
    assert result.document.metadata.file_type == "text"
    assert result.document.metadata.word_count == 4
    assert result.metadata["quality_score"] == 1.0


@pytest.mark.asyncio
async def test_process_text_with_embeddings(service_with_embedding):
    """Test text processing with embeddings."""
    # Initialize service
    await service_with_embedding.initialize()
    
    # Mock embedding service
    embedding_result = EmbeddingResult(
        id="test",
        text="Sample text for processing.",
        embedding=[0.1, 0.2, 0.3],
        model="test-model"
    )
    batch_result = BatchEmbeddingResult(
        results=[embedding_result],
        model="test-model"
    )
    service_with_embedding.embedding_service.embed_batch.return_value = batch_result
    
    # Process text with embeddings
    result = await service_with_embedding.process_text(
        "Sample text for processing.",
        generate_embeddings=True
    )
    
    # Check embedding service was called
    service_with_embedding.embedding_service.embed_batch.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_document(service_with_embedding, sample_document):
    """Test document summarization."""
    # Initialize service
    await service_with_embedding.initialize()
    
    # Mock embedding service
    summary_result = SummaryResult(
        id="test",
        text="Sample document text for testing.",
        summary="This is a test document.",
        model="test-model"
    )
    service_with_embedding.embedding_service.summarize.return_value = summary_result
    
    # Summarize document
    summary = await service_with_embedding.summarize_document(sample_document)
    
    # Check result
    assert summary == "This is a test document."
    
    # Check embedding service was called
    service_with_embedding.embedding_service.summarize.assert_called_once_with(
        "Sample document text for testing.",
        max_length=1000
    )


@pytest.mark.asyncio
async def test_summarize_document_without_embedding_service(service, sample_document):
    """Test document summarization without embedding service."""
    # Initialize service
    await service.initialize()
    
    # Try to summarize document
    with pytest.raises(ProcessingError):
        await service.summarize_document(sample_document)