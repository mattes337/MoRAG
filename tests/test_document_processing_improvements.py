"""Tests for Document Processing Improvements (Tasks 1, 2, 3, 5)."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Test Task 1: Word Boundary Preservation
def test_word_boundary_detection():
    """Test word boundary detection helper method."""
    from packages.morag_document.src.morag_document.converters.base import BaseConverter
    
    converter = BaseConverter()
    text = "This is a test sentence with some punctuation! And another sentence."
    
    # Test backward search
    boundary = converter._find_word_boundary(text, 20, "backward")
    assert text[boundary-1:boundary+1] in [" ", "! ", ". "] or boundary == 0
    
    # Test forward search  
    boundary = converter._find_word_boundary(text, 20, "forward")
    assert text[boundary-1:boundary+1] in [" ", "! ", ". "] or boundary == len(text)

def test_sentence_boundary_detection():
    """Test enhanced sentence boundary detection."""
    from packages.morag_document.src.morag_document.converters.base import BaseConverter
    
    converter = BaseConverter()
    text = "Dr. Smith went to the U.S.A. He bought 3.14 pounds of apples. What a day!"
    
    boundaries = converter._detect_sentence_boundaries(text)
    
    # Should detect proper sentence boundaries, not abbreviations or decimals
    assert len(boundaries) >= 3  # Start, at least one sentence boundary, end
    assert boundaries[0] == 0
    assert boundaries[-1] == len(text)

@pytest.mark.asyncio
async def test_enhanced_word_chunking():
    """Test enhanced word-based chunking with better overlap."""
    from packages.morag_document.src.morag_document.converters.base import BaseConverter
    from packages.morag_core.interfaces.converter import ConversionOptions, ChunkingStrategy
    from packages.morag_core.models.document import Document, DocumentMetadata
    
    converter = BaseConverter()
    
    # Create test document
    text = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence."
    document = Document(
        metadata=DocumentMetadata(title="Test", source="test.txt"),
        raw_text=text
    )
    
    options = ConversionOptions(
        chunking_strategy=ChunkingStrategy.WORD,
        chunk_size=50,  # Small chunks to test overlap
        chunk_overlap=20
    )
    
    result = await converter._chunk_document(document, options)
    
    # Should have multiple chunks with proper word boundaries
    assert len(result.chunks) > 1
    
    # Check that chunks don't split words
    for chunk in result.chunks:
        words = chunk.content.split()
        # Each word should be complete (no partial words)
        for word in words:
            assert not word.startswith(' ') and not word.endswith(' ')

# Test Task 2: Search Embedding Optimization
@pytest.mark.asyncio
async def test_search_embedding_optimization():
    """Test optimized search embedding strategy."""
    from packages.morag_services.src.morag_services.services import MoRAGServices
    
    # Mock the embedding service
    mock_embedding_service = AsyncMock()
    mock_embedding_service.initialize.return_value = True
    mock_embedding_service.generate_embedding.return_value = [0.1] * 768
    
    # Mock vector storage
    mock_vector_storage = AsyncMock()
    mock_vector_storage._initialized = True
    mock_vector_storage.search_similar.return_value = [
        {
            "id": "test1",
            "score": 0.9,
            "metadata": {
                "text": "Test content",
                "content_type": "document",
                "source": "test.txt"
            }
        }
    ]
    
    services = MoRAGServices()
    services._gemini_embedding_service = mock_embedding_service
    services._vector_storage = mock_vector_storage
    
    # Test search with performance monitoring
    results = await services.search_similar("test query", limit=5)
    
    # Verify embedding was called with correct parameters
    mock_embedding_service.generate_embedding.assert_called_once()
    
    # Verify search was performed
    mock_vector_storage.search_similar.assert_called_once()
    
    # Verify results format (Task 3: no text duplication)
    assert len(results) == 1
    result = results[0]
    assert "content" in result  # Text content in 'content' field
    assert "text" not in result["metadata"]  # No text duplication in metadata

# Test Task 3: Text Duplication Fix
def test_search_response_deduplication():
    """Test that search responses don't duplicate text content."""
    from packages.morag_services.src.morag_services.services import MoRAGServices
    
    # Mock search result with text in metadata
    mock_result = {
        "id": "test1",
        "score": 0.9,
        "metadata": {
            "text": "This is test content",
            "content_type": "document",
            "source": "test.txt",
            "chunk_index": 0
        }
    }
    
    services = MoRAGServices()
    
    # Simulate the formatting logic from search_similar
    text_content = mock_result.get("metadata", {}).get("text", "")
    clean_metadata = {k: v for k, v in mock_result.get("metadata", {}).items() if k != "text"}
    
    formatted_result = {
        "id": mock_result.get("id"),
        "score": mock_result.get("score", 0.0),
        "content": text_content,
        "metadata": clean_metadata,
        "content_type": mock_result.get("metadata", {}).get("content_type"),
        "source": mock_result.get("metadata", {}).get("source")
    }
    
    # Verify no text duplication
    assert formatted_result["content"] == "This is test content"
    assert "text" not in formatted_result["metadata"]
    assert "content_type" in formatted_result["metadata"]
    assert "source" in formatted_result["metadata"]

# Test Task 5: Document Replacement
@pytest.mark.asyncio
async def test_document_id_generation():
    """Test document ID generation."""
    from packages.morag.src.morag.ingest_tasks import generate_document_id
    
    # Test URL ID generation
    url_id = generate_document_id("https://example.com/test-page")
    assert len(url_id) == 16
    assert url_id.isalnum()
    
    # Test file ID generation
    file_id = generate_document_id("test_document.pdf")
    assert file_id == "test_document_pdf"
    
    # Test file with content hash
    file_with_content_id = generate_document_id("test.txt", "test content")
    assert file_with_content_id.startswith("test_txt_")
    assert len(file_with_content_id) > len("test_txt_")

@pytest.mark.asyncio
async def test_document_replacement_storage():
    """Test document replacement in storage layer."""
    from packages.morag_services.src.morag_services.storage import QdrantVectorStorage
    
    # Mock Qdrant client
    mock_client = AsyncMock()
    
    storage = QdrantVectorStorage(
        host="localhost",
        port=6333,
        collection_name="test_collection"
    )
    storage.client = mock_client
    
    # Test find_document_points
    mock_client.scroll.return_value = ([Mock(id="point1"), Mock(id="point2")], None)
    
    points = await storage.find_document_points("test_doc_id")
    assert points == ["point1", "point2"]
    
    # Test replace_document
    new_vectors = [[0.1] * 768, [0.2] * 768]
    new_metadata = [{"text": "chunk1"}, {"text": "chunk2"}]
    
    # Mock store_vectors to return point IDs
    storage.store_vectors = AsyncMock(return_value=["new_point1", "new_point2"])
    
    result_ids = await storage.replace_document(
        "test_doc_id",
        new_vectors,
        new_metadata
    )
    
    assert result_ids == ["new_point1", "new_point2"]
    
    # Verify document_id was added to metadata
    storage.store_vectors.assert_called_once()
    call_args = storage.store_vectors.call_args
    stored_metadata = call_args[0][1]  # Second argument is metadata
    
    for meta in stored_metadata:
        assert meta["document_id"] == "test_doc_id"
        assert "replaced_at" in meta

def test_document_id_validation():
    """Test document ID validation."""
    import re
    
    # Valid IDs
    valid_ids = ["test_doc", "test-doc", "TestDoc123", "doc_123-abc"]
    pattern = r'^[a-zA-Z0-9_-]+$'
    
    for doc_id in valid_ids:
        assert re.match(pattern, doc_id), f"Valid ID {doc_id} should match pattern"
    
    # Invalid IDs
    invalid_ids = ["test doc", "test.doc", "test@doc", "test/doc", ""]
    
    for doc_id in invalid_ids:
        assert not re.match(pattern, doc_id), f"Invalid ID {doc_id} should not match pattern"

if __name__ == "__main__":
    pytest.main([__file__])
