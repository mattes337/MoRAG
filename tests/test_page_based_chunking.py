import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from morag.processors.document import DocumentProcessor, DocumentChunk, DocumentParseResult
from morag.core.config import settings


@pytest.fixture
def document_processor():
    """Create a document processor instance for testing."""
    return DocumentProcessor()


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            text="This is the first paragraph on page 1.",
            chunk_type="text",
            page_number=1,
            element_id="chunk_1",
            metadata={"element_type": "Text"}
        ),
        DocumentChunk(
            text="This is the second paragraph on page 1.",
            chunk_type="text",
            page_number=1,
            element_id="chunk_2",
            metadata={"element_type": "Text"}
        ),
        DocumentChunk(
            text="This is a title on page 2.",
            chunk_type="title",
            page_number=2,
            element_id="chunk_3",
            metadata={"element_type": "Title"}
        ),
        DocumentChunk(
            text="This is content on page 2.",
            chunk_type="text",
            page_number=2,
            element_id="chunk_4",
            metadata={"element_type": "Text"}
        ),
        DocumentChunk(
            text="This is the only content on page 3.",
            chunk_type="text",
            page_number=3,
            element_id="chunk_5",
            metadata={"element_type": "Text"}
        )
    ]


@pytest.fixture
def sample_parse_result(sample_chunks):
    """Create a sample parse result for testing."""
    return DocumentParseResult(
        chunks=sample_chunks,
        metadata={
            "parser": "test",
            "file_name": "test.pdf",
            "total_chunks": len(sample_chunks)
        },
        images=[],
        total_pages=3,
        word_count=50
    )


class TestPageBasedChunking:
    """Test page-based chunking functionality."""

    @pytest.mark.asyncio
    async def test_apply_page_based_chunking_basic(self, document_processor, sample_parse_result):
        """Test basic page-based chunking functionality."""
        # Apply page-based chunking
        result = await document_processor._apply_page_based_chunking(sample_parse_result)
        
        # Should have 3 chunks (one per page)
        assert len(result.chunks) == 3
        
        # Check page 1 chunk
        page1_chunk = result.chunks[0]
        assert page1_chunk.page_number == 1
        assert page1_chunk.chunk_type == "page"
        assert "This is the first paragraph on page 1." in page1_chunk.text
        assert "This is the second paragraph on page 1." in page1_chunk.text
        assert page1_chunk.metadata["page_based_chunking"] is True
        assert page1_chunk.metadata["original_chunks_count"] == 2
        
        # Check page 2 chunk
        page2_chunk = result.chunks[1]
        assert page2_chunk.page_number == 2
        assert page2_chunk.chunk_type == "page"
        assert "This is a title on page 2." in page2_chunk.text
        assert "This is content on page 2." in page2_chunk.text
        assert page2_chunk.metadata["page_based_chunking"] is True
        assert page2_chunk.metadata["original_chunks_count"] == 2
        
        # Check page 3 chunk
        page3_chunk = result.chunks[2]
        assert page3_chunk.page_number == 3
        assert page3_chunk.chunk_type == "page"
        assert page3_chunk.text == "This is the only content on page 3."
        assert page3_chunk.metadata["page_based_chunking"] is True
        assert page3_chunk.metadata["original_chunks_count"] == 1

    @pytest.mark.asyncio
    async def test_apply_page_based_chunking_large_page(self, document_processor):
        """Test page-based chunking with a page that exceeds max size."""
        # Create a large chunk that exceeds max_page_chunk_size
        large_text = "This is a very long text. " * 500  # About 13,500 characters
        
        large_chunks = [
            DocumentChunk(
                text=large_text,
                chunk_type="text",
                page_number=1,
                element_id="large_chunk",
                metadata={"element_type": "Text"}
            )
        ]
        
        parse_result = DocumentParseResult(
            chunks=large_chunks,
            metadata={"parser": "test"},
            images=[],
            total_pages=1,
            word_count=1000
        )
        
        # Mock settings to use a smaller max size for testing
        with patch.object(settings, 'max_page_chunk_size', 1000):
            result = await document_processor._apply_page_based_chunking(parse_result)
        
        # Should split into multiple chunks
        assert len(result.chunks) > 1
        
        # All chunks should be from page 1
        for chunk in result.chunks:
            assert chunk.page_number == 1
            assert chunk.chunk_type == "page"
            assert chunk.metadata["is_partial_page"] is True

    @pytest.mark.asyncio
    async def test_apply_page_based_chunking_empty_chunks(self, document_processor):
        """Test page-based chunking with empty chunk list."""
        empty_parse_result = DocumentParseResult(
            chunks=[],
            metadata={"parser": "test"},
            images=[],
            total_pages=0,
            word_count=0
        )
        
        result = await document_processor._apply_page_based_chunking(empty_parse_result)
        
        # Should return the same result
        assert len(result.chunks) == 0
        assert result.metadata == empty_parse_result.metadata

    @pytest.mark.asyncio
    async def test_parse_document_with_page_chunking(self, document_processor):
        """Test document parsing with page-based chunking enabled."""
        # Create a temporary PDF file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(b"dummy pdf content")
        
        try:
            # Mock the parsing methods to return controlled results
            mock_chunks = [
                DocumentChunk(
                    text="Page 1 content",
                    chunk_type="text",
                    page_number=1,
                    element_id="p1_chunk1"
                ),
                DocumentChunk(
                    text="More page 1 content",
                    chunk_type="text",
                    page_number=1,
                    element_id="p1_chunk2"
                )
            ]
            
            mock_result = DocumentParseResult(
                chunks=mock_chunks,
                metadata={"parser": "mock"},
                images=[],
                total_pages=1,
                word_count=10
            )
            
            with patch.object(document_processor, '_parse_with_unstructured', return_value=mock_result):
                with patch.object(settings, 'default_chunking_strategy', 'page'):
                    with patch.object(settings, 'enable_page_based_chunking', True):
                        result = await document_processor.parse_document(
                            tmp_path,
                            use_docling=False,
                            chunking_strategy="page"
                        )
            
            # Should have applied page-based chunking
            assert len(result.chunks) == 1  # Combined into one page chunk
            assert result.chunks[0].chunk_type == "page"
            assert result.chunks[0].page_number == 1
            assert "Page 1 content" in result.chunks[0].text
            assert "More page 1 content" in result.chunks[0].text
            assert result.metadata["page_based_chunking_applied"] is True
            
        finally:
            # Clean up
            tmp_path.unlink()

    def test_chunking_strategy_configuration(self):
        """Test that chunking strategy configuration is properly set."""
        # Test default values
        assert hasattr(settings, 'default_chunking_strategy')
        assert hasattr(settings, 'enable_page_based_chunking')
        assert hasattr(settings, 'max_page_chunk_size')
        
        # Test that page strategy is available
        from morag.services.chunking import SemanticChunker
        chunker = SemanticChunker()
        
        # This should not raise an exception
        assert hasattr(chunker, '_page_chunking')


if __name__ == "__main__":
    pytest.main([__file__])
