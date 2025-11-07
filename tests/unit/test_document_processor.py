import pytest
import tempfile
from pathlib import Path
from morag_document import document_processor, DocumentType
from morag_core.exceptions import ValidationError

def test_document_type_detection():
    """Test document type detection."""
    assert document_processor.detect_document_type("test.pdf") == DocumentType.PDF
    assert document_processor.detect_document_type("test.docx") == DocumentType.DOCX
    assert document_processor.detect_document_type("test.md") == DocumentType.MARKDOWN
    assert document_processor.detect_document_type("test.txt") == DocumentType.TXT

    with pytest.raises(ValidationError):
        document_processor.detect_document_type("test.xyz")

@pytest.mark.asyncio
async def test_markdown_parsing():
    """Test parsing of markdown files."""
    # Create a test markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""
# Test Document

This is a test document with multiple sections.

## Section 1

Some content in section 1.

## Section 2

Some content in section 2.
        """)
        temp_path = f.name

    try:
        result = await document_processor.parse_document(temp_path)

        assert len(result.chunks) > 0
        assert result.word_count > 0
        assert result.metadata["parser"] in ["unstructured", "basic_text"]
        assert result.metadata["file_name"] == Path(temp_path).name

    finally:
        Path(temp_path).unlink()

@pytest.mark.asyncio
async def test_text_file_parsing():
    """Test parsing of text files."""
    # Create a test text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a simple text file for testing.\n\nIt has multiple paragraphs.")
        temp_path = f.name

    try:
        result = await document_processor.parse_document(temp_path)

        assert len(result.chunks) > 0
        assert result.word_count > 0
        assert result.metadata["parser"] in ["unstructured", "basic_text"]

    finally:
        Path(temp_path).unlink()

def test_file_validation():
    """Test file validation."""
    # Test non-existent file
    with pytest.raises(ValidationError, match="File not found"):
        document_processor.validate_file("non_existent_file.pdf")

    # Test unsupported file type
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValidationError, match="Unsupported file type"):
            document_processor.validate_file(temp_path)
    finally:
        Path(temp_path).unlink()

def test_file_size_validation():
    """Test file size validation."""
    # Create a small file that should pass validation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Small test file")
        temp_path = f.name

    try:
        # Should pass with default max size
        assert document_processor.validate_file(temp_path) == True

        # Should fail with very small max size
        with pytest.raises(ValidationError, match="File too large"):
            document_processor.validate_file(temp_path, max_size_mb=0.000001)

    finally:
        Path(temp_path).unlink()

@pytest.mark.asyncio
async def test_docling_fallback():
    """Test docling fallback when docling is not available."""
    # Create a test PDF-like file (actually text)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        f.write("This is a fake PDF file for testing fallback.")
        temp_path = f.name

    try:
        # This should fallback to unstructured since docling might not be installed
        result = await document_processor.parse_document(temp_path, use_docling=True)

        # Should still get a result (from unstructured fallback)
        assert result is not None
        assert result.metadata["parser"] in ["unstructured", "docling", "basic_text"]

    finally:
        Path(temp_path).unlink()

def test_table_to_markdown():
    """Test table to markdown conversion."""
    # Create a mock table element
    class MockTable:
        def __init__(self, text):
            self.text = text
            self.metadata = None

    table = MockTable("Header1 | Header2\nValue1 | Value2")
    result = document_processor._table_to_markdown(table)

    assert "**Table:**" in result
    # When unstructured is not available, we get a different message
    if "content not available" not in result:
        assert "Header1 | Header2" in result

@pytest.mark.asyncio
async def test_empty_file_handling():
    """Test handling of empty files."""
    # Create an empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name

    try:
        result = await document_processor.parse_document(temp_path)

        # Should handle empty file gracefully
        assert result is not None
        assert len(result.chunks) == 0
        assert result.word_count == 0

    finally:
        Path(temp_path).unlink()

@pytest.mark.asyncio
async def test_large_document_chunking():
    """Test processing of larger documents."""
    # Create a larger test document
    large_content = "\n\n".join([f"This is paragraph {i} with some content." for i in range(50)])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(large_content)
        temp_path = f.name

    try:
        result = await document_processor.parse_document(temp_path)

        assert len(result.chunks) > 0
        assert result.word_count > 100
        assert result.metadata["total_chunks"] == len(result.chunks)

    finally:
        Path(temp_path).unlink()

def test_supported_file_types():
    """Test that all supported file types are recognized."""
    supported_extensions = [".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"]

    for ext in supported_extensions:
        doc_type = document_processor.detect_document_type(f"test{ext}")
        assert doc_type in DocumentType

def test_case_insensitive_extensions():
    """Test that file extensions are case insensitive."""
    assert document_processor.detect_document_type("test.PDF") == DocumentType.PDF
    assert document_processor.detect_document_type("test.DOCX") == DocumentType.DOCX
    assert document_processor.detect_document_type("test.MD") == DocumentType.MARKDOWN
