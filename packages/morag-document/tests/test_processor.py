"""Tests for document processor."""

import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from morag_core.interfaces.converter import ChunkingStrategy, ConversionOptions
from morag_core.interfaces.processor import ProcessingConfig
from morag_core.models.document import Document
from morag_core.exceptions import ValidationError, ProcessingError

from morag_document.processor import DocumentProcessor
from morag_document.converters.base import DocumentConverter


@pytest.fixture
def processor():
    """Create document processor fixture."""
    return DocumentProcessor()


@pytest.fixture
def mock_converter():
    """Create mock document converter fixture."""
    converter = MagicMock(spec=DocumentConverter)
    converter.supported_formats = {"test"}
    converter.convert = AsyncMock()
    return converter


@pytest.fixture
def sample_document():
    """Create sample document fixture."""
    return Document(
        raw_text="Sample document text",
        metadata={
            "title": "Test Document",
            "file_type": "test",
            "word_count": 3,
        }
    )


@pytest.mark.asyncio
async def test_processor_initialization(processor):
    """Test processor initialization."""
    # Check that converters are registered
    assert processor.converters
    assert "pdf" in processor.converters
    assert "docx" in processor.converters
    assert "txt" in processor.converters
    assert "xlsx" in processor.converters
    assert "pptx" in processor.converters


@pytest.mark.asyncio
async def test_supports_format(processor):
    """Test format support check."""
    # Check supported formats
    assert await processor.supports_format("pdf")
    assert await processor.supports_format("docx")
    assert await processor.supports_format("txt")
    assert await processor.supports_format("xlsx")
    assert await processor.supports_format("pptx")

    # Check case insensitivity
    assert await processor.supports_format("PDF")
    assert await processor.supports_format("DOCX")

    # Check unsupported format
    assert not await processor.supports_format("invalid")


@pytest.mark.asyncio
async def test_validate_input(processor, tmp_path):
    """Test input validation."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")

    # Valid input
    config = ProcessingConfig(file_path=str(test_file))
    assert await processor.validate_input(config)

    # Missing file path
    with pytest.raises(ValidationError):
        await processor.validate_input(ProcessingConfig())

    # Non-existent file
    with pytest.raises(ValidationError):
        await processor.validate_input(ProcessingConfig(file_path="non_existent.txt"))

    # Directory instead of file
    with pytest.raises(ValidationError):
        await processor.validate_input(ProcessingConfig(file_path=str(tmp_path)))


@pytest.mark.asyncio
async def test_process_with_custom_converter(processor, mock_converter, sample_document, tmp_path):
    """Test processing with custom converter."""
    # Create test file
    test_file = tmp_path / "test.test"
    test_file.write_text("Test content")

    # Set up mock converter
    mock_converter.convert.return_value = MagicMock(
        document=sample_document,
        quality=MagicMock(score=0.9, issues=[]),
        warnings=[],
    )

    # Register mock converter
    processor.converters["test"] = mock_converter

    # Process file
    with patch("morag_document.processor.detect_format", return_value="test"):
        result = await processor.process_file(test_file)

    # Check result
    assert result.document == sample_document
    assert result.metadata["quality_score"] == 0.9
    assert result.metadata["quality_issues"] == []
    assert result.metadata["warnings"] == []

    # Check converter was called with correct arguments
    mock_converter.convert.assert_called_once()
    call_args = mock_converter.convert.call_args[0]
    assert call_args[0] == test_file
    assert isinstance(call_args[1], ConversionOptions)
    assert call_args[1].format_type == "test"
    assert call_args[1].chunking_strategy == ChunkingStrategy.PARAGRAPH


@pytest.mark.asyncio
async def test_process_with_unsupported_format(processor, tmp_path):
    """Test processing with unsupported format."""
    # Create test file
    test_file = tmp_path / "test.unsupported"
    test_file.write_text("Test content")

    # Process file with unsupported format
    with patch("morag_document.processor.detect_format", return_value="unsupported"):
        with pytest.raises(ProcessingError):
            await processor.process_file(test_file)


@pytest.mark.asyncio
async def test_process_with_custom_options(processor, mock_converter, sample_document, tmp_path):
    """Test processing with custom options."""
    # Create test file
    test_file = tmp_path / "test.test"
    test_file.write_text("Test content")

    # Set up mock converter
    mock_converter.convert.return_value = MagicMock(
        document=sample_document,
        quality=MagicMock(score=0.9, issues=[]),
        warnings=[],
    )

    # Register mock converter
    processor.converters["test"] = mock_converter

    # Process file with custom options
    with patch("morag_document.processor.detect_format", return_value="test"):
        result = await processor.process_file(
            test_file,
            chunking_strategy=ChunkingStrategy.SENTENCE,
            chunk_size=500,
            chunk_overlap=50,
            extract_metadata=False,
        )

    # Check converter was called with correct options
    mock_converter.convert.assert_called_once()
    call_args = mock_converter.convert.call_args[0]
    assert call_args[1].chunking_strategy == ChunkingStrategy.SENTENCE
    assert call_args[1].chunk_size == 500
    assert call_args[1].chunk_overlap == 50
    assert call_args[1].extract_metadata is False
