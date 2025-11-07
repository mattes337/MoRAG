"""Integration tests for universal document conversion system."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.morag.converters import (
    ChunkingStrategy,
    ConversionOptions,
    ConversionResult,
    DocumentConverter,
    QualityScore,
)
from src.morag.services.universal_converter import UniversalConverterService


@pytest.fixture
def converter_service():
    """Create a fresh converter service for testing."""
    return UniversalConverterService()


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"Sample PDF content for testing")
        yield Path(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    content = """# Test Document

This is a test document with multiple sections.

## Introduction

This document is used for testing the universal converter.

## Main Content

Here is some main content with:
- Bullet points
- Multiple paragraphs
- Various formatting

## Conclusion

This concludes the test document.
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        yield Path(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


class TestUniversalConverterService:
    """Test the universal converter service."""

    def test_service_initialization(self, converter_service):
        """Test service initializes correctly."""
        assert converter_service.document_converter is not None
        assert converter_service.stats["total_conversions"] == 0
        assert converter_service.stats["successful_conversions"] == 0

    def test_get_supported_formats(self, converter_service):
        """Test getting supported formats."""
        formats = converter_service.get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        # Should include at least PDF and audio formats
        assert any("pdf" in fmt.lower() for fmt in formats)

    def test_get_converter_info(self, converter_service):
        """Test getting converter information."""
        info = converter_service.get_converter_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_get_statistics(self, converter_service):
        """Test getting conversion statistics."""
        stats = converter_service.get_statistics()
        assert isinstance(stats, dict)
        assert "total_conversions" in stats
        assert "successful_conversions" in stats
        assert "failed_conversions" in stats

    def test_reset_statistics(self, converter_service):
        """Test resetting statistics."""
        # Modify stats first
        converter_service.stats["total_conversions"] = 10
        converter_service.stats["successful_conversions"] = 8

        # Reset
        converter_service.reset_statistics()

        # Verify reset
        assert converter_service.stats["total_conversions"] == 0
        assert converter_service.stats["successful_conversions"] == 0

    @pytest.mark.asyncio
    async def test_convert_document_validation_error(self, converter_service):
        """Test document conversion with validation error."""
        non_existent_file = Path("non_existent_file.pdf")

        with pytest.raises(Exception):  # Should raise ValidationError
            await converter_service.convert_document(non_existent_file)

    @pytest.mark.asyncio
    async def test_convert_document_file_too_large(self, converter_service):
        """Test document conversion with file too large."""
        # Create a large temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            # Write content larger than max size
            large_content = b"x" * (200 * 1024 * 1024)  # 200MB
            tmp.write(large_content)
            large_file = Path(tmp.name)

        try:
            with pytest.raises(Exception):  # Should raise ValidationError
                await converter_service.convert_document(large_file)
        finally:
            large_file.unlink(missing_ok=True)


class TestConversionOptions:
    """Test conversion options functionality."""

    def test_default_options(self):
        """Test default conversion options."""
        options = ConversionOptions()

        assert options.preserve_formatting is True
        assert options.extract_images is True
        assert options.include_metadata is True
        assert options.chunking_strategy == ChunkingStrategy.PAGE
        assert options.min_quality_threshold == 0.7
        assert options.enable_fallback is True

    def test_format_specific_options(self):
        """Test format-specific option creation."""
        pdf_options = ConversionOptions.for_format("pdf")
        assert "use_ocr" in pdf_options.format_options
        assert pdf_options.format_options["use_ocr"] is True

        audio_options = ConversionOptions.for_format("audio")
        assert "enable_diarization" in audio_options.format_options
        assert "confidence_threshold" in audio_options.format_options

        video_options = ConversionOptions.for_format("video")
        assert "enable_diarization" in video_options.format_options
        assert "include_timestamps" in video_options.format_options

        web_options = ConversionOptions.for_format("web")
        assert "follow_redirects" in web_options.format_options
        assert "extract_main_content" in web_options.format_options


class TestQualityAssessment:
    """Test quality assessment functionality."""

    def test_quality_score_creation(self):
        """Test quality score creation and validation."""
        quality = QualityScore(
            overall_score=0.8,
            completeness_score=0.9,
            readability_score=0.7,
            structure_score=0.8,
            metadata_preservation=0.6,
        )

        assert quality.overall_score == 0.8
        assert quality.completeness_score == 0.9

    def test_quality_score_validation(self):
        """Test quality score validation."""
        with pytest.raises(ValueError):
            QualityScore(
                overall_score=1.5,  # Invalid: > 1.0
                completeness_score=0.9,
                readability_score=0.7,
                structure_score=0.8,
                metadata_preservation=0.6,
            )

        with pytest.raises(ValueError):
            QualityScore(
                overall_score=0.8,
                completeness_score=-0.1,  # Invalid: < 0.0
                readability_score=0.7,
                structure_score=0.8,
                metadata_preservation=0.6,
            )


class TestDocumentConverter:
    """Test document converter registry."""

    def test_format_detection(self):
        """Test format detection."""
        converter = DocumentConverter()

        assert converter.detect_format("test.pdf") == "pdf"
        assert converter.detect_format("test.docx") == "word"
        assert converter.detect_format("test.mp3") == "audio"
        assert converter.detect_format("test.mp4") == "video"
        assert converter.detect_format("test.html") == "web"
        assert converter.detect_format("test.unknown") == "unknown"

    def test_converter_registration(self):
        """Test converter registration."""
        converter = DocumentConverter()

        # Test getting converters
        pdf_converter = converter.get_converter("pdf")
        assert pdf_converter is not None
        assert pdf_converter.name == "MoRAG PDF Converter"

        # Test supported formats
        formats = converter.list_supported_formats()
        assert "pdf" in formats
        assert "audio" in formats


@pytest.mark.integration
class TestEndToEndConversion:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_text_file_conversion_mock(self, converter_service, sample_text_file):
        """Test text file conversion with mocked processors."""

        # Mock the document processor to avoid dependency issues
        with patch("src.morag.converters.pdf.document_processor") as mock_processor:
            # Create mock parse result
            mock_parse_result = Mock()
            mock_parse_result.metadata = {
                "title": "Test Document",
                "filename": sample_text_file.name,
                "parser_used": "mock",
                "chunking_strategy": "page",
            }
            mock_parse_result.chunks = [
                Mock(
                    content="# Test Document\n\nThis is a test document.",
                    metadata={"page_number": 1},
                ),
                Mock(
                    content="## Introduction\n\nThis document is used for testing.",
                    metadata={"page_number": 2},
                ),
            ]
            mock_parse_result.images = []
            mock_parse_result.total_pages = 2
            mock_parse_result.word_count = 20

            mock_processor.parse_document = AsyncMock(return_value=mock_parse_result)

            # Mock chunking and embedding services
            with patch.object(
                converter_service, "chunking_service"
            ) as mock_chunking, patch.object(
                converter_service, "embedding_service"
            ) as mock_embedding:
                mock_chunking.chunk_with_metadata = AsyncMock(
                    return_value=[
                        Mock(text="Test chunk 1", metadata={}),
                        Mock(text="Test chunk 2", metadata={}),
                    ]
                )

                mock_embedding.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])

                # Convert the file (treating it as PDF for testing)
                options = ConversionOptions.for_format("pdf")
                result = await converter_service.convert_document(
                    sample_text_file, options=options, generate_embeddings=True
                )

                # Verify results
                assert result["success"] is True
                assert len(result["content"]) > 0
                assert "Test Document" in result["content"]
                assert len(result["chunks"]) == 2
                assert len(result["embeddings"]) == 2
                assert result["quality_score"] is not None
                assert result["processing_time"] > 0

    @pytest.mark.asyncio
    async def test_conversion_statistics_update(
        self, converter_service, sample_text_file
    ):
        """Test that conversion statistics are updated correctly."""

        initial_stats = converter_service.get_statistics()
        initial_total = initial_stats["total_conversions"]

        # Mock successful conversion
        with patch("src.morag.converters.pdf.document_processor") as mock_processor:
            mock_parse_result = Mock()
            mock_parse_result.metadata = {"title": "Test"}
            mock_parse_result.chunks = [Mock(content="Test content", metadata={})]
            mock_parse_result.images = []
            mock_parse_result.total_pages = 1
            mock_parse_result.word_count = 2

            mock_processor.parse_document = AsyncMock(return_value=mock_parse_result)

            with patch.object(
                converter_service, "chunking_service"
            ) as mock_chunking, patch.object(
                converter_service, "embedding_service"
            ) as mock_embedding:
                mock_chunking.chunk_with_metadata = AsyncMock(return_value=[])
                mock_embedding.embed_text = AsyncMock(return_value=[])

                # Perform conversion
                await converter_service.convert_document(sample_text_file)

                # Check statistics updated
                updated_stats = converter_service.get_statistics()
                assert updated_stats["total_conversions"] == initial_total + 1
                assert (
                    updated_stats["successful_conversions"]
                    >= initial_stats["successful_conversions"]
                )


@pytest.mark.performance
class TestPerformance:
    """Performance tests for the universal converter."""

    @pytest.mark.asyncio
    async def test_conversion_performance(self, converter_service, sample_text_file):
        """Test conversion performance meets requirements."""
        import time

        with patch("src.morag.converters.pdf.document_processor") as mock_processor:
            mock_parse_result = Mock()
            mock_parse_result.metadata = {"title": "Performance Test"}
            mock_parse_result.chunks = [
                Mock(content="Performance test content", metadata={})
            ]
            mock_parse_result.images = []
            mock_parse_result.total_pages = 1
            mock_parse_result.word_count = 3

            mock_processor.parse_document = AsyncMock(return_value=mock_parse_result)

            with patch.object(
                converter_service, "chunking_service"
            ) as mock_chunking, patch.object(
                converter_service, "embedding_service"
            ) as mock_embedding:
                mock_chunking.chunk_with_metadata = AsyncMock(return_value=[])
                mock_embedding.embed_text = AsyncMock(return_value=[])

                start_time = time.time()
                result = await converter_service.convert_document(sample_text_file)
                end_time = time.time()

                processing_time = end_time - start_time

                # Performance assertions
                assert (
                    processing_time < 10.0
                )  # Should complete within 10 seconds for small file
                assert result["success"] is True
                assert result["processing_time"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
