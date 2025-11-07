"""Tests for markitdown-based converters."""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from morag_core.interfaces.converter import ConversionOptions, ConversionResult
from morag_core.models.document import Document, DocumentMetadata

from morag_document.converters.pdf import PDFConverter
from morag_document.converters.word import WordConverter
from morag_document.converters.excel import ExcelConverter
from morag_document.converters.presentation import PresentationConverter
# from morag_document.converters.image import ImageConverter  # Module not available


@pytest.mark.asyncio
class TestMarkitdownConverters:
    """Test suite for markitdown-based converters."""

    @pytest.fixture
    def mock_markitdown_service(self):
        """Mock markitdown service."""
        service = AsyncMock()
        service.convert_file.return_value = "# Test Document\n\nThis is test content."
        service.supports_format.return_value = True
        return service

    @pytest.fixture
    def sample_file_path(self, tmp_path):
        """Create a sample file for testing."""
        file_path = tmp_path / "test_document.pdf"
        file_path.write_text("Sample content")
        return file_path

    async def test_pdf_converter_initialization(self):
        """Test PDF converter initialization."""
        converter = PDFConverter()
        assert "pdf" in converter.supported_formats
        assert await converter.supports_format("pdf")
        assert not await converter.supports_format("docx")

    async def test_word_converter_initialization(self):
        """Test Word converter initialization."""
        converter = WordConverter()
        assert "docx" in converter.supported_formats
        assert "doc" in converter.supported_formats
        assert "word" in converter.supported_formats
        assert await converter.supports_format("docx")
        assert await converter.supports_format("doc")
        assert not await converter.supports_format("pdf")

    async def test_excel_converter_initialization(self):
        """Test Excel converter initialization."""
        converter = ExcelConverter()
        assert "xlsx" in converter.supported_formats
        assert "xls" in converter.supported_formats
        assert "excel" in converter.supported_formats
        assert await converter.supports_format("xlsx")
        assert await converter.supports_format("xls")
        assert not await converter.supports_format("pdf")

    async def test_presentation_converter_initialization(self):
        """Test PowerPoint converter initialization."""
        converter = PresentationConverter()
        assert "pptx" in converter.supported_formats
        assert "ppt" in converter.supported_formats
        assert "powerpoint" in converter.supported_formats
        assert await converter.supports_format("pptx")
        assert await converter.supports_format("ppt")
        assert not await converter.supports_format("pdf")

    async def test_image_converter_initialization(self):
        """Test Image converter initialization."""
        converter = ImageConverter()
        assert "jpg" in converter.supported_formats
        assert "png" in converter.supported_formats
        assert "gif" in converter.supported_formats
        assert await converter.supports_format("jpg")
        assert await converter.supports_format("png")
        assert not await converter.supports_format("pdf")

    @patch('morag_document.converters.markitdown_base.MarkitdownService')
    async def test_pdf_converter_convert(self, mock_service_class, sample_file_path, mock_markitdown_service):
        """Test PDF converter conversion."""
        mock_service_class.return_value = mock_markitdown_service

        converter = PDFConverter()
        options = ConversionOptions()

        result = await converter.convert(sample_file_path, options)

        assert isinstance(result, ConversionResult)
        assert result.success
        assert result.content
        assert isinstance(result.document, Document)
        mock_markitdown_service.convert_file.assert_called_once()

    @patch('morag_document.converters.markitdown_base.MarkitdownService')
    async def test_word_converter_convert(self, mock_service_class, tmp_path, mock_markitdown_service):
        """Test Word converter conversion."""
        mock_service_class.return_value = mock_markitdown_service

        # Create a Word file
        file_path = tmp_path / "test_document.docx"
        file_path.write_text("Sample content")

        converter = WordConverter()
        options = ConversionOptions()

        result = await converter.convert(file_path, options)

        assert isinstance(result, ConversionResult)
        assert result.success
        assert result.content
        assert isinstance(result.document, Document)
        mock_markitdown_service.convert_file.assert_called_once()

    @patch('morag_document.converters.markitdown_base.MarkitdownService')
    async def test_excel_converter_convert(self, mock_service_class, tmp_path, mock_markitdown_service):
        """Test Excel converter conversion."""
        mock_service_class.return_value = mock_markitdown_service

        # Create an Excel file
        file_path = tmp_path / "test_document.xlsx"
        file_path.write_text("Sample content")

        converter = ExcelConverter()
        options = ConversionOptions()

        result = await converter.convert(file_path, options)

        assert isinstance(result, ConversionResult)
        assert result.success
        assert result.content
        assert isinstance(result.document, Document)
        mock_markitdown_service.convert_file.assert_called_once()

    @patch('morag_document.converters.markitdown_base.MarkitdownService')
    async def test_presentation_converter_convert(self, mock_service_class, tmp_path, mock_markitdown_service):
        """Test PowerPoint converter conversion."""
        mock_service_class.return_value = mock_markitdown_service

        # Create a PowerPoint file
        file_path = tmp_path / "test_document.pptx"
        file_path.write_text("Sample content")

        converter = PresentationConverter()
        options = ConversionOptions()

        result = await converter.convert(file_path, options)

        assert isinstance(result, ConversionResult)
        assert result.success
        assert result.content
        assert isinstance(result.document, Document)
        mock_markitdown_service.convert_file.assert_called_once()

    @patch('morag_document.converters.markitdown_base.MarkitdownService')
    async def test_image_converter_convert(self, mock_service_class, tmp_path, mock_markitdown_service):
        """Test Image converter conversion."""
        mock_service_class.return_value = mock_markitdown_service

        # Create an image file
        file_path = tmp_path / "test_image.jpg"
        file_path.write_text("Sample content")

        converter = ImageConverter()
        options = ConversionOptions()

        result = await converter.convert(file_path, options)

        assert isinstance(result, ConversionResult)
        assert result.success
        assert result.content
        assert isinstance(result.document, Document)
        mock_markitdown_service.convert_file.assert_called_once()

    async def test_converter_format_detection(self):
        """Test format detection across all converters."""
        converters = [
            (PDFConverter(), ["pdf"]),
            (WordConverter(), ["docx", "doc", "word"]),
            (ExcelConverter(), ["xlsx", "xls", "excel"]),
            (PresentationConverter(), ["pptx", "ppt", "powerpoint"]),
            (ImageConverter(), ["jpg", "png", "gif", "bmp", "tiff", "webp", "svg"])
        ]

        for converter, formats in converters:
            for format_type in formats:
                assert await converter.supports_format(format_type)
                assert await converter.supports_format(format_type.upper())

            # Test unsupported format
            assert not await converter.supports_format("unsupported")


@pytest.mark.asyncio
class TestMarkitdownIntegration:
    """Integration tests for markitdown converters."""

    async def test_all_converters_inherit_from_markitdown_base(self):
        """Test that all converters inherit from MarkitdownConverter."""
        from morag_document.converters.markitdown_base import MarkitdownConverter

        converters = [
            PDFConverter(),
            WordConverter(),
            ExcelConverter(),
            PresentationConverter(),
            ImageConverter()
        ]

        for converter in converters:
            assert isinstance(converter, MarkitdownConverter)

    async def test_converter_registration_in_processor(self):
        """Test that all converters are properly registered in the processor."""
        from morag_document.processor import DocumentProcessor

        processor = DocumentProcessor()

        # Check that all expected formats are registered
        expected_formats = {
            "pdf", "docx", "doc", "word", "xlsx", "xls", "excel",
            "pptx", "ppt", "powerpoint", "jpg", "jpeg", "png", "gif",
            "bmp", "tiff", "webp", "svg", "txt", "md", "html", "htm"
        }

        registered_formats = set(processor.converters.keys())

        # Check that markitdown formats are registered
        markitdown_formats = {
            "pdf", "docx", "doc", "word", "xlsx", "xls", "excel",
            "pptx", "ppt", "powerpoint", "jpg", "jpeg", "png", "gif",
            "bmp", "tiff", "webp", "svg"
        }

        for format_type in markitdown_formats:
            assert format_type in registered_formats, f"Format {format_type} not registered"
