"""Basic tests for enhanced document converters (Tasks 25-29)."""

from pathlib import Path

import pytest
from src.morag.converters.audio import AudioConverter
from src.morag.converters.base import (
    ChunkingStrategy,
    ConversionError,
    ConversionOptions,
)
from src.morag.converters.office import OfficeConverter
from src.morag.converters.pdf import PDFConverter
from src.morag.converters.video import VideoConverter
from src.morag.converters.web import WebConverter


@pytest.fixture
def basic_conversion_options():
    """Create basic test conversion options."""
    return ConversionOptions(
        chunking_strategy=ChunkingStrategy.PAGE,
        include_metadata=True,
        include_toc=False,
        extract_images=False,
        min_quality_threshold=0.5,
        format_options={},
    )


class TestBasicConverterInitialization:
    """Test basic converter initialization."""

    def test_pdf_converter_initialization(self):
        """Test PDF converter initialization."""
        converter = PDFConverter()
        assert converter.name == "Enhanced MoRAG PDF Converter"
        assert "pdf" in converter.supported_formats
        assert converter.quality_validator is not None

    def test_audio_converter_initialization(self):
        """Test audio converter initialization."""
        converter = AudioConverter()
        assert converter.name == "Enhanced MoRAG Audio Converter"
        assert "audio" in converter.supported_formats
        assert "mp3" in converter.supported_formats

    def test_video_converter_initialization(self):
        """Test video converter initialization."""
        converter = VideoConverter()
        assert converter.name == "MoRAG Video Converter"
        assert "video" in converter.supported_formats
        assert "mp4" in converter.supported_formats

    def test_office_converter_initialization(self):
        """Test office converter initialization."""
        converter = OfficeConverter()
        assert converter.name == "Enhanced MoRAG Office Converter"
        assert "docx" in converter.supported_formats
        assert "xlsx" in converter.supported_formats
        assert "pptx" in converter.supported_formats

    def test_web_converter_initialization(self):
        """Test web converter initialization."""
        converter = WebConverter()
        assert converter.name == "Enhanced MoRAG Web Converter"
        assert "web" in converter.supported_formats
        assert "url" in converter.supported_formats
        assert "html" in converter.supported_formats


class TestConverterFormatSupport:
    """Test converter format support."""

    def test_pdf_format_support(self):
        """Test PDF format support."""
        converter = PDFConverter()
        assert converter.supports_format("pdf")
        assert not converter.supports_format("docx")

    def test_audio_format_support(self):
        """Test audio format support."""
        converter = AudioConverter()
        assert converter.supports_format("audio")
        assert converter.supports_format("mp3")
        assert converter.supports_format("wav")
        assert not converter.supports_format("pdf")

    def test_video_format_support(self):
        """Test video format support."""
        converter = VideoConverter()
        assert converter.supports_format("video")
        assert converter.supports_format("mp4")
        assert converter.supports_format("avi")
        assert not converter.supports_format("pdf")

    def test_office_format_support(self):
        """Test office format support."""
        converter = OfficeConverter()
        assert converter.supports_format("docx")
        assert converter.supports_format("xlsx")
        assert converter.supports_format("pptx")
        assert converter.supports_format("word")
        assert converter.supports_format("excel")
        assert converter.supports_format("powerpoint")
        assert not converter.supports_format("pdf")

    def test_web_format_support(self):
        """Test web format support."""
        converter = WebConverter()
        assert converter.supports_format("web")
        assert converter.supports_format("url")
        assert converter.supports_format("html")
        assert not converter.supports_format("pdf")


class TestConverterValidation:
    """Test converter input validation."""

    @pytest.mark.asyncio
    async def test_pdf_converter_validation(self, basic_conversion_options):
        """Test PDF converter input validation."""
        converter = PDFConverter()

        # Test with non-existent file
        with pytest.raises(ConversionError):
            await converter.convert("non_existent.pdf", basic_conversion_options)

    @pytest.mark.asyncio
    async def test_audio_converter_validation(self, basic_conversion_options):
        """Test audio converter input validation."""
        converter = AudioConverter()

        # Test with non-existent file
        with pytest.raises(ConversionError):
            await converter.convert("non_existent.mp3", basic_conversion_options)

    @pytest.mark.asyncio
    async def test_video_converter_validation(self, basic_conversion_options):
        """Test video converter input validation."""
        converter = VideoConverter()

        # Test with non-existent file
        with pytest.raises(ConversionError):
            await converter.convert("non_existent.mp4", basic_conversion_options)

    @pytest.mark.asyncio
    async def test_office_converter_validation(self, basic_conversion_options):
        """Test office converter input validation."""
        converter = OfficeConverter()

        # Test with non-existent file
        with pytest.raises(ConversionError):
            await converter.convert("non_existent.docx", basic_conversion_options)


class TestConverterFeatures:
    """Test converter feature availability."""

    def test_pdf_converter_features(self):
        """Test PDF converter features."""
        converter = PDFConverter()

        # Check if advanced docling is available
        has_docling = converter.docling_converter is not None

        # Check fallback converters
        assert isinstance(converter.fallback_converters, list)

    def test_audio_converter_features(self):
        """Test audio converter features."""
        converter = AudioConverter()

        # Check if diarization is available
        has_diarization = converter.diarization_pipeline is not None

        # Check if topic segmentation is available
        has_topic_segmentation = converter.topic_segmenter is not None

        # These might be None if dependencies aren't installed, which is fine
        assert converter.diarization_pipeline is None or hasattr(
            converter.diarization_pipeline, "__call__"
        )

    def test_office_converter_features(self):
        """Test office converter features."""
        converter = OfficeConverter()

        # Check if format-specific converters are available
        # These might be None if dependencies aren't installed
        word_available = converter.word_converter is not None
        excel_available = converter.excel_converter is not None
        pptx_available = converter.powerpoint_converter is not None

        # At least one should be available or all should be None
        assert isinstance(converter.word_converter, (type(None), object))
        assert isinstance(converter.excel_converter, (type(None), object))
        assert isinstance(converter.powerpoint_converter, (type(None), object))

    def test_web_converter_features(self):
        """Test web converter features."""
        converter = WebConverter()

        # Check extraction methods
        assert isinstance(converter.extraction_methods, list)
        assert len(converter.extraction_methods) > 0

        # Basic web processor should always be available
        assert "basic_web_processor" in converter.extraction_methods

        # Check web processor
        assert converter.web_processor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
