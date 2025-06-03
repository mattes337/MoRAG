"""Tests for universal document conversion system."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.morag.converters import (
    DocumentConverter,
    ConversionOptions,
    ConversionResult,
    QualityScore,
    BaseConverter,
    ChunkingStrategy
)
from src.morag.converters.config import ConversionConfig


class TestBaseConverter:
    """Test base converter functionality."""
    
    def test_conversion_options_creation(self):
        """Test ConversionOptions creation and defaults."""
        options = ConversionOptions()
        
        assert options.preserve_formatting is True
        assert options.extract_images is True
        assert options.include_metadata is True
        assert options.chunking_strategy == ChunkingStrategy.PAGE
        assert options.min_quality_threshold == 0.7
        assert options.enable_fallback is True
    
    def test_conversion_options_for_format(self):
        """Test format-specific option creation."""
        pdf_options = ConversionOptions.for_format('pdf')
        
        assert 'use_ocr' in pdf_options.format_options
        assert pdf_options.format_options['use_ocr'] is True
        assert pdf_options.format_options['extract_tables'] is True
        
        audio_options = ConversionOptions.for_format('audio')
        assert 'enable_diarization' in audio_options.format_options
        assert audio_options.format_options['confidence_threshold'] == 0.8
    
    def test_quality_score_validation(self):
        """Test QualityScore validation."""
        # Valid scores
        quality = QualityScore(
            overall_score=0.8,
            completeness_score=0.9,
            readability_score=0.7,
            structure_score=0.8,
            metadata_preservation=0.6
        )
        assert quality.overall_score == 0.8
        
        # Invalid scores should raise ValueError
        with pytest.raises(ValueError):
            QualityScore(
                overall_score=1.5,  # Invalid: > 1.0
                completeness_score=0.9,
                readability_score=0.7,
                structure_score=0.8,
                metadata_preservation=0.6
            )
    
    def test_conversion_result_properties(self):
        """Test ConversionResult properties."""
        quality = QualityScore(
            overall_score=0.85,
            completeness_score=0.9,
            readability_score=0.8,
            structure_score=0.9,
            metadata_preservation=0.8
        )
        
        result = ConversionResult(
            content="# Test Document\n\nThis is a test.",
            metadata={'title': 'Test'},
            quality_score=quality
        )
        
        assert result.is_high_quality is True  # > 0.8
        assert result.word_count == 7  # "# Test Document This is a test."
        
        # Test low quality
        low_quality = QualityScore(
            overall_score=0.5,
            completeness_score=0.5,
            readability_score=0.5,
            structure_score=0.5,
            metadata_preservation=0.5
        )
        
        low_result = ConversionResult(
            content="Test",
            metadata={},
            quality_score=low_quality
        )
        
        assert low_result.is_high_quality is False


class MockConverter(BaseConverter):
    """Mock converter for testing."""
    
    def __init__(self, name: str, supported_formats: list):
        super().__init__(name)
        self.supported_formats = supported_formats
    
    def supports_format(self, format_type: str) -> bool:
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path, options):
        return ConversionResult(
            content=f"# Mock Conversion\n\nConverted {file_path} using {self.name}",
            metadata={'converter': self.name, 'file': str(file_path)},
            quality_score=QualityScore(
                overall_score=0.9,
                completeness_score=0.9,
                readability_score=0.9,
                structure_score=0.9,
                metadata_preservation=0.9
            ),
            success=True,
            converter_used=self.name
        )


class TestDocumentConverter:
    """Test DocumentConverter registry and orchestration."""
    
    def test_converter_registration(self):
        """Test converter registration."""
        converter = DocumentConverter()
        mock_converter = MockConverter("Test Converter", ["test"])
        
        converter.register_converter("test", mock_converter)
        
        assert converter.get_converter("test") == mock_converter
        assert "test" in converter.list_supported_formats()
    
    def test_fallback_converter_registration(self):
        """Test fallback converter registration."""
        converter = DocumentConverter()
        primary = MockConverter("Primary", ["test"])
        fallback = MockConverter("Fallback", ["test"])
        
        converter.register_converter("test", primary, is_primary=True)
        converter.register_converter("test", fallback, is_primary=False)
        
        assert converter.get_converter("test") == primary
        assert fallback in converter.get_fallback_converters("test")
    
    def test_format_detection(self):
        """Test format detection."""
        converter = DocumentConverter()
        
        assert converter.detect_format("test.pdf") == "pdf"
        assert converter.detect_format("test.docx") == "word"
        assert converter.detect_format("test.mp3") == "audio"
        assert converter.detect_format("test.mp4") == "video"
        assert converter.detect_format("test.html") == "web"
    
    @pytest.mark.asyncio
    async def test_successful_conversion(self):
        """Test successful document conversion."""
        converter = DocumentConverter()
        mock_converter = MockConverter("Test Converter", ["test"])
        converter.register_converter("test", mock_converter)
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix=".test", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = Path(tmp.name)
        
        try:
            options = ConversionOptions()
            result = await converter.convert_to_markdown(tmp_path, options)
            
            assert result.success is True
            assert result.converter_used == "Test Converter"
            assert "Mock Conversion" in result.content
            assert result.original_format == "test"
            
        finally:
            tmp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        converter = DocumentConverter()
        
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".unsupported", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(Exception):  # Should raise UnsupportedFormatError
                await converter.convert_to_markdown(tmp_path)
        finally:
            tmp_path.unlink()
    
    def test_converter_info(self):
        """Test converter information retrieval."""
        converter = DocumentConverter()
        primary = MockConverter("Primary", ["test"])
        fallback = MockConverter("Fallback", ["test"])
        
        converter.register_converter("test", primary, is_primary=True)
        converter.register_converter("test", fallback, is_primary=False)
        
        info = converter.get_converter_info()
        
        assert "test" in info
        assert info["test"]["primary_converter"] == "Primary"
        assert "Fallback" in info["test"]["fallback_converters"]


class TestConversionConfig:
    """Test conversion configuration system."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = ConversionConfig()
        
        assert config.default_options['preserve_formatting'] is True
        assert config.default_options['chunking_strategy'] == 'page'
        assert 'pdf' in config.format_specific
        assert 'audio' in config.format_specific
    
    def test_format_options_retrieval(self):
        """Test format-specific options retrieval."""
        config = ConversionConfig()
        
        pdf_options = config.get_format_options('pdf')
        assert 'use_docling' in pdf_options
        assert pdf_options['use_docling'] is True
        
        # Non-existent format should return empty dict
        unknown_options = config.get_format_options('unknown')
        assert unknown_options == {}
    
    def test_format_options_update(self):
        """Test format options updating."""
        config = ConversionConfig()
        
        config.update_format_options('pdf', {'new_option': True})
        
        pdf_options = config.get_format_options('pdf')
        assert pdf_options['new_option'] is True
        assert pdf_options['use_docling'] is True  # Existing option preserved
    
    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        config = ConversionConfig()
        config.default_options['test_option'] = True
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save configuration
            config.save_to_file(tmp_path)
            assert tmp_path.exists()
            
            # Load configuration
            loaded_config = ConversionConfig.load_from_file(tmp_path)
            assert loaded_config.default_options['test_option'] is True
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the universal converter system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_conversion_flow(self):
        """Test complete conversion flow."""
        # This test would require actual file processing
        # For now, we'll test the flow with mocked components
        
        converter = DocumentConverter()
        mock_converter = MockConverter("Integration Test", ["pdf"])
        converter.register_converter("pdf", mock_converter)
        
        # Create test PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp_path = Path(tmp.name)
        
        try:
            options = ConversionOptions.for_format('pdf')
            result = await converter.convert_to_markdown(tmp_path, options)
            
            assert result.success is True
            assert result.original_format == "pdf"
            assert result.quality_score is not None
            assert result.quality_score.overall_score > 0.8
            
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
