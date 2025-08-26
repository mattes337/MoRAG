"""Tests for markitdown base converter."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from morag_document.converters.markitdown_base import MarkitdownConverter
from morag_core.interfaces.converter import ConversionOptions, ChunkingStrategy
from morag_core.exceptions import ConversionError, UnsupportedFormatError
from morag_core.models.document import DocumentType


class TestMarkitdownConverter:
    """Test markitdown base converter functionality."""

    @pytest.fixture
    def converter(self):
        """Create markitdown converter instance."""
        converter = MarkitdownConverter()
        converter.supported_formats = {'pdf', 'docx', 'txt'}
        return converter

    @pytest.fixture
    def mock_markitdown_service(self):
        """Mock markitdown service."""
        service = Mock()
        service.convert_file = AsyncMock(return_value="# Test Document\n\nThis is test content.")
        service.supports_format = AsyncMock(return_value=True)
        return service

    @pytest.mark.asyncio
    async def test_converter_initialization(self, converter):
        """Test converter initialization."""
        assert isinstance(converter.supported_formats, set)
        assert converter._markitdown_service is None

    @pytest.mark.asyncio
    async def test_get_markitdown_service(self, converter):
        """Test getting markitdown service instance."""
        service1 = await converter._get_markitdown_service()
        service2 = await converter._get_markitdown_service()
        
        # Should return the same instance
        assert service1 is service2

    @pytest.mark.asyncio
    async def test_supports_format(self, converter, mock_markitdown_service):
        """Test format support checking."""
        converter._markitdown_service = mock_markitdown_service
        
        # Test supported format
        result = await converter.supports_format('pdf')
        assert result is True
        
        # Test unsupported format
        converter.supported_formats = {'pdf'}
        result = await converter.supports_format('docx')
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_input_success(self, converter, tmp_path):
        """Test successful input validation."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        result = await converter.validate_input(test_file)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_input_file_not_found(self, converter):
        """Test validation with non-existent file."""
        with pytest.raises(ConversionError, match="File not found"):
            await converter.validate_input("non_existent.txt")

    @pytest.mark.asyncio
    async def test_validate_input_not_a_file(self, converter, tmp_path):
        """Test validation with directory."""
        with pytest.raises(ConversionError, match="Not a file"):
            await converter.validate_input(tmp_path)

    @pytest.mark.asyncio
    async def test_validate_input_empty_file(self, converter, tmp_path):
        """Test validation with empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.touch()
        
        with pytest.raises(ConversionError, match="Empty file"):
            await converter.validate_input(test_file)

    @pytest.mark.asyncio
    async def test_detect_format(self, converter, tmp_path):
        """Test format detection."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")
        
        with patch('morag_core.utils.file_handling.detect_format', return_value='pdf'):
            format_type = converter.detect_format(test_file)
            assert format_type == 'pdf'

    def test_map_format_to_document_type(self, converter):
        """Test format to document type mapping."""
        assert converter._map_format_to_document_type('pdf') == DocumentType.PDF
        assert converter._map_format_to_document_type('docx') == DocumentType.WORD
        assert converter._map_format_to_document_type('xlsx') == DocumentType.EXCEL
        assert converter._map_format_to_document_type('pptx') == DocumentType.POWERPOINT
        assert converter._map_format_to_document_type('txt') == DocumentType.TEXT
        assert converter._map_format_to_document_type('jpg') == DocumentType.IMAGE
        assert converter._map_format_to_document_type('unknown') == DocumentType.UNKNOWN

    @pytest.mark.asyncio
    async def test_convert_success(self, converter, tmp_path, mock_markitdown_service):
        """Test successful document conversion."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Mock dependencies
        converter._markitdown_service = mock_markitdown_service
        
        with patch('morag_core.utils.file_handling.get_file_info') as mock_get_file_info:
            mock_get_file_info.return_value = {
                'file_name': 'test.txt',
                'mime_type': 'text/plain',
                'file_size': 12
            }
            
            with patch('morag_core.utils.file_handling.detect_format', return_value='txt'):
                options = ConversionOptions(chunking_strategy=ChunkingStrategy.NONE)
                result = await converter.convert(test_file, options)
                
                assert result.success is True
                assert result.content == "# Test Document\n\nThis is test content."
                assert result.document.raw_text == "# Test Document\n\nThis is test content."
                assert result.metadata['converter'] == 'markitdown'
                assert result.metadata['format'] == 'txt'

    @pytest.mark.asyncio
    async def test_convert_markitdown_disabled(self, converter, tmp_path):
        """Test conversion when markitdown is disabled."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Mock settings with markitdown disabled
        mock_settings = Mock()
        mock_settings.markitdown_enabled = False
        converter.settings = mock_settings
        
        with pytest.raises(ConversionError, match="Markitdown is disabled"):
            await converter.convert(test_file)

    @pytest.mark.asyncio
    async def test_convert_unsupported_format(self, converter, tmp_path):
        """Test conversion with unsupported format."""
        # Create test file
        test_file = tmp_path / "test.unknown"
        test_file.write_text("Test content")
        
        converter.supported_formats = {'pdf'}
        
        with patch('morag_core.utils.file_handling.detect_format', return_value='unknown'):
            with pytest.raises(UnsupportedFormatError):
                await converter.convert(test_file)

    @pytest.mark.asyncio
    async def test_convert_with_chunking(self, converter, tmp_path, mock_markitdown_service):
        """Test conversion with chunking enabled."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Mock dependencies
        converter._markitdown_service = mock_markitdown_service
        
        with patch('morag_core.utils.file_handling.get_file_info') as mock_get_file_info:
            mock_get_file_info.return_value = {
                'file_name': 'test.txt',
                'mime_type': 'text/plain',
                'file_size': 12
            }
            
            with patch('morag_core.utils.file_handling.detect_format', return_value='txt'):
                options = ConversionOptions(
                    chunking_strategy=ChunkingStrategy.PARAGRAPH,
                    chunk_size=100,
                    chunk_overlap=20
                )
                result = await converter.convert(test_file, options)
                
                assert result.success is True
                assert len(result.document.chunks) > 0

    @pytest.mark.asyncio
    async def test_assess_quality_no_content(self, converter):
        """Test quality assessment with no content."""
        from morag_core.models.document import Document, DocumentMetadata, DocumentType
        
        metadata = DocumentMetadata(
            source_type=DocumentType.TEXT,
            source_name="test.txt"
        )
        document = Document(metadata=metadata)
        
        quality = await converter.assess_quality(document)
        assert quality.overall_score == 0.0
        assert "No text content extracted" in quality.issues_detected

    @pytest.mark.asyncio
    async def test_assess_quality_short_content(self, converter):
        """Test quality assessment with short content."""
        from morag_core.models.document import Document, DocumentMetadata, DocumentType
        
        metadata = DocumentMetadata(
            source_type=DocumentType.TEXT,
            source_name="test.txt"
        )
        document = Document(metadata=metadata)
        document.raw_text = "Short"
        
        quality = await converter.assess_quality(document)
        assert quality.overall_score < 1.0
        assert "Very short text content" in quality.issues_detected

    def test_get_markitdown_options(self, converter):
        """Test markitdown options mapping."""
        options = ConversionOptions()
        markitdown_options = converter._get_markitdown_options(options)
        
        assert isinstance(markitdown_options, dict)

    def test_find_word_boundary_backward(self, converter):
        """Test finding word boundary in backward direction."""
        text = "This is a test sentence."
        
        # Test finding boundary at space
        boundary = converter._find_word_boundary(text, 10, "backward")
        assert boundary <= 10
        assert text[boundary-1:boundary+1] in [" i", "s ", "is"]

    def test_find_word_boundary_forward(self, converter):
        """Test finding word boundary in forward direction."""
        text = "This is a test sentence."
        
        # Test finding boundary at space
        boundary = converter._find_word_boundary(text, 10, "forward")
        assert boundary >= 10

    def test_detect_sentence_boundaries(self, converter):
        """Test sentence boundary detection."""
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        
        boundaries = converter._detect_sentence_boundaries(text)
        assert len(boundaries) >= 2  # At least start and end
        assert boundaries[0] == 0
        assert boundaries[-1] == len(text)
