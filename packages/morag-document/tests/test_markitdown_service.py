"""Tests for markitdown service."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from morag_document.services.markitdown_service import MarkitdownService
from morag_core.exceptions import ConversionError, UnsupportedFormatError


class TestMarkitdownService:
    """Test markitdown service functionality."""

    @pytest.fixture
    def service(self):
        """Create markitdown service instance."""
        return MarkitdownService()

    @pytest.fixture
    def mock_markitdown_result(self):
        """Mock markitdown conversion result."""
        result = Mock()
        result.text_content = "# Test Document\n\nThis is test content."
        return result

    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test service initialization."""
        assert service._markitdown is None
        assert not service._initialized

        # Initialize service
        await service._initialize()

        assert service._initialized
        assert service._markitdown is not None

    @pytest.mark.asyncio
    async def test_get_supported_formats(self, service):
        """Test getting supported formats."""
        formats = await service.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert 'pdf' in formats
        assert 'docx' in formats
        assert 'xlsx' in formats
        assert 'pptx' in formats

    @pytest.mark.asyncio
    async def test_supports_format(self, service):
        """Test format support checking."""
        # Test supported formats
        assert await service.supports_format('pdf')
        assert await service.supports_format('docx')
        assert await service.supports_format('xlsx')

        # Test unsupported format
        assert not await service.supports_format('unknown')

    @pytest.mark.asyncio
    async def test_convert_file_success(self, service, tmp_path, mock_markitdown_result):
        """Test successful file conversion."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with patch.object(service, '_convert_sync', return_value=mock_markitdown_result):
            result = await service.convert_file(test_file)

            assert result == "# Test Document\n\nThis is test content."

    @pytest.mark.asyncio
    async def test_convert_file_not_found(self, service):
        """Test conversion with non-existent file."""
        with pytest.raises(ConversionError, match="File not found"):
            await service.convert_file("non_existent_file.txt")

    @pytest.mark.asyncio
    async def test_convert_file_empty_file(self, service, tmp_path):
        """Test conversion with empty file."""
        # Create empty file
        test_file = tmp_path / "empty.txt"
        test_file.touch()

        with pytest.raises(ConversionError, match="Empty file"):
            await service.convert_file(test_file)

    @pytest.mark.asyncio
    async def test_convert_file_directory(self, service, tmp_path):
        """Test conversion with directory instead of file."""
        with pytest.raises(ConversionError, match="Not a file"):
            await service.convert_file(tmp_path)

    @pytest.mark.asyncio
    async def test_convert_file_empty_result(self, service, tmp_path):
        """Test conversion with empty result."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Mock empty result
        empty_result = Mock()
        empty_result.text_content = ""

        with patch.object(service, '_convert_sync', return_value=empty_result):
            with pytest.raises(ConversionError, match="empty content"):
                await service.convert_file(test_file)

    @pytest.mark.asyncio
    async def test_convert_sync_unsupported_format(self, service):
        """Test synchronous conversion with unsupported format."""
        with patch('markitdown.MarkItDown') as mock_markitdown_class:
            mock_instance = Mock()
            mock_instance.convert.side_effect = Exception("Format not supported")
            mock_markitdown_class.return_value = mock_instance

            service._markitdown = mock_instance

            with pytest.raises(UnsupportedFormatError):
                service._convert_sync("test.unknown", {})

    @pytest.mark.asyncio
    async def test_get_conversion_info(self, service, tmp_path):
        """Test getting conversion information."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        info = await service.get_conversion_info(test_file)

        assert info['file_path'] == str(test_file)
        assert info['format'] == 'pdf'
        assert info['supported'] is True
        assert info['file_size'] > 0
        assert info['service'] == 'markitdown'

    @pytest.mark.asyncio
    async def test_get_conversion_info_not_found(self, service):
        """Test getting conversion info for non-existent file."""
        with pytest.raises(ConversionError, match="File not found"):
            await service.get_conversion_info("non_existent.pdf")

    @pytest.mark.asyncio
    async def test_configuration_with_azure(self, service):
        """Test configuration with Azure Document Intelligence."""
        # Mock settings with Azure enabled
        mock_settings = Mock()
        mock_settings.markitdown_use_azure_doc_intel = True
        mock_settings.markitdown_azure_endpoint = "https://test.cognitiveservices.azure.com/"

        service.settings = mock_settings

        # Test configuration (should not raise errors)
        await service._configure_markitdown()

    @pytest.mark.asyncio
    async def test_configuration_with_llm_image_description(self, service):
        """Test configuration with LLM image description."""
        # Mock settings with LLM image description enabled
        mock_settings = Mock()
        mock_settings.markitdown_use_llm_image_description = True

        service.settings = mock_settings

        # Test configuration (should not raise errors)
        await service._configure_markitdown()

    @pytest.mark.asyncio
    async def test_import_error_handling(self, service):
        """Test handling of markitdown import errors."""
        with patch('builtins.__import__', side_effect=ImportError("markitdown not found")):
            with pytest.raises(ConversionError, match="Markitdown is not installed"):
                await service._initialize()

    @pytest.mark.asyncio
    async def test_conversion_options_mapping(self, service):
        """Test conversion options mapping."""
        options = {"test_option": "test_value"}
        markitdown_options = service._get_markitdown_options(Mock())

        # Currently returns empty dict, but structure is in place
        assert isinstance(markitdown_options, dict)
