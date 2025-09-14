"""Tests for enhanced document converters (Tasks 25-29)."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from src.morag.converters.pdf import PDFConverter
from src.morag.converters.audio import AudioConverter
from src.morag.converters.video import VideoConverter
from src.morag.converters.office import OfficeConverter
from src.morag.converters.web import WebConverter
from src.morag.converters.base import ConversionOptions, ChunkingStrategy


@pytest.fixture
def conversion_options():
    """Create test conversion options."""
    return ConversionOptions(
        chunking_strategy=ChunkingStrategy.PAGE,
        include_metadata=True,
        include_toc=True,
        extract_images=True,
        min_quality_threshold=0.7,
        format_options={
            'use_advanced_docling': True,
            'extract_tables': True,
            'use_ocr': True,
            'enable_diarization': True,
            'enable_topic_segmentation': True,
            'extract_keyframes': True,
            'include_audio': True,
            'extract_main_content': True,
            'include_links': True
        }
    )


class TestEnhancedPDFConverter:
    """Test enhanced PDF converter with advanced docling features."""
    
    def test_pdf_converter_initialization(self):
        """Test PDF converter initialization."""
        converter = PDFConverter()
        assert converter.name == "Enhanced MoRAG PDF Converter"
        assert 'pdf' in converter.supported_formats
        assert converter.quality_validator is not None
    
    @pytest.mark.asyncio
    async def test_pdf_conversion_with_advanced_docling(self, conversion_options):
        """Test PDF conversion with advanced docling features."""
        converter = PDFConverter()
        
        # Mock docling converter
        with patch.object(converter, 'docling_converter') as mock_docling:
            mock_result = Mock()
            mock_result.status.name = "SUCCESS"
            mock_result.document.pages = [Mock(), Mock()]  # 2 pages
            mock_result.document.name = "Test Document"
            mock_result.input.file.name = "test.pdf"
            mock_docling.convert.return_value = mock_result
            
            # Mock file path
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                
                try:
                    result = await converter.convert(tmp_path, conversion_options)
                    
                    assert result.success
                    assert result.content
                    assert "Test Document" in result.content
                    assert result.metadata['processing_method'] == 'advanced_docling'
                    assert result.quality_score.overall_score > 0
                    
                finally:
                    os.unlink(tmp_path)
    
    @pytest.mark.asyncio
    async def test_pdf_fallback_conversion(self, conversion_options):
        """Test PDF conversion fallback when advanced docling fails."""
        converter = PDFConverter()
        
        # Mock docling converter to fail
        with patch.object(converter, 'docling_converter', None):
            with patch('src.morag.processors.document.document_processor.parse_document') as mock_parse:
                mock_result = Mock()
                mock_result.metadata = {'filename': 'test.pdf'}
                mock_result.chunks = [Mock()]
                mock_result.images = []
                mock_parse.return_value = mock_result
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    
                    try:
                        result = await converter.convert(tmp_path, conversion_options)
                        
                        assert result.success
                        assert result.fallback_used
                        
                    finally:
                        os.unlink(tmp_path)


class TestEnhancedAudioConverter:
    """Test enhanced audio converter with speaker diarization."""
    
    def test_audio_converter_initialization(self):
        """Test audio converter initialization."""
        converter = AudioConverter()
        assert converter.name == "Enhanced MoRAG Audio Converter"
        assert 'audio' in converter.supported_formats
        assert 'mp3' in converter.supported_formats
    
    @pytest.mark.asyncio
    async def test_audio_conversion_with_diarization(self, conversion_options):
        """Test audio conversion with speaker diarization."""
        converter = AudioConverter()
        
        # Mock audio processor
        with patch('src.morag.processors.audio.audio_processor.process_audio') as mock_process:
            mock_result = Mock()
            mock_result.transcript = "Hello world. This is a test."
            mock_result.metadata = {'duration': 30.0, 'filename': 'test.mp3'}
            mock_result.summary = "Test audio summary"
            mock_result.segments = []
            mock_process.return_value = mock_result
            
            # Mock speaker diarization
            with patch.object(converter, 'diarization_pipeline') as mock_diarization:
                mock_diarization_result = Mock()
                mock_diarization_result.itertracks.return_value = [
                    (Mock(start=0, end=15), None, 'SPEAKER_00'),
                    (Mock(start=15, end=30), None, 'SPEAKER_01')
                ]
                mock_diarization.return_value = mock_diarization_result
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    
                    try:
                        result = await converter.convert(tmp_path, conversion_options)
                        
                        assert result.success
                        assert result.content
                        assert "Audio Transcription" in result.content
                        assert result.metadata.get('diarization_used')
                        
                    finally:
                        os.unlink(tmp_path)


class TestEnhancedOfficeConverter:
    """Test enhanced office converter with full format support."""
    
    def test_office_converter_initialization(self):
        """Test office converter initialization."""
        converter = OfficeConverter()
        assert converter.name == "Enhanced MoRAG Office Converter"
        assert 'docx' in converter.supported_formats
        assert 'xlsx' in converter.supported_formats
        assert 'pptx' in converter.supported_formats
    
    @pytest.mark.asyncio
    async def test_word_document_conversion(self, conversion_options):
        """Test Word document conversion."""
        converter = OfficeConverter()
        
        # Mock Word converter
        with patch.object(converter, 'word_converter') as mock_word_converter:
            mock_result = Mock()
            mock_result.success = True
            mock_result.content = "# Test Document\n\nContent here"
            mock_result.metadata = {'office_type': 'word'}
            mock_result.quality_score = Mock(overall_score=0.9)
            mock_word_converter.convert.return_value = mock_result
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                
                try:
                    result = await converter.convert(tmp_path, conversion_options)
                    
                    assert result.success
                    assert result.content
                    assert "Test Document" in result.content
                    assert result.metadata['office_type'] == 'word'
                    
                finally:
                    os.unlink(tmp_path)
    
    @pytest.mark.asyncio
    async def test_fallback_conversion(self, conversion_options):
        """Test fallback conversion for unsupported formats."""
        converter = OfficeConverter()
        
        # Mock no converters available
        converter.word_converter = None
        converter.excel_converter = None
        converter.powerpoint_converter = None
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            try:
                result = await converter.convert(tmp_path, conversion_options)
                
                assert result.success
                assert result.warnings
                assert "Limited support" in result.warnings[0]
                
            finally:
                os.unlink(tmp_path)


class TestEnhancedWebConverter:
    """Test enhanced web converter with dynamic content extraction."""
    
    def test_web_converter_initialization(self):
        """Test web converter initialization."""
        converter = WebConverter()
        assert converter.name == "Enhanced MoRAG Web Converter"
        assert 'web' in converter.supported_formats
        assert 'url' in converter.supported_formats
        assert 'html' in converter.supported_formats
    
    @pytest.mark.asyncio
    async def test_url_conversion(self, conversion_options):
        """Test URL conversion."""
        converter = WebConverter()
        
        # Mock web processor
        with patch('src.morag.processors.web.web_processor.process_url') as mock_process:
            mock_result = Mock()
            mock_result.content = "Test web content"
            mock_result.metadata = {
                'url': 'https://example.com',
                'title': 'Test Page',
                'extraction_method': 'MoRAG Web Processor'
            }
            mock_result.success = True
            mock_result.links = []
            mock_result.images = []
            mock_process.return_value = mock_result
            
            result = await converter.convert("https://example.com", conversion_options)
            
            assert result.success
            assert result.content
            assert "Test Page" in result.content
            assert "Test web content" in result.content
    
    @pytest.mark.asyncio
    async def test_local_html_file_conversion(self, conversion_options):
        """Test local HTML file conversion."""
        converter = WebConverter()
        
        html_content = """
        <html>
            <head><title>Test HTML</title></head>
            <body><h1>Test Content</h1><p>This is a test.</p></body>
        </html>
        """
        
        # Mock content converter
        with patch('src.morag.services.content_converter.content_converter.html_to_markdown') as mock_convert:
            mock_result = Mock()
            mock_result.content = "# Test Content\n\nThis is a test."
            mock_result.metadata = {'word_count': 5}
            mock_result.success = True
            mock_convert.return_value = mock_result
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(html_content)
                tmp_path = Path(tmp_file.name)
                
                try:
                    result = await converter.convert(tmp_path, conversion_options)
                    
                    assert result.success
                    assert result.content
                    assert "Test Content" in result.content
                    
                finally:
                    os.unlink(tmp_path)


class TestVideoConverter:
    """Test video converter (already implemented)."""
    
    def test_video_converter_initialization(self):
        """Test video converter initialization."""
        converter = VideoConverter()
        assert converter.name == "MoRAG Video Converter"
        assert 'video' in converter.supported_formats
        assert 'mp4' in converter.supported_formats
    
    @pytest.mark.asyncio
    async def test_video_conversion(self, conversion_options):
        """Test video conversion."""
        converter = VideoConverter()
        
        # Mock video processor
        with patch('src.morag.processors.video.video_processor.process_video') as mock_process:
            mock_result = Mock()
            mock_result.metadata = {
                'filename': 'test.mp4',
                'duration': 120.0,
                'resolution': '1920x1080',
                'fps': 30
            }
            mock_result.summary = "Test video summary"
            mock_result.transcript = "Video transcript content"
            mock_result.keyframes = [
                {'timestamp': 10.0, 'description': 'Scene 1'},
                {'timestamp': 60.0, 'description': 'Scene 2'}
            ]
            mock_process.return_value = mock_result
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                
                try:
                    result = await converter.convert(tmp_path, conversion_options)
                    
                    assert result.success
                    assert result.content
                    assert "Video Analysis" in result.content
                    assert "test" in result.content.lower()
                    
                finally:
                    os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__])
