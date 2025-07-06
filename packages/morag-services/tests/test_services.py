"""Tests for MoRAG Services.

This module contains tests for the MoRAGServices class.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from morag_services.services import MoRAGServices, ServiceConfig, ContentType, ProcessingResult

@pytest.fixture
def services():
    """Create a MoRAGServices instance for testing."""
    # Create services with mocked specialized services
    services = MoRAGServices()
    
    # Mock all specialized services
    services.document_service = AsyncMock()
    services.audio_service = AsyncMock()
    services.video_service = AsyncMock()
    services.image_service = AsyncMock()
    services.embedding_service = AsyncMock()
    services.web_service = AsyncMock()
    services.youtube_service = AsyncMock()
    
    return services

class TestContentTypeDetection:
    """Tests for content type detection."""
    
    def test_detect_document(self, services):
        """Test document detection."""
        assert services.detect_content_type("test.pdf") == ContentType.DOCUMENT
        assert services.detect_content_type("test.docx") == ContentType.DOCUMENT
        assert services.detect_content_type("test.txt") == ContentType.DOCUMENT
    
    def test_detect_audio(self, services):
        """Test audio detection."""
        assert services.detect_content_type("test.mp3") == ContentType.AUDIO
        assert services.detect_content_type("test.wav") == ContentType.AUDIO
        assert services.detect_content_type("test.flac") == ContentType.AUDIO
    
    def test_detect_video(self, services):
        """Test video detection."""
        assert services.detect_content_type("test.mp4") == ContentType.VIDEO
        assert services.detect_content_type("test.avi") == ContentType.VIDEO
        assert services.detect_content_type("test.mkv") == ContentType.VIDEO
    
    def test_detect_image(self, services):
        """Test image detection."""
        assert services.detect_content_type("test.jpg") == ContentType.IMAGE
        assert services.detect_content_type("test.png") == ContentType.IMAGE
        assert services.detect_content_type("test.gif") == ContentType.IMAGE
    
    def test_detect_web(self, services):
        """Test web URL detection."""
        assert services.detect_content_type("http://example.com") == ContentType.WEB
        assert services.detect_content_type("https://example.com/page") == ContentType.WEB
    
    def test_detect_youtube(self, services):
        """Test YouTube URL detection."""
        assert services.detect_content_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == ContentType.YOUTUBE
        assert services.detect_content_type("https://youtu.be/dQw4w9WgXcQ") == ContentType.YOUTUBE
    
    def test_detect_unknown(self, services):
        """Test unknown content type detection."""
        assert services.detect_content_type("test.unknown") == ContentType.UNKNOWN

class TestProcessContent:
    """Tests for content processing."""
    
    @pytest.mark.asyncio
    async def test_process_document(self, services):
        """Test document processing."""
        # Mock document service response
        mock_result = MagicMock()
        mock_result.text = "Document text"
        mock_result.metadata = {"pages": 5}
        mock_result.extracted_files = [Path("extracted.txt")]
        mock_result.processing_time = 1.5
        mock_result.success = True
        mock_result.error_message = None
        
        services.document_service.process_document.return_value = mock_result
        
        # Process document
        result = await services.process_document("test.pdf")
        
        # Verify result
        assert result.content_type == ContentType.DOCUMENT
        assert result.content_path == "test.pdf"
        assert result.text_content == "Document text"
        assert result.metadata == {"pages": 5}
        assert result.extracted_files == ["extracted.txt"]
        assert result.processing_time == 1.5
        assert result.success is True
        assert result.error_message is None
        
        # Verify service call
        services.document_service.process_document.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_audio(self, services):
        """Test audio processing."""
        # Mock audio service response
        mock_result = MagicMock()
        mock_result.transcription = "Audio transcription"
        mock_result.metadata = {"duration": 120}
        mock_result.processing_time = 2.0
        mock_result.success = True
        mock_result.error_message = None
        
        services.audio_service.process_audio.return_value = mock_result
        
        # Process audio
        result = await services.process_audio("test.mp3")
        
        # Verify result
        assert result.content_type == ContentType.AUDIO
        assert result.content_path == "test.mp3"
        assert result.text_content == "Audio transcription"
        assert result.metadata == {"duration": 120}
        assert result.processing_time == 2.0
        assert result.success is True
        assert result.error_message is None
        
        # Verify service call
        services.audio_service.process_audio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_video(self, services):
        """Test video processing."""
        # Mock video service response for process_file method
        mock_json_result = {
            "success": True,
            "processing_time": 5.0,
            "content": {
                "title": "Test Video",
                "topics": [
                    {
                        "title": "Topic 1",
                        "sentences": [
                            {"text": "Video transcription"}
                        ]
                    }
                ]
            },
            "metadata": {"duration": 300, "resolution": "1080p"},
            "thumbnails": ["thumbnail.jpg"],
            "keyframes": [],
            "error": None
        }

        # Configure the mock to return the dictionary directly
        async def mock_process_file(*args, **kwargs):
            return mock_json_result

        services.video_service.process_file = mock_process_file
        services.video_service.config = MagicMock()
        services.video_service.config.generate_thumbnails = True

        # Process video
        result = await services.process_video("test.mp4")

        # Verify result
        assert result.content_type == ContentType.VIDEO
        assert result.content_path == "test.mp4"
        assert "Test Video" in result.text_content
        assert "Video transcription" in result.text_content
        assert result.metadata == {"duration": 300, "resolution": "1080p"}
        assert result.extracted_files == ["thumbnail.jpg"]
        assert result.processing_time == 5.0
        assert result.success is True
        assert result.error_message is None

        # Test passed - video processing works correctly
    
    @pytest.mark.asyncio
    async def test_process_image(self, services):
        """Test image processing."""
        # Mock image service response
        mock_result = MagicMock()
        mock_result.text = "Image text"
        mock_result.metadata = {"width": 800, "height": 600}
        mock_result.processing_time = 1.0
        mock_result.success = True
        mock_result.error_message = None
        
        services.image_service.process_image.return_value = mock_result
        
        # Process image
        result = await services.process_image("test.jpg")
        
        # Verify result
        assert result.content_type == ContentType.IMAGE
        assert result.content_path == "test.jpg"
        assert result.text_content == "Image text"
        assert result.metadata == {"width": 800, "height": 600}
        assert result.processing_time == 1.0
        assert result.success is True
        assert result.error_message is None
        
        # Verify service call
        services.image_service.process_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_url(self, services):
        """Test URL processing."""
        # Mock web service response
        mock_result = MagicMock()
        mock_result.content = "Web content"
        mock_result.metadata = {"title": "Example Page"}
        mock_result.processing_time = 0.5
        mock_result.success = True
        mock_result.error_message = None
        
        services.web_service.process_url.return_value = mock_result
        
        # Process URL
        result = await services.process_url("https://example.com")
        
        # Verify result
        assert result.content_type == ContentType.WEB
        assert result.content_url == "https://example.com"
        assert result.text_content == "Web content"
        assert result.metadata == {"title": "Example Page"}
        assert result.processing_time == 0.5
        assert result.success is True
        assert result.error_message is None
        
        # Verify service call
        services.web_service.process_url.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_youtube(self, services):
        """Test YouTube processing."""
        # Mock YouTube metadata
        mock_metadata = MagicMock()
        mock_metadata.id = "dQw4w9WgXcQ"
        mock_metadata.title = "Test Video"
        mock_metadata.description = "Test Description"
        mock_metadata.uploader = "Test Uploader"
        mock_metadata.upload_date = "20210101"
        mock_metadata.duration = 213
        mock_metadata.view_count = 1000
        mock_metadata.like_count = 100
        mock_metadata.comment_count = 10
        mock_metadata.tags = ["test", "video"]
        mock_metadata.categories = ["Music"]
        mock_metadata.thumbnail_url = "https://example.com/thumbnail.jpg"
        mock_metadata.webpage_url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        mock_metadata.channel_id = "UC123"
        mock_metadata.channel_url = "https://youtube.com/channel/UC123"
        
        # Mock YouTube service response
        mock_result = MagicMock()
        mock_result.metadata = mock_metadata
        mock_result.video_path = Path("video.mp4")
        mock_result.audio_path = Path("audio.mp3")
        mock_result.subtitle_paths = [Path("subtitles.vtt")]
        mock_result.thumbnail_paths = [Path("thumbnail.jpg")]
        mock_result.processing_time = 10.0
        mock_result.success = True
        mock_result.error_message = None
        
        services.youtube_service.process_video.return_value = mock_result
        
        # Process YouTube URL
        result = await services.process_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        
        # Verify result
        assert result.content_type == ContentType.YOUTUBE
        assert result.content_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert result.metadata["id"] == "dQw4w9WgXcQ"
        assert result.metadata["title"] == "Test Video"
        assert "video.mp4" in result.extracted_files
        assert "audio.mp3" in result.extracted_files
        assert "subtitles.vtt" in result.extracted_files
        assert "thumbnail.jpg" in result.extracted_files
        assert result.processing_time == 10.0
        assert result.success is True
        assert result.error_message is None
        
        # Verify service call
        services.youtube_service.process_video.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_content_auto_detect(self, services):
        """Test content processing with auto-detection."""
        # Mock document service
        mock_doc_result = MagicMock()
        mock_doc_result.text = "Document text"
        mock_doc_result.metadata = {"pages": 5}
        mock_doc_result.success = True
        services.document_service.process_document.return_value = mock_doc_result
        
        # Mock web service
        mock_web_result = MagicMock()
        mock_web_result.content = "Web content"
        mock_web_result.metadata = {"title": "Example Page"}
        mock_web_result.success = True
        services.web_service.process_url.return_value = mock_web_result
        
        # Process with auto-detection
        doc_result = await services.process_content("test.pdf")
        web_result = await services.process_content("https://example.com")
        
        # Verify document result
        assert doc_result.content_type == ContentType.DOCUMENT
        assert doc_result.text_content == "Document text"
        
        # Verify web result
        assert web_result.content_type == ContentType.WEB
        assert web_result.text_content == "Web content"
    
    @pytest.mark.asyncio
    async def test_process_content_error_handling(self, services):
        """Test error handling in content processing."""
        # Mock service to raise exception
        services.document_service.process_document.side_effect = Exception("Test error")
        
        # Process with error
        result = await services.process_content("test.pdf")
        
        # Verify error handling
        assert result.content_type == ContentType.DOCUMENT
        assert result.content_path == "test.pdf"
        assert result.success is False
        assert result.error_message == "Test error"

@pytest.mark.asyncio
class TestBatchProcessing:
    """Tests for batch processing."""
    
    async def test_process_batch(self, services):
        """Test batch processing."""
        # Mock document service
        mock_doc_result = MagicMock()
        mock_doc_result.text = "Document text"
        mock_doc_result.success = True
        services.document_service.process_document.return_value = mock_doc_result
        
        # Mock web service
        mock_web_result = MagicMock()
        mock_web_result.content = "Web content"
        mock_web_result.success = True
        services.web_service.process_url.return_value = mock_web_result
        
        # Process batch
        items = ["test.pdf", "https://example.com"]
        results = await services.process_batch(items)
        
        # Verify results
        assert len(results) == 2
        assert results["test.pdf"].content_type == ContentType.DOCUMENT
        assert results["https://example.com"].content_type == ContentType.WEB
        assert results["test.pdf"].success is True
        assert results["https://example.com"].success is True
    
    async def test_process_batch_with_errors(self, services):
        """Test batch processing with errors."""
        # Mock document service to succeed
        mock_doc_result = MagicMock()
        mock_doc_result.text = "Document text"
        mock_doc_result.success = True
        services.document_service.process_document.return_value = mock_doc_result
        
        # Mock web service to fail
        services.web_service.process_url.side_effect = Exception("Web error")
        
        # Process batch
        items = ["test.pdf", "https://example.com"]
        results = await services.process_batch(items)
        
        # Verify results
        assert len(results) == 2
        assert results["test.pdf"].success is True
        assert results["https://example.com"].success is False
        assert results["https://example.com"].error_message == "Web error"

@pytest.mark.asyncio
async def test_generate_embeddings(services):
    """Test embedding generation."""
    # Mock embedding service
    services.embedding_service.generate_embeddings.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    # Generate embeddings
    texts = ["Text 1", "Text 2"]
    embeddings = await services.generate_embeddings(texts)
    
    # Verify results
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    
    # Verify service call
    services.embedding_service.generate_embeddings.assert_called_once_with(
        texts,
        config=None
    )