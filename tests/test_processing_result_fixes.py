"""Tests for processing result fixes and thumbnail functionality."""

import pytest
import json
import base64
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from morag.server import create_app, normalize_processing_result, encode_thumbnails_to_base64
from morag_services.services import ProcessingResult as ServicesProcessingResult
from morag_core.models.config import ProcessingResult as CoreProcessingResult


class TestProcessingResultNormalization:
    """Test processing result normalization fixes."""
    
    def test_normalize_services_processing_result(self):
        """Test normalizing ProcessingResult from services (Pydantic model)."""
        # Create a services ProcessingResult (Pydantic model)
        services_result = ServicesProcessingResult(
            content_type="video",
            text_content="Test video content",
            metadata={"duration": 30.0},
            processing_time=5.0,
            success=True
        )
        
        # Normalize it
        normalized = normalize_processing_result(services_result)
        
        # Should be converted to CoreProcessingResult with content field
        assert isinstance(normalized, CoreProcessingResult)
        assert normalized.content == "Test video content"
        assert normalized.success is True
        assert normalized.processing_time == 5.0
        assert normalized.metadata["duration"] == 30.0
    
    def test_normalize_core_processing_result_with_content(self):
        """Test normalizing ProcessingResult that already has content field."""
        # Create a core ProcessingResult that already has content
        core_result = CoreProcessingResult(
            success=True,
            task_id="test",
            source_type="document",
            content="Existing content",
            processing_time=2.0
        )
        
        # Normalize it
        normalized = normalize_processing_result(core_result)
        
        # Should return as-is since it already has content
        assert normalized is core_result
        assert normalized.content == "Existing content"
    
    def test_normalize_result_without_text_content(self):
        """Test normalizing result without text_content."""
        # Create a mock result without text_content or raw_result
        mock_result = Mock()
        mock_result.success = True
        mock_result.processing_time = 1.0
        mock_result.metadata = {}
        mock_result.error_message = None
        # Explicitly set these to None to prevent Mock from creating them
        mock_result.text_content = None
        mock_result.raw_result = None

        # Normalize it
        normalized = normalize_processing_result(mock_result)

        # Should create CoreProcessingResult with empty content
        assert isinstance(normalized, CoreProcessingResult)
        assert normalized.content == ""
        assert normalized.success is True


class TestThumbnailEncoding:
    """Test thumbnail encoding functionality."""
    
    def test_encode_thumbnails_to_base64(self, tmp_path):
        """Test encoding thumbnail files to base64."""
        # Create test image files
        test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        
        thumb1 = tmp_path / "thumb1.png"
        thumb2 = tmp_path / "thumb2.jpg"
        
        thumb1.write_bytes(test_image_data)
        thumb2.write_bytes(test_image_data)
        
        # Encode thumbnails
        encoded = encode_thumbnails_to_base64([str(thumb1), str(thumb2)])
        
        assert len(encoded) == 2
        assert encoded[0].startswith("data:image/png;base64,")
        assert encoded[1].startswith("data:image/jpeg;base64,")
        
        # Verify base64 encoding
        for encoded_thumb in encoded:
            base64_part = encoded_thumb.split(',')[1]
            decoded = base64.b64decode(base64_part)
            assert decoded == test_image_data
    
    def test_encode_nonexistent_thumbnails(self):
        """Test encoding thumbnails that don't exist."""
        # Try to encode non-existent files
        encoded = encode_thumbnails_to_base64(["/nonexistent/thumb1.jpg", "/nonexistent/thumb2.png"])
        
        # Should return empty list (failed encodings are skipped)
        assert encoded == []


class TestFileUploadWithThumbnails:
    """Test file upload API with thumbnail functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def mock_video_processing_result(self, tmp_path):
        """Create mock video processing result with thumbnails."""
        # Create test thumbnail files
        test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        
        thumb1 = tmp_path / "thumb1.jpg"
        thumb2 = tmp_path / "thumb2.jpg"
        thumb1.write_bytes(test_image_data)
        thumb2.write_bytes(test_image_data)
        
        # Create mock result with thumbnails
        mock_result = ServicesProcessingResult(
            content_type="video",
            text_content="Video processing completed",
            metadata={"duration": 30.0},
            processing_time=5.0,
            success=True,
            raw_result={
                "thumbnails": [str(thumb1), str(thumb2)],
                "content": "Video processing completed"
            }
        )
        
        return mock_result
    
    def test_video_upload_with_thumbnails_enabled(self, client, mock_video_processing_result):
        """Test video upload with thumbnails enabled."""
        with patch('morag.server.MoRAGAPI') as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.process_file = AsyncMock(return_value=mock_video_processing_result)
            
            test_content = b"Test video content"
            options = {"include_thumbnails": True}
            
            response = client.post(
                "/process/file",
                files={"file": ("test.mp4", test_content, "video/mp4")},
                data={
                    "content_type": "video",
                    "options": json.dumps(options)
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["content"] == "Video processing completed"
            assert data["thumbnails"] is not None
            assert len(data["thumbnails"]) == 2
            assert all(thumb.startswith("data:image/jpeg;base64,") for thumb in data["thumbnails"])
    
    def test_video_upload_with_thumbnails_disabled(self, client, mock_video_processing_result):
        """Test video upload with thumbnails disabled (default)."""
        with patch('morag.server.MoRAGAPI') as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.process_file = AsyncMock(return_value=mock_video_processing_result)
            
            test_content = b"Test video content"
            options = {"include_thumbnails": False}  # Explicitly disabled
            
            response = client.post(
                "/process/file",
                files={"file": ("test.mp4", test_content, "video/mp4")},
                data={
                    "content_type": "video",
                    "options": json.dumps(options)
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["content"] == "Video processing completed"
            assert data["thumbnails"] is None  # No thumbnails when disabled
    
    def test_video_upload_without_thumbnail_option(self, client, mock_video_processing_result):
        """Test video upload without thumbnail option (should default to disabled)."""
        with patch('morag.server.MoRAGAPI') as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.process_file = AsyncMock(return_value=mock_video_processing_result)
            
            test_content = b"Test video content"
            
            response = client.post(
                "/process/file",
                files={"file": ("test.mp4", test_content, "video/mp4")},
                data={"content_type": "video"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["content"] == "Video processing completed"
            assert data["thumbnails"] is None  # No thumbnails by default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
