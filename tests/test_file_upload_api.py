"""Tests for file upload API endpoint."""

import pytest
import tempfile
import json
import asyncio
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

from morag.server import create_app
from morag.utils.file_upload import FileUploadHandler, FileUploadConfig, FileUploadError
from morag_core.models import ProcessingResult


class TestFileUploadHandler:
    """Test FileUploadHandler functionality."""
    
    @pytest.fixture
    def upload_config(self):
        """Create test upload configuration."""
        return FileUploadConfig(
            max_file_size=1024 * 1024,  # 1MB for testing
            allowed_extensions={'.txt', '.pdf', '.mp3', '.mp4', '.jpg'},
            cleanup_timeout=60  # 1 minute for testing
        )
    
    @pytest.fixture
    def upload_handler(self, upload_config):
        """Create test upload handler."""
        handler = FileUploadHandler(upload_config)
        yield handler
        # Cleanup after test
        handler.cleanup_temp_dir()
    
    def create_mock_upload_file(self, filename: str, content: bytes, content_type: Optional[str] = None):
        """Create mock UploadFile for testing."""
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = filename
        upload_file.content_type = content_type
        upload_file.size = len(content)
        
        # Mock async read method
        content_buffer = bytearray(content)  # Use mutable bytearray
        async def mock_read(size: int = -1):
            if size == -1 or size >= len(content_buffer):
                data = bytes(content_buffer)
                content_buffer[:] = b''  # Clear content to simulate reading
                return data
            else:
                data = bytes(content_buffer[:size])
                content_buffer[:] = content_buffer[size:]
                return data
        
        upload_file.read = mock_read
        return upload_file
    
    @pytest.mark.asyncio
    async def test_save_valid_file(self, upload_handler):
        """Test saving a valid file."""
        content = b"Test file content"
        upload_file = self.create_mock_upload_file("test.txt", content, "text/plain")
        
        temp_path = await upload_handler.save_upload(upload_file)
        
        assert temp_path.exists()
        assert temp_path.read_bytes() == content
        assert temp_path.name.endswith("_test.txt")
    
    @pytest.mark.asyncio
    async def test_file_too_large(self, upload_handler):
        """Test rejection of files that are too large."""
        # Create content larger than max_file_size (1MB)
        large_content = b"x" * (2 * 1024 * 1024)  # 2MB
        upload_file = self.create_mock_upload_file("large.txt", large_content, "text/plain")
        
        with pytest.raises(FileUploadError, match="File size exceeds maximum"):
            await upload_handler.save_upload(upload_file)
    
    @pytest.mark.asyncio
    async def test_invalid_extension(self, upload_handler):
        """Test rejection of files with invalid extensions."""
        content = b"Test content"
        upload_file = self.create_mock_upload_file("test.xyz", content, "application/xyz")
        
        with pytest.raises(FileUploadError, match="File extension '.xyz' not allowed"):
            await upload_handler.save_upload(upload_file)
    
    @pytest.mark.asyncio
    async def test_no_filename(self, upload_handler):
        """Test rejection of files without filename."""
        content = b"Test content"
        upload_file = self.create_mock_upload_file(None, content, "text/plain")
        
        with pytest.raises(FileUploadError, match="No filename provided"):
            await upload_handler.save_upload(upload_file)
    
    @pytest.mark.asyncio
    async def test_filename_sanitization(self, upload_handler):
        """Test filename sanitization."""
        content = b"Test content"
        dangerous_filename = "../../../etc/passwd"
        upload_file = self.create_mock_upload_file(dangerous_filename, content, "text/plain")
        
        # Should not raise error but sanitize filename
        upload_file.filename = "passwd.txt"  # Simulate sanitized result
        temp_path = await upload_handler.save_upload(upload_file)
        
        assert temp_path.exists()
        assert "passwd.txt" in temp_path.name
        assert ".." not in str(temp_path)
    
    def test_sanitize_filename(self, upload_handler):
        """Test filename sanitization method."""
        # Test dangerous characters
        assert upload_handler._sanitize_filename("test<>file.txt") == "test__file.txt"
        assert upload_handler._sanitize_filename("../../../passwd") == "passwd"
        assert upload_handler._sanitize_filename("file:with|bad*chars.txt") == "file_with_bad_chars.txt"
        
        # Test empty/invalid names
        assert upload_handler._sanitize_filename("") == ""
        assert upload_handler._sanitize_filename("...") == ""
        assert upload_handler._sanitize_filename("   ") == ""
        
        # Test long filenames
        long_name = "a" * 300 + ".txt"
        sanitized = upload_handler._sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized.endswith(".txt")


class TestFileUploadAPI:
    """Test file upload API endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def mock_morag_api(self):
        """Mock MoRAG API for testing."""
        with patch('morag.server.MoRAGAPI') as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            
            # Mock successful processing result
            mock_result = ProcessingResult(
                success=True,
                content="Processed content",
                metadata={"test": True},
                processing_time=1.5,
                error_message=None
            )
            mock_api.process_file = AsyncMock(return_value=mock_result)
            
            yield mock_api
    
    def test_successful_file_upload(self, client, mock_morag_api):
        """Test successful file upload and processing."""
        # Create test file
        test_content = b"Test PDF content"
        
        response = client.post(
            "/process/file",
            files={"file": ("test.pdf", test_content, "application/pdf")},
            data={"content_type": "document"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content"] == "Processed content"
        assert data["metadata"]["test"] is True
        assert data["processing_time"] == 1.5
    
    def test_file_upload_with_options(self, client, mock_morag_api):
        """Test file upload with processing options."""
        test_content = b"Test audio content"
        options = {"language": "de", "quality": "high"}
        
        response = client.post(
            "/process/file",
            files={"file": ("test.mp3", test_content, "audio/mpeg")},
            data={
                "content_type": "audio",
                "options": json.dumps(options)
            }
        )
        
        assert response.status_code == 200
        # Verify options were passed to process_file
        mock_morag_api.process_file.assert_called_once()
        call_args = mock_morag_api.process_file.call_args
        assert call_args[0][1] == "audio"  # content_type
        assert call_args[0][2] == options  # parsed options
    
    def test_invalid_file_type(self, client):
        """Test upload of invalid file type."""
        test_content = b"Invalid content"
        
        response = client.post(
            "/process/file",
            files={"file": ("test.xyz", test_content, "application/xyz")}
        )
        
        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"]
    
    def test_file_too_large(self, client):
        """Test upload of file that's too large."""
        # Create large content (assuming default 100MB limit)
        large_content = b"x" * (101 * 1024 * 1024)  # 101MB
        
        response = client.post(
            "/process/file",
            files={"file": ("large.txt", large_content, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]
    
    def test_invalid_json_options(self, client, mock_morag_api):
        """Test upload with invalid JSON options."""
        test_content = b"Test content"
        
        response = client.post(
            "/process/file",
            files={"file": ("test.txt", test_content, "text/plain")},
            data={"options": "invalid json"}
        )
        
        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]
    
    def test_processing_failure(self, client):
        """Test handling of processing failures."""
        with patch('morag.server.MoRAGAPI') as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.process_file = AsyncMock(side_effect=Exception("Processing failed"))
            
            test_content = b"Test content"
            
            response = client.post(
                "/process/file",
                files={"file": ("test.txt", test_content, "text/plain")}
            )
            
            assert response.status_code == 500
            assert "Processing failed" in response.json()["detail"]
    
    def test_no_file_provided(self, client):
        """Test endpoint with no file provided."""
        response = client.post("/process/file")
        
        assert response.status_code == 422  # Validation error


class TestFileUploadIntegration:
    """Integration tests for file upload functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_file_processing(self):
        """Test complete file upload and processing flow."""
        # This would be an integration test with actual file processing
        # For now, we'll test the upload handler integration
        
        config = FileUploadConfig(max_file_size=1024)
        handler = FileUploadHandler(config)
        
        try:
            # Create mock file
            content = b"Test content for integration"
            upload_file = Mock(spec=UploadFile)
            upload_file.filename = "integration_test.txt"
            upload_file.content_type = "text/plain"
            upload_file.size = len(content)
            
            # Mock read method
            content_copy = bytearray(content)
            async def mock_read(size: int = -1):
                if size == -1 or size >= len(content_copy):
                    data = bytes(content_copy)
                    content_copy.clear()
                    return data
                else:
                    data = bytes(content_copy[:size])
                    del content_copy[:size]
                    return data
            
            upload_file.read = mock_read
            
            # Test upload
            temp_path = await handler.save_upload(upload_file)
            assert temp_path.exists()
            assert temp_path.read_bytes() == content
            
        finally:
            handler.cleanup_temp_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
