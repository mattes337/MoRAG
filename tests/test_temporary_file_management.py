"""Tests for temporary file management system."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock

from morag.server import create_app
from morag.services.temporary_file_service import TemporaryFileService


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_service(temp_dir):
    """Create temporary file service for testing."""
    return TemporaryFileService(
        base_dir=str(temp_dir),
        retention_hours=1,  # Short retention for testing
        max_session_size_mb=10,  # Small limit for testing
        cleanup_interval_minutes=1  # Fast cleanup for testing
    )


class TestTemporaryFileService:
    """Test cases for temporary file service."""
    
    @pytest.mark.asyncio
    async def test_create_session_dir(self, temp_service):
        """Test session directory creation."""
        session_id = "test-session-123"
        session_dir = await temp_service.create_session_dir(session_id)
        
        assert session_dir.exists()
        assert session_dir.is_dir()
        assert (session_dir / "artifacts").exists()
        assert (session_dir / "artifacts" / "thumbnails").exists()
        assert (session_dir / "artifacts" / "chunks").exists()
        assert (session_dir / "artifacts" / "analysis").exists()
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_file(self, temp_service):
        """Test storing and retrieving files."""
        session_id = "test-session-456"
        filename = "test.txt"
        content = b"Hello, world!"
        
        # Store file
        metadata = await temp_service.store_file(session_id, filename, content, "text/plain")
        
        assert metadata["filename"] == filename
        assert metadata["size_bytes"] == len(content)
        assert metadata["content_type"] == "text/plain"
        assert metadata["session_id"] == session_id
        
        # Retrieve file
        retrieved_content, retrieved_metadata = await temp_service.get_file(session_id, filename)
        
        assert retrieved_content == content
        assert retrieved_metadata["filename"] == filename
        assert retrieved_metadata["size_bytes"] == len(content)
    
    @pytest.mark.asyncio
    async def test_store_text_file(self, temp_service):
        """Test storing text files."""
        session_id = "test-session-789"
        filename = "document.md"
        content = "# Test Document\n\nThis is a test."
        
        metadata = await temp_service.store_text_file(session_id, filename, content)
        
        assert metadata["filename"] == filename
        assert metadata["content_type"] == "text/markdown"
        assert metadata["size_bytes"] == len(content.encode('utf-8'))
        
        # Retrieve and verify
        retrieved_content, _ = await temp_service.get_file(session_id, filename)
        assert retrieved_content.decode('utf-8') == content
    
    @pytest.mark.asyncio
    async def test_list_session_files(self, temp_service):
        """Test listing files in a session."""
        session_id = "test-session-list"
        
        # Store multiple files
        await temp_service.store_file(session_id, "file1.txt", b"Content 1", "text/plain")
        await temp_service.store_file(session_id, "file2.json", b'{"key": "value"}', "application/json")
        
        # List files
        session_info = await temp_service.list_session_files(session_id)
        
        assert session_info["session_id"] == session_id
        assert len(session_info["files"]) == 2
        assert session_info["total_size_bytes"] > 0
        assert session_info["expires_at"] is not None
        
        # Check file details
        filenames = [f["filename"] for f in session_info["files"]]
        assert "file1.txt" in filenames
        assert "file2.json" in filenames
    
    @pytest.mark.asyncio
    async def test_session_size_limit(self, temp_service):
        """Test session size limit enforcement."""
        session_id = "test-session-limit"
        
        # Create content that exceeds the limit (10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        
        with pytest.raises(ValueError, match="Session size limit exceeded"):
            await temp_service.store_file(session_id, "large.bin", large_content)
    
    @pytest.mark.asyncio
    async def test_invalid_session_id(self, temp_service):
        """Test validation of session IDs."""
        invalid_ids = ["", "test/../hack", "test\\hack", "test<script>"]
        
        for invalid_id in invalid_ids:
            with pytest.raises(ValueError, match="Invalid session ID"):
                temp_service.get_session_dir(invalid_id)
    
    @pytest.mark.asyncio
    async def test_invalid_filename(self, temp_service):
        """Test validation of filenames."""
        session_id = "test-session-filename"
        invalid_filenames = ["", "../hack.txt", "test/hack.txt", "test\\hack.txt"]
        
        for invalid_filename in invalid_filenames:
            with pytest.raises(ValueError, match="Invalid filename"):
                await temp_service.store_file(session_id, invalid_filename, b"content")
    
    @pytest.mark.asyncio
    async def test_delete_session(self, temp_service):
        """Test session deletion."""
        session_id = "test-session-delete"
        
        # Store some files
        await temp_service.store_file(session_id, "file1.txt", b"Content 1")
        await temp_service.store_file(session_id, "file2.txt", b"Content 2")
        
        # Verify files exist
        session_info = await temp_service.list_session_files(session_id)
        assert len(session_info["files"]) == 2
        
        # Delete session
        success = await temp_service.delete_session(session_id)
        assert success is True
        
        # Verify files are gone
        session_info = await temp_service.list_session_files(session_id)
        assert len(session_info["files"]) == 0
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, temp_service):
        """Test handling of non-existent files."""
        session_id = "test-session-notfound"
        
        with pytest.raises(FileNotFoundError):
            await temp_service.get_file(session_id, "nonexistent.txt")
        
        with pytest.raises(FileNotFoundError):
            await temp_service.get_file_path(session_id, "nonexistent.txt")


class TestTemporaryFileEndpoints:
    """Test cases for temporary file REST endpoints."""
    
    @patch('morag.api_models.endpoints.temp_files.get_temp_file_service')
    def test_list_session_files_endpoint(self, mock_get_service, client):
        """Test listing session files via REST API."""
        mock_service = AsyncMock()
        mock_service.list_session_files.return_value = {
            "session_id": "test-123",
            "files": [
                {
                    "filename": "test.txt",
                    "size_bytes": 100,
                    "content_type": "text/plain",
                    "created_at": "2024-01-01T12:00:00Z"
                }
            ],
            "total_size_bytes": 100,
            "expires_at": "2024-01-02T12:00:00Z"
        }
        mock_get_service.return_value = mock_service
        
        response = client.get("/api/files/temp/test-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-123"
        assert len(data["files"]) == 1
        assert data["files"][0]["filename"] == "test.txt"
    
    def test_invalid_session_id_endpoint(self, client):
        """Test endpoint with invalid session ID."""
        response = client.get("/api/files/temp/x")  # Too short
        
        assert response.status_code == 400
        assert "Invalid session ID" in response.json()["detail"]
    
    @patch('morag.api_models.endpoints.temp_files.get_temp_file_service')
    def test_delete_session_endpoint(self, mock_get_service, client):
        """Test session deletion via REST API."""
        mock_service = AsyncMock()
        mock_service.delete_session.return_value = True
        mock_get_service.return_value = mock_service
        
        response = client.delete("/api/files/temp/test-session-delete")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["message"]
    
    @patch('morag.api_models.endpoints.temp_files.get_temp_file_service')
    def test_manual_cleanup_endpoint(self, mock_get_service, client):
        """Test manual cleanup via REST API."""
        mock_service = AsyncMock()
        mock_service.cleanup_expired_sessions.return_value = 3
        mock_get_service.return_value = mock_service
        
        response = client.post("/api/files/temp/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cleaned_sessions"] == 3
    
    @patch('morag.api_models.endpoints.temp_files.get_temp_file_service')
    def test_get_file_info_endpoint(self, mock_get_service, client):
        """Test getting file info via REST API."""
        mock_service = AsyncMock()
        mock_service.get_file.return_value = (
            b"content",
            {
                "filename": "test.txt",
                "size_bytes": 7,
                "content_type": "text/plain",
                "created_at": "2024-01-01T12:00:00Z"
            }
        )
        mock_get_service.return_value = mock_service
        
        response = client.get("/api/files/temp/test-123/test.txt/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["size_bytes"] == 7
        assert data["content_type"] == "text/plain"
    
    @patch('morag.api_models.endpoints.temp_files.get_temp_file_service')
    def test_file_not_found_endpoint(self, mock_get_service, client):
        """Test file not found handling in endpoints."""
        mock_service = AsyncMock()
        mock_service.get_file.side_effect = FileNotFoundError("File not found")
        mock_get_service.return_value = mock_service
        
        response = client.get("/api/files/temp/test-123/nonexistent.txt/info")
        
        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])
