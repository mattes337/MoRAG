"""Tests for ingestion API functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

from morag.api.main import create_app
from morag.utils.file_handling import file_handler
from morag.api.models import SourceType

class TestFileHandling:
    """Test file handling utilities."""
    
    def test_supported_mime_types(self):
        """Test that all expected MIME types are supported."""
        expected_types = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'audio/mpeg',
            'audio/wav',
            'video/mp4',
            'image/jpeg',
            'image/png'
        ]
        
        for mime_type in expected_types:
            assert mime_type in file_handler.supported_mimes
    
    def test_file_extension_validation(self):
        """Test file extension validation for different source types."""
        # Document extensions
        assert file_handler._is_valid_for_source_type('pdf', 'document')
        assert file_handler._is_valid_for_source_type('docx', 'document')
        assert not file_handler._is_valid_for_source_type('mp3', 'document')
        
        # Audio extensions
        assert file_handler._is_valid_for_source_type('mp3', 'audio')
        assert file_handler._is_valid_for_source_type('wav', 'audio')
        assert not file_handler._is_valid_for_source_type('pdf', 'audio')
        
        # Video extensions
        assert file_handler._is_valid_for_source_type('mp4', 'video')
        assert file_handler._is_valid_for_source_type('mov', 'video')
        assert not file_handler._is_valid_for_source_type('jpg', 'video')
        
        # Image extensions
        assert file_handler._is_valid_for_source_type('jpg', 'image')
        assert file_handler._is_valid_for_source_type('png', 'image')
        assert not file_handler._is_valid_for_source_type('mp4', 'image')
    
    def test_max_file_sizes(self):
        """Test maximum file size limits."""
        assert file_handler._get_max_size_for_type('document') == 100 * 1024 * 1024
        assert file_handler._get_max_size_for_type('audio') == 500 * 1024 * 1024
        assert file_handler._get_max_size_for_type('video') == 2 * 1024 * 1024 * 1024
        assert file_handler._get_max_size_for_type('image') == 50 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_file_validation_success(self):
        """Test successful file validation."""
        # Create mock upload file
        content = b"Test PDF content"
        upload_file = UploadFile(
            filename="test.pdf",
            file=io.BytesIO(content),
            size=len(content)
        )
        upload_file.content_type = "application/pdf"
        
        extension, file_info = file_handler.validate_file(upload_file, 'document')
        
        assert extension == 'pdf'
        assert file_info['original_filename'] == 'test.pdf'
        assert file_info['mime_type'] == 'application/pdf'
        assert file_info['file_extension'] == 'pdf'
        assert file_info['size'] == len(content)
    
    @pytest.mark.asyncio
    async def test_file_validation_unsupported_type(self):
        """Test file validation with unsupported type."""
        upload_file = UploadFile(
            filename="test.xyz",
            file=io.BytesIO(b"content"),
            size=7
        )
        upload_file.content_type = "application/xyz"
        
        with pytest.raises(Exception):  # Should raise ValidationError
            file_handler.validate_file(upload_file, 'document')
    
    @pytest.mark.asyncio
    async def test_file_validation_wrong_source_type(self):
        """Test file validation with wrong source type."""
        upload_file = UploadFile(
            filename="test.pdf",
            file=io.BytesIO(b"content"),
            size=7
        )
        upload_file.content_type = "application/pdf"
        
        with pytest.raises(Exception):  # Should raise ValidationError
            file_handler.validate_file(upload_file, 'audio')

class TestIngestionAPI:
    """Test ingestion API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for API requests."""
        return {"Authorization": "Bearer test-api-key"}
    
    def test_file_upload_endpoint_structure(self, client, auth_headers):
        """Test file upload endpoint structure (without actual processing)."""
        # Create a small test file
        test_content = b"Test document content"
        
        with patch('morag.tasks.document_tasks.process_document_task') as mock_task:
            # Mock the Celery task
            mock_result = MagicMock()
            mock_result.id = "test-task-id"
            mock_task.delay.return_value = mock_result
            
            with patch('morag.utils.file_handling.file_handler.save_uploaded_file') as mock_save:
                mock_save.return_value = (Path("/tmp/test.pdf"), {
                    'original_filename': 'test.pdf',
                    'mime_type': 'application/pdf',
                    'file_extension': 'pdf',
                    'size': len(test_content),
                    'file_path': '/tmp/test.pdf',
                    'file_hash': 'abcd1234'
                })
                
                response = client.post(
                    "/api/v1/ingest/file",
                    headers=auth_headers,
                    data={
                        "source_type": "document",
                        "metadata": json.dumps({"test": True})
                    },
                    files={"file": ("test.pdf", test_content, "application/pdf")}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert "File ingestion started" in data["message"]
        assert "estimated_time" in data
    
    def test_url_ingestion_endpoint(self, client, auth_headers):
        """Test URL ingestion endpoint."""
        with patch('morag.tasks.web_tasks.process_web_task') as mock_task:
            mock_result = MagicMock()
            mock_result.id = "test-task-id"
            mock_task.delay.return_value = mock_result
            
            response = client.post(
                "/api/v1/ingest/url",
                headers=auth_headers,
                json={
                    "source_type": "web",
                    "url": "https://example.com",
                    "metadata": {"test": True}
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert "URL ingestion started" in data["message"]
    
    def test_youtube_ingestion_endpoint(self, client, auth_headers):
        """Test YouTube URL ingestion."""
        with patch('morag.tasks.video_tasks.process_video_task') as mock_task:
            mock_result = MagicMock()
            mock_result.id = "test-task-id"
            mock_task.delay.return_value = mock_result
            
            response = client.post(
                "/api/v1/ingest/url",
                headers=auth_headers,
                json={
                    "source_type": "youtube",
                    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
    
    def test_batch_ingestion_endpoint(self, client, auth_headers):
        """Test batch ingestion endpoint."""
        with patch('morag.tasks.web_tasks.process_web_task') as mock_web_task:
            with patch('morag.tasks.video_tasks.process_video_task') as mock_video_task:
                mock_result1 = MagicMock()
                mock_result1.id = "task-1"
                mock_result2 = MagicMock()
                mock_result2.id = "task-2"
                
                mock_web_task.delay.return_value = mock_result1
                mock_video_task.delay.return_value = mock_result2
                
                response = client.post(
                    "/api/v1/ingest/batch",
                    headers=auth_headers,
                    json={
                        "items": [
                            {
                                "source_type": "web",
                                "url": "https://example.com"
                            },
                            {
                                "source_type": "youtube",
                                "url": "https://www.youtube.com/watch?v=test"
                            }
                        ]
                    }
                )
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert len(data["task_ids"]) == 2
        assert data["total_items"] == 2
    
    def test_authentication_required(self, client):
        """Test that authentication is required."""
        response = client.post(
            "/api/v1/ingest/url",
            json={
                "source_type": "web",
                "url": "https://example.com"
            }
        )
        
        assert response.status_code == 401
    
    def test_invalid_url_format(self, client, auth_headers):
        """Test invalid URL format validation."""
        response = client.post(
            "/api/v1/ingest/url",
            headers=auth_headers,
            json={
                "source_type": "web",
                "url": "not-a-valid-url"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_unsupported_source_type_for_url(self, client, auth_headers):
        """Test unsupported source type for URL ingestion."""
        response = client.post(
            "/api/v1/ingest/url",
            headers=auth_headers,
            json={
                "source_type": "document",  # Not supported for URL
                "url": "https://example.com"
            }
        )
        
        assert response.status_code == 500  # Should be handled as ValidationError

class TestStatusAPI:
    """Test status API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for API requests."""
        return {"Authorization": "Bearer test-api-key"}
    
    def test_task_status_endpoint(self, client, auth_headers):
        """Test task status endpoint."""
        with patch('morag.services.task_manager.task_manager.get_task_status') as mock_status:
            from morag.services.task_manager import TaskInfo, TaskStatus
            from datetime import datetime
            
            mock_task_info = TaskInfo(
                task_id="test-task-id",
                status=TaskStatus.PROGRESS,
                progress=0.5,
                result=None,
                error=None,
                created_at=datetime.now(),
                started_at=datetime.now(),
                completed_at=None
            )
            mock_status.return_value = mock_task_info
            
            response = client.get(
                "/api/v1/status/test-task-id",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-id"
        assert data["status"] == "PROGRESS"
        assert data["progress"] == 0.5
    
    def test_list_active_tasks(self, client, auth_headers):
        """Test listing active tasks."""
        with patch('morag.services.task_manager.task_manager.get_active_tasks') as mock_active:
            mock_active.return_value = ["task-1", "task-2", "task-3"]
            
            response = client.get(
                "/api/v1/status/",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["active_tasks"]) == 3
    
    def test_queue_stats(self, client, auth_headers):
        """Test queue statistics endpoint."""
        with patch('morag.services.task_manager.task_manager.get_queue_stats') as mock_stats:
            mock_stats.return_value = {
                "pending": 5,
                "active": 2,
                "completed": 100,
                "failed": 3
            }
            
            response = client.get(
                "/api/v1/status/stats/queues",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["pending"] == 5
        assert data["active"] == 2
        assert data["completed"] == 100
        assert data["failed"] == 3

if __name__ == "__main__":
    pytest.main([__file__])
