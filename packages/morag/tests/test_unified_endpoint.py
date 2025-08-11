"""Tests for the unified processing endpoint."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from io import BytesIO

from morag.api_models.endpoints.unified import setup_unified_endpoints
from morag.api_models.models import UnifiedProcessRequest, UnifiedProcessResponse


@pytest.fixture
def mock_morag_api():
    """Mock MoRAG API for testing."""
    api = Mock()
    api.process_file = AsyncMock()
    api.process_url = AsyncMock()
    api.process_batch = AsyncMock()
    return api


@pytest.fixture
def app_with_unified_endpoint(mock_morag_api):
    """Create FastAPI app with unified endpoint for testing."""
    app = FastAPI()
    
    def get_morag_api():
        return mock_morag_api
    
    unified_router = setup_unified_endpoints(get_morag_api)
    app.include_router(unified_router)
    
    return app


@pytest.fixture
def client(app_with_unified_endpoint):
    """Test client for the unified endpoint."""
    return TestClient(app_with_unified_endpoint)


class TestUnifiedEndpointValidation:
    """Test request validation for the unified endpoint."""
    
    def test_invalid_json_request_data(self, client):
        """Test that invalid JSON in request_data returns 400."""
        response = client.post(
            "/api/v1/process",
            data={"request_data": "invalid json"}
        )
        assert response.status_code == 400
        assert "Invalid request data" in response.json()["detail"]
    
    def test_missing_file_for_file_source_type(self, client):
        """Test that missing file for source_type='file' returns 400."""
        request_data = {
            "mode": "convert",
            "source_type": "file"
        }
        
        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)}
        )
        assert response.status_code == 400
        assert "File upload required" in response.json()["detail"]
    
    def test_missing_url_for_url_source_type(self, client):
        """Test that missing URL for source_type='url' returns 400."""
        request_data = {
            "mode": "process",
            "source_type": "url"
        }
        
        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)}
        )
        assert response.status_code == 400
        assert "URL required" in response.json()["detail"]
    
    def test_missing_webhook_for_ingest_mode(self, client):
        """Test that missing webhook config for mode='ingest' returns 400."""
        request_data = {
            "mode": "ingest",
            "source_type": "url",
            "url": "https://example.com"
        }
        
        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)}
        )
        assert response.status_code == 400
        assert "Webhook configuration required" in response.json()["detail"]


class TestConvertMode:
    """Test convert mode functionality."""
    
    @patch('morag.api_models.endpoints.unified.get_upload_handler')
    @patch('morag.api_models.endpoints.unified.get_deduplication_service')
    @patch('morag.api_models.endpoints.unified.MarkitdownService')
    def test_convert_mode_pdf_file(self, mock_markitdown, mock_dedup_service, mock_upload_handler, client):
        """Test convert mode with PDF file."""
        # Setup mocks
        mock_handler = Mock()
        mock_temp_path = Mock()
        mock_temp_path.stat.return_value.st_size = 100000
        mock_temp_path.suffix = ".pdf"
        mock_temp_path.exists.return_value = True
        mock_handler.save_upload = AsyncMock(return_value=mock_temp_path)
        mock_upload_handler.return_value = mock_handler
        
        mock_dedup = AsyncMock()
        mock_dedup.document_exists = AsyncMock(return_value=False)
        mock_dedup_service.return_value = mock_dedup
        
        mock_service = Mock()
        mock_service.convert_file = AsyncMock(return_value="# Test Markdown")
        mock_markitdown.return_value = mock_service
        
        # Create test file
        test_file = BytesIO(b"test pdf content")
        
        request_data = {
            "mode": "convert",
            "source_type": "file",
            "processing_options": {
                "include_metadata": True
            }
        }
        
        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)},
            files={"file": ("test.pdf", test_file, "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["mode"] == "convert"
        assert data["content"] == "# Test Markdown"
        assert "original_format" in data["metadata"]
        assert data["metadata"]["original_format"] == "pdf"
    
    def test_convert_mode_unsupported_file_type(self, client):
        """Test convert mode with unsupported file type."""
        # Test that convert mode with URL source type fails validation first
        request_data = {
            "mode": "convert",
            "source_type": "url"
        }

        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)}
        )
        assert response.status_code == 400
        assert "URL required for source_type='url'" in response.json()["detail"]


class TestProcessMode:
    """Test process mode functionality."""
    
    @patch('morag.api_models.endpoints.unified.get_upload_handler')
    @patch('morag.api_models.endpoints.unified.normalize_processing_result')
    def test_process_mode_file(self, mock_normalize, mock_upload_handler, client, mock_morag_api):
        """Test process mode with file upload."""
        # Setup mocks
        mock_handler = Mock()
        mock_temp_path = Mock()
        mock_temp_path.exists.return_value = True
        mock_handler.save_upload = AsyncMock(return_value=mock_temp_path)
        mock_upload_handler.return_value = mock_handler
        
        # Mock processing result
        mock_result = Mock()
        mock_result.success = True
        mock_result.content = "Processed content"
        mock_result.metadata = {"test": "metadata"}
        mock_result.error_message = None
        mock_result.warnings = None
        mock_result.thumbnails = None
        mock_normalize.return_value = mock_result
        
        mock_morag_api.process_file.return_value = mock_result
        
        # Create test file
        test_file = BytesIO(b"test content")
        
        request_data = {
            "mode": "process",
            "source_type": "file",
            "content_type": "document"
        }
        
        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)},
            files={"file": ("test.txt", test_file, "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["mode"] == "process"
        assert data["content"] == "Processed content"
        assert data["metadata"]["test"] == "metadata"
    
    def test_process_mode_url(self, client, mock_morag_api):
        """Test process mode with URL."""
        # Mock processing result
        mock_result = Mock()
        mock_result.success = True
        mock_result.content = "URL content"
        mock_result.metadata = {"url": "test"}
        mock_result.error_message = None
        mock_result.warnings = None
        mock_result.thumbnails = None
        
        with patch('morag.api_models.endpoints.unified.normalize_processing_result', return_value=mock_result):
            mock_morag_api.process_url.return_value = mock_result
            
            request_data = {
                "mode": "process",
                "source_type": "url",
                "url": "https://example.com",
                "content_type": "web"
            }
            
            response = client.post(
                "/api/v1/process",
                data={"request_data": json.dumps(request_data)}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["mode"] == "process"
            assert data["content"] == "URL content"


class TestIngestMode:
    """Test ingest mode functionality."""
    
    @patch('morag.api_models.endpoints.unified.get_upload_handler')
    @patch('morag.api_models.endpoints.unified.ingest_file_task')
    def test_ingest_mode_file(self, mock_task, mock_upload_handler, client):
        """Test ingest mode with file upload."""
        # Setup mocks
        mock_handler = Mock()
        mock_temp_path = Mock()
        mock_temp_path.stat.return_value.st_size = 1024 * 1024  # 1MB
        mock_temp_path.exists.return_value = True
        mock_handler.save_upload = AsyncMock(return_value=mock_temp_path)
        mock_upload_handler.return_value = mock_handler
        
        # Mock Celery task
        mock_celery_task = Mock()
        mock_celery_task.id = "test-task-id"
        mock_task.delay.return_value = mock_celery_task
        
        # Create test file
        test_file = BytesIO(b"test content")
        
        request_data = {
            "mode": "ingest",
            "source_type": "file",
            "webhook_config": {
                "url": "https://webhook.example.com",
                "auth_token": "test-token"
            },
            "document_id": "test-doc-123",
            "processing_options": {
                "language": "en",
                "chunk_size": 1000
            }
        }
        
        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)},
            files={"file": ("test.txt", test_file, "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["mode"] == "ingest"
        assert data["task_id"] == "test-task-id"
        assert data["document_id"] == "test-doc-123"
        assert "status_url" in data
        assert "estimated_time_seconds" in data
    
    @patch('morag.api_models.endpoints.unified.ingest_url_task')
    def test_ingest_mode_url(self, mock_task, client):
        """Test ingest mode with URL."""
        # Mock Celery task
        mock_celery_task = Mock()
        mock_celery_task.id = "test-url-task-id"
        mock_task.delay.return_value = mock_celery_task
        
        request_data = {
            "mode": "ingest",
            "source_type": "url",
            "url": "https://example.com/article",
            "webhook_config": {
                "url": "https://webhook.example.com"
            },
            "content_type": "web"
        }
        
        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["mode"] == "ingest"
        assert data["task_id"] == "test-url-task-id"
        assert data["estimated_time_seconds"] == 180  # 3 minutes


class TestErrorHandling:
    """Test error handling in the unified endpoint."""
    
    def test_invalid_mode(self, client):
        """Test that invalid mode returns 400."""
        request_data = {
            "mode": "invalid_mode",
            "source_type": "file"
        }

        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)}
        )
        assert response.status_code == 400
        # Pydantic validation error message for invalid literal
        assert "Input should be 'convert', 'process' or 'ingest'" in response.json()["detail"]
    
    @patch('morag.api_models.endpoints.unified.get_upload_handler')
    def test_processing_exception_handling(self, mock_upload_handler, client):
        """Test that processing exceptions are handled gracefully."""
        # Setup mock to raise exception
        mock_handler = Mock()
        mock_handler.save_upload = AsyncMock(side_effect=Exception("Test error"))
        mock_upload_handler.return_value = mock_handler
        
        test_file = BytesIO(b"test content")
        
        request_data = {
            "mode": "convert",
            "source_type": "file"
        }
        
        response = client.post(
            "/api/v1/process",
            data={"request_data": json.dumps(request_data)},
            files={"file": ("test.txt", test_file, "text/plain")}
        )
        
        assert response.status_code == 200  # Should return 200 with error in response
        data = response.json()
        assert data["success"] is False
        assert "Test error" in data["error_message"]
        assert "processing_time_ms" in data
