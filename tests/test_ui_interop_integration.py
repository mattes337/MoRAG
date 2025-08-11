"""Integration tests for UI interoperability endpoints."""

import pytest
import tempfile
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock

from morag.server import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# Integration Test Document\n\nThis is a comprehensive test document for UI interoperability.")
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        # Write minimal PDF content
        f.write(b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n')
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestUIInteropIntegration:
    """Integration tests for complete UI interoperability workflow."""
    
    @patch('morag.api_models.endpoints.conversion.get_deduplication_service')
    def test_markdown_conversion_workflow(self, mock_dedup_service, client, sample_text_file):
        """Test complete markdown conversion workflow."""
        # Mock deduplication service
        mock_service = AsyncMock()
        mock_service.validate_document_id.return_value = True
        mock_service.check_document_exists.return_value = None
        mock_service.generate_document_id.return_value = "test-doc-integration"
        mock_dedup_service.return_value = mock_service
        
        # Test markdown conversion
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "document_id": "test-doc-integration",
                    "include_metadata": "true",
                    "language": "en"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["success"] is True
        assert "Integration Test Document" in data["markdown"]
        assert data["metadata"]["document_id"] == "test-doc-integration"
        assert data["metadata"]["original_format"] == "txt"
        assert data["metadata"]["language"] == "en"
        assert data["processing_time_ms"] > 0
    
    @patch('morag.tasks.enhanced_processing_task.enhanced_process_ingest_task.delay')
    @patch('morag.api_models.endpoints.conversion.get_deduplication_service')
    def test_processing_with_webhooks_workflow(self, mock_dedup_service, mock_task, client, sample_text_file):
        """Test complete processing with webhooks workflow."""
        # Mock deduplication service
        mock_service = AsyncMock()
        mock_service.validate_document_id.return_value = True
        mock_service.check_document_exists.return_value = None
        mock_dedup_service.return_value = mock_service
        
        # Mock Celery task
        mock_task.return_value = Mock(id="integration-task-123")
        
        # Test processing with webhooks
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "webhook_url": "https://api.example.com/webhooks/test",
                    "document_id": "test-doc-processing",
                    "webhook_auth_token": "test-token-123",
                    "collection_name": "integration_tests",
                    "language": "en",
                    "chunking_strategy": "semantic",
                    "chunk_size": "2000",
                    "chunk_overlap": "100",
                    "metadata": json.dumps({"test": "integration", "category": "ui-interop"})
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["success"] is True
        assert data["task_id"] == "integration-task-123"
        assert data["document_id"] == "test-doc-processing"
        assert data["estimated_time_seconds"] > 0
        assert data["status_url"] == "/api/v1/status/integration-task-123"
        assert "webhook" in data["message"]
        
        # Verify task was called with correct parameters
        mock_task.assert_called_once()
        call_args = mock_task.call_args
        file_path, options = call_args[0]
        
        assert options["webhook_url"] == "https://api.example.com/webhooks/test"
        assert options["document_id"] == "test-doc-processing"
        assert options["webhook_auth_token"] == "test-token-123"
        assert options["collection_name"] == "integration_tests"
        assert options["language"] == "en"
        assert options["chunking_strategy"] == "semantic"
        assert options["chunk_size"] == 2000
        assert options["chunk_overlap"] == 100
        assert options["metadata"]["test"] == "integration"
    
    @patch('morag.api_models.endpoints.conversion.get_deduplication_service')
    def test_duplicate_document_handling(self, mock_dedup_service, client, sample_text_file):
        """Test duplicate document handling workflow."""
        # Mock deduplication service to return existing document
        mock_service = AsyncMock()
        mock_service.validate_document_id.return_value = True
        mock_service.check_document_exists.return_value = {
            "document_id": "duplicate-doc",
            "created_at": "2024-01-01T12:00:00Z",
            "status": "completed",
            "facts_count": 10,
            "keywords_count": 5,
            "chunks_count": 3
        }
        mock_service.create_duplicate_error_response.return_value = {
            "error": "duplicate_document",
            "message": "Document with ID 'duplicate-doc' already exists",
            "existing_document": {
                "document_id": "duplicate-doc",
                "status": "completed"
            },
            "options": {
                "update_url": "/api/process/update/duplicate-doc",
                "delete_url": "/api/process/delete/duplicate-doc"
            }
        }
        mock_dedup_service.return_value = mock_service
        
        # Test markdown conversion with existing document
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")},
                data={"document_id": "duplicate-doc"}
            )
        
        # Should return existing document info
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["metadata"]["status"] == "already_exists"
        assert data["metadata"]["document_id"] == "duplicate-doc"
        
        # Test processing with existing document (should return 409)
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "webhook_url": "https://api.example.com/webhooks/test",
                    "document_id": "duplicate-doc"
                }
            )
        
        assert response.status_code == 409
        error_data = response.json()["detail"]
        assert error_data["error"] == "duplicate_document"
        assert "duplicate-doc" in error_data["message"]
    
    @patch('morag.api_models.endpoints.temp_files.get_temp_file_service')
    def test_temporary_file_management_workflow(self, mock_get_service, client):
        """Test complete temporary file management workflow."""
        # Mock temporary file service
        mock_service = AsyncMock()
        
        # Mock session files listing
        mock_service.list_session_files.return_value = {
            "session_id": "integration-session-123",
            "files": [
                {
                    "filename": "original.txt",
                    "size_bytes": 1024,
                    "content_type": "text/plain",
                    "created_at": "2024-01-15T10:30:00Z"
                },
                {
                    "filename": "markdown.md",
                    "size_bytes": 2048,
                    "content_type": "text/markdown",
                    "created_at": "2024-01-15T10:31:00Z"
                },
                {
                    "filename": "metadata.json",
                    "size_bytes": 512,
                    "content_type": "application/json",
                    "created_at": "2024-01-15T10:31:30Z"
                }
            ],
            "total_size_bytes": 3584,
            "expires_at": "2024-01-16T10:30:00Z"
        }
        
        # Mock file info retrieval
        mock_service.get_file.return_value = (
            b'{"test": "metadata"}',
            {
                "filename": "metadata.json",
                "size_bytes": 512,
                "content_type": "application/json",
                "created_at": "2024-01-15T10:31:30Z"
            }
        )
        
        # Mock session deletion
        mock_service.delete_session.return_value = True
        
        # Mock cleanup
        mock_service.cleanup_expired_sessions.return_value = 2
        
        mock_get_service.return_value = mock_service
        
        # Test listing session files
        response = client.get("/api/files/temp/integration-session-123")
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "integration-session-123"
        assert len(data["files"]) == 3
        assert data["total_size_bytes"] == 3584
        assert data["expires_at"] is not None
        
        # Verify file types
        filenames = [f["filename"] for f in data["files"]]
        assert "original.txt" in filenames
        assert "markdown.md" in filenames
        assert "metadata.json" in filenames
        
        # Test getting file info
        response = client.get("/api/files/temp/integration-session-123/metadata.json/info")
        assert response.status_code == 200
        info_data = response.json()
        
        assert info_data["filename"] == "metadata.json"
        assert info_data["content_type"] == "application/json"
        assert info_data["size_bytes"] == 512
        
        # Test session deletion
        response = client.delete("/api/files/temp/integration-session-123")
        assert response.status_code == 200
        delete_data = response.json()
        
        assert delete_data["success"] is True
        assert "deleted successfully" in delete_data["message"]
        
        # Test manual cleanup
        response = client.post("/api/files/temp/cleanup")
        assert response.status_code == 200
        cleanup_data = response.json()
        
        assert cleanup_data["success"] is True
        assert cleanup_data["cleaned_sessions"] == 2
    
    def test_api_documentation_endpoints(self, client):
        """Test that API documentation endpoints are accessible."""
        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert openapi_spec["info"]["title"] == "MoRAG API"
        assert openapi_spec["info"]["version"] == "1.0.0"
        
        # Verify UI interop endpoints are documented
        paths = openapi_spec["paths"]
        assert "/api/convert/markdown" in paths
        assert "/api/convert/process-ingest" in paths
        assert "/api/files/temp/{session_id}" in paths
        
        # Verify security schemes
        assert "securitySchemes" in openapi_spec["components"]
        assert "BearerAuth" in openapi_spec["components"]["securitySchemes"]
        
        # Test Swagger UI (should return HTML)
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Test ReDoc (should return HTML)
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_error_handling_consistency(self, client):
        """Test consistent error handling across endpoints."""
        # Test invalid session ID
        response = client.get("/api/files/temp/x")  # Too short
        assert response.status_code == 400
        error_data = response.json()
        assert "detail" in error_data
        assert "Invalid session ID" in error_data["detail"]
        
        # Test missing required fields
        response = client.post("/api/convert/process-ingest")  # No file
        assert response.status_code == 422  # Validation error
        
        # Test file not found
        response = client.get("/api/files/temp/nonexistent-session/file.txt/info")
        assert response.status_code == 404
        
        # Test invalid webhook URL format (would need proper validation)
        response = client.post(
            "/api/convert/process-ingest",
            files={"file": ("test.txt", b"content", "text/plain")},
            data={"webhook_url": "invalid-url"}
        )
        # Should return validation error or 500 with proper error message
        assert response.status_code in [422, 500]


if __name__ == "__main__":
    pytest.main([__file__])
