"""Integration tests for the ingestion API."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from morag.api.main import create_app


class TestIngestionAPIIntegration:
    """Integration tests for the ingestion API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for API requests."""
        return {"Authorization": "Bearer test-api-key"}

    def test_health_endpoint(self, client):
        """Test health endpoint works."""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    def test_api_documentation_available(self, client):
        """Test that API documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "MoRAG Ingestion Pipeline"

    def test_ingestion_endpoints_require_auth(self, client):
        """Test that ingestion endpoints require authentication."""
        # Test file upload without auth
        response = client.post(
            "/api/v1/ingest/file",
            data={"source_type": "document"},
            files={"file": ("test.pdf", b"test content", "application/pdf")},
        )
        assert response.status_code == 401

        # Test URL ingestion without auth
        response = client.post(
            "/api/v1/ingest/url",
            json={"source_type": "web", "url": "https://example.com"},
        )
        assert response.status_code == 401

    def test_status_endpoints_require_auth(self, client):
        """Test that status endpoints require authentication."""
        response = client.get("/api/v1/status/test-task-id")
        assert response.status_code == 401

        response = client.get("/api/v1/status/")
        assert response.status_code == 401

    @patch("morag.tasks.document_tasks.process_document_task")
    @patch("morag.utils.file_handling.file_handler.save_uploaded_file")
    def test_document_upload_flow(
        self, mock_save_file, mock_task, client, auth_headers
    ):
        """Test complete document upload flow."""
        # Mock file saving
        test_path = Path("/tmp/test.pdf")
        file_info = {
            "original_filename": "test.pdf",
            "mime_type": "application/pdf",
            "file_extension": "pdf",
            "size": 1000,
            "file_path": str(test_path),
            "file_hash": "abcd1234",
        }
        mock_save_file.return_value = (test_path, file_info)

        # Mock task creation
        mock_result = MagicMock()
        mock_result.id = "test-task-123"
        mock_task.delay.return_value = mock_result

        # Test file upload
        response = client.post(
            "/api/v1/ingest/file",
            headers=auth_headers,
            data={
                "source_type": "document",
                "metadata": json.dumps({"priority": 1, "tags": ["test"]}),
            },
            files={"file": ("test.pdf", b"PDF content", "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-123"
        assert data["status"] == "pending"
        assert "File ingestion started" in data["message"]
        assert data["estimated_time"] == 60

        # Verify task was called with correct parameters
        mock_task.delay.assert_called_once()
        call_args = mock_task.delay.call_args
        assert call_args[1]["file_path"] == str(test_path)
        assert call_args[1]["source_type"] == "document"
        assert "metadata" in call_args[1]
        assert call_args[1]["use_docling"] is False

    @patch("morag.tasks.web_tasks.process_web_url")
    def test_web_url_ingestion_flow(self, mock_task, client, auth_headers):
        """Test complete web URL ingestion flow."""
        # Mock task creation
        mock_result = MagicMock()
        mock_result.id = "web-task-456"
        mock_task.delay.return_value = mock_result

        # Test URL ingestion
        response = client.post(
            "/api/v1/ingest/url",
            headers=auth_headers,
            json={
                "source_type": "web",
                "url": "https://example.com/article",
                "metadata": {"category": "news"},
                "webhook_url": "https://webhook.example.com/notify",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "web-task-456"
        assert data["status"] == "pending"
        assert "URL ingestion started" in data["message"]
        assert data["estimated_time"] == 120

        # Verify task was called with correct parameters
        mock_task.delay.assert_called_once()
        call_args = mock_task.delay.call_args
        assert call_args[1]["url"] == "https://example.com/article"
        assert "metadata" in call_args[1]
        assert (
            call_args[1]["metadata"]["webhook_url"]
            == "https://webhook.example.com/notify"
        )

    def test_invalid_url_validation(self, client, auth_headers):
        """Test URL validation."""
        response = client.post(
            "/api/v1/ingest/url",
            headers=auth_headers,
            json={"source_type": "web", "url": "not-a-valid-url"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_unsupported_source_type_for_url(self, client, auth_headers):
        """Test unsupported source type for URL ingestion."""
        response = client.post(
            "/api/v1/ingest/url",
            headers=auth_headers,
            json={
                "source_type": "document",  # Not supported for URL
                "url": "https://example.com",
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    @patch("morag.services.task_manager.task_manager.get_task_status")
    def test_task_status_retrieval(self, mock_get_status, client, auth_headers):
        """Test task status retrieval."""
        from datetime import datetime

        from src.morag.services.task_manager import TaskInfo, TaskStatus

        # Mock task status
        mock_task_info = TaskInfo(
            task_id="test-task-123",
            status=TaskStatus.PROGRESS,
            progress=0.75,
            result=None,
            error=None,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            started_at=datetime(2024, 1, 1, 12, 0, 5),
            completed_at=None,
        )
        mock_get_status.return_value = mock_task_info

        response = client.get("/api/v1/status/test-task-123", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-123"
        assert data["status"] == "PROGRESS"
        assert data["progress"] == 0.75
        assert data["created_at"] == "2024-01-01T12:00:00"
        assert data["started_at"] == "2024-01-01T12:00:05"
        assert data["completed_at"] is None
        assert "estimated_time_remaining" in data

    @patch("morag.services.task_manager.task_manager.get_active_tasks")
    def test_list_active_tasks(self, mock_get_active, client, auth_headers):
        """Test listing active tasks."""
        mock_get_active.return_value = ["task-1", "task-2", "task-3"]

        response = client.get("/api/v1/status/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["active_tasks"]) == 3
        assert "task-1" in data["active_tasks"]

    @patch("morag.services.task_manager.task_manager.get_queue_stats")
    def test_queue_statistics(self, mock_get_stats, client, auth_headers):
        """Test queue statistics endpoint."""
        mock_get_stats.return_value = {
            "pending": 5,
            "active": 2,
            "completed": 100,
            "failed": 3,
        }

        response = client.get("/api/v1/status/stats/queues", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["pending"] == 5
        assert data["active"] == 2
        assert data["completed"] == 100
        assert data["failed"] == 3


if __name__ == "__main__":
    pytest.main([__file__])
