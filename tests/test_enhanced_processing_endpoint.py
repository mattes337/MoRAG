"""Tests for enhanced processing endpoint with webhooks."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from morag.server import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("# Test Document\n\nThis is a test document for enhanced processing.")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_webhook_server():
    """Mock webhook server for testing."""
    return "http://example.com/webhook"


class TestEnhancedProcessingEndpoint:
    """Test cases for enhanced processing endpoint with webhooks."""

    @patch("morag.tasks.enhanced_processing_task.enhanced_process_ingest_task.delay")
    def test_process_with_ingestion_success(
        self, mock_task, client, sample_text_file, mock_webhook_server
    ):
        """Test successful submission of enhanced processing task."""
        # Mock the Celery task
        mock_task.return_value = Mock(id="test-task-123")

        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "webhook_url": mock_webhook_server,
                    "document_id": "test-doc-123",
                    "collection_name": "test_collection",
                    "language": "en",
                },
            )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["task_id"] == "test-task-123"
        assert data["document_id"] == "test-doc-123"
        assert data["estimated_time_seconds"] > 0
        assert data["status_url"] == "/api/v1/status/test-task-123"
        assert mock_webhook_server in data["message"]

        # Verify task was called with correct parameters
        mock_task.assert_called_once()
        call_args = mock_task.call_args
        file_path, options = call_args[0]

        assert options["webhook_url"] == mock_webhook_server
        assert options["document_id"] == "test-doc-123"
        assert options["collection_name"] == "test_collection"
        assert options["language"] == "en"

    @patch("morag.tasks.enhanced_processing_task.enhanced_process_ingest_task.delay")
    def test_process_with_auto_generated_document_id(
        self, mock_task, client, sample_text_file, mock_webhook_server
    ):
        """Test processing with auto-generated document ID."""
        mock_task.return_value = Mock(id="test-task-456")

        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"webhook_url": mock_webhook_server},
            )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["document_id"] is not None
        assert len(data["document_id"]) > 0  # Should be a UUID

        # Verify task was called with generated document ID
        call_args = mock_task.call_args
        file_path, options = call_args[0]
        assert options["document_id"] == data["document_id"]

    @patch("morag.tasks.enhanced_processing_task.enhanced_process_ingest_task.delay")
    def test_process_with_metadata(
        self, mock_task, client, sample_text_file, mock_webhook_server
    ):
        """Test processing with additional metadata."""
        mock_task.return_value = Mock(id="test-task-789")

        metadata = {
            "source": "test",
            "category": "documentation",
            "tags": ["test", "sample"],
        }

        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "webhook_url": mock_webhook_server,
                    "metadata": json.dumps(metadata),
                    "chunk_size": "1000",
                    "chunk_overlap": "100",
                },
            )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True

        # Verify metadata was passed correctly
        call_args = mock_task.call_args
        file_path, options = call_args[0]

        assert options["chunk_size"] == 1000
        assert options["chunk_overlap"] == 100
        assert "source" in options["metadata"]
        assert options["metadata"]["source"] == "test"
        assert options["metadata"]["category"] == "documentation"

    def test_process_missing_webhook_url(self, client, sample_text_file):
        """Test processing fails without webhook URL."""
        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={"document_id": "test-doc"},
            )

        assert response.status_code == 422  # Validation error

    @patch("morag.tasks.enhanced_processing_task.enhanced_process_ingest_task.delay")
    def test_process_with_auth_token(
        self, mock_task, client, sample_text_file, mock_webhook_server
    ):
        """Test processing with webhook authentication token."""
        mock_task.return_value = Mock(id="test-task-auth")

        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "webhook_url": mock_webhook_server,
                    "webhook_auth_token": "secret-token-123",
                },
            )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True

        # Verify auth token was passed
        call_args = mock_task.call_args
        file_path, options = call_args[0]
        assert options["webhook_auth_token"] == "secret-token-123"

    def test_process_invalid_metadata_json(
        self, client, sample_text_file, mock_webhook_server
    ):
        """Test processing with invalid metadata JSON."""
        with open(sample_text_file, "rb") as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "webhook_url": mock_webhook_server,
                    "metadata": "invalid-json-{{",
                },
            )

        # Should still succeed but with empty metadata
        assert response.status_code == 200

    @patch("morag.tasks.enhanced_processing_task.enhanced_process_ingest_task.delay")
    def test_estimated_time_calculation(self, mock_task, client, mock_webhook_server):
        """Test that estimated time is calculated based on file size."""
        mock_task.return_value = Mock(id="test-task-time")

        # Create a larger file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write about 1MB of content
            content = "This is test content. " * 50000
            f.write(content)
            large_file = Path(f.name)

        try:
            with open(large_file, "rb") as f:
                response = client.post(
                    "/api/convert/process-ingest",
                    files={"file": ("large.txt", f, "text/plain")},
                    data={"webhook_url": mock_webhook_server},
                )

            assert response.status_code == 200
            data = response.json()

            # Should have a reasonable estimated time (at least 60 seconds)
            assert data["estimated_time_seconds"] >= 60

        finally:
            if large_file.exists():
                large_file.unlink()


class TestWebhookService:
    """Test cases for webhook service functionality."""

    def test_validate_webhook_url_valid(self):
        """Test webhook URL validation with valid URLs."""
        from morag.services.enhanced_webhook_service import EnhancedWebhookService

        service = EnhancedWebhookService()

        assert service.validate_webhook_url("https://example.com/webhook") is True
        assert (
            service.validate_webhook_url("http://api.example.com/notifications") is True
        )

    def test_validate_webhook_url_localhost_blocked(self):
        """Test webhook URL validation blocks localhost by default."""
        from morag.services.enhanced_webhook_service import EnhancedWebhookService

        service = EnhancedWebhookService()

        assert service.validate_webhook_url("http://localhost:8080/webhook") is False
        assert service.validate_webhook_url("http://127.0.0.1:3000/webhook") is False
        assert service.validate_webhook_url("http://192.168.1.100/webhook") is False

    def test_validate_webhook_url_localhost_allowed(self):
        """Test webhook URL validation allows localhost when enabled."""
        from morag.services.enhanced_webhook_service import EnhancedWebhookService

        service = EnhancedWebhookService()

        assert (
            service.validate_webhook_url(
                "http://localhost:8080/webhook", allow_localhost=True
            )
            is True
        )
        assert (
            service.validate_webhook_url(
                "http://127.0.0.1:3000/webhook", allow_localhost=True
            )
            is True
        )

    def test_validate_webhook_url_invalid_scheme(self):
        """Test webhook URL validation rejects invalid schemes."""
        from morag.services.enhanced_webhook_service import EnhancedWebhookService

        service = EnhancedWebhookService()

        assert service.validate_webhook_url("ftp://example.com/webhook") is False
        assert service.validate_webhook_url("file:///tmp/webhook") is False
        assert service.validate_webhook_url("javascript:alert('xss')") is False


if __name__ == "__main__":
    pytest.main([__file__])
