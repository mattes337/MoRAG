"""Tests for document deduplication service."""

import pytest
import tempfile
import json
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, Mock

from morag.server import create_app
from morag.services.document_deduplication_service import DocumentDeduplicationService


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# Test Document\n\nThis is a test document for deduplication testing.")
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_vector_storage():
    """Mock vector storage for testing."""
    storage = AsyncMock()
    storage.search_by_metadata.return_value = []
    return storage


@pytest.fixture
def dedup_service(mock_vector_storage):
    """Create deduplication service with mocked storage."""
    return DocumentDeduplicationService(mock_vector_storage)


class TestDocumentDeduplicationService:
    """Test cases for document deduplication service."""
    
    def test_validate_document_id_valid(self, dedup_service):
        """Test document ID validation with valid IDs."""
        assert dedup_service.validate_document_id("test-doc-123") is True
        assert dedup_service.validate_document_id("user_document_456") is True
        assert dedup_service.validate_document_id("simple-id") is True
        assert dedup_service.validate_document_id("a" * 255) is True  # Max length
    
    def test_validate_document_id_invalid(self, dedup_service):
        """Test document ID validation with invalid IDs."""
        assert dedup_service.validate_document_id("") is False
        assert dedup_service.validate_document_id(None) is False
        assert dedup_service.validate_document_id("a" * 256) is False  # Too long
        assert dedup_service.validate_document_id("doc<script>") is False  # Invalid chars
        assert dedup_service.validate_document_id("doc\nwith\nnewlines") is False
        assert dedup_service.validate_document_id("doc'with'quotes") is False
    
    def test_generate_document_id(self, dedup_service):
        """Test document ID generation."""
        doc_id = dedup_service.generate_document_id()
        assert doc_id is not None
        assert len(doc_id) > 0
        assert dedup_service.validate_document_id(doc_id) is True
        
        # Should generate different IDs
        doc_id2 = dedup_service.generate_document_id()
        assert doc_id != doc_id2
    
    @pytest.mark.asyncio
    async def test_check_document_exists_not_found(self, dedup_service, mock_vector_storage):
        """Test checking for non-existent document."""
        mock_vector_storage.search_by_metadata.return_value = []
        
        result = await dedup_service.check_document_exists("non-existent-doc")
        assert result is None
        
        mock_vector_storage.search_by_metadata.assert_called_once_with(
            filter_dict={"document_id": "non-existent-doc"},
            limit=1
        )
    
    @pytest.mark.asyncio
    async def test_check_document_exists_found(self, dedup_service, mock_vector_storage):
        """Test checking for existing document."""
        mock_point = Mock()
        mock_point.payload = {
            "document_id": "existing-doc",
            "created_at": "2024-01-01T12:00:00Z",
            "original_filename": "test.txt",
            "content_type": "text/plain",
            "file_size_bytes": 1024
        }
        mock_vector_storage.search_by_metadata.return_value = [mock_point]
        
        result = await dedup_service.check_document_exists("existing-doc")
        
        assert result is not None
        assert result["document_id"] == "existing-doc"
        assert result["status"] == "completed"
        assert result["metadata"]["original_filename"] == "test.txt"
    
    def test_create_duplicate_error_response(self, dedup_service):
        """Test creation of duplicate error response."""
        existing_doc = {
            "document_id": "test-doc-123",
            "created_at": "2024-01-01T12:00:00Z",
            "status": "completed",
            "facts_count": 10,
            "keywords_count": 5,
            "chunks_count": 3
        }
        
        response = dedup_service.create_duplicate_error_response(existing_doc)
        
        assert response["error"] == "duplicate_document"
        assert "test-doc-123" in response["message"]
        assert response["existing_document"]["document_id"] == "test-doc-123"
        assert response["existing_document"]["facts_count"] == 10
        assert "update_url" in response["options"]
        assert "version_url" in response["options"]
        assert "delete_url" in response["options"]


class TestMarkdownConversionWithDeduplication:
    """Test markdown conversion endpoint with deduplication."""
    
    @patch('morag.api_models.endpoints.conversion.get_deduplication_service')
    def test_convert_with_new_document_id(self, mock_get_service, client, sample_text_file):
        """Test conversion with new document ID."""
        mock_service = AsyncMock()
        mock_service.validate_document_id.return_value = True
        mock_service.check_document_exists.return_value = None
        mock_get_service.return_value = mock_service
        
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")},
                data={"document_id": "test-doc-123"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["metadata"]["document_id"] == "test-doc-123"
    
    @patch('morag.api_models.endpoints.conversion.get_deduplication_service')
    def test_convert_with_existing_document_id(self, mock_get_service, client, sample_text_file):
        """Test conversion with existing document ID."""
        mock_service = AsyncMock()
        mock_service.validate_document_id.return_value = True
        mock_service.check_document_exists.return_value = {
            "document_id": "existing-doc",
            "created_at": "2024-01-01T12:00:00Z",
            "status": "completed",
            "metadata": {
                "content_type": "text/plain",
                "file_size_bytes": 1024
            }
        }
        mock_get_service.return_value = mock_service
        
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")},
                data={"document_id": "existing-doc"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["metadata"]["status"] == "already_exists"
        assert data["metadata"]["document_id"] == "existing-doc"
    
    @patch('morag.api_models.endpoints.conversion.get_deduplication_service')
    def test_convert_with_invalid_document_id(self, mock_get_service, client, sample_text_file):
        """Test conversion with invalid document ID."""
        mock_service = Mock()
        mock_service.validate_document_id.return_value = False
        mock_get_service.return_value = mock_service
        
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")},
                data={"document_id": "invalid<id>"}
            )
        
        assert response.status_code == 400
        assert "Invalid document ID format" in response.json()["detail"]
    
    @patch('morag.api_models.endpoints.conversion.get_deduplication_service')
    def test_convert_without_document_id(self, mock_get_service, client, sample_text_file):
        """Test conversion without document ID (auto-generation)."""
        mock_service = AsyncMock()
        mock_service.generate_document_id.return_value = "auto-generated-123"
        mock_get_service.return_value = mock_service
        
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/markdown",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["metadata"]["document_id"] == "auto-generated-123"


class TestEnhancedProcessingWithDeduplication:
    """Test enhanced processing endpoint with deduplication."""
    
    @patch('morag.tasks.enhanced_processing_task.enhanced_process_ingest_task.delay')
    @patch('morag.api_models.endpoints.conversion.get_deduplication_service')
    def test_process_with_duplicate_document_id(self, mock_get_service, mock_task, client, sample_text_file):
        """Test processing with duplicate document ID returns 409 error."""
        mock_service = AsyncMock()
        mock_service.validate_document_id.return_value = True
        mock_service.check_document_exists.return_value = {
            "document_id": "duplicate-doc",
            "created_at": "2024-01-01T12:00:00Z",
            "status": "completed",
            "facts_count": 10,
            "keywords_count": 5
        }
        mock_service.create_duplicate_error_response.return_value = {
            "error": "duplicate_document",
            "message": "Document with ID 'duplicate-doc' already exists"
        }
        mock_get_service.return_value = mock_service
        
        with open(sample_text_file, 'rb') as f:
            response = client.post(
                "/api/convert/process-ingest",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "webhook_url": "http://example.com/webhook",
                    "document_id": "duplicate-doc"
                }
            )
        
        assert response.status_code == 409
        assert "duplicate_document" in response.json()["detail"]["error"]
        
        # Task should not be called for duplicates
        mock_task.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
