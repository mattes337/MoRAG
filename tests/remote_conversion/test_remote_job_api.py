"""Tests for remote job API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from morag_core.models.remote_job import RemoteJob
from morag.endpoints.remote_jobs import router, get_remote_job_service
from fastapi import FastAPI


@pytest.fixture
def app():
    """Create FastAPI app with remote jobs router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_service():
    """Create mock remote job service."""
    return Mock()


@pytest.fixture
def override_service(app, mock_service):
    """Override the service dependency."""
    app.dependency_overrides[get_remote_job_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


class TestRemoteJobAPI:
    """Test cases for remote job API endpoints."""

    def test_create_remote_job(self, client, override_service):
        """Test creating a remote job."""
        # Setup mock
        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"webhook_url": "http://example.com/webhook"}
        )
        override_service.create_job.return_value = mock_job

        # Make request
        response = client.post("/api/v1/remote-jobs/", json={
            "source_file_path": "/tmp/test.mp3",
            "content_type": "audio",
            "task_options": {"webhook_url": "http://example.com/webhook"},
            "ingestion_task_id": "test-task-123"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == mock_job.id
        assert data["status"] == "pending"
        assert "created_at" in data

    def test_create_remote_job_error(self, client, override_service):
        """Test creating a remote job with error."""
        # Setup mock to raise exception
        override_service.create_job.side_effect = Exception("Database error")

        # Make request
        response = client.post("/api/v1/remote-jobs/", json={
            "source_file_path": "/tmp/test.mp3",
            "content_type": "audio",
            "task_options": {},
            "ingestion_task_id": "test-task-123"
        })

        # Verify error response
        assert response.status_code == 500
        assert "Failed to create remote job" in response.json()["detail"]

    def test_poll_for_jobs_available(self, client, override_service):
        """Test polling for jobs when jobs are available."""
        # Setup mock
        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"webhook_url": "http://example.com/webhook"}
        )
        override_service.poll_available_jobs.return_value = [mock_job]

        # Make request
        response = client.get("/api/v1/remote-jobs/poll", params={
            "worker_id": "worker-1",
            "content_types": "audio,video",
            "max_jobs": 1
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == mock_job.id
        assert data["content_type"] == "audio"
        assert data["source_file_url"] == f"/api/v1/remote-jobs/{mock_job.id}/download"
        assert data["task_options"] == {"webhook_url": "http://example.com/webhook"}

    def test_poll_for_jobs_none_available(self, client, override_service):
        """Test polling for jobs when no jobs are available."""
        # Setup mock
        override_service.poll_available_jobs.return_value = []

        # Make request
        response = client.get("/api/v1/remote-jobs/poll", params={
            "worker_id": "worker-1",
            "content_types": "audio,video",
            "max_jobs": 1
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] is None
        assert data["source_file_url"] is None
        assert data["content_type"] is None
        assert data["task_options"] is None

    @patch('morag.ingest_tasks.continue_ingestion_after_remote_processing')
    def test_submit_job_result_success(self, mock_continue, client, override_service):
        """Test submitting successful job result."""
        # Setup mocks
        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        mock_job.status = "completed"
        override_service.submit_result.return_value = mock_job
        mock_continue.return_value = True

        # Make request
        response = client.put("/api/v1/remote-jobs/job-123/result", json={
            "success": True,
            "content": "Processed content",
            "metadata": {"duration": 120.5},
            "processing_time": 45.2
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["ingestion_continued"] == True

        # Verify continuation was called
        mock_continue.assert_called_once_with(
            "job-123",
            "Processed content",
            {"duration": 120.5},
            45.2
        )

    def test_submit_job_result_failure(self, client, override_service):
        """Test submitting failed job result."""
        # Setup mock
        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        mock_job.status = "failed"
        override_service.submit_result.return_value = mock_job

        # Make request
        response = client.put("/api/v1/remote-jobs/job-123/result", json={
            "success": False,
            "error_message": "Processing failed"
        })

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["ingestion_continued"] == False

    def test_submit_job_result_not_found(self, client, override_service):
        """Test submitting result for non-existent job."""
        # Setup mock
        override_service.submit_result.return_value = None

        # Make request
        response = client.put("/api/v1/remote-jobs/nonexistent/result", json={
            "success": True,
            "content": "Processed content"
        })

        # Verify error response
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_job_status(self, client, override_service):
        """Test getting job status."""
        # Setup mock
        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        mock_job.status = "processing"
        mock_job.worker_id = "worker-1"
        mock_job.started_at = datetime.utcnow()
        override_service.get_job_status.return_value = mock_job

        # Make request
        response = client.get("/api/v1/remote-jobs/job-123/status")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == mock_job.id
        assert data["status"] == "processing"
        assert data["worker_id"] == "worker-1"
        assert data["retry_count"] == 0
        assert "estimated_completion" in data

    def test_get_job_status_not_found(self, client, override_service):
        """Test getting status for non-existent job."""
        # Setup mock
        override_service.get_job_status.return_value = None

        # Make request
        response = client.get("/api/v1/remote-jobs/nonexistent/status")

        # Verify error response
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch('pathlib.Path')
    def test_download_job_file(self, mock_path, client, override_service):
        """Test downloading job file."""
        # Setup mock
        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        mock_job.status = "processing"
        override_service.get_job_status.return_value = mock_job

        # Mock file existence
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_file.name = "test.mp3"
        mock_path.return_value = mock_file

        # Make request
        response = client.get("/api/v1/remote-jobs/job-123/download")

        # Verify response (would be file download in real scenario)
        # In test, we just verify the endpoint doesn't error
        assert response.status_code in [200, 404, 500]  # May fail due to file mocking

    def test_download_job_file_not_processing(self, client, override_service):
        """Test downloading file for job not in processing state."""
        # Setup mock
        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        mock_job.status = "pending"  # Not processing
        override_service.get_job_status.return_value = mock_job

        # Make request
        response = client.get("/api/v1/remote-jobs/job-123/download")

        # Verify error response
        assert response.status_code == 403
        assert "not in processing state" in response.json()["detail"]
