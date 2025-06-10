"""Tests for remote job service."""

import pytest
from unittest.mock import Mock, patch
from morag.services.remote_job_service import RemoteJobService
from morag.models.remote_job_api import CreateRemoteJobRequest, SubmitResultRequest
from morag_core.models.remote_job import RemoteJob


class TestRemoteJobService:
    """Test cases for RemoteJobService."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository."""
        return Mock()

    @pytest.fixture
    def service(self, mock_repository):
        """Create a service instance with mock repository."""
        return RemoteJobService(repository=mock_repository)

    def test_create_job(self, service, mock_repository):
        """Test creating a job through service."""
        # Setup mock
        mock_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"webhook_url": "http://example.com/webhook"}
        )
        mock_repository.create_job.return_value = mock_job
        
        # Create request
        request = CreateRemoteJobRequest(
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"webhook_url": "http://example.com/webhook"},
            ingestion_task_id="test-task-123"
        )
        
        # Call service
        result = service.create_job(request)
        
        # Verify
        assert result == mock_job
        mock_repository.create_job.assert_called_once_with(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"webhook_url": "http://example.com/webhook"}
        )

    def test_poll_available_jobs(self, service, mock_repository):
        """Test polling for available jobs."""
        # Setup mock
        mock_jobs = [
            RemoteJob.create_new("task-1", "/tmp/test1.mp3", "audio", {}),
            RemoteJob.create_new("task-2", "/tmp/test2.mp3", "audio", {})
        ]
        mock_repository.poll_available_jobs.return_value = mock_jobs
        
        # Call service
        result = service.poll_available_jobs("worker-1", ["audio"], max_jobs=2)
        
        # Verify
        assert result == mock_jobs
        mock_repository.poll_available_jobs.assert_called_once_with(
            "worker-1", ["audio"], 2
        )

    def test_submit_result(self, service, mock_repository):
        """Test submitting job result."""
        # Setup mock
        mock_job = RemoteJob.create_new("task-1", "/tmp/test.mp3", "audio", {})
        mock_job.status = "completed"
        mock_repository.submit_result.return_value = mock_job
        
        # Create request
        request = SubmitResultRequest(
            success=True,
            content="Processed content",
            metadata={"duration": 120.5},
            processing_time=45.2
        )
        
        # Call service
        result = service.submit_result("job-123", request)
        
        # Verify
        assert result == mock_job
        mock_repository.submit_result.assert_called_once_with(
            job_id="job-123",
            success=True,
            content="Processed content",
            metadata={"duration": 120.5},
            error_message=None,
            processing_time=45.2
        )

    def test_get_job_status(self, service, mock_repository):
        """Test getting job status."""
        # Setup mock
        mock_job = RemoteJob.create_new("task-1", "/tmp/test.mp3", "audio", {})
        mock_repository.get_job.return_value = mock_job
        
        # Call service
        result = service.get_job_status("job-123")
        
        # Verify
        assert result == mock_job
        mock_repository.get_job.assert_called_once_with("job-123")

    def test_cleanup_expired_jobs(self, service, mock_repository):
        """Test cleaning up expired jobs."""
        # Setup mock
        mock_repository.cleanup_expired_jobs.return_value = 3
        
        # Call service
        result = service.cleanup_expired_jobs()
        
        # Verify
        assert result == 3
        mock_repository.cleanup_expired_jobs.assert_called_once()

    def test_cleanup_old_jobs(self, service, mock_repository):
        """Test cleaning up old jobs."""
        # Setup mock
        mock_repository.cleanup_old_jobs.return_value = 5
        
        # Call service
        result = service.cleanup_old_jobs(days_old=10)
        
        # Verify
        assert result == 5
        mock_repository.cleanup_old_jobs.assert_called_once_with(10)
