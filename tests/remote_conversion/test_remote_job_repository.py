"""Tests for remote job repository."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from morag.repositories.remote_job_repository import RemoteJobRepository
from morag_core.models.remote_job import RemoteJob


class TestRemoteJobRepository:
    """Test cases for RemoteJobRepository."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def repository(self, temp_dir):
        """Create a repository instance with temporary directory."""
        return RemoteJobRepository(data_dir=temp_dir)

    def test_create_job(self, repository):
        """Test creating a new job."""
        job = repository.create_job(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"webhook_url": "http://example.com/webhook"}
        )
        
        assert job.id is not None
        assert job.ingestion_task_id == "test-task-123"
        assert job.source_file_path == "/tmp/test.mp3"
        assert job.content_type == "audio"
        assert job.status == "pending"
        assert job.timeout_at is not None
        
        # Verify file was created
        job_file = Path(repository.data_dir) / "pending" / f"{job.id}.json"
        assert job_file.exists()

    def test_get_job(self, repository):
        """Test retrieving a job."""
        # Create a job
        created_job = repository.create_job(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        
        # Retrieve the job
        retrieved_job = repository.get_job(created_job.id)
        
        assert retrieved_job is not None
        assert retrieved_job.id == created_job.id
        assert retrieved_job.ingestion_task_id == created_job.ingestion_task_id
        assert retrieved_job.source_file_path == created_job.source_file_path
        assert retrieved_job.content_type == created_job.content_type

    def test_get_nonexistent_job(self, repository):
        """Test retrieving a non-existent job."""
        job = repository.get_job("nonexistent-id")
        assert job is None

    def test_update_job_status_change(self, repository):
        """Test updating a job with status change."""
        # Create a job
        job = repository.create_job(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        
        # Update job status
        job.status = "processing"
        job.worker_id = "worker-123"
        job.started_at = datetime.utcnow()
        
        success = repository.update_job(job)
        assert success
        
        # Verify job moved to processing directory
        pending_file = Path(repository.data_dir) / "pending" / f"{job.id}.json"
        processing_file = Path(repository.data_dir) / "processing" / f"{job.id}.json"
        
        assert not pending_file.exists()
        assert processing_file.exists()
        
        # Verify updated data
        retrieved_job = repository.get_job(job.id)
        assert retrieved_job.status == "processing"
        assert retrieved_job.worker_id == "worker-123"
        assert retrieved_job.started_at is not None

    def test_poll_available_jobs(self, repository):
        """Test polling for available jobs."""
        # Create multiple jobs
        job1 = repository.create_job(
            ingestion_task_id="task-1",
            source_file_path="/tmp/test1.mp3",
            content_type="audio",
            task_options={}
        )
        
        job2 = repository.create_job(
            ingestion_task_id="task-2",
            source_file_path="/tmp/test2.mp4",
            content_type="video",
            task_options={}
        )
        
        job3 = repository.create_job(
            ingestion_task_id="task-3",
            source_file_path="/tmp/test3.pdf",
            content_type="document",
            task_options={}
        )
        
        # Poll for audio jobs
        audio_jobs = repository.poll_available_jobs("worker-1", ["audio"], max_jobs=2)
        assert len(audio_jobs) == 1
        assert audio_jobs[0].content_type == "audio"
        assert audio_jobs[0].status == "processing"
        assert audio_jobs[0].worker_id == "worker-1"
        
        # Poll for video jobs
        video_jobs = repository.poll_available_jobs("worker-2", ["video"], max_jobs=2)
        assert len(video_jobs) == 1
        assert video_jobs[0].content_type == "video"
        
        # Poll for audio/video jobs (should get video since audio is already claimed)
        av_jobs = repository.poll_available_jobs("worker-3", ["audio", "video"], max_jobs=2)
        assert len(av_jobs) == 0  # video is already claimed
        
        # Poll for document jobs (not supported by workers)
        doc_jobs = repository.poll_available_jobs("worker-4", ["audio", "video"], max_jobs=2)
        assert len(doc_jobs) == 0

    def test_submit_result_success(self, repository):
        """Test submitting successful result."""
        # Create and claim a job
        job = repository.create_job(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        
        # Claim the job
        jobs = repository.poll_available_jobs("worker-1", ["audio"], max_jobs=1)
        assert len(jobs) == 1
        
        # Submit successful result
        result_job = repository.submit_result(
            job_id=job.id,
            success=True,
            content="Processed audio content",
            metadata={"duration": 120.5},
            processing_time=45.2
        )
        
        assert result_job is not None
        assert result_job.status == "completed"
        assert result_job.completed_at is not None
        assert result_job.result_data["content"] == "Processed audio content"
        assert result_job.result_data["metadata"]["duration"] == 120.5
        assert result_job.result_data["processing_time"] == 45.2
        
        # Verify job moved to completed directory
        completed_file = Path(repository.data_dir) / "completed" / f"{job.id}.json"
        assert completed_file.exists()

    def test_submit_result_failure(self, repository):
        """Test submitting failed result."""
        # Create and claim a job
        job = repository.create_job(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={}
        )
        
        jobs = repository.poll_available_jobs("worker-1", ["audio"], max_jobs=1)
        assert len(jobs) == 1
        
        # Submit failed result
        result_job = repository.submit_result(
            job_id=job.id,
            success=False,
            error_message="Processing failed due to corrupted file"
        )
        
        assert result_job is not None
        assert result_job.status == "failed"
        assert result_job.error_message == "Processing failed due to corrupted file"
        assert result_job.retry_count == 1
        
        # Verify job moved to failed directory
        failed_file = Path(repository.data_dir) / "failed" / f"{job.id}.json"
        assert failed_file.exists()

    def test_find_jobs_by_status(self, repository):
        """Test finding jobs by status."""
        # Create jobs with different statuses
        job1 = repository.create_job("task-1", "/tmp/test1.mp3", "audio", {})
        job2 = repository.create_job("task-2", "/tmp/test2.mp3", "audio", {})
        
        # Claim one job
        repository.poll_available_jobs("worker-1", ["audio"], max_jobs=1)
        
        # Find pending jobs
        pending_jobs = repository.find_jobs_by_status("pending")
        assert len(pending_jobs) == 1
        
        # Find processing jobs
        processing_jobs = repository.find_jobs_by_status("processing")
        assert len(processing_jobs) == 1

    def test_find_jobs_by_content_type(self, repository):
        """Test finding jobs by content type."""
        # Create jobs with different content types
        repository.create_job("task-1", "/tmp/test1.mp3", "audio", {})
        repository.create_job("task-2", "/tmp/test2.mp4", "video", {})
        repository.create_job("task-3", "/tmp/test3.mp3", "audio", {})
        
        # Find audio jobs
        audio_jobs = repository.find_jobs_by_content_type("audio")
        assert len(audio_jobs) == 2
        
        # Find video jobs
        video_jobs = repository.find_jobs_by_content_type("video")
        assert len(video_jobs) == 1

    def test_cleanup_expired_jobs(self, repository):
        """Test cleaning up expired jobs."""
        # Create a job and make it expired
        job = repository.create_job("task-1", "/tmp/test1.mp3", "audio", {})
        job.timeout_at = datetime.utcnow() - timedelta(hours=1)  # Expired
        repository.update_job(job)
        
        # Run cleanup
        expired_count = repository.cleanup_expired_jobs()
        assert expired_count == 1
        
        # Verify job moved to timeout directory
        timeout_file = Path(repository.data_dir) / "timeout" / f"{job.id}.json"
        assert timeout_file.exists()
        
        # Verify job status updated
        updated_job = repository.get_job(job.id)
        assert updated_job.status == "timeout"
        assert updated_job.error_message == "Job exceeded maximum processing time"

    def test_cleanup_old_jobs(self, repository):
        """Test cleaning up old completed jobs."""
        # Create a completed job
        job = repository.create_job("task-1", "/tmp/test1.mp3", "audio", {})
        job.status = "completed"
        repository.update_job(job)
        
        # Manually set file modification time to be old
        completed_file = Path(repository.data_dir) / "completed" / f"{job.id}.json"
        old_time = datetime.now().timestamp() - (8 * 24 * 60 * 60)  # 8 days ago
        import os
        os.utime(completed_file, (old_time, old_time))
        
        # Run cleanup (7 days old)
        cleaned_count = repository.cleanup_old_jobs(days_old=7)
        assert cleaned_count == 1
        
        # Verify file was deleted
        assert not completed_file.exists()

    def test_delete_job(self, repository):
        """Test deleting a job."""
        # Create a job
        job = repository.create_job("task-1", "/tmp/test1.mp3", "audio", {})
        
        # Verify job exists
        assert repository.get_job(job.id) is not None
        
        # Delete job
        success = repository.delete_job(job.id)
        assert success
        
        # Verify job no longer exists
        assert repository.get_job(job.id) is None
