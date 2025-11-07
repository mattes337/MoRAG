"""Tests for remote job model."""

from datetime import datetime, timedelta

import pytest
from morag_core.models.remote_job import RemoteJob


class TestRemoteJob:
    """Test cases for RemoteJob model."""

    def test_create_new_job(self):
        """Test creating a new remote job."""
        job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"webhook_url": "http://example.com/webhook"},
        )

        assert job.id is not None
        assert job.ingestion_task_id == "test-task-123"
        assert job.source_file_path == "/tmp/test.mp3"
        assert job.content_type == "audio"
        assert job.task_options == {"webhook_url": "http://example.com/webhook"}
        assert job.status == "pending"
        assert job.worker_id is None
        assert job.created_at is not None
        assert job.retry_count == 0
        assert job.max_retries == 3

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original_job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={"webhook_url": "http://example.com/webhook"},
        )

        # Convert to dict
        job_dict = original_job.to_dict()

        # Verify dict structure
        assert job_dict["id"] == original_job.id
        assert job_dict["ingestion_task_id"] == "test-task-123"
        assert job_dict["source_file_path"] == "/tmp/test.mp3"
        assert job_dict["content_type"] == "audio"
        assert job_dict["status"] == "pending"
        assert isinstance(job_dict["created_at"], str)  # Should be ISO string

        # Convert back to object
        restored_job = RemoteJob.from_dict(job_dict)

        # Verify restoration
        assert restored_job.id == original_job.id
        assert restored_job.ingestion_task_id == original_job.ingestion_task_id
        assert restored_job.source_file_path == original_job.source_file_path
        assert restored_job.content_type == original_job.content_type
        assert restored_job.status == original_job.status
        assert restored_job.created_at == original_job.created_at

    def test_can_retry(self):
        """Test retry logic."""
        job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={},
        )

        # Initially can't retry (not failed)
        assert not job.can_retry()

        # After failure, can retry
        job.status = "failed"
        assert job.can_retry()

        # After max retries, can't retry
        job.retry_count = 3
        assert not job.can_retry()

        # Timeout status can be retried
        job.retry_count = 0
        job.status = "timeout"
        assert job.can_retry()

    def test_is_expired(self):
        """Test expiration logic."""
        job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={},
        )

        # No timeout set
        assert not job.is_expired

        # Future timeout
        job.timeout_at = datetime.utcnow() + timedelta(hours=1)
        assert not job.is_expired

        # Past timeout
        job.timeout_at = datetime.utcnow() - timedelta(hours=1)
        assert job.is_expired

    def test_processing_duration(self):
        """Test processing duration calculation."""
        job = RemoteJob.create_new(
            ingestion_task_id="test-task-123",
            source_file_path="/tmp/test.mp3",
            content_type="audio",
            task_options={},
        )

        # No start time
        assert job.processing_duration == 0.0

        # With start time, no end time (ongoing)
        job.started_at = datetime.utcnow() - timedelta(seconds=30)
        duration = job.processing_duration
        assert 25 <= duration <= 35  # Allow some variance

        # With both start and end time
        job.completed_at = job.started_at + timedelta(seconds=60)
        assert job.processing_duration == 60.0
